"""基础加载、PEFT 注入与精度设置的唯一构建入口。"""

from collections import defaultdict
import inspect
import re

import torch
from torch import nn

from openpi.tempo_lora.config import model_config


def require_patch():
    import transformers
    from transformers.models.gemma import modeling_gemma
    from transformers.models.siglip import modeling_siglip

    if transformers.__version__ != "4.53.2":
        raise RuntimeError(f"TEMPO 补丁要求 transformers 4.53.2，当前 {transformers.__version__}")
    if not hasattr(modeling_siglip.SiglipEncoderLayer, "_temporal_attention") or "adarms_cond" not in inspect.signature(modeling_gemma.GemmaModel.forward).parameters:
        raise RuntimeError("Transformers 补丁未安装，请执行 trained_model/scripts/setup.sh")


def inject_lora(model, cfg):
    from peft.tuners.lora.layer import Linear

    model.requires_grad_(False)
    rules = [
        (r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))", "vlm"),
        (r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))", "expert"),
        (r"paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)", "siglip"),
    ]
    matched = defaultdict(list)
    for path, layer in list(model.named_modules()):
        for pattern, family in rules:
            match = re.fullmatch(pattern, path)
            if match is None:
                continue
            if family == "siglip" and (int(match.group(1)) + 1) % cfg.temporal_attn_period:
                continue
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"LoRA 命中非线性层 {path}")
            rank = getattr(cfg, f"lora_{family}_rank")
            alpha = getattr(cfg, f"lora_{family}_alpha")
            parent_path, name = path.rsplit(".", 1)
            # 直接使用 PEFT Linear，保留联合前向访问的 .weight、.in_features 和模块层级。
            replacement = Linear(layer, adapter_name="default", r=rank, lora_alpha=rank if alpha is None else alpha,
                                 lora_dropout=cfg.lora_dropout, init_lora_weights=True)
            setattr(model.get_submodule(parent_path), name, replacement)
            matched[family].append(path)
    for family in ("vlm", "expert", "siglip"):
        if not matched[family]:
            raise ValueError(f"LoRA 未命中 {family}，请核查实际模块树")
    for name in ("sam2_fusion", "action_history_vlm_tokens", "action_history_cond_path",
                 "action_in_proj", "action_out_proj"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(True)
    if cfg.train_time_projections:
        model.time_mlp_in.requires_grad_(True)
        model.time_mlp_out.requires_grad_(True)
    if getattr(model, "sam2_fusion", None) is not None:
        # 双零初始化会让 gate 与 out_proj 同时得不到梯度，仅输出层保留零初始化。
        with torch.no_grad():
            model.sam2_fusion.gate.fill_(1)
    return dict(matched)


def _cast_linear_input(module, args):
    # 不对整个主干 no_grad，保持冻结层到 adapter 和新模块的反向传播路径。
    return (args[0].to(module.weight.dtype), *args[1:])


def set_precision(model, dtype):
    if dtype == "float32":
        model.float()
    else:
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # LayerNorm 计算和参数保留 FP32，避免 FP32 图像残差混入 BF16 归一化权重。
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.float()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            module.register_forward_pre_hook(_cast_linear_input)
    decoder = getattr(model, "_ut_decoder", None)
    if decoder is not None:
        decoder.float().requires_grad_(False)


def trainable_report(model, matched):
    groups = defaultdict(int)
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameters.append({"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                               "numel": parameter.numel()})
            groups[parameter_group(name)] += parameter.numel()
    return {"targets": matched, "groups": dict(groups), "trainable": sum(groups.values()),
            "total": sum(p.numel() for p in model.parameters()), "parameters": parameters}


def parameter_group(name):
    if ".lora_A." in name or ".lora_B." in name:
        return "lora"
    if name.startswith(("action_in_proj.", "action_out_proj.")):
        return "ut"
    return "tempo"


def build_model(cfg, *, train=False, state=None):
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    require_patch()
    model = PI0Pytorch(model_config(cfg))
    base = safetensors.torch.load_file(cfg["paths"]["base_pytorch"])
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    head = "paligemma_with_expert.paligemma.lm_head.weight"
    if embedding not in base and head in base:
        base[embedding] = base[head]
    if head not in base and embedding in base:
        base[head] = base[embedding]
    if cfg["model"]["predict_ut"]:
        # 32D 预训练动作投影与独立 64D flow 头形状不同，明确重新初始化。
        base = {key: value for key, value in base.items() if not key.startswith(("action_in_proj.", "action_out_proj."))}
    missing, unexpected = model.load_state_dict(base, strict=False)
    allowed = ("sam2_fusion.", "action_history_", "action_in_proj.", "action_out_proj.",
               "paligemma_with_expert.paligemma.lm_head.")
    bad_missing = [key for key in missing if not key.startswith(allowed)]
    if bad_missing or unexpected:
        raise ValueError(f"基础权重不匹配 missing={bad_missing}, unexpected={unexpected}")
    del base
    matched = inject_lora(model, model.config) if cfg["model"]["lora_enabled"] else {}
    set_precision(model, cfg["model"]["dtype"])
    if state is not None:
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError(f"检查点参数不匹配 missing={expected - set(state)}, extra={set(state) - expected}")
        model.load_state_dict(state, strict=False)
    model.train(train)
    if train:
        model.gradient_checkpointing_enable()
    return model, trainable_report(model, matched)


def optimizer_groups(model, cfg):
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32, name
            groups[parameter_group(name)].append(parameter)
    return [{"params": values, "lr": cfg[f"{name}_lr"], "name": name}
            for name, values in groups.items()]
