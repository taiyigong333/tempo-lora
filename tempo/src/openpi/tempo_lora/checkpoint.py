"""原子保存完整训练状态，adapter 与额外模块共用严格加载协议。"""

import math
from pathlib import Path
import random

import numpy as np
import torch

from openpi.tempo_lora.config import ACTION_SPACE, UPSTREAM_COMMIT, artifact_root, digest, file_digest
from openpi.tempo_lora.data import contract, split_data


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def scheduler(optimizer, warmup, total, minimum=.1):
    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(total - warmup, 1))
        return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))
    # 同一个倍率乘各组初始学习率，保留 LoRA/TEMPO/u_t 的学习率比例。
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def artifact_references(cfg):
    cache = artifact_root(cfg)
    files = {"norm": cache / "norm/norm_stats.json", "split": cache / "split.json",
             "base": Path(cfg["paths"]["base_pytorch"])}
    if cfg["model"]["predict_ut"]:
        files.update(mvae=cache / "mvae/best.pt", decoder=cache / "decoder/best.pt",
                     ut_manifest=cache / "ut/manifest.json")
    return {key: {"path": str(path), "sha256": file_digest(path)} for key, path in files.items()}


def policy_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def save_policy(path, model, optimizer, schedule, step, cfg, references, **extra):
    from openpi.shared.normalize import load, serialize_json

    atomic_save({"format_version": 1, "upstream_commit": UPSTREAM_COMMIT,
                 "action_space": ACTION_SPACE, "contract": digest(contract(cfg)),
                 "split": split_data(cfg), "config": cfg, "references": references,
                 "norm_stats": serialize_json(load(artifact_root(cfg) / "norm")),
                 "trainable": policy_state(model), "optimizer": optimizer.state_dict(),
                 "scheduler": schedule.state_dict(), "step": step, "rng": rng_state(),
                 "ema_enabled": False, **extra}, path)


def load_policy_checkpoint(path, cfg=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != 1 or saved.get("action_space") != ACTION_SPACE:
        raise ValueError("检查点格式或动作语义不兼容")
    if cfg is not None and digest(cfg) != digest(saved["config"]):
        raise ValueError("恢复配置与检查点不一致，请使用保存的配置并保持调度参数")
    # 基础模型与辅助产物是 checkpoint 的组成部分，变化时拒绝静默加载。
    for name, reference in saved["references"].items():
        if file_digest(reference["path"]) != reference["sha256"]:
            raise ValueError(f"检查点引用 {name} 已变化，请恢复匹配产物")
    return saved
