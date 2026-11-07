"""真实完整模型集成检查，数步更新、冻结、重载及恢复后更新逐项对比。"""

import gc
import hashlib
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from openpi.tempo_lora.checkpoint import artifact_references, load_policy_checkpoint, policy_state, restore_rng, save_policy, scheduler, seed_all
from openpi.tempo_lora.config import ROOT
from openpi.tempo_lora.data import WindowDataset, json_write, policy_batch, split_data
from openpi.tempo_lora.model import build_model, optimizer_groups
from openpi.tempo_lora.train import check_caches


def frozen_digest(model):
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            value.update(name.encode())
            value.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def family(name):
    if "lora_" in name:
        if "vision_tower" in name:
            return "lora_siglip"
        return "lora_expert" if "gemma_expert" in name else "lora_vlm"
    return name.split(".")[0]


def update(model, optimizer, observation, actions):
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss = model(observation, actions).float().mean()
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert parameter.dtype == parameter.grad.dtype == torch.float32, name
            assert torch.isfinite(parameter.grad).all(), name
            key = family(name)
            gradients[key] = gradients.get(key, 0.) + parameter.grad.float().norm().item()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    return loss.item(), gradients


def verify(cfg):
    check_caches(cfg)
    seed_all(cfg["seed"])
    output = ROOT / "trained_model/testing/full_model"
    output.mkdir(parents=True, exist_ok=True)
    ds = WindowDataset(cfg, split_data(cfg)["train_episodes"], "policy")
    batch = default_collate([ds[i] for i in range(cfg["training"]["batch_size"])])
    observation, actions = policy_batch(batch, "cuda")
    model, report = build_model(cfg, train=True)
    model.cuda()
    json_write(output / "trainable_parameters.json", report)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg["training"]), weight_decay=cfg["training"]["weight_decay"])
    schedule = scheduler(optimizer, cfg["training"]["warmup_steps"], cfg["training"]["steps"])
    frozen_before, trainable_before = frozen_digest(model), policy_state(model)
    losses, gradients = [], {}
    for _ in range(3):
        loss, gradient = update(model, optimizer, observation, actions)
        schedule.step()
        losses.append(loss)
        for name, value in gradient.items():
            gradients[name] = gradients.get(name, 0.) + value
    changes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = family(name)
            change = (parameter.detach().cpu() - trainable_before[name]).abs().max().item()
            changes[key] = max(changes.get(key, 0.), change)
    assert frozen_digest(model) == frozen_before
    assert all(value > 0 for value in changes.values()), changes
    assert all(value > 0 for value in gradients.values()), gradients
    del trainable_before
    references = artifact_references(cfg)
    checkpoint = output / "checkpoint.pt"
    save_policy(checkpoint, model, optimizer, schedule, 3, cfg, references, best=float("inf"), stale=0)
    noise = torch.randn(len(actions), model.flow_positions, model.flow_dim, device="cuda")
    model.eval()
    with torch.no_grad():
        before_reload = model.sample_actions("cuda", observation, noise=noise, num_steps=2).cpu()
    # 基准下一步从保存的随机状态开始，恢复分支必须产生相同更新。
    saved = load_policy_checkpoint(checkpoint, cfg)
    restore_rng(saved["rng"])
    next_loss, _ = update(model, optimizer, observation, actions)
    schedule.step()
    next_parameters = policy_state(model)
    del model, optimizer, schedule
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = build_model(cfg, train=True, state=saved["trainable"])
    model.cuda()
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg["training"]), weight_decay=cfg["training"]["weight_decay"])
    schedule = scheduler(optimizer, cfg["training"]["warmup_steps"], cfg["training"]["steps"])
    optimizer.load_state_dict(saved["optimizer"])
    schedule.load_state_dict(saved["scheduler"])
    model.eval()
    with torch.no_grad():
        after_reload = model.sample_actions("cuda", observation, noise=noise, num_steps=2).cpu()
    torch.testing.assert_close(before_reload, after_reload, rtol=0, atol=0)
    restore_rng(saved["rng"])
    resumed_loss, _ = update(model, optimizer, observation, actions)
    schedule.step()
    update_error = 0.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.detach().cpu()
            torch.testing.assert_close(value, next_parameters[name], rtol=1e-5, atol=1e-7)
            update_error = max(update_error, (value - next_parameters[name]).abs().max().item())
    result = {"losses": losses, "gradient_norm_sums": gradients, "max_parameter_updates": changes,
              "frozen_sha256_unchanged": frozen_before, "reload_output_max_error": (before_reload - after_reload).abs().max().item(),
              "resume_next_loss": resumed_loss, "baseline_next_loss": next_loss,
              "resume_update_max_error": update_error, "trainable_groups": report["groups"],
              "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9, "data": "real_training_batch"}
    json_write(output / "verification.json", result)
    print(result, flush=True)
    del model, optimizer, schedule, saved, next_parameters, observation, actions
    gc.collect()
    torch.cuda.empty_cache()
    # 走公开 create_trained_policy 路径，核对输入变换和原生输出裁剪。
    from openpi.policies.policy_config import create_trained_policy
    from openpi.tempo_lora.config import make_train_config
    policy = create_trained_policy(make_train_config(cfg), checkpoint, sample_kwargs={"num_steps": 2})
    from openpi.tempo_lora.inference import OnlinePolicy
    online = OnlinePolicy(policy, cfg)
    ep, _ = ds.index[0]
    state = ds.rows[ep]["observation.state"]
    frames = []
    for t in range(3):
        frames.append({"timestamp": t / cfg["data"]["fps"], "state": state[t].astype(np.float32),
                       "images": {cam: ds.array("rgb", ep, cam)[t].copy() for cam in cfg["data"]["cameras"]}})
    started = time.monotonic()
    output_action = online.infer({"episode_id": "verification_a", "feedback": frames})["actions"]
    assert output_action.shape == (cfg["model"]["action_horizon"], cfg["data"]["action_dim"])
    assert np.isfinite(output_action).all()
    online.infer({"episode_id": "verification_b", "feedback": [frames[0]]})
    assert online.frame == 0 and online.inputs()["action_history_is_pad"].all()
    result.update(normal_policy_entry=True, online_episode_reset=True,
                  online_native_action_shape=list(output_action.shape),
                  online_two_inferences_seconds=time.monotonic() - started)
    json_write(output / "verification.json", result)
    return result
