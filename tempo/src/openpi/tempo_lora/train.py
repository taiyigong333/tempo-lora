"""单服务器 LoRA 训练，保持参数组倍率与可精确恢复的采样位置。"""

import json
from pathlib import Path
import time
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from openpi.tempo_lora.checkpoint import artifact_references, load_policy_checkpoint, restore_rng, rng_state, save_policy, scheduler, seed_all
from openpi.tempo_lora.config import artifact_root
from openpi.tempo_lora.data import WindowDataset, check_manifest, json_write, metadata, policy_batch, split_data
from openpi.tempo_lora.model import build_model, optimizer_groups
from openpi.tempo_lora.tracking import Tracker


class ResumeBatchSampler(Sampler):
    def __init__(self, size, batch_size, seed, start, stop):
        self.size, self.batch_size, self.seed, self.start, self.stop = size, batch_size, seed, start, stop
        self.per_epoch = size // batch_size
        if self.per_epoch == 0:
            raise ValueError("数据量小于 batch_size")

    def __iter__(self):
        previous_epoch, permutation = -1, None
        for step in range(self.start, self.stop):
            epoch, offset = divmod(step, self.per_epoch)
            if epoch != previous_epoch:
                permutation = np.random.default_rng(self.seed + epoch).permutation(self.size)
                previous_epoch = epoch
            begin = offset * self.batch_size
            yield permutation[begin:begin + self.batch_size].tolist()

    def __len__(self):
        return self.stop - self.start


def check_caches(cfg):
    cache = artifact_root(cfg)
    for kind in ("sam2", "rgb", "norm"):
        check_manifest(cfg, cache / kind, kind)
    if cfg["model"]["predict_ut"]:
        check_manifest(cfg, cache / "ut", "ut")
        from openpi.tempo_lora.config import ACTION_SPACE, file_digest
        saved = torch.load(cache / "decoder/best.pt", map_location="cpu", weights_only=False)
        if saved.get("action_semantics") != ACTION_SPACE or saved["normalization_sha256"] != file_digest(cache / "norm/norm_stats.json"):
            raise ValueError("动作解码器动作语义或归一化版本不匹配")
    _, episodes = metadata(cfg)
    for ep in episodes:
        index, length = ep["episode_index"], ep["length"]
        expected = [(cache / "sam2" / f"episode_{index:06d}_latent.npy", (length, 256, 8, 8))]
        expected.extend((cache / "rgb" / f"episode_{index:06d}_{camera}.npy", (length, 224, 224, 3)) for camera in cfg["data"]["cameras"])
        if cfg["model"]["predict_ut"]:
            expected.append((cache / "ut" / f"episode_{index:06d}_u.npy",
                             (max(0, length - cfg["auxiliary"]["window"] + 1), cfg["model"]["ut_dim"])))
        for path, shape in expected:
            if np.load(path, mmap_mode="r").shape != shape:
                raise ValueError(f"缓存形状不匹配 {path}")


@torch.no_grad()
def validation_loss(model, loader, batches):
    saved_rng = rng_state()
    was_training = model.training
    model.eval()
    seed_all(1234)
    total, count = 0., 0
    try:
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            observation, actions = policy_batch(batch, "cuda")
            loss = model(observation, actions).float().mean()
            total += loss.item() * len(actions)
            count += len(actions)
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
    return total / count


def train_policy(cfg, *, resume=False, steps=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("当前 Piper-X 入口使用单 GPU，请使用一个训练进程")
    not_before = cfg["training"].get("not_before")
    if not_before and time.time() < datetime.fromisoformat(not_before).timestamp():
        raise RuntimeError(f"正式训练计划从 {not_before} 开始，请等待定时任务")
    check_caches(cfg)
    seed_all(cfg["seed"])
    tc = cfg["training"]
    total_steps = tc["steps"] if steps is None else steps
    output = Path(cfg["paths"]["run"])
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest.json"
    if latest_path.exists() and not resume:
        raise FileExistsError(f"已有训练 {output}，请 resume 或使用新实验目录")
    saved = load_policy_checkpoint(json.loads(latest_path.read_text())["path"], cfg) if resume else None
    model, report = build_model(cfg, train=True, state=saved["trainable"] if saved else None)
    model.cuda()
    json_write(output / "trainable_parameters.json", report)
    for item in report["parameters"]:
        print(f"可训练 {item['name']} {item['shape']} {item['dtype']}", flush=True)
    print(json.dumps({"groups": report["groups"], "trainable": report["trainable"], "total": report["total"]}), flush=True)
    optimizer = torch.optim.AdamW(optimizer_groups(model, tc), weight_decay=tc["weight_decay"])
    schedule = scheduler(optimizer, tc["warmup_steps"], tc["steps"], tc["min_lr_ratio"])
    start, best, stale = 0, float("inf"), 0
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        start, best, stale = saved["step"], saved["best"], saved["stale"]
    split = split_data(cfg)
    train_ds = WindowDataset(cfg, split["train_episodes"], "policy")
    has_validation = bool(split["val_episodes"])
    monitor_ds = WindowDataset(cfg, split["val_episodes"], "policy") if has_validation else train_ds
    accumulation = tc["grad_accum_steps"]
    batches = ResumeBatchSampler(len(train_ds), tc["batch_size"], cfg["seed"], start * accumulation, total_steps * accumulation)
    loader = DataLoader(train_ds, batch_sampler=batches, num_workers=tc["num_workers"],
                        pin_memory=True, persistent_workers=tc["num_workers"] > 0,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    val_loader = DataLoader(monitor_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc["num_workers"],
                            pin_memory=True, persistent_workers=tc["num_workers"] > 0,
                            generator=torch.Generator().manual_seed(0))
    references = saved["references"] if saved else artifact_references(cfg)
    json_write(output / "config.json", cfg)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    timer, accumulated_loss, current = time.monotonic(), 0., start
    tracker = Tracker(cfg, "policy", output, resume=resume)
    if saved:
        restore_rng(saved["rng"])
    with (output / "metrics.jsonl").open("a") as metrics:
        for microstep, batch in enumerate(loader, start=start * accumulation):
            observation, actions = policy_batch(batch, "cuda")
            loss = model(observation, actions).float().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"策略 loss 非有限值 microstep={microstep}")
            (loss / accumulation).backward()
            accumulated_loss += loss.item() / accumulation
            if (microstep + 1) % accumulation:
                continue
            gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                      tc["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            current += 1
            row = {"step": current, "loss": accumulated_loss, "grad_norm": gradient.item(),
                   "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
                   "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "seconds": time.monotonic() - timer}
            accumulated_loss = 0.
            if current % tc["eval_interval"] == 0 or current == total_steps:
                monitor_key = "val_flow_mse" if has_validation else "train_monitor_flow_mse"
                row[monitor_key] = validation_loss(model, val_loader, tc["eval_batches"])
                improved = row[monitor_key] < best
                best = min(best, row[monitor_key])
                stale = 0 if improved else stale + 1
                if improved:
                    save_policy(output / "best.pt", model, optimizer, schedule, current, cfg, references, best=best, stale=stale)
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            if current == start + 1 or current % tc["log_interval"] == 0 or any(key.endswith("flow_mse") for key in row):
                print(json.dumps(row), flush=True)
                tracker.log({**{key: value for key, value in row.items() if key != "lr"},
                             **{f"lr/{name}": value for name, value in row["lr"].items()}}, current)
            early_stop = has_validation and stale >= tc["patience"]
            if current % tc["save_interval"] == 0 or current == total_steps or early_stop:
                destination = output / f"step_{current:08d}.pt"
                save_policy(destination, model, optimizer, schedule, current, cfg, references, best=best, stale=stale)
                json_write(latest_path, {"step": current, "path": str(destination)})
            if early_stop:
                print(f"策略验证损失连续 {stale} 次未改善，早停", flush=True)
                break
    tracker.finish()
    return {"step": current, "planned_steps": total_steps, "best_monitor_flow_mse": best,
            "early_stopped": current < total_steps, "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
