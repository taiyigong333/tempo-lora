"""用真实数据和完整优化器更新测量 batch 吞吐，产物与正式训练隔离。"""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader

from openpi.tempo_lora.checkpoint import scheduler, seed_all
from openpi.tempo_lora.config import DEFAULT_CONFIG, configure_environment, read_config
from openpi.tempo_lora.data import WindowDataset, json_write, policy_batch, split_data
from openpi.tempo_lora.model import build_model, optimizer_groups
from openpi.tempo_lora.train import ResumeBatchSampler, check_caches
from openpi.tempo_lora.tracking import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--batches", default="4,8,16,24,32,40,48,52,56")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    configure_environment(cfg)
    check_caches(cfg)
    seed_all(cfg["seed"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = WindowDataset(cfg, split_data(cfg)["train_episodes"], "policy")
    model, _ = build_model(cfg, train=True)
    model.cuda()
    tc = cfg["training"]
    optimizer = torch.optim.AdamW(optimizer_groups(model, tc), weight_decay=tc["weight_decay"])
    schedule = scheduler(optimizer, tc["warmup_steps"], tc["steps"], tc["min_lr_ratio"])
    accumulation = tc["grad_accum_steps"]
    tracker = Tracker(cfg, "batch_tuning", output)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    results = []
    try:
        for candidate_index, batch_size in enumerate(map(int, args.batches.split(","))):
            # 重建加载器并释放上一候选的图，显存峰值与数据传输均纳入实测。
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            seed_all(cfg["seed"])
            sampler = ResumeBatchSampler(len(dataset), batch_size, cfg["seed"], 0,
                                         (args.warmup + args.measure) * accumulation)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=tc["num_workers"],
                                pin_memory=True, generator=torch.Generator().manual_seed(cfg["seed"]))
            iterator = iter(loader)
            durations = []
            losses = []
            observation = actions = batch = loss = None
            try:
                for update in range(args.warmup + args.measure):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.
                    for _ in range(accumulation):
                        batch = next(iterator)
                        observation, actions = policy_batch(batch, "cuda")
                        loss = model(observation, actions).float().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("batch 测试出现非有限 loss")
                        (loss / accumulation).backward()
                        total_loss += loss.item() / accumulation
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                  tc["grad_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    schedule.step()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    print(f"batch={batch_size} update={update} seconds={elapsed:.4f} loss={total_loss:.6f}", flush=True)
                    if update >= args.warmup:
                        durations.append(elapsed)
                        losses.append(total_loss)
                seconds = statistics.mean(durations)
                row = {"batch_size": batch_size, "grad_accum_steps": accumulation,
                       "effective_batch": batch_size * accumulation, "status": "ok",
                       "seconds_per_update": seconds, "samples_per_second": batch_size * accumulation / seconds,
                       "measured_seconds": durations, "losses": losses,
                       "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                       "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                       "reserved_memory_fraction": torch.cuda.max_memory_reserved() / total_memory}
            except torch.OutOfMemoryError as error:
                row = {"batch_size": batch_size, "status": "out_of_memory",
                       "error": str(error),
                       "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                       "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
            finally:
                optimizer.zero_grad(set_to_none=True)
                del iterator, loader, observation, actions, batch, loss
                gc.collect()
                torch.cuda.empty_cache()
            results.append(row)
            json_write(output / "results.json", {"config": cfg, "results": results,
                                                 "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
                                                 "timing": "real_data_optimizer_step_including_accumulation_and_loading"})
            tracker.log({key: value for key, value in row.items() if isinstance(value, (int, float))}, candidate_index)
            print(json.dumps(row), flush=True)
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
