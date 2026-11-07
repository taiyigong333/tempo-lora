"""复用 TEMPO 的 PoE-MVAE 和 decoder，统一 7D 窗口和策略归一化。"""

from dataclasses import asdict
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openpi.tempo_lora.checkpoint import atomic_save, restore_rng, rng_state, scheduler, seed_all
from openpi.tempo_lora.config import ACTION_SPACE, ROOT, artifact_root, digest, file_digest
from openpi.tempo_lora.data import WindowDataset, check_manifest, contract, metadata, split_data, write_manifest
from openpi.tempo_lora.tracking import Tracker

sys.path.insert(0, str(ROOT / "tempo"))
from tools.ut.mvae import ActionDecoder, MVAEConfig, PoEMVAE, VideoLatentDecoder


def auxiliary_config(cfg):
    aux = cfg["auxiliary"]
    return MVAEConfig(window=aux["window"], sam_dim=cfg["model"]["sam2_token_dim"], sam_grid=1,
                      action_dim=cfg["data"]["action_dim"], hidden_dim=aux["hidden_dim"],
                      latent_dim=aux["latent_dim"], enc_layers=aux["enc_layers"], dec_layers=aux["dec_layers"],
                      nhead=aux["nhead"], beta=aux["beta"], action_loss_weight=aux["action_loss_weight"])


def train_auxiliary(cfg, stage, *, resume=False, epochs=None):
    seed_all(cfg["seed"])
    aux = cfg["auxiliary"]
    output = artifact_root(cfg) / stage
    check_manifest(cfg, artifact_root(cfg) / "sam2", "sam2")
    check_manifest(cfg, artifact_root(cfg) / "norm", "norm")
    if stage == "decoder":
        check_manifest(cfg, artifact_root(cfg) / "ut", "ut")
    manifest = write_manifest(cfg, output, stage, model_cfg=asdict(auxiliary_config(cfg)),
                              norm_sha256=file_digest(artifact_root(cfg) / "norm/norm_stats.json"))
    mc = auxiliary_config(cfg)
    model = PoEMVAE(mc) if stage == "mvae" else torch.nn.ModuleDict(
        {"act_dec": ActionDecoder(mc), "sam_dec": VideoLatentDecoder(mc)})
    model = model.cuda().float()
    split = split_data(cfg)
    parts = ["train", "val"] if split["val_episodes"] else ["train"]
    datasets = {part: WindowDataset(cfg, split[f"{part}_episodes"], stage) for part in parts}
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders = {part: DataLoader(ds, batch_size=aux["batch_size"], shuffle=part == "train",
                               num_workers=cfg["training"]["num_workers"], pin_memory=True,
                               generator=generator if part == "train" else torch.Generator().manual_seed(0),
                               persistent_workers=cfg["training"]["num_workers"] > 0)
               for part, ds in datasets.items()}
    total_epochs = aux[f"{stage}_epochs"] if epochs is None else epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=aux["lr"], weight_decay=aux["weight_decay"])
    schedule = scheduler(optimizer, aux["warmup_steps"], total_epochs * len(loaders["train"]))
    start_epoch, best, stale, step = 0, float("inf"), 0, 0
    checkpoint_path = output / "last.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"已有训练产物 {checkpoint_path}，请显式 resume")
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["manifest"] != manifest or saved["config"] != cfg or saved["total_epochs"] != total_epochs:
            raise ValueError("辅助模型恢复配置不兼容")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        generator.set_state(saved["loader_rng"])
        start_epoch, best, stale, step = saved["epoch"] + 1, saved["best"], saved["stale"], saved["step"]
    log = (output / "metrics.jsonl").open("a")
    tracker = Tracker(cfg, stage, output, resume=resume)
    if resume:
        # 登录及数据装配可能消耗随机数，恢复点放在实际训练循环前。
        restore_rng(saved["rng"])
    try:
        for epoch in range(start_epoch, total_epochs):
            timer = time.monotonic()
            metrics = {}
            for part, loader in loaders.items():
                training = part == "train"
                model.train(training)
                sums = np.zeros(4)
                for batch_index, batch in enumerate(loader):
                    latent = batch["latent"].cuda(non_blocking=True).float()
                    action = batch["action"].cuda(non_blocking=True).float()
                    with torch.set_grad_enabled(training):
                        if stage == "mvae":
                            result = model(latent, action)
                            loss, loss_v, loss_a = result["loss"], result["mse_latent"], result["mse_action"]
                        else:
                            u = batch["u"].cuda(non_blocking=True).float()
                            loss_a = F.mse_loss(model["act_dec"](u), action)
                            loss_v = F.mse_loss(model["sam_dec"](u), latent)
                            loss = loss_a + loss_v
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{stage} loss 非有限值")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        schedule.step()
                        step += 1
                    n = len(action)
                    sums += (loss.item() * n, loss_a.item() * n, loss_v.item() * n, n)
                    if training and batch_index % 50 == 0:
                        print(f"{stage} epoch={epoch} batch={batch_index}/{len(loader)} loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
                        tracker.log({"batch/loss": loss.item(), "batch/action_mse": loss_a.item(),
                                     "batch/video_mse": loss_v.item(), "lr": optimizer.param_groups[0]["lr"]}, step)
                metrics[part] = dict(zip(("loss", "action_mse", "video_mse"), (sums[:3] / sums[3]).tolist()))
            monitor = "val" if "val" in metrics else "train"
            improved = metrics[monitor]["loss"] < best
            best = min(best, metrics[monitor]["loss"])
            stale = 0 if improved else stale + 1
            saved = {"model": model.state_dict(), "model_cfg": asdict(mc), "config": cfg,
                     "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                     "rng": rng_state(), "loader_rng": generator.get_state(), "epoch": epoch,
                     "total_epochs": total_epochs, "best": best, "stale": stale, "step": step,
                     "selection_metric": f"{monitor}_loss",
                     "split": split, "manifest": manifest, "action_space": "policy_norm",
                     "action_semantics": ACTION_SPACE,
                     "normalization_sha256": file_digest(artifact_root(cfg) / "norm/norm_stats.json")}
            if stage == "decoder":
                saved.update(act_dec=model["act_dec"].state_dict(), sam_dec=model["sam_dec"].state_dict())
            atomic_save(saved, checkpoint_path)
            if improved:
                atomic_save(saved, output / "best.pt")
            row = {"epoch": epoch, "step": step, "lr": optimizer.param_groups[0]["lr"],
                   **metrics, "seconds": time.monotonic() - timer,
                   "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            tracker.log({**{f"{part}/{key}": value for part in parts for key, value in metrics[part].items()},
                         "epoch": epoch, "peak_memory_gb": row["peak_memory_gb"]}, step)
            if "val" in metrics and stale >= aux["patience"]:
                print(f"{stage} 验证损失连续 {stale} 个 epoch 未改善，早停", flush=True)
                break
    finally:
        log.close()
        tracker.finish()


@torch.no_grad()
def encode_ut(cfg):
    directory = artifact_root(cfg)
    checkpoint = directory / "mvae/best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["manifest"]["contract"] != digest(contract(cfg)):
        raise ValueError("MVAE 契约不兼容")
    model = PoEMVAE(MVAEConfig(**saved["model_cfg"])).cuda().eval()
    model.load_state_dict(saved["model"])
    output = directory / "ut"
    manifest = write_manifest(cfg, output, "ut", mvae_sha256=file_digest(checkpoint),
                              window=cfg["auxiliary"]["window"], ut_dim=cfg["model"]["ut_dim"], label="mu_v")
    _, episodes = metadata(cfg)
    window = cfg["auxiliary"]["window"]
    for ep in episodes:
        index, length = ep["episode_index"], ep["length"]
        destination = output / f"episode_{index:06d}_u.npy"
        if destination.exists():
            if np.load(destination, mmap_mode="r").shape != (max(0, length - window + 1), cfg["model"]["ut_dim"]):
                raise ValueError(f"u_t 缓存形状不兼容 {destination}")
            continue
        latent = np.load(directory / "sam2" / f"episode_{index:06d}_latent.npy", mmap_mode="r")
        pooled = latent.mean((2, 3), keepdims=True)
        windows = np.stack([pooled[t:t + window] for t in range(length - window + 1)]) if length >= window else np.empty((0, window, 256, 1, 1), np.float32)
        values = []
        for start in range(0, len(windows), cfg["auxiliary"]["batch_size"]):
            mu, _ = model.video_enc(torch.from_numpy(windows[start:start + cfg["auxiliary"]["batch_size"]]).cuda())
            values.append(mu.cpu().numpy())
        u = np.concatenate(values) if values else np.zeros((0, cfg["model"]["ut_dim"]), np.float32)
        temporary = destination.with_suffix(".tmp.npy")
        np.save(temporary, u)
        temporary.replace(destination)
        np.savez(output / f"episode_{index:06d}.npz", u_window=u, ep_len=length, contract=manifest["contract"])
        print(f"编码 u_t episode={index} windows={len(u)}", flush=True)
