"""来源划分、窗口动作语义与离线/在线共享历史。"""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from openpi import transforms
from openpi.tempo_lora.config import ACTION_SPACE, CONVERSION_COMMIT, artifact_root, digest, file_digest


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def metadata(cfg):
    root = Path(cfg["data"]["root"])
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    return info, sorted(episodes, key=lambda row: row["episode_index"])


def episode_path(cfg, episode, *, video_key=None):
    info, _ = metadata(cfg)
    template = info["data_path"] if video_key is None else info["video_path"]
    return Path(cfg["data"]["root"]) / template.format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=video_key)


def read_episode(cfg, episode):
    table = pq.read_table(episode_path(cfg, episode))
    return {key: np.asarray(table[key].to_pylist()) for key in table.column_names}


def delta_mask(dim):
    if dim not in (7, 14):
        raise ValueError(f"不支持的动作维度 {dim}")
    return tuple(i % 7 != 6 for i in range(dim))


def action_window(action, state, start, window):
    # 磁盘标签已经向前对齐一帧，这里直接从 start 读取，避免重复偏移。
    indices = np.minimum(np.arange(start, start + window), len(action) - 1)
    sample = {"state": np.array(state[start], dtype=np.float32),
              "actions": np.array(action[indices], dtype=np.float32)}
    return transforms.DeltaActions(delta_mask(action.shape[-1]))(sample)["actions"]


def history_buckets(actions, frame, steps=10, width=30):
    # 部分桶的启动 padding 参与均值，严格复用 TEMPO 原有桶边界。
    indices = np.arange(frame - steps * width, frame)
    valid = (indices >= 0) & (indices < len(actions))
    values = np.zeros((steps * width, actions.shape[-1]), np.float32)
    values[valid] = actions[indices[valid]]
    return values.reshape(steps, width, -1).mean(1), (~valid).reshape(steps, width).all(1)


def contract(cfg):
    root = Path(cfg["data"]["root"])
    return {
        "version": 1, "conversion_commit": CONVERSION_COMMIT,
        "data_id": cfg["data"]["repo_id"], "action_space": ACTION_SPACE,
        "action_dim": cfg["data"]["action_dim"], "units": "joint_rad_gripper_m",
        "window": cfg["auxiliary"]["window"], "ut_dim": cfg["model"]["ut_dim"],
        "data_metadata_sha256": {name: file_digest(root / "meta" / name)
                                 for name in ("info.json", "conversion.json", "source_episodes.jsonl")},
        "data_config": cfg["data"],
    }


def split_data(cfg):
    path = artifact_root(cfg) / "split.json"
    signature = digest(contract(cfg))
    if path.exists():
        split = json.loads(path.read_text())
        if split.get("contract") != signature:
            raise ValueError("划分契约改变，请使用新的 artifacts 目录")
        return split
    rows = [json.loads(line) for line in
            (Path(cfg["data"]["root"]) / "meta/source_episodes.jsonl").read_text().splitlines()]
    if cfg["data"].get("split_mode") == "all_train":
        split = {"train_episodes": sorted(row["episode_index"] for row in rows), "val_episodes": [],
                 "source_groups": [], "method": "all_train_user_requested", "seed": cfg["seed"], "contract": signature}
        json_write(path, split)
        json_write(artifact_root(cfg) / "contract.json", contract(cfg))
        return split
    # 连续采集按时间间隔组成整组，相邻录制不跨集合，原全局 episode_index 保留。
    groups = []
    previous = None
    for row in sorted(rows, key=lambda row: row["source"]):
        recording = Path(row["source"]).parents[1].name
        stamp = datetime.strptime(recording, "%Y%m%dT%H%M%S%f")
        key = (row["batch"], str(Path(row["source"]).parents[2]))
        if previous is None or key != previous[0] or (stamp - previous[1]).total_seconds() > cfg["data"]["split_group_gap_seconds"]:
            groups.append([])
        groups[-1].append(row["episode_index"])
        previous = key, stamp
    if len(groups) < 2:
        raise ValueError("来源只有一个连续采集组，无法建立隔离验证集")
    order = np.random.default_rng(cfg["seed"]).permutation(len(groups))
    val_groups = []
    target = len(rows) * cfg["data"]["val_fraction"]
    for index in order[:-1]:
        if sum(map(len, val_groups)) >= target:
            break
        val_groups.append(groups[index])
    val = sorted(ep for group in val_groups for ep in group)
    train = sorted(set(row["episode_index"] for row in rows) - set(val))
    split = {"train_episodes": train, "val_episodes": val, "source_groups": groups,
             "method": "continuous_recording_groups", "seed": cfg["seed"], "contract": signature}
    json_write(path, split)
    json_write(artifact_root(cfg) / "contract.json", contract(cfg))
    return split


def validate_data(cfg):
    import av

    info, episodes = metadata(cfg)
    if info["fps"] != cfg["data"]["fps"] or info["features"]["action"]["shape"] != [cfg["data"]["action_dim"]]:
        raise ValueError("元信息与配置的帧率或动作维度不一致")
    conversion = json.loads((Path(cfg["data"]["root"]) / "meta/conversion.json").read_text())
    if conversion["config"]["action_source"] != "next_observed_state":
        raise ValueError("当前数据并非下一实测状态代理")
    counts, minimum, maximum = [], [], []
    for ep in episodes:
        index, length = ep["episode_index"], ep["length"]
        row = read_episode(cfg, index)
        state, action = row["observation.state"], row["action"]
        assert len(state) == length and np.isfinite(state).all() and np.isfinite(action).all()
        np.testing.assert_allclose(action, state[np.minimum(np.arange(length) + 1, length - 1)], atol=1e-6)
        np.testing.assert_array_equal(row["frame_index"], np.arange(length))
        np.testing.assert_array_equal(row["episode_index"], np.full(length, index))
        np.testing.assert_allclose(row["timestamp"], np.arange(length) / info["fps"], atol=1e-5)
        for camera in cfg["data"]["cameras"]:
            path = episode_path(cfg, index, video_key=f"observation.images.{camera}")
            with av.open(str(path)) as container:
                count = sum(1 for _ in container.decode(video=0))
            if count != length:
                raise ValueError(f"视频长度不匹配 {path}: {count} != {length}")
        counts.append(length)
        minimum.append(state.min(0))
        maximum.append(state.max(0))
        if len(counts) % 25 == 0:
            print(f"数据校验 {len(counts)}/{len(episodes)}", flush=True)
    assert len(episodes) == info["total_episodes"] and sum(counts) == info["total_frames"]
    split = split_data(cfg)
    history_window = int(cfg["data"]["fps"] * cfg["data"]["action_history_seconds"])
    valid_ratio = sum(np.minimum(np.arange(n), history_window).sum() for n in counts) / (sum(counts) * history_window)
    result = {"episodes": len(counts), "frames": sum(counts), "videos": len(counts) * len(cfg["data"]["cameras"]),
              "train_episodes": len(split["train_episodes"]), "val_episodes": len(split["val_episodes"]),
              "source_groups": len(split["source_groups"]), "state_min": np.min(minimum, 0).tolist(),
              "state_max": np.max(maximum, 0).tolist(), "history_valid_frame_ratio": float(valid_ratio),
              "proxy_check": "all_frames_passed", "contract": digest(contract(cfg)),
              "short_window_episodes": [ep["episode_index"] for ep in episodes if ep["length"] < cfg["auxiliary"]["window"]]}
    json_write(artifact_root(cfg) / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def write_manifest(cfg, directory, kind, **extra):
    content = {"kind": kind, "contract": digest(contract(cfg)),
               "split": digest(split_data(cfg)), **extra}
    path = Path(directory) / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != content:
        raise ValueError(f"产物版本不匹配，请新建目录 {directory}")
    json_write(path, content)
    return content


def check_manifest(cfg, directory, kind):
    value = json.loads((Path(directory) / "manifest.json").read_text())
    if value["kind"] != kind or value["contract"] != digest(contract(cfg)) or value["split"] != digest(split_data(cfg)):
        raise ValueError(f"产物契约不匹配，请重建 {directory}")
    return value


def norm_stats(cfg):
    from openpi.shared.normalize import NormStats, save

    states, actions = [], []
    window = cfg["auxiliary"]["window"]
    for ep in split_data(cfg)["train_episodes"]:
        row = read_episode(cfg, ep)
        state, action = row["observation.state"], row["action"]
        # 与策略支持的完整窗口一致，末尾不足窗口不参与辅助模型或统计。
        starts = range(max(0, len(state) - window + 1))
        states.extend(state[list(starts)])
        actions.extend(action_window(action, state, t, window) for t in starts)
    arrays = {"state": np.asarray(states, np.float32), "actions": np.concatenate(actions)}
    stats = {name: NormStats(mean=a.mean(0), std=a.std(0), q01=np.quantile(a, .01, axis=0),
                             q99=np.quantile(a, .99, axis=0)) for name, a in arrays.items()}
    output = artifact_root(cfg) / "norm"
    manifest = write_manifest(cfg, output, "norm", normalization="train_window_quantile_v1")
    save(output, stats)
    json_write(output / "version.json", {**manifest, "sha256": file_digest(output / "norm_stats.json")})
    return stats


class WindowDataset(Dataset):
    """复用上游 MVAE 和策略输入变换，窗口均从同一个 action[t] 开始。"""

    def __init__(self, cfg, episodes, stage):
        self.cfg, self.stage = cfg, stage
        self.cache = artifact_root(cfg)
        self.window = cfg["auxiliary"]["window"]
        self.rows = {}
        self.index = []
        for ep in episodes:
            row = read_episode(cfg, ep)
            self.rows[ep] = row
            if len(row["action"]) < self.window:
                print(f"episode={ep} 长度={len(row['action'])}，保留轨迹但缺少完整 {self.window} 帧监督窗口", flush=True)
            self.index.extend((ep, t) for t in range(max(0, len(row["action"]) - self.window + 1)))
        if not self.index:
            raise ValueError(f"没有可用的 {stage} 窗口")
        self._arrays = {}
        from openpi.shared.normalize import load
        self.stats = load(self.cache / "norm")
        self.normalize = transforms.Normalize(self.stats, use_quantiles=True)
        if stage == "policy":
            from openpi.tempo_lora.config import make_train_config, model_config
            dc = make_train_config(cfg).data.create(self.cache, model_config(cfg))
            self.prepare = transforms.compose([*dc.data_transforms.inputs, self.normalize,
                                                *dc.model_transforms.inputs])

    def array(self, directory, ep, name):
        key = directory, ep, name
        if key not in self._arrays:
            path = self.cache / directory / f"episode_{ep:06d}_{name}.npy"
            self._arrays[key] = np.load(path, mmap_mode="r")
        return self._arrays[key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ep, t = self.index[index]
        row = self.rows[ep]
        action = action_window(row["action"], row["observation.state"], t, self.window)
        latent = self.array("sam2", ep, "latent")
        if self.stage != "policy":
            sample = {"latent": np.asarray(latent[t:t + self.window]).mean((2, 3), keepdims=True).copy(),
                      "action": self.normalize({"actions": action})["actions"].astype(np.float32)}
            if self.stage == "decoder":
                sample["u"] = self.array("ut", ep, "u")[t].copy()
            return sample
        mc, dc = self.cfg["model"], self.cfg["data"]
        width = round(dc["fps"] * dc["action_history_seconds"] / mc["action_history_steps"])
        history, padding = history_buckets(row["action"], t, mc["action_history_steps"], width)
        offsets = np.maximum(0, t - np.arange(mc["obs_history"] - 1, -1, -1) * round(mc["obs_history_stride_s"] * dc["fps"]))
        images = {cam: self.array("rgb", ep, cam)[offsets].copy() for cam in dc["cameras"]}
        indices = np.minimum(np.arange(t, t + self.window), len(row["action"]) - 1)
        sample = {"state": row["observation.state"][t].astype(np.float32), "images": images,
                  "actions": row["action"][indices].astype(np.float32), "prompt": dc["prompt"],
                  "sam2_tokens": np.asarray(latent[t]).transpose(1, 2, 0).reshape(-1, mc["sam2_token_dim"]).copy(),
                  "action_history": history, "action_history_is_pad": padding}
        if mc["predict_ut"]:
            sample.update(ut_target=self.array("ut", ep, "u")[t].copy(), ut_target_is_pad=np.bool_(False))
        return self.prepare(sample)


def policy_batch(batch, device):
    from openpi.models.model import Observation

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        tensor = torch.as_tensor(value, device=device)
        # quantile 序列化可能恢复为 float64，统一策略浮点输入，避免隐式双精度 flow。
        return tensor.float() if tensor.is_floating_point() else tensor

    sample = convert(batch)
    actions = sample.pop("actions")
    return Observation.from_dict(sample), actions
