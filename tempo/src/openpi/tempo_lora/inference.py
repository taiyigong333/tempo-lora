"""未合并 PEFT 推理，在线输入始终由实测反馈重建。"""

from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from openpi_client.base_policy import BasePolicy
import torch

from openpi import transforms
from openpi.policies.policy import Policy
from openpi.shared.normalize import deserialize_json
from openpi.tempo_lora.checkpoint import load_policy_checkpoint
from openpi.tempo_lora.config import artifact_root, configure_environment, make_train_config, model_config
from openpi.tempo_lora.data import WindowDataset, history_buckets, json_write, split_data
from openpi.tempo_lora.model import build_model


def checkpoint_file(path):
    path = Path(path)
    if path.is_file():
        return path
    if (path / "best.pt").exists():
        return path / "best.pt"
    return Path(json.loads((path / "latest.json").read_text())["path"])


def create_policy(checkpoint, *, pytorch_device=None, sample_kwargs=None):
    saved = load_policy_checkpoint(checkpoint_file(checkpoint))
    cfg = saved["config"]
    configure_environment(cfg)
    model, _ = build_model(cfg, state=saved["trainable"])
    dc = make_train_config(cfg).data.create(artifact_root(cfg), model_config(cfg))
    norm = deserialize_json(saved["norm_stats"])
    def native_actions(result):
        # 归一化统计经 JSON 加载可能是 float64，控制接口固定接收 float32。
        return {**result, "actions": np.asarray(result["actions"], dtype=np.float32)}
    return Policy(model, is_pytorch=True, pytorch_device=pytorch_device or "cuda",
                  transforms=[*dc.data_transforms.inputs, transforms.Normalize(norm, use_quantiles=True),
                              *dc.model_transforms.inputs],
                  output_transforms=[*dc.model_transforms.outputs, transforms.Unnormalize(norm, use_quantiles=True),
                                     *dc.data_transforms.outputs, native_actions],
                  sample_kwargs=sample_kwargs or {"num_steps": cfg["inference"]["denoise_steps"]},
                  metadata={"config": cfg["name"], "native_action_dim": cfg["data"]["action_dim"],
                            "action_horizon": cfg["model"]["action_horizon"],
                            "action_space": "absolute_joint_rad_gripper_m", "feedback_fps": cfg["data"]["fps"],
                            "execute_steps": cfg["inference"]["execute_steps"], "ema_enabled": False})


class OnlinePolicy(BasePolicy):
    def __init__(self, policy, cfg, *, sam_stream=None):
        self.policy, self.cfg = policy, cfg
        if sam_stream is None:
            from openpi.tempo_lora.sam import SAM2Stream, build_sam2_tiny
            sam_stream = SAM2Stream(build_sam2_tiny("cuda"), precision=cfg["model"]["dtype"])
        self.sam = sam_stream
        mc, dc = cfg["model"], cfg["data"]
        self.history_width = round(dc["fps"] * dc["action_history_seconds"] / mc["action_history_steps"])
        self.max_history = self.history_width * mc["action_history_steps"]
        self.image_stride = round(mc["obs_history_stride_s"] * dc["fps"])
        self.image_length = (mc["obs_history"] - 1) * self.image_stride + 1
        self.reset()

    def reset(self):
        self.episode = None
        self.frame = -1
        self.last_timestamp = None
        self.origin_timestamp = None
        self.feedback = deque(maxlen=self.max_history)
        self.images = deque(maxlen=self.image_length)
        self.sam.reset()
        self.latest_tokens = None

    def ingest(self, observation):
        timestamp = float(observation["timestamp"])
        if not np.isfinite(timestamp):
            raise ValueError("反馈时间戳必须有限")
        if self.last_timestamp is not None:
            if abs(timestamp - self.last_timestamp - 1 / self.cfg["data"]["fps"]) > .002:
                raise ValueError("需要完整连续的 30Hz 实测反馈，请通过 feedback 批量传入缺失帧")
        else:
            self.origin_timestamp = timestamp
        state = np.asarray(observation["state"], np.float32)
        if state.shape != (self.cfg["data"]["action_dim"],) or not np.isfinite(state).all():
            raise ValueError("反馈状态维度或数值不合法")
        images = {camera: np.asarray(observation["images"][camera]) for camera in self.cfg["data"]["cameras"]}
        if any(value.shape != (224, 224, 3) or value.dtype != np.uint8 for value in images.values()):
            raise ValueError("在线图像应与转换数据一致，为 224×224 RGB uint8 等比补黑边图像")
        self.frame += 1
        if self.frame > 0:
            # state[t] 对应 action[t-1]，仅在该反馈实际到达后添加历史。
            self.feedback.append(state.copy())
        self.images.append({key: value.copy() for key, value in images.items()})
        self.latest_tokens = self.sam.step(images["head"]).transpose(1, 2, 0).reshape(64, 256)
        self.last_state = state.copy()
        self.last_timestamp = timestamp

    def inputs(self):
        mc = self.cfg["model"]
        actions = np.array(self.feedback, np.float32).reshape(-1, self.cfg["data"]["action_dim"])
        history, padding = history_buckets(actions, len(actions), mc["action_history_steps"], self.history_width)
        images = list(self.images)
        indices = [max(0, len(images) - 1 - offset * self.image_stride)
                   for offset in range(mc["obs_history"] - 1, -1, -1)]
        return {"state": self.last_state, "images": {cam: np.stack([images[i][cam] for i in indices])
                for cam in self.cfg["data"]["cameras"]}, "prompt": self.cfg["data"]["prompt"],
                "sam2_tokens": self.latest_tokens, "action_history": history, "action_history_is_pad": padding}

    def infer(self, obs, *, noise=None):
        if "episode_id" not in obs:
            raise ValueError("在线输入必须携带 episode_id，防止跨轨迹状态泄漏")
        new_episode = obs.get("reset", False) or self.episode != obs["episode_id"]
        frames = obs.get("feedback")
        if frames is None:
            frames = [obs]
        if not frames:
            raise ValueError("feedback 不能为空")
        # 先核验整批反馈，避免输入错误在处理一半后污染在线记忆。
        previous = None if new_episode else self.last_timestamp
        for frame in frames:
            stamp = float(frame["timestamp"])
            if not np.isfinite(stamp) or (previous is not None and abs(stamp - previous - 1 / self.cfg["data"]["fps"]) > .002):
                raise ValueError("需要完整连续的 30Hz 实测反馈")
            state = np.asarray(frame["state"])
            if state.shape != (self.cfg["data"]["action_dim"],) or not np.isfinite(state).all():
                raise ValueError("反馈状态维度或数值不合法")
            for camera in self.cfg["data"]["cameras"]:
                rgb = np.asarray(frame["images"][camera])
                if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
                    raise ValueError("在线图像需要 224×224 RGB uint8")
            previous = stamp
        if new_episode:
            self.reset()
            self.episode = obs["episode_id"]
        timer = time.monotonic()
        for frame in frames:
            self.ingest(frame)
        result = self.policy.infer(self.inputs(), noise=noise)
        result["end_to_end_ms"] = (time.monotonic() - timer) * 1000
        return result

    @property
    def metadata(self):
        return self.policy.metadata


def serve(cfg, checkpoint=None, *, port=None):
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    path = checkpoint_file(checkpoint or cfg["paths"]["run"])
    saved = load_policy_checkpoint(path)
    cfg = saved["config"]
    configure_environment(cfg)
    policy = OnlinePolicy(create_policy(path), cfg)
    WebsocketPolicyServer(policy, host=cfg["inference"]["host"], port=port or cfg["inference"]["port"],
                          metadata=policy.metadata).serve_forever()


@torch.no_grad()
def evaluate(cfg, checkpoint=None, *, limit=None):
    path = checkpoint_file(checkpoint or cfg["paths"]["run"])
    policy = create_policy(path)
    saved = load_policy_checkpoint(path)
    cfg = saved["config"]
    split = split_data(cfg)
    partition = "val" if split["val_episodes"] else "train"
    ds = WindowDataset(cfg, split[f"{partition}_episodes"], "policy")
    count = min(len(ds), 32 if limit is None else limit)
    errors, durations = [], []
    directory = artifact_root(cfg)
    for index in np.linspace(0, len(ds) - 1, count, dtype=int):
        ep, t = ds.index[index]
        row = ds.rows[ep]
        mc, dc = cfg["model"], cfg["data"]
        offsets = np.maximum(0, t - np.arange(mc["obs_history"] - 1, -1, -1) * round(mc["obs_history_stride_s"] * dc["fps"]))
        width = round(dc["fps"] * dc["action_history_seconds"] / mc["action_history_steps"])
        history, padding = history_buckets(row["action"], t, mc["action_history_steps"], width)
        latent = np.load(directory / "sam2" / f"episode_{ep:06d}_latent.npy", mmap_mode="r")
        observation = {"state": row["observation.state"][t].astype(np.float32),
                       "images": {cam: np.load(directory / "rgb" / f"episode_{ep:06d}_{cam}.npy", mmap_mode="r")[offsets].copy()
                                  for cam in dc["cameras"]}, "prompt": dc["prompt"],
                       "action_history": history, "action_history_is_pad": padding,
                       "sam2_tokens": latent[t].transpose(1, 2, 0).reshape(64, 256).copy()}
        noise = np.random.default_rng(ep * 10000 + t).standard_normal((1 if mc["predict_ut"] else mc["action_horizon"],
                                                                     mc["ut_dim"] if mc["predict_ut"] else mc["action_dim"])).astype(np.float32)
        begin = time.monotonic()
        prediction = policy.infer(observation, noise=noise)["actions"]
        durations.append((time.monotonic() - begin) * 1000)
        target = row["action"][t:t + mc["action_horizon"]]
        errors.append(((prediction - target) ** 2).mean(0))
    result = {"checkpoint": str(path), "samples": count, "native_action_mse_per_dim": np.mean(errors, 0).tolist(),
              "policy_latency_ms_mean": float(np.mean(durations)), "policy_latency_ms_p95": float(np.quantile(durations, .95)),
              "closed_loop_success_rate": None, "evaluation": "offline_feedback_proxy", "data_split": partition}
    json_write(Path(cfg["paths"]["run"]) / "evaluation.json", result)
    print(json.dumps(result), flush=True)
    return result
