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
from openpi.tempo_lora.checkpoint import find_latest_checkpoint, load_policy_checkpoint
from openpi.tempo_lora.config import artifact_root, configure_environment, make_train_config, model_config
from openpi.tempo_lora.data import WindowDataset, delta_mask, history_buckets, json_write, split_data
from openpi.tempo_lora.model import build_model


def checkpoint_file(path):
    path = Path(path)
    if path.is_file():
        return path
    if (path / "best.pt").exists():
        return path / "best.pt"
    latest = find_latest_checkpoint(path)
    if latest is None:
        raise FileNotFoundError(f"没有可用的策略检查点 {path}")
    return latest


def create_policy(checkpoint, *, pytorch_device=None, sample_kwargs=None, compile_inference=False):
    if type(compile_inference) is not bool:
        raise ValueError("compile_inference必须为布尔值")
    saved = load_policy_checkpoint(checkpoint_file(checkpoint))
    cfg = saved["config"]
    configure_environment(cfg)
    model, _ = build_model(cfg, state=saved["trainable"])
    if compile_inference:
        # 仅编译重复调用的去噪子图，避开整段在线历史和RTC调度的动态控制流。
        # 推理冻结参数仍允许RTC对输入潜变量计算VJP，减少无关参数的图记录。
        model.requires_grad_(False)
        model.denoise_step = torch.compile(model.denoise_step, dynamic=True, mode="default")
        if model.predict_ut:
            model._ut_decoder.forward = torch.compile(model._ut_decoder.forward, dynamic=False, mode="default")
    dc = make_train_config(cfg).data.create(artifact_root(cfg), model_config(cfg))
    norm = deserialize_json(saved["norm_stats"])
    def native_actions(result):
        # 归一化统计经 JSON 加载可能是 float64，控制接口固定接收 float32。
        return {**result, "actions": np.asarray(result["actions"], dtype=np.float32)}
    policy = Policy(model, is_pytorch=True, pytorch_device=pytorch_device or "cuda",
                  transforms=[*dc.data_transforms.inputs, transforms.Normalize(norm, use_quantiles=True),
                              *dc.model_transforms.inputs],
                  output_transforms=[*dc.model_transforms.outputs, transforms.Unnormalize(norm, use_quantiles=True),
                                     *dc.data_transforms.outputs, native_actions],
                  sample_kwargs=sample_kwargs or {"num_steps": cfg["inference"]["denoise_steps"]},
                  metadata={"config": cfg["name"], "native_action_dim": cfg["data"]["action_dim"],
                            "action_horizon": cfg["model"]["action_horizon"],
                            "action_space": "absolute_joint_rad_gripper_m", "feedback_fps": cfg["data"]["fps"],
                            "execute_steps": cfg["inference"]["execute_steps"], "ema_enabled": False,
                            "modes": ["tempo", "temporal_ensemble", "rtc"],
                            "rtc_algorithm": "decoded_action_vjp_soft_prefix",
                            "task_prompts": [cfg["data"]["prompt"]], "protocol_version": 2,
                            "denoise_steps": (sample_kwargs or {}).get("num_steps", cfg["inference"]["denoise_steps"]),
                            "compile_inference": bool(compile_inference)})
    policy.tempo_norm_stats = norm
    return policy


class OnlinePolicy(BasePolicy):
    def __init__(self, policy, cfg, *, sam_stream=None, rtc_max_guidance=10.0,
                 parallel_sam=False, compile_inference=False):
        self.policy, self.cfg = policy, cfg
        if type(parallel_sam) is not bool:
            raise ValueError("parallel_sam必须为布尔值")
        self.parallel_sam = parallel_sam
        if not np.isfinite(rtc_max_guidance) or rtc_max_guidance <= 0:
            raise ValueError("RTC引导上限必须为有限正数")
        self.rtc_max_guidance = float(rtc_max_guidance)
        if sam_stream is None:
            from openpi.tempo_lora.sam import SAM2Stream, build_sam2_tiny
            sam_stream = SAM2Stream(build_sam2_tiny("cuda"), precision=cfg["model"]["dtype"],
                                    compile_model=compile_inference)
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

    def ingest(self, observation, *, encode_sam=True):
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
        if encode_sam:
            self.latest_tokens = self.sam.step(images["head"]).transpose(1, 2, 0).reshape(64, 256)
        self.last_state = state.copy()
        self.last_timestamp = timestamp

    def inputs(self, *, sam2_tokens=None):
        mc = self.cfg["model"]
        actions = np.array(self.feedback, np.float32).reshape(-1, self.cfg["data"]["action_dim"])
        history, padding = history_buckets(actions, len(actions), mc["action_history_steps"], self.history_width)
        images = list(self.images)
        indices = [max(0, len(images) - 1 - offset * self.image_stride)
                   for offset in range(mc["obs_history"] - 1, -1, -1)]
        return {"state": self.last_state, "images": {cam: np.stack([images[i][cam] for i in indices])
                for cam in self.cfg["data"]["cameras"]}, "prompt": self.cfg["data"]["prompt"],
                "sam2_tokens": self.latest_tokens if sam2_tokens is None else sam2_tokens, "action_history": history, "action_history_is_pad": padding}

    def infer(self, obs, *, noise=None):
        if "episode_id" not in obs:
            raise ValueError("在线输入必须携带 episode_id，防止跨轨迹状态泄漏")
        mode = obs.get("mode", "tempo")
        if mode not in ("tempo", "temporal_ensemble", "rtc"):
            raise ValueError("不支持的推理模式")
        prompt = obs.get("prompt", self.cfg["data"]["prompt"])
        if prompt != self.cfg["data"]["prompt"]:
            raise ValueError("所选任务与当前模型训练任务不匹配")
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
        sampling = {}
        prefix = obs.get("rtc_prefix")
        if prefix is not None:
            from openpi.tempo_lora.rtc import prefix_weights
            prefix = np.asarray(prefix, np.float32)
            horizon, dim = self.cfg["model"]["action_horizon"], self.cfg["data"]["action_dim"]
            if (mode != "rtc" or prefix.ndim != 2 or prefix.shape[1] != dim
                    or not 1 <= len(prefix) <= horizon or not np.isfinite(prefix).all()):
                raise ValueError("RTC前缀必须为原生绝对关节和夹爪动作")
            weights = prefix_weights(obs.get("rtc_delay", 0), len(prefix), horizon)
            # 历史动作块属于旧state锚点，换到本轮state后再使用训练归一化。
            target = transforms.DeltaActions(delta_mask(dim))({"actions": prefix.copy(), "state": np.asarray(frames[-1]["state"], np.float32)})
            norm = self.policy.tempo_norm_stats
            target = transforms.Normalize(norm, use_quantiles=True)(target)["actions"]
            padded = np.zeros((1, horizon, self.cfg["model"]["action_dim"]), np.float32)
            padded[0, :len(prefix), :dim] = target
            mask = np.zeros_like(padded)
            mask[0, :, :dim] = weights[:, None]
            sampling = {"rtc_prefix": padded, "rtc_weights": mask, "rtc_max_guidance": self.rtc_max_guidance}
        if new_episode:
            self.reset()
            self.episode = obs["episode_id"]
        timer = time.monotonic()
        provider = None
        if self.parallel_sam:
            provider = self.sam.encode_async([frame["images"]["head"] for frame in frames])
            sampling["sam2_tokens_provider"] = provider
        try:
            for frame in frames:
                self.ingest(frame, encode_sam=not self.parallel_sam)
            ingest_ms = (time.monotonic() - timer) * 1000
            placeholder = np.zeros((64, 256), np.float32) if self.parallel_sam else None
            result = self.policy.infer(self.inputs(sam2_tokens=placeholder), noise=noise,
                                       **({"sample_kwargs": sampling} if sampling else {}))
        finally:
            # 出错同样排空当前批SAM2任务，下一episode重置时不会与旧任务竞争记忆。
            if provider is not None:
                tokens = provider()
        if provider is not None:
            self.latest_tokens = tokens[0].detach().cpu().numpy()
        result["end_to_end_ms"] = (time.monotonic() - timer) * 1000
        result["feedback_ingest_ms"] = ingest_ms
        result["mode"] = mode
        result["rtc_prefix_length"] = 0 if prefix is None else len(prefix)
        return result

    @property
    def metadata(self):
        return {**self.policy.metadata, "rtc_max_guidance": self.rtc_max_guidance,
                "parallel_sam": self.parallel_sam}


def serve(cfg, checkpoint=None, *, port=None, host=None):
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    rtc_max_guidance = cfg["inference"].get("rtc_max_guidance", 10.0)
    compile_inference = cfg["inference"].get("compile_inference", False)
    parallel_sam = cfg["inference"].get("parallel_sam", False)
    path = checkpoint_file(checkpoint or cfg["paths"]["run"])
    saved = load_policy_checkpoint(path)
    cfg = saved["config"]
    configure_environment(cfg)
    policy = OnlinePolicy(create_policy(path, compile_inference=compile_inference), cfg, rtc_max_guidance=rtc_max_guidance,
                          parallel_sam=parallel_sam, compile_inference=compile_inference)
    # 先预热原生与RTC路径，再开放端口，首轮控制沿用正常帧龄保护。
    state = np.asarray(policy.policy.tempo_norm_stats["state"].mean, np.float32)[:cfg["data"]["action_dim"]]
    images = {camera: np.zeros((224, 224, 3), np.uint8) for camera in cfg["data"]["cameras"]}
    warm = policy.infer({"episode_id": "__startup_warmup__", "timestamp": 0., "state": state, "images": images})
    policy.infer({"episode_id": "__startup_warmup__", "mode": "rtc",
                  "feedback": [{"timestamp": index / cfg["data"]["fps"], "state": state, "images": images}
                               for index in range(1, 4)],
                  "rtc_prefix": warm["actions"], "rtc_delay": 1})
    policy.reset()
    # 服务地址是部署参数，独立于检查点内不可变的训练配置。
    metadata = {**policy.metadata, "checkpoint": path.name, "step": saved["step"]}
    WebsocketPolicyServer(policy, host=host if host is not None else cfg["inference"]["host"],
                          port=port if port is not None else cfg["inference"]["port"],
                          metadata=metadata, single_client=True).serve_forever()


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
