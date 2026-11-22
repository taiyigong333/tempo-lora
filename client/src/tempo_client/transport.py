"""校验有线出口、模型身份和时序，网络失败后由新会话重建在线记忆。"""

import socket
import time
import uuid

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect
from tempo_client.settings import validate_sampling


def validate_frames(frames, fps, previous=None):
    if not frames:
        raise ValueError("反馈不能为空")
    for frame in frames:
        stamp = float(frame["timestamp"])
        if not np.isfinite(stamp) or (previous is not None and abs(stamp - previous - 1 / fps) > .002):
            raise ValueError("反馈必须为连续30Hz时间网格")
        state = np.asarray(frame["state"])
        if state.shape != (7,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError("state 必须是 float32[7] 六关节弧度与夹爪米")
        for camera in ("head", "left_wrist"):
            rgb = np.asarray(frame["images"][camera])
            if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
                raise ValueError("图像必须为224×224 RGB uint8")
        previous = stamp
    return previous


class Client:
    def __init__(self, cfg):
        self.cfg = cfg
        connection = cfg["connection"]
        raw = socket.create_connection(
            (connection["host"], connection["port"]), connection["connect_timeout_s"],
            source_address=(connection["local_ip"], 0))
        self.ws = None
        try:
            if raw.getsockname()[0] != connection["local_ip"]:
                raise RuntimeError("网络出口不是配置中的有线IP")
            self.ws = connect(f"ws://{connection['host']}:{connection['port']}", sock=raw,
                              proxy=None, compression=None, max_size=None,
                              open_timeout=connection["connect_timeout_s"], close_timeout=2, ping_timeout=300)
            self.metadata = self._receive()
            for key in ("config", "step", "action_horizon", "action_space"):
                if self.metadata.get(key) != cfg["model"][key]:
                    raise ValueError(f"服务器模型身份不匹配 {key}: {self.metadata.get(key)}")
            if (self.metadata.get("native_action_dim") != cfg["model"]["action_dim"]
                    or self.metadata.get("feedback_fps") != cfg["model"]["fps"]):
                raise ValueError("服务器动作维度或反馈帧率不匹配")
            # 论文参数配置显式核验采样方式，防止连接到同权重的其他推理配置。
            sampling = validate_sampling(cfg.get("sampling", {}))
            if sampling and not self.metadata.get("runtime_sampling_parameters"):
                raise ValueError("服务尚未支持客户端采样参数，请停止服务后重新启用")
            for key in (() if sampling else ('inference_profile', 'denoise_steps', 'rtc_max_guidance')):
                if key in cfg['model'] and self.metadata.get(key) != cfg['model'][key]:
                    raise ValueError(f'服务器推理参数不匹配 {key}: {self.metadata.get(key)}')
            # 模型历史契约独立于可编辑的推理参数。
            for key in ('action_history_source', 'action_history_frames', 'action_history_bucket_frames'):
                if key in cfg['model'] and self.metadata.get(key) != cfg['model'][key]:
                    raise ValueError(f'服务器状态历史契约不匹配 {key}: {self.metadata.get(key)}')
            if "predict_ut" in cfg["model"] and self.metadata.get("predict_ut") is not cfg["model"]["predict_ut"]:
                raise ValueError("服务器动作预测方式与所选模型不匹配")
            expected_checkpoint = cfg["model"].get("checkpoint_contains")
            if expected_checkpoint and expected_checkpoint not in self.metadata.get("checkpoint", ""):
                raise ValueError("服务器检查点来源与所选模型不匹配")
            self.local_ip = raw.getsockname()[0]
        except BaseException:
            self.close()
            raw.close()
            raise
        self.episode = uuid.uuid4().hex
        self.previous = None
        self.packer = msgpack_numpy.Packer()
        self.reserved = False

    def _receive(self):
        response = self.ws.recv(timeout=self.cfg["connection"]["inference_timeout_s"])
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack_numpy.unpackb(response)

    def validate_mode(self, mode, prompt):
        if mode not in self.metadata.get("modes", ["tempo"]):
            raise ValueError("服务器尚未支持所选推理模式，请启动更新后的模型服务")
        if prompt not in self.metadata.get("task_prompts", [prompt]):
            raise ValueError("所选任务与当前服务模型不匹配")
        if mode == "rtc":
            # 旧模型配置仍按潜变量 RTC 协议校验。
            expected = ("direct_action_vjp_soft_prefix" if self.cfg["model"].get("predict_ut") is False
                        else "decoded_action_vjp_soft_prefix")
            if self.metadata.get("rtc_algorithm") != expected:
                raise ValueError("服务器RTC引导算法与所选模型不匹配")
        algorithms = {"dynamicvla": "continuous_latency_aware_streaming", "vlash": "future_state_rollforward"}
        if mode in algorithms and self.metadata.get("inference_algorithms", {}).get(mode) != algorithms[mode]:
            raise ValueError("服务器推理算法实现与所选模式不匹配")

    def reserve(self):
        if self.reserved:
            return
        if self.metadata.get("protocol_version", 0) < 2:
            raise ValueError("模型服务需要更新到任务与运动控制协议，请启动新版服务")
        try:
            self.ws.send(self.packer.pack({"operation": "claim"}))
            if self._receive().get("claimed") is not True:
                raise RuntimeError("服务器推理会话预约失败")
        except BaseException:
            self.close()
            raise
        self.reserved = True

    def infer(self, frames, *, mode="tempo", prompt=None, rtc_prefix=None, rtc_delay=0,
              future_state=None, future_state_offset_steps=0):
        stamp = validate_frames(frames, self.cfg["model"]["fps"], self.previous)
        if mode not in ("tempo", "temporal_ensemble", "rtc", "dynamicvla", "vlash"):
            raise ValueError("推理模式不支持")
        request = {"episode_id": self.episode, "feedback": frames, "mode": mode}
        sampling = validate_sampling(self.cfg.get("sampling", {}))
        if sampling:
            if not self.metadata.get("runtime_sampling_parameters"):
                raise ValueError("服务尚未支持客户端采样参数，请重新启用模型加载新版服务")
            request["sampling_parameters"] = sampling
        if type(future_state_offset_steps) is not int:
            raise ValueError("VLASH时间偏移必须为整数")
        if future_state is not None:
            state = np.asarray(future_state, np.float32)
            if (mode != "vlash" or state.shape != (7,) or not np.isfinite(state).all()
                    or type(future_state_offset_steps) is not int
                    or not 1 <= future_state_offset_steps < self.cfg["model"]["action_horizon"]):
                raise ValueError("VLASH未来状态或时间偏移不合法")
            request.update(future_state=state.copy(), future_state_offset_steps=future_state_offset_steps)
        elif future_state_offset_steps != 0:
            raise ValueError("VLASH时间偏移需要对应的未来状态")
        if prompt is not None:
            request["prompt"] = prompt
        if rtc_prefix is not None:
            prefix = np.asarray(rtc_prefix, np.float32)
            if (mode != "rtc" or prefix.ndim != 2 or prefix.shape[1] != 7
                    or not 1 <= len(prefix) <= self.cfg["model"]["action_horizon"]
                    or not np.isfinite(prefix).all() or type(rtc_delay) is not int or not 0 <= rtc_delay <= len(prefix)):
                raise ValueError("RTC绝对动作前缀或延迟不合法")
            request.update(rtc_prefix=prefix, rtc_delay=rtc_delay)
        begin = time.monotonic()
        # 失败后关闭连接，避免服务端已消费输入时重发同一批并污染历史。
        try:
            self.ws.send(self.packer.pack(request))
            result = self._receive()
            if any(result.get("sampling_parameters", {}).get(key) != value for key, value in sampling.items()):
                raise ValueError("服务端实际采样参数与客户端设置不一致")
            actions = np.asarray(result["actions"])
            expected = (self.cfg["model"]["action_horizon"], self.cfg["model"]["action_dim"])
            if actions.shape != expected or actions.dtype != np.float32 or not np.isfinite(actions).all():
                raise ValueError("服务器返回动作形状、类型或有限性错误")
            if result.get("mode", "tempo") != mode:
                raise ValueError("服务端实际推理模式与请求不匹配")
            if mode in ("dynamicvla", "vlash") and result.get("action_start_offset_steps") != future_state_offset_steps:
                raise ValueError("服务端动作时间偏移与请求不一致")
        except BaseException:
            self.close()
            raise
        self.previous = stamp
        result["round_trip_ms"] = (time.monotonic() - begin) * 1000
        return result

    def close(self):
        if self.ws is not None:
            self.ws.close()

    def abort(self):
        # GUI停止不能等待模型返回；关闭当前socket中断recv，新任务重建episode。
        if self.ws is not None:
            try:
                self.ws.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
