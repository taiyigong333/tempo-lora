"""校验有线出口、模型身份和时序，网络失败后由新会话重建在线记忆。"""

import socket
import time
import uuid

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


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
                              open_timeout=connection["connect_timeout_s"], ping_timeout=300)
            self.metadata = self._receive()
            for key in ("config", "step", "action_horizon", "action_space"):
                if self.metadata.get(key) != cfg["model"][key]:
                    raise ValueError(f"服务器模型身份不匹配 {key}: {self.metadata.get(key)}")
            if (self.metadata.get("native_action_dim") != cfg["model"]["action_dim"]
                    or self.metadata.get("feedback_fps") != cfg["model"]["fps"]):
                raise ValueError("服务器动作维度或反馈帧率不匹配")
            self.local_ip = raw.getsockname()[0]
        except BaseException:
            self.close()
            raw.close()
            raise
        self.episode = uuid.uuid4().hex
        self.previous = None
        self.packer = msgpack_numpy.Packer()

    def _receive(self):
        response = self.ws.recv(timeout=self.cfg["connection"]["inference_timeout_s"])
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack_numpy.unpackb(response)

    def infer(self, frames):
        stamp = validate_frames(frames, self.cfg["model"]["fps"], self.previous)
        begin = time.monotonic()
        # 失败后关闭连接，避免服务端已消费输入时重发同一批并污染历史。
        try:
            self.ws.send(self.packer.pack({"episode_id": self.episode, "feedback": frames}))
            result = self._receive()
            actions = np.asarray(result["actions"])
            expected = (self.cfg["model"]["action_horizon"], self.cfg["model"]["action_dim"])
            if actions.shape != expected or actions.dtype != np.float32 or not np.isfinite(actions).all():
                raise ValueError("服务器返回动作形状、类型或有限性错误")
        except BaseException:
            self.close()
            raise
        self.previous = stamp
        result["round_trip_ms"] = (time.monotonic() - begin) * 1000
        return result

    def close(self):
        if self.ws is not None:
            self.ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
