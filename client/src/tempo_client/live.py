"""只读取 CAN 与双相机，在固定时间网格上对实测关节和夹爪插值。"""

from collections import deque
import importlib
import contextlib
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np


# 保留原导入路径，图像处理与相机进程共享同一实现。
from tempo_client.cameras import CameraStream, letterbox


def interpolate(samples, stamp, max_gap):
    before = [value for value in samples if value[0] <= stamp]
    after = [value for value in samples if value[0] >= stamp]
    if not before or not after:
        raise LookupError("等待时间网格两侧的实测反馈")
    t0, v0 = before[-1]
    t1, v1 = after[0]
    if t1 - t0 > max_gap:
        raise RuntimeError("实测反馈缺口超过阈值")
    weight = 0.0 if t1 == t0 else (stamp - t0) / (t1 - t0)
    return np.asarray(v0 + weight * (v1 - v0), np.float32)


class LiveFeedback:
    def __init__(self, cfg, *, cameras=None, robot=None):
        self.cfg = cfg["live"]
        self.fps = cfg["model"]["fps"]
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.arm = None
        self.robot = robot
        self.threads = []
        self.joints, self.gripper = deque(maxlen=180), deque(maxlen=180)
        self.cameras = cameras
        self.owns_cameras = cameras is None
        self.pending = queue.Queue(maxsize=self.cfg["max_pending_frames"])

    def _worker(self, function, *args):
        try:
            function(*args)
        except Exception as error:
            with self.lock:
                self.error = error
            self.stop.set()

    def _start(self, function, *args):
        thread = threading.Thread(target=self._worker, args=(function, *args), daemon=True)
        self.threads.append(thread)
        thread.start()

    def _connect_arm(self):
        if self.robot is not None:
            self.arm, self.effector = self.robot.arm, self.robot.gripper
            return
        sdk_path = Path(self.cfg["sdk_path"]).resolve()
        sys.path.insert(0, str(sdk_path))
        sdk = importlib.import_module("pyAgxArm")
        if not Path(sdk.__file__).resolve().is_relative_to(sdk_path):
            raise RuntimeError("SDK 来源与配置不符")
        self.arm = sdk.AgxArmFactory.create_arm(sdk.create_agx_arm_config(
            robot=sdk.ArmModel.PIPER_X, firmeware_version=self.cfg["firmware_version"],
            interface="socketcan", channel=self.cfg["can_name"]))
        self.arm.connect()
        self.effector = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

    def _sample_arm(self):
        while not self.stop.is_set():
            with self.robot.lock if self.robot is not None else contextlib.nullcontext():
                # getter返回可变缓存，锁内保存时间戳与数值，并在时间戳之后读取当前时刻。
                values = (self.arm.get_joint_angles(), self.effector.get_gripper_status())
                with self.lock:
                    for index, (history, feedback) in enumerate(zip((self.joints, self.gripper), values)):
                        if feedback is None or feedback.hz <= 0:
                            continue
                        stamp = float(feedback.timestamp)
                        age = time.time() - stamp
                        if not 0 <= age <= self.cfg["max_feedback_age_s"]:
                            raise RuntimeError(f"CAN反馈陈旧或时钟异常，通道{index}，帧龄{age:.6f}秒")
                        if index == 1 and feedback.msg.mode != "width":
                            raise RuntimeError("夹爪反馈应为米制开口")
                        value = np.asarray(feedback.msg if index == 0 else [feedback.msg.value], np.float32)
                        if value.shape != ((6,) if index == 0 else (1,)) or not np.isfinite(value).all():
                            raise RuntimeError("CAN反馈维度或数值错误")
                        if not history or stamp > history[-1][0]:
                            history.append((stamp, value.copy()))
            self.stop.wait(1 / self.cfg["sample_hz"])

    def _frame(self, stamp):
        with self.lock:
            state = np.concatenate([interpolate(history, stamp, self.cfg["max_feedback_age_s"])
                                    for history in (self.joints, self.gripper)])
        capture = self.cameras.snapshot(stamp)
        return {"timestamp": stamp, "state": state, **capture}

    def _produce(self, origin):
        index = 0
        while not self.stop.is_set():
            stamp = origin + index / self.fps
            deadline = time.monotonic() + self.cfg["interpolation_wait_s"]
            while not self.stop.is_set():
                try:
                    frame = self._frame(stamp)
                    break
                except LookupError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("时间网格缺少真实反馈，结束当前episode")
                    self.stop.wait(.002)
            if self.stop.is_set():
                return
            try:
                self.pending.put_nowait(frame)
            except queue.Full as error:
                raise RuntimeError("推理积压超过上限，结束当前episode，不能丢帧继续") from error
            index += 1
            self.stop.wait(max(0, origin + index / self.fps - time.time()))

    def take(self, *, max_frames=None):
        while True:
            if self.error is not None:
                raise RuntimeError(str(self.error)) from self.error
            try:
                frames = [self.pending.get(timeout=.1)]
                break
            except queue.Empty:
                if self.stop.is_set():
                    raise RuntimeError("反馈采样已停止")
        maximum = self.cfg["max_batch_frames"] if max_frames is None else max_frames
        while len(frames) < maximum:
            try:
                frames.append(self.pending.get_nowait())
            except queue.Empty:
                break
        return frames

    def __enter__(self):
        try:
            self._connect_arm()
            self._start(self._sample_arm)
            deadline = time.monotonic() + self.cfg["startup_timeout_s"]
            while not self.stop.is_set():
                if self.error is not None:
                    raise RuntimeError(str(self.error)) from self.error
                with self.lock:
                    ready = len(self.joints) > 1 and len(self.gripper) > 1
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("CAN或双相机启动超时")
                self.stop.wait(.02)
            if self.error is not None:
                raise RuntimeError(str(self.error)) from self.error
            if self.stop.is_set():
                raise InterruptedError("硬件启动已取消")
            if self.owns_cameras:
                self.cameras = CameraStream({"live": self.cfg})
                self.cameras.start()
            # 每轮只接收本轮CAN就绪之后的新图，持续视频流不携带上一轮历史。
            self.cameras.wait_ready(self.stop, after=time.time())
            self.origin_wall = time.time()
            self._start(self._produce, self.origin_wall)
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *args):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)
        try:
            if self.arm is not None and self.robot is None:
                self.arm.disconnect()
        finally:
            if self.owns_cameras and self.cameras is not None:
                self.cameras.close()
