"""只读取 CAN 与双相机，在固定时间网格上对实测关节和夹爪插值。"""

from collections import deque
import importlib
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np


def letterbox(rgb):
    height, width = rgb.shape[:2]
    scale = min(224 / width, 224 / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas = np.zeros((224, 224, 3), np.uint8)
    left, top = (224 - new_width) // 2, (224 - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = cv2.resize(
        rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return canvas


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
    def __init__(self, cfg):
        self.cfg = cfg["live"]
        self.fps = cfg["model"]["fps"]
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.arm = None
        self.threads = []
        self.joints, self.gripper = deque(maxlen=180), deque(maxlen=180)
        self.images = {name: deque(maxlen=60) for name in self.cfg["cameras"]}
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
            values = (self.arm.get_joint_angles(), self.effector.get_gripper_status())
            now = time.time()
            with self.lock:
                for index, (history, feedback) in enumerate(zip((self.joints, self.gripper), values)):
                    if feedback is None or feedback.hz <= 0:
                        continue
                    stamp = float(feedback.timestamp)
                    if not 0 <= now - stamp <= self.cfg["max_feedback_age_s"]:
                        raise RuntimeError("CAN反馈陈旧或时钟异常")
                    if index == 1 and feedback.msg.mode != "width":
                        raise RuntimeError("夹爪反馈应为米制开口")
                    value = np.asarray(feedback.msg if index == 0 else [feedback.msg.value], np.float32)
                    if value.shape != ((6,) if index == 0 else (1,)) or not np.isfinite(value).all():
                        raise RuntimeError("CAN反馈维度或数值错误")
                    if not history or stamp > history[-1][0]:
                        history.append((stamp, value.copy()))
            self.stop.wait(1 / self.cfg["sample_hz"])

    def _sample_camera(self, name, settings):
        import pyrealsense2 as rs
        parameters = json.loads(Path(self.cfg["camera_parameters"]).read_text())
        options = parameters["devices"][settings["name"]]["configured_options"]
        pipe, config = rs.pipeline(), rs.config()
        config.enable_device(settings["serial"])
        config.enable_stream(rs.stream.color, settings["width"], settings["height"], rs.format.rgb8, settings["fps"])
        started = False
        try:
            profile = pipe.start(config)
            started = True
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
                for key in sorted(options, key=lambda key: (key not in ("enable_auto_exposure", "enable_auto_white_balance"), key)):
                    option = getattr(rs.option, key)
                    if sensor.supports(option):
                        sensor.set_option(option, options[key])
            while not self.stop.is_set():
                frame = pipe.wait_for_frames(500).get_color_frame()
                if not frame or frame.get_frame_timestamp_domain() not in (
                        rs.timestamp_domain.global_time, rs.timestamp_domain.system_time):
                    continue
                stamp = frame.get_timestamp() / 1000
                if not 0 <= time.time() - stamp <= self.cfg["max_camera_age_s"]:
                    continue
                image = letterbox(np.asanyarray(frame.get_data()))
                with self.lock:
                    history = self.images[name]
                    if not history or stamp > history[-1][0]:
                        history.append((stamp, image))
        finally:
            if started:
                pipe.stop()

    def _frame(self, stamp):
        with self.lock:
            state = np.concatenate([interpolate(history, stamp, self.cfg["max_feedback_age_s"])
                                    for history in (self.joints, self.gripper)])
            images, stamps = {}, []
            for name, history in self.images.items():
                available = [value for value in history if value[0] <= stamp]
                if not available:
                    raise LookupError("等待相机帧")
                image_stamp, image = available[-1]
                if stamp - image_stamp > self.cfg["max_camera_age_s"]:
                    raise RuntimeError("相机帧陈旧")
                images[name] = image.copy()
                stamps.append(image_stamp)
            if max(stamps) - min(stamps) > self.cfg["max_pair_skew_s"]:
                raise RuntimeError("双相机时间差过大")
        return {"timestamp": stamp, "state": state, "images": images,
                "capture_timestamps": dict(zip(images, stamps))}

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

    def take(self):
        while True:
            if self.error is not None:
                raise RuntimeError(str(self.error)) from self.error
            try:
                frames = [self.pending.get(timeout=.1)]
                break
            except queue.Empty:
                if self.stop.is_set():
                    raise RuntimeError("反馈采样已停止")
        while len(frames) < self.cfg["max_batch_frames"]:
            try:
                frames.append(self.pending.get_nowait())
            except queue.Empty:
                break
        return frames

    def __enter__(self):
        try:
            self._connect_arm()
            self._start(self._sample_arm)
            for name, settings in self.cfg["cameras"].items():
                self._start(self._sample_camera, name, settings)
            deadline = time.monotonic() + self.cfg["startup_timeout_s"]
            while not self.stop.is_set():
                if self.error is not None:
                    raise RuntimeError(str(self.error)) from self.error
                with self.lock:
                    ready = len(self.joints) > 1 and len(self.gripper) > 1 and all(self.images.values())
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("CAN或双相机启动超时")
                self.stop.wait(.02)
            if self.error is not None:
                raise RuntimeError(str(self.error)) from self.error
            if self.stop.is_set():
                raise InterruptedError("硬件启动已取消")
            self._start(self._produce, time.time())
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *args):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)
        if self.arm is not None:
            self.arm.disconnect()
