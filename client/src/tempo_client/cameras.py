"""相机进程由应用持有，持续排空设备帧，仅保留时间对齐所需的短缓存。"""

import json
import math
import multiprocessing as mp
from pathlib import Path
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


class RealSenseCapture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipelines = []

    def start(self, stop):
        import pyrealsense2 as rs
        parameters = json.loads(Path(self.cfg["camera_parameters"]).read_text())
        for name, settings in self.cfg["cameras"].items():
            if stop.is_set():
                raise InterruptedError("相机启动已取消")
            pipe, config = rs.pipeline(), rs.config()
            config.enable_device(settings["serial"])
            config.enable_stream(rs.stream.color, settings["width"], settings["height"], rs.format.rgb8, settings["fps"])
            profile = pipe.start(config)
            self.pipelines.append((name, pipe))
            options = parameters["devices"][settings["name"]]["configured_options"]
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
                for key in sorted(options, key=lambda key: (key not in ("enable_auto_exposure", "enable_auto_white_balance"), key)):
                    option = getattr(rs.option, key)
                    if sensor.supports(option):
                        sensor.set_option(option, options[key])

    def read_batch(self):
        import pyrealsense2 as rs
        images = {}
        for name, pipe in self.pipelines:
            frame = pipe.wait_for_frames(500).get_color_frame()
            # 持续排空设备队列，空闲时也不会把未使用的画面积累到下一轮。
            while True:
                newer = pipe.poll_for_frames()
                if not newer:
                    break
                frame = newer.get_color_frame()
            if not frame or frame.get_frame_timestamp_domain() not in (
                    rs.timestamp_domain.global_time, rs.timestamp_domain.system_time):
                raise LookupError("等待相机时间域稳定")
            stamp = frame.get_timestamp() / 1000
            if not 0 <= time.time() - stamp <= self.cfg["max_camera_age_s"]:
                raise LookupError("等待新鲜相机帧")
            images[name] = (stamp, letterbox(np.asanyarray(frame.get_data())))
        return images

    def close(self):
        # 即使一路关闭失败，也继续释放已打开的其他pipeline。
        errors = []
        for _, pipe in self.pipelines:
            try:
                pipe.stop()
            except Exception as error:
                errors.append(error)
        self.pipelines.clear()
        if errors:
            raise errors[0]


def _capture_worker(cfg, capacity, pixels, stamps, counts, lock, stop, channel, capture_factory):
    capture = capture_factory(cfg)
    names = list(cfg["cameras"])
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(len(names), capacity, 224, 224, 3)
    times = np.frombuffer(stamps, dtype=np.float64).reshape(len(names), capacity)
    try:
        capture.start(stop)
        while not stop.is_set():
            try:
                batch = capture.read_batch()
            except LookupError:
                stop.wait(.005)
                continue
            with lock:
                for index, name in enumerate(names):
                    stamp, rgb = batch[name]
                    if counts[index] and stamp <= times[index, (counts[index] - 1) % capacity]:
                        continue
                    slot = counts[index] % capacity
                    view[index, slot] = rgb
                    times[index, slot] = stamp
                    counts[index] += 1
    except Exception as error:
        channel.send(f"{type(error).__name__}: {error}")
    finally:
        try:
            capture.close()
        except Exception as error:
            channel.send(f"相机关闭失败: {error}")
        channel.close()


class CameraStream:
    def __init__(self, cfg, *, capture_factory=RealSenseCapture):
        self.cfg = cfg["live"]
        self.names = list(self.cfg["cameras"])
        if set(self.names) != {"head", "left_wrist"}:
            raise ValueError("需要head与left_wrist两路相机")
        fps = max(value["fps"] for value in self.cfg["cameras"].values())
        # 只保留反馈插值等待窗口，未被消费的旧帧自动覆盖。
        self.capacity = max(2, math.ceil((self.cfg["max_camera_age_s"] + self.cfg["interpolation_wait_s"]) * fps) + 2)
        self.context = mp.get_context("spawn")
        self.capture_factory = capture_factory
        self.process = None
        self.error = None
        self.started = False
        self.status_lock = threading.Lock()

    def start(self):
        if self.started:
            return
        self.stop = self.context.Event()
        self.lock = self.context.Lock()
        self.pixels = self.context.RawArray("B", len(self.names) * self.capacity * 224 * 224 * 3)
        self.stamps = self.context.RawArray("d", len(self.names) * self.capacity)
        self.counts = self.context.RawArray("Q", len(self.names))
        self.channel, child = self.context.Pipe(duplex=False)
        self.process = self.context.Process(target=_capture_worker, args=(self.cfg, self.capacity,
            self.pixels, self.stamps, self.counts, self.lock, self.stop, child, self.capture_factory),
            name="tempo-cameras", daemon=True)
        try:
            self.process.start()
        except BaseException:
            self.channel.close()
            self.process = None
            raise
        finally:
            child.close()
        self.started = True

    def snapshot(self, stamp=None):
        with self.status_lock:
            if not self.started or self.process is None:
                raise RuntimeError("相机流尚未开启")
            while self.channel.poll():
                try:
                    self.error = self.channel.recv()
                except EOFError:
                    break
            if self.error is not None:
                raise RuntimeError(self.error)
            if not self.process.is_alive():
                raise RuntimeError(f"相机进程已退出，exitcode={self.process.exitcode}")
        stamp = time.time() if stamp is None else stamp
        if not self.lock.acquire(timeout=self.cfg.get("camera_lock_timeout_s", .05)):
            raise RuntimeError("相机共享帧锁超时")
        try:
            view = np.frombuffer(self.pixels, np.uint8).reshape(len(self.names), self.capacity, 224, 224, 3)
            times = np.frombuffer(self.stamps, np.float64).reshape(len(self.names), self.capacity)
            images, capture_times = {}, {}
            for index, name in enumerate(self.names):
                available = np.flatnonzero((times[index] > 0) & (times[index] <= stamp))
                if not len(available):
                    raise LookupError("等待时间网格之前的相机帧")
                slot = available[np.argmax(times[index, available])]
                image_stamp = float(times[index, slot])
                if stamp - image_stamp > self.cfg["max_camera_age_s"]:
                    raise RuntimeError("相机帧陈旧")
                images[name] = view[index, slot].copy()
                capture_times[name] = image_stamp
        finally:
            self.lock.release()
        if max(capture_times.values()) - min(capture_times.values()) > self.cfg["max_pair_skew_s"]:
            raise RuntimeError("双相机时间差过大")
        return {"images": images, "capture_timestamps": capture_times}

    def wait_ready(self, cancel, *, after=None):
        deadline = time.monotonic() + self.cfg["startup_timeout_s"]
        while not cancel.is_set():
            try:
                frame = self.snapshot()
                if after is None or min(frame["capture_timestamps"].values()) > after:
                    return frame
            except LookupError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("双相机等待新帧超时")
            cancel.wait(.01)
        raise InterruptedError("相机等待已取消，视频流保留")

    def close(self):
        if self.process is None:
            return
        self.stop.set()
        timeout = self.cfg.get("camera_stop_timeout_s", 2)
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout)
        if self.process.is_alive():
            raise RuntimeError("相机进程尚未退出，请检查设备")
        self.channel.close()
        self.process = None
