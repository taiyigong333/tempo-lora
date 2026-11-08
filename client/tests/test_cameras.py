"""真实子进程中的合成相机回归，验证持续开流、短缓存与跨轮隔离。"""

import copy
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tempo_client.cameras import CameraStream
from tempo_client.live import LiveFeedback
from tempo_client.session import SessionWorker


class SyntheticCapture:
    def __init__(self, cfg):
        self.cfg = cfg

    def mark(self, event):
        with Path(self.cfg["capture_marker"]).open("a") as stream:
            stream.write(f"{os.getpid()} {event}\n")

    def start(self, stop):
        self.stop = stop
        self.mark("start")

    def read_batch(self):
        self.stop.wait(1 / 30)
        stamp = time.time()
        return {name: (stamp, np.full((224, 224, 3), 64 + index, np.uint8))
                for index, name in enumerate(self.cfg["cameras"])}

    def close(self):
        self.mark("close")


class FailedCapture(SyntheticCapture):
    def read_batch(self):
        raise RuntimeError("合成相机故障")


class BlockedCapture(SyntheticCapture):
    def read_batch(self):
        time.sleep(30)


class SyntheticArm:
    def get_joint_angles(self):
        return SimpleNamespace(msg=np.zeros(6, np.float32), timestamp=time.time(), hz=90)

    def get_gripper_status(self):
        return SimpleNamespace(msg=SimpleNamespace(mode="width", value=.05), timestamp=time.time(), hz=90)

    def disconnect(self):
        pass


class SyntheticFeedback(LiveFeedback):
    def _connect_arm(self):
        self.arm = self.effector = SyntheticArm()


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parameters = self.root / "parameters.json"
        self.parameters.write_text("{}")
        self.cfg = {"model": {"fps": 30}, "live": {
            "cameras": {name: {"fps": 30} for name in ("head", "left_wrist")},
            "camera_parameters": str(self.parameters), "capture_marker": str(self.root / "capture.log"),
            "max_camera_age_s": .2, "interpolation_wait_s": .1, "max_pair_skew_s": .1,
            "startup_timeout_s": 5, "camera_stop_timeout_s": .2, "camera_lock_timeout_s": .1,
            "max_feedback_age_s": .1, "sample_hz": 90, "max_pending_frames": 60, "max_batch_frames": 8}}
        self.streams = []
        self.worker = None

    def tearDown(self):
        if self.worker is not None:
            self.worker.close_cameras()
        for stream in self.streams:
            stream.close()
        self.temporary.cleanup()

    def make_stream(self, factory=SyntheticCapture):
        stream = CameraStream(self.cfg, capture_factory=factory)
        self.streams.append(stream)
        stream.start()
        return stream

    def test_idle_keeps_capture_running_with_bounded_latest_frames(self):
        stream = self.make_stream()
        first = stream.wait_ready(threading.Event())
        pid = stream.process.pid
        time.sleep(.5)
        latest = stream.snapshot()
        self.assertEqual(stream.process.pid, pid)
        self.assertGreater(min(latest["capture_timestamps"].values()), max(first["capture_timestamps"].values()))
        self.assertGreater(stream.counts[0], stream.capacity)
        self.assertEqual(len(stream.stamps), 2 * stream.capacity)
        lines = (self.root / "capture.log").read_text().splitlines()
        self.assertEqual(lines, [f"{pid} start"])

    def test_multiple_feedback_rounds_share_process_and_use_new_frames(self):
        stream = self.make_stream()
        stream.wait_ready(threading.Event())
        pid = stream.process.pid
        previous = None
        for _ in range(2):
            boundary = time.time()
            with SyntheticFeedback(self.cfg, cameras=stream) as source:
                batch = source.take()
                self.assertGreater(min(batch[0]["capture_timestamps"].values()), boundary)
                self.assertEqual(batch[0]["state"].shape, (7,))
                self.assertEqual(stream.process.pid, pid)
                if previous is not None:
                    self.assertGreater(batch[0]["timestamp"], previous)
                previous = batch[-1]["timestamp"]
            self.assertTrue(stream.process.is_alive())
            time.sleep(.1)
        self.assertEqual((self.root / "capture.log").read_text().count("start"), 1)

    def test_manager_reuses_and_only_rebuilds_for_changed_configuration(self):
        self.worker = SessionWorker()
        def factory(cfg):
            return CameraStream(cfg, capture_factory=SyntheticCapture)
        with patch("tempo_client.session.CameraStream", side_effect=factory):
            first = self.worker._prepare_cameras(self.cfg)
            second = self.worker._prepare_cameras(self.cfg)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["process_id"], second["process_id"])
            changed = copy.deepcopy(self.cfg)
            changed["live"]["cameras"]["head"]["fps"] = 25
            third = self.worker._prepare_cameras(changed)
            self.assertFalse(third["reused"])
            self.assertNotEqual(first["process_id"], third["process_id"])
        self.worker.close_cameras()
        self.assertEqual((self.root / "capture.log").read_text().count("start"), 2)
        self.assertEqual((self.root / "capture.log").read_text().count("close"), 2)

    def test_preview_pause_discards_frames_while_capture_continues(self):
        self.worker = SessionWorker()
        with patch("tempo_client.session.CameraStream", side_effect=lambda cfg: CameraStream(cfg, capture_factory=SyntheticCapture)):
            self.worker._prepare_cameras(self.cfg)
        count = self.worker.cameras.counts[0]
        self.worker.preview_enabled.clear()
        self.worker.take_preview()
        time.sleep(.3)
        self.assertIsNone(self.worker.take_preview())
        self.assertGreater(self.worker.cameras.counts[0], count)
        self.worker.preview_enabled.set()
        time.sleep(.3)
        self.assertIsNotNone(self.worker.take_preview())

    def test_failed_stream_is_not_automatically_reopened(self):
        stream = self.make_stream(FailedCapture)
        pid = stream.process.pid
        with self.assertRaisesRegex(RuntimeError, "合成相机故障"):
            stream.wait_ready(threading.Event())
        stream.start()
        self.assertEqual(stream.process.pid, pid)
        self.assertEqual((self.root / "capture.log").read_text().count("start"), 1)

    def test_close_recovers_a_blocked_owned_process(self):
        stream = self.make_stream(BlockedCapture)
        process = stream.process
        deadline = time.monotonic() + 5
        while not (self.root / "capture.log").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue((self.root / "capture.log").exists())
        stream.close()
        self.assertFalse(process.is_alive())

    def test_partial_process_start_failure_can_be_released(self):
        stream = CameraStream(self.cfg)
        self.streams.append(stream)
        process = Mock()
        process.start.side_effect = RuntimeError("无法创建相机进程")
        with patch.object(stream.context, "Process", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "无法创建相机进程"):
                stream.start()
        self.assertIsNone(stream.process)
        self.assertTrue(stream.channel.closed)
        stream.close()


if __name__ == "__main__":
    unittest.main()
