"""三种真实调度分支、过期停止和实机出口的合成回归。"""

import io
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np

from tempo_client.cli import DEFAULT_CONFIG, load_config
from tempo_client.control import InferenceRun, Plan, Plans
from tempo_client.settings import select_runtime
from tempo_client.transport import validate_frames


class TimedSource:
    def __init__(self):
        self.origin_wall = time.time()
        self.stop = threading.Event()
        self.index = 0
        self.image = np.zeros((224, 224, 3), np.uint8)

    def take(self, *, max_frames):
        while not self.stop.is_set():
            latest = int((time.time() - self.origin_wall) * 30)
            if latest >= self.index:
                end = min(latest + 1, self.index + max_frames)
                frames = [{"timestamp": self.origin_wall + index / 30,
                           "state": np.zeros(7, np.float32),
                           "images": {name: self.image for name in ("head", "left_wrist")}}
                          for index in range(self.index, end)]
                self.index = end
                return frames
            self.stop.wait(.002)
        raise InterruptedError("合成采样停止")


class TimedClient:
    def __init__(self, latency=.01):
        self.latency = latency
        self.stopped = threading.Event()
        self.entered = threading.Event()
        self.previous = None
        self.requests = []

    def infer(self, frames, **kwargs):
        self.previous = validate_frames(frames, 30, self.previous)
        self.requests.append(kwargs)
        self.entered.set()
        if self.stopped.wait(self.latency):
            raise OSError("合成网络已关闭")
        prefix = kwargs.get("rtc_prefix")
        return {"actions": np.full((16, 7), .05, np.float32), "round_trip_ms": self.latency * 1000,
                "mode": kwargs["mode"], "rtc_prefix_length": 0 if prefix is None else len(prefix)}

    def abort(self):
        self.stopped.set()


class TargetRobot:
    def __init__(self):
        self.sent_commands = 0
        self.targets = []
        self.held = False
        self.check_target = Mock()

    def read_state(self):
        return np.zeros(7, np.float32)

    def check_status(self):
        pass

    def send_action(self, action, cancel):
        if cancel.is_set() or self.held:
            raise InterruptedError("合成机器人已停止")
        self.targets.append(np.asarray(action).copy())
        self.sent_commands += 2

    def hold(self):
        self.held = True


class ControlTests(unittest.TestCase):
    def make_run(self, mode, root, *, execute=False, latency=.01):
        cfg = select_runtime(load_config(DEFAULT_CONFIG), mode=mode)
        summary = {"batches": 0, "feedback_frames": 0, "log_dir": str(root)}
        client, source, robot = TimedClient(latency), TimedSource(), TargetRobot()
        run = InferenceRun(cfg, client, source, robot, threading.Event(), queue.Queue(),
                           io.StringIO(), summary, execute)
        return run, client, robot, summary

    def test_native_readonly_has_planned_targets_and_zero_commands(self):
        with tempfile.TemporaryDirectory() as root:
            run, client, robot, summary = self.make_run("tempo", root)
            run.run(.15)
        self.assertGreater(summary["planned_targets"], 0)
        self.assertEqual(summary["robot_commands"], 0)
        self.assertFalse(robot.targets)
        self.assertTrue(all(request["mode"] == "tempo" for request in client.requests))

    def test_native_motion_dispatches_and_holds(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, robot, summary = self.make_run("tempo", root, execute=True)
            run.run(.12)
        self.assertTrue(robot.targets)
        self.assertTrue(robot.held)
        self.assertEqual(summary["robot_commands"], 2 * len(robot.targets))

    def test_temporal_ensemble_weights_same_target_time(self):
        plans = Plans("temporal_ensemble", 16, .1)
        plans.add(Plan(0, np.zeros((16, 7), np.float32), time.time()))
        plans.add(Plan(1, np.full((16, 7), 2, np.float32), time.time()))
        action, _ = plans.action(3, time.time(), .5)
        expected = 2 * np.exp(-.2) / (np.exp(-.3) + np.exp(-.2))
        np.testing.assert_allclose(action, expected, atol=1e-6)
        self.assertIsNone(plans.action(17, time.time(), .5))

    def test_async_modes_run_real_control_loop(self):
        for mode in ("temporal_ensemble", "rtc"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                run, client, _, summary = self.make_run(mode, root)
                run.run(.2)
                self.assertGreaterEqual(summary["planned_targets"], 4)
                self.assertEqual(summary["robot_commands"], 0)
                if mode == "rtc":
                    self.assertGreater(summary["rtc_guided_requests"], 0)
                    self.assertTrue(any(request.get("rtc_prefix") is not None for request in client.requests))

    def test_old_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, robot, _ = self.make_run("tempo", root, execute=True, latency=.55)
            with self.assertRaisesRegex(RuntimeError, "观测过期"):
                run.run(.2)
        self.assertFalse(robot.targets)
        self.assertTrue(robot.held)

    def test_async_result_outside_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, robot, _ = self.make_run("temporal_ensemble", root, latency=.65)
            with self.assertRaisesRegex(RuntimeError, "动作窗口"):
                run.run(.2)
        self.assertFalse(robot.targets)

    def test_readonly_invalid_target_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, robot, summary = self.make_run("tempo", root)
            robot.check_target.side_effect = ValueError("目标越界")
            run.run(.1)
        self.assertGreater(summary["safety_violations"], 0)
        self.assertFalse(robot.targets)

    def test_motion_invalid_target_stops_before_command(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, robot, _ = self.make_run("tempo", root, execute=True)
            robot.check_target.side_effect = ValueError("目标越界")
            with self.assertRaisesRegex(ValueError, "目标越界"):
                run.run(.1)
        self.assertFalse(robot.targets)
        self.assertTrue(robot.held)
