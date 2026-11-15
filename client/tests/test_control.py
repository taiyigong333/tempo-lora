"""三种真实调度分支、过期停止和实机出口的合成回归。"""

import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_native_anchor_with_virtual_latency_and_eight_step_tail(self):
        for latency in (.13, .2, .3):
            for execute_steps in (1, 8):
                with self.subTest(latency=latency, execute_steps=execute_steps), tempfile.TemporaryDirectory() as root:
                    run, _, robot, summary = self.make_run("tempo", root, execute=True)
                    run.inference["execute_steps"] = execute_steps
                    clock = SimpleNamespace(elapsed=0.)

                    class Stop:
                        stopped = False

                        def is_set(self):
                            return self.stopped

                        def set(self):
                            self.stopped = True

                        def wait(self, seconds):
                            # 模拟网格唤醒的微小正抖动，避免恰好落在浮点边界。
                            clock.elapsed += max(0, seconds) + 1e-7

                    run.stop = Stop()
                    run.origin_wall, run.origin_mono = 1000., 0.
                    latest = -1

                    def take(**kwargs):
                        nonlocal latest
                        end = int(clock.elapsed * 30)
                        frames = [{"timestamp": 1000 + i / 30, "state": np.zeros(7, np.float32)}
                                  for i in range(latest + 1, end + 1)]
                        latest = end
                        return frames

                    def infer(frames, **kwargs):
                        clock.elapsed += latency
                        return {"actions": np.repeat(np.arange(1, 17, dtype=np.float32)[:, None], 7, axis=1),
                                "round_trip_ms": latency * 1000}

                    run.source.take = take
                    run.client.infer = infer
                    fake_time = SimpleNamespace(time=lambda: 1000 + clock.elapsed, monotonic=lambda: clock.elapsed)
                    with patch("tempo_client.control.time", fake_time), patch("tempo_client.control.threading.Thread"):
                        run.run(1.2)
                    rows = [json.loads(line) for line in (Path(root) / "controls.jsonl").read_text().splitlines()]
                    self.assertGreater(len(rows), 1)
                    self.assertEqual(rows[0]["sources"][0]["prediction_step"], int(latency * 30) + 1)
                    for row in rows:
                        source = row["sources"][0]
                        self.assertEqual(row["action"][0], source["prediction_step"])
                        self.assertAlmostEqual(row["target_timestamp"], source["observed"] + source["prediction_step"] / 30)
                        self.assertLessEqual(row["age_s"], .5)
                        self.assertGreaterEqual(row["schedule_lag_s"], 0)
                    self.assertEqual(summary["planned_targets"], len(rows))
                    self.assertTrue(robot.held)

    def test_rtc_bootstrap_and_next_prefix_keep_observation_anchors(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, _, _ = self.make_run("rtc", root)
            run.origin_wall = 1000.
            run.tick = Mock(return_value=6)
            frames = [[{"timestamp": 1000.}], [{"timestamp": 1000 + 4 / 30}]]
            run.source.take = Mock(side_effect=frames)
            actions = np.repeat(np.arange(1, 17, dtype=np.float32)[:, None], 7, axis=1)
            run._request = Mock(side_effect=[({"actions": actions, "round_trip_ms": 200.}, batch[-1]["timestamp"])
                                             for batch in frames])
            run._active = Mock(side_effect=[True, True, True, True, False])
            run._async()
            self.assertTrue(run.errors.empty())
            self.assertEqual(run.plans.latest_start(), 5)
            second = run._request.call_args_list[1]
            np.testing.assert_array_equal(second.args[0], actions[4:])
            self.assertEqual(second.args[1], 9)
            # 首块以观测后一帧为起点，响应落在tick6时，tick7必须取第7步。
            first = Plans("rtc", 16, .05)
            first.add(Plan(1, actions, 1000.))
            self.assertEqual(first.action(7, 1000 + 7 / 30, .5)[0][0], 7)

    def test_te_source_steps_match_same_physical_target_time(self):
        plans = Plans("temporal_ensemble", 16, .05)
        actions = np.repeat(np.arange(1, 17, dtype=np.float32)[:, None], 7, axis=1)
        plans.add(Plan(1, actions, 1000.))
        plans.add(Plan(4, actions, 1000.1))
        target, observed, sources = plans.action(9, 1000.3, .5, include_sources=True)
        self.assertEqual([value["prediction_step"] for value in sources], [9, 6])
        self.assertAlmostEqual(observed, 1000.)
        self.assertAlmostEqual(sum(value["weight"] for value in sources), 1.)
        for value in sources:
            self.assertAlmostEqual(value["observed"] + value["prediction_step"] / 30, 1000.3)
        self.assertAlmostEqual(float(target[0]), sum(value["weight"] * value["prediction_step"] for value in sources), places=5)

    def test_te_200ms_loop_records_executed_prediction_steps(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, _, summary = self.make_run("temporal_ensemble", root, latency=.2)
            run.run(.35)
            rows = [json.loads(line) for line in (Path(root) / "controls.jsonl").read_text().splitlines()]
            requests = [json.loads(line) for line in run.predictions.getvalue().splitlines()]
            self.assertGreaterEqual(len(rows), 8)
            self.assertGreaterEqual(min(value["prediction_step"] for value in rows[0]["sources"]), 6)
            for row in rows:
                self.assertGreaterEqual(row["dispatch_end_timestamp"], row["time"])
                for source in row["sources"]:
                    self.assertEqual(source["plan_start_tick"] + source["prediction_step"] - 1, row["tick"])
                    self.assertAlmostEqual(source["observed"] + source["prediction_step"] / 30,
                                           row["target_timestamp"], places=5)
            for request in requests:
                self.assertAlmostEqual(request["action_target_timestamps"][0], request["timestamp"] + 1 / 30)
            self.assertAlmostEqual(sum(summary["prediction_step_weight_sum"].values()), len(rows))
