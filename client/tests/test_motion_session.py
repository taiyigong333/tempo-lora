"""真实任务控制器与合成SDK联调，覆盖归位、执行、停止及跨轮相机复用。"""

import copy
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tempo_client.cameras import CameraStream
from tempo_client.cli import DEFAULT_CONFIG, load_config
from tempo_client.robot import Robot
from tempo_client.session import SessionWorker
from tempo_client.transport import Client
from test_cameras import SyntheticCapture
from test_control import TimedClient


class SimulatedArm:
    ARM_STATUS = SimpleNamespace(CtrlMode=SimpleNamespace(CAN_CTRL=1), ModeFeedback=SimpleNamespace(MOVE_J=1))

    def __init__(self, target):
        self.joints = np.asarray(target, float).copy()
        self.width = .05
        self.enabled = False
        self.mode = 0
        self.operations = []

    def get_joint_angles(self):
        return SimpleNamespace(msg=self.joints.copy(), hz=90, timestamp=time.time())

    def get_gripper_status(self):
        return SimpleNamespace(msg=SimpleNamespace(mode="width", value=self.width), hz=90, timestamp=time.time())

    def fk(self, joints):
        value = np.asarray(joints)
        return [.2 + .01 * value[0], .01 * value[1], .3 + .01 * value[2], *(.01 * value[3:])]

    def get_flange_pose(self):
        return SimpleNamespace(msg=self.fk(self.joints), hz=90, timestamp=time.time())

    def get_arm_status(self):
        return SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(
            arm_status=0, err_code=0, ctrl_mode=self.mode, mode_feedback=self.mode))

    def get_joints_enable_status_list(self):
        return [self.enabled] * 6

    def enable(self):
        self.enabled = True
        self.operations.append(("enable", True))
        return True

    def set_speed_percent(self, value):
        self.operations.append(("speed", value))

    def set_motion_mode(self, mode):
        assert mode == "j"
        self.mode = 1
        self.operations.append(("mode", mode))

    def move_j(self, joints):
        self.joints = np.asarray(joints, float)
        self.operations.append(("joint_target", self.joints.tolist()))

    def move_gripper_m(self, *, value, force):
        self.width = value
        self.operations.append(("gripper_target", value))

    def electronic_emergency_stop(self):
        self.operations.append(("emergency_stop", True))

    def disconnect(self):
        self.operations.append(("disconnect", True))


class SimulatedRobot(Robot):
    instances = []

    def connect(self):
        self.simulated_arm = SimulatedArm(self.home["joint_positions_rad"])
        self.arm = self.gripper = self.simulated_arm
        self.diagnostics = {"hardware": "synthetic"}
        self.instances.append(self)


class SimulatedClient(TimedClient):
    validate_mode = Client.validate_mode

    def __init__(self, cfg):
        super().__init__(.01)
        self.target = np.r_[cfg["home"]["joint_positions_rad"], .05].astype(np.float32)
        self.local_ip = "192.168.50.1"
        self.metadata = {"config": cfg["model"]["config"], "step": 10000, "checkpoint": "step_00010000.pt",
                         "action_horizon": 16, "native_action_dim": 7, "modes": ["tempo", "temporal_ensemble", "rtc"],
                         "rtc_algorithm": "decoded_action_vjp_soft_prefix", "protocol_version": 2,
                         "task_prompts": ["grab the rotating red square"]}

    def reserve(self):
        pass

    def infer(self, frames, **kwargs):
        result = super().infer(frames, **kwargs)
        result["actions"] = np.tile(self.target, (16, 1))
        result["actions"][:, 0] += np.arange(16, dtype=np.float32) * .0001
        return result

    def close(self):
        self.abort()


class MotionSessionTests(unittest.TestCase):
    def test_all_modes_readonly_and_confirmed_motion_reuse_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
            cfg["live"]["capture_marker"] = str(Path(directory) / "camera.log")
            worker = SessionWorker()
            summaries = []
            SimulatedRobot.instances = []
            factory = lambda config: CameraStream(config, capture_factory=SyntheticCapture)
            try:
                with patch("tempo_client.session.load_config", return_value=cfg), \
                     patch("tempo_client.session.CameraStream", side_effect=factory), \
                     patch("tempo_client.session.Client", SimulatedClient), \
                     patch("tempo_client.session.Robot", SimulatedRobot):
                    for mode in ("tempo", "temporal_ensemble", "rtc"):
                        for execute in (False, True):
                            with self.subTest(mode=mode, execute=execute):
                                worker.start("live", DEFAULT_CONFIG, duration=.18, inference_mode=mode,
                                             execute=execute, motion_confirmed=execute, log_root=directory)
                                worker.thread.join(8)
                                self.assertFalse(worker.thread.is_alive())
                                events = []
                                while not worker.events.empty():
                                    events.append(worker.events.get())
                                summary = next(value for kind, value in events if kind == "done")
                                self.assertEqual(summary["outcome"], "completed", summary)
                                self.assertGreater(summary["planned_targets"], 0)
                                operations = SimulatedRobot.instances[-1].simulated_arm.operations
                                if execute:
                                    self.assertGreater(summary["robot_commands"], 0)
                                    self.assertIn(("speed", 10), operations)
                                    self.assertIn(("speed", 90), operations)
                                    self.assertTrue(SimulatedRobot.instances[-1].simulated_arm.enabled)
                                    self.assertNotIn(("emergency_stop", True), operations)
                                    self.assertEqual(summary["robot_diagnostics"]["hold_joint_enabled"], [True] * 6)
                                else:
                                    self.assertEqual(summary["robot_commands"], 0)
                                    self.assertFalse(any(name == "enable" for name, _ in operations))
                                if mode == "rtc":
                                    self.assertGreater(summary["rtc_guided_requests"], 0)
                                summaries.append(summary)
                    self.assertEqual(len({item["camera_stream"]["process_id"] for item in summaries}), 1)
                    self.assertTrue(worker.cameras.process.is_alive())
            finally:
                worker.stop()
                if worker.thread is not None:
                    worker.thread.join(8)
                worker.close_cameras()


if __name__ == "__main__":
    unittest.main()
