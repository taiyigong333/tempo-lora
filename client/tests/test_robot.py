"""实机执行出口使用合成SDK验证，测试不接触真实CAN。"""

import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from tempo_client.cli import DEFAULT_CONFIG, load_config
from tempo_client.robot import Robot
from tempo_client.session import SessionWorker
from tempo_client.settings import select_runtime


class RobotTests(unittest.TestCase):
    def setUp(self):
        self.cfg = select_runtime(load_config(DEFAULT_CONFIG))
        self.robot = Robot(self.cfg)
        self.robot.arm = Mock()
        self.robot.gripper = Mock()
        self.target = np.asarray([0., .5, -.5, 0., 0., 0., .05])
        self.robot.target_pose = Mock(return_value=np.asarray([.2, 0., .3, 0., 0., 0.]))
        self.robot.read_pose = Mock(return_value=np.asarray([.2, 0., .3, 0., 0., 0.]))
        self.robot.read_joints = Mock(return_value=self.target[:6].copy())
        self.robot.read_state = Mock(return_value=self.target.astype(np.float32))

    def test_motion_requires_confirmation_before_thread_start(self):
        worker = SessionWorker()
        with self.assertRaisesRegex(ValueError, "现场确认"):
            worker.start("live", DEFAULT_CONFIG, execute=True)
        self.assertIsNone(worker.thread)

    def test_send_uses_joint_radians_and_gripper_metres(self):
        self.robot.enabled, self.robot.holding = True, False
        self.robot.send_action(self.target, threading.Event())
        self.robot.arm.move_j.assert_called_once_with(self.target[:6].tolist())
        self.robot.gripper.move_gripper_m.assert_called_once_with(value=.05, force=1.)
        self.assertEqual(self.robot.sent_commands, 2)

    def test_stop_blocks_subsequent_target(self):
        self.robot.enabled, self.robot.holding = True, False
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            self.robot.send_action(self.target, stop)
        self.robot.arm.move_j.assert_not_called()

    def test_joint_gripper_and_workspace_guards(self):
        for index, invalid in ((0, 20.), (6, .2)):
            with self.subTest(index=index):
                target = self.target.copy()
                target[index] = invalid
                with self.assertRaises(ValueError):
                    self.robot.check_target(target)
        self.robot.target_pose.return_value[2] = -1
        with self.assertRaisesRegex(ValueError, "工作空间"):
            self.robot.check_target(self.target)
        self.robot.arm.move_j.assert_not_called()

    def test_hold_latches_once_and_preserves_gripper(self):
        self.robot.enabled, self.robot.holding = True, False
        self.robot.hold()
        self.robot.hold()
        self.robot.arm.move_j.assert_called_once()
        self.robot.gripper.move_gripper_m.assert_called_once()
        self.assertTrue(self.robot.holding)

    def test_old_mode_feedback_does_not_allow_target(self):
        arm = self.robot.arm
        arm.enable.return_value = True
        arm.ARM_STATUS = SimpleNamespace(CtrlMode=SimpleNamespace(CAN_CTRL=1), ModeFeedback=SimpleNamespace(MOVE_J=1))
        arm.get_joints_enable_status_list.return_value = [True] * 6
        arm.get_arm_status.return_value = SimpleNamespace(timestamp=time.time() - 1,
                                                          msg=SimpleNamespace(ctrl_mode=1, mode_feedback=1))
        self.robot.check_status = Mock()
        self.robot.safety["control_ready_timeout_s"] = .02
        self.robot.safety["control_ready_poll_s"] = .001
        with self.assertRaisesRegex(TimeoutError, "尚未确认"):
            self.robot.prepare_motion(threading.Event(), 10)
        arm.move_j.assert_not_called()

    def test_partial_enable_does_not_allow_target(self):
        arm = self.robot.arm
        arm.enable.return_value = True
        arm.ARM_STATUS = SimpleNamespace(CtrlMode=SimpleNamespace(CAN_CTRL=1), ModeFeedback=SimpleNamespace(MOVE_J=1))
        arm.get_joints_enable_status_list.return_value = [True] * 5 + [False]
        arm.get_arm_status.side_effect = lambda: SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(ctrl_mode=1, mode_feedback=1))
        self.robot.check_status = Mock()
        self.robot.safety["control_ready_timeout_s"] = .02
        self.robot.safety["control_ready_poll_s"] = .001
        with self.assertRaises(TimeoutError):
            self.robot.prepare_motion(threading.Event(), 10)
        arm.move_j.assert_not_called()

    def test_home_guard_precedes_enable(self):
        self.robot.read_pose.return_value[0] = 2
        with self.assertRaisesRegex(ValueError, "归位起点"):
            self.robot.go_home(threading.Event())
        self.robot.arm.enable.assert_not_called()
