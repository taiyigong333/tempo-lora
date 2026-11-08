"""重现SDK可变反馈缓存与采样时刻的竞争，继续拒绝真实过期和未来时间戳。"""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from tempo_client.cli import DEFAULT_CONFIG, load_config
from tempo_client.live import LiveFeedback
from tempo_client.robot import Robot


class UpdatingFeedback:
    hz = 90
    def __init__(self, clock, value):
        self.clock, self.msg = clock, value
    @property
    def timestamp(self):
        self.clock[0] += .001
        return self.clock[0]


class FeedbackSnapshotTests(unittest.TestCase):
    def make_source(self, joint, gripper):
        source = LiveFeedback(load_config(DEFAULT_CONFIG), robot=SimpleNamespace(lock=threading.RLock()))
        source.arm = SimpleNamespace(get_joint_angles=lambda: joint)
        source.effector = SimpleNamespace(get_gripper_status=lambda: gripper)
        source.stop.wait = Mock(side_effect=lambda interval: source.stop.set())
        return source

    def test_sdk_timestamp_update_is_snapshotted_before_clock(self):
        clock = [100.]
        joints = UpdatingFeedback(clock, [0.] * 6)
        gripper = UpdatingFeedback(clock, SimpleNamespace(mode="width", value=.05))
        source = self.make_source(joints, gripper)
        with patch("tempo_client.live.time.time", side_effect=lambda: clock[0]):
            source._sample_arm()
        self.assertEqual(len(source.joints), 1)
        self.assertEqual(len(source.gripper), 1)

    def test_actual_stale_feedback_is_rejected(self):
        joints = SimpleNamespace(hz=90, timestamp=99., msg=[0.] * 6)
        source = self.make_source(joints, None)
        with patch("tempo_client.live.time.time", return_value=100.), self.assertRaisesRegex(RuntimeError, "帧龄1.000000"):
            source._sample_arm()

    def test_actual_future_feedback_is_rejected(self):
        joints = SimpleNamespace(hz=90, timestamp=101., msg=[0.] * 6)
        source = self.make_source(joints, None)
        with patch("tempo_client.live.time.time", return_value=100.), self.assertRaisesRegex(RuntimeError, "帧龄-1.000000"):
            source._sample_arm()

    def test_robot_feedback_readers_snapshot_before_clock(self):
        robot = Robot(load_config(DEFAULT_CONFIG))
        clock = [100.]
        with patch("tempo_client.robot.time.time", side_effect=lambda: clock[0]):
            value = robot._feedback(UpdatingFeedback(clock, [0.] * 6), (6,))
            np.testing.assert_array_equal(value, 0.)
            robot.read_joints = Mock(return_value=np.zeros(6))
            robot.gripper = SimpleNamespace(get_gripper_status=lambda: UpdatingFeedback(clock, SimpleNamespace(mode="width", value=.05)))
            self.assertAlmostEqual(float(robot.read_state()[-1]), .05)
            status = SimpleNamespace(arm_status=0, err_code=0)
            robot.arm = SimpleNamespace(get_arm_status=lambda: UpdatingFeedback(clock, status))
            robot.check_status()
