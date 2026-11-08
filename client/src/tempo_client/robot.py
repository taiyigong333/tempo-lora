"""关节/夹爪唯一运动出口，沿用OpenPI的反馈、模式确认与停止保持。"""

from pathlib import Path
import re
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


class Robot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.settings, self.safety, self.home = cfg["live"], cfg["safety"], cfg["home"]
        self.lock = threading.RLock()
        self.arm = self.gripper = None
        self.enabled = False
        self.holding = True
        self.hold_target = None
        self.policy_started = False
        self.heartbeat = time.monotonic()
        self.sent_commands = 0
        self.diagnostics = {}

    @staticmethod
    def firmware_driver(version):
        match = re.fullmatch(r"S-V(\d+)\.(\d+)-(\d+)", version)
        if match is None:
            raise RuntimeError("无法识别机器人固件版本 " + str(version))
        value = tuple(map(int, match.groups()))
        for minimum, driver in (((1, 8, 9), "v189"), ((1, 8, 8), "v188"), ((1, 8, 3), "v183")):
            if value >= minimum:
                return driver
        return "default"

    def connect(self):
        sdk_path = Path(self.settings["sdk_path"]).resolve()
        sys.path.insert(0, str(sdk_path))
        import pyAgxArm
        if not Path(pyAgxArm.__file__).resolve().is_relative_to(sdk_path):
            raise RuntimeError("机器人SDK来源与配置不匹配")
        self.arm = pyAgxArm.AgxArmFactory.create_arm(pyAgxArm.create_agx_arm_config(
            robot=pyAgxArm.ArmModel.PIPER_X, firmeware_version=self.settings["firmware_version"],
            interface="socketcan", channel=self.settings["can_name"]))
        self.arm.connect()
        deadline = time.monotonic() + self.safety["feedback_wait_timeout_s"]
        while True:
            try:
                self.read_joints()
                self.read_pose()
                if self.arm.get_fps() > 0:
                    break
            except RuntimeError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("机器人CAN反馈未就绪")
            time.sleep(.02)
        # SDK在尚无反馈时查询固件可能清空固件分片，先等待接收流。
        firmware = self.arm.get_firmware(timeout=self.safety["firmware_query_timeout_s"])
        if not isinstance(firmware, dict) or not firmware.get("software_version"):
            raise RuntimeError("机器人固件查询失败")
        if self.firmware_driver(firmware["software_version"]) != self.settings["firmware_version"]:
            raise RuntimeError("实际机器人固件与配置驱动不匹配")
        self.diagnostics = {"sdk": pyAgxArm.__file__, "firmware": firmware}
        self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        deadline = time.monotonic() + self.safety["feedback_wait_timeout_s"]
        while True:
            try:
                self.read_state()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("夹爪反馈未就绪")
                time.sleep(.01)

    def _feedback(self, value, shape):
        if value is None or value.hz <= 0 or not 0 <= time.time() - value.timestamp <= self.safety["feedback_max_age_s"]:
            raise RuntimeError("机器人反馈缺失或过期")
        result = np.asarray(value.msg, np.float64)
        if result.shape != shape or not np.isfinite(result).all():
            raise RuntimeError("机器人反馈维度或数值异常")
        return result.copy()

    def read_joints(self):
        with self.lock:
            return self._feedback(self.arm.get_joint_angles(), (6,))

    def read_pose(self):
        with self.lock:
            return self._feedback(self.arm.get_flange_pose(), (6,))

    def read_state(self):
        with self.lock:
            joints = self.read_joints()
            feedback = self.gripper.get_gripper_status()
            if (feedback is None or feedback.hz <= 0 or feedback.msg.mode != "width"
                    or not 0 <= time.time() - feedback.timestamp <= self.safety["feedback_max_age_s"]):
                raise RuntimeError("夹爪反馈缺失或过期")
            width = float(feedback.msg.value)
            lo, hi = self.safety["gripper_range_m"]
            tolerance = self.safety["gripper_feedback_tolerance_m"]
            if not np.isfinite(width) or not lo - tolerance <= width <= hi + tolerance:
                raise RuntimeError("夹爪实测开口越界")
            return np.r_[joints, width].astype(np.float32)

    def check_status(self):
        with self.lock:
            feedback = self.arm.get_arm_status()
            if feedback is None or not 0 <= time.time() - feedback.timestamp <= self.safety["feedback_max_age_s"]:
                raise RuntimeError("机械臂状态反馈过期")
            status = feedback.msg
            if getattr(status.arm_status, "value", status.arm_status) != 0 or status.err_code:
                raise RuntimeError("机械臂控制器报告故障")
            self.read_state()
            if (self.enabled and not self.holding and self.policy_started
                    and time.monotonic() - self.heartbeat > self.safety["watchdog_s"]):
                raise TimeoutError("模型动作断流")

    def _within_limits(self, joints, tolerance=0.):
        limits = np.asarray(self.home["joint_limits_rad"], float)
        return bool(np.all(joints >= limits[:, 0] - tolerance) and np.all(joints <= limits[:, 1] + tolerance))

    def target_pose(self, joints):
        with self.lock:
            result = np.asarray(self.arm.fk(np.asarray(joints).tolist()), float)
        if result.shape != (6,) or not np.isfinite(result).all():
            raise RuntimeError("SDK正运动学结果无效")
        return result

    def check_target(self, action):
        action = np.asarray(action, np.float64)
        if action.shape != (7,) or not np.isfinite(action).all() or not self._within_limits(action[:6]):
            raise ValueError("模型关节目标无效或超出关节限位")
        lo, hi = self.safety["gripper_range_m"]
        if not lo <= action[6] <= hi:
            raise ValueError("模型夹爪目标超出物理范围")
        target, current = self.target_pose(action[:6]), self.read_pose()
        bounds = np.asarray(self.safety["workspace_m"])
        if np.any(target[:3] < bounds[:, 0]) or np.any(target[:3] > bounds[:, 1]):
            raise ValueError("模型关节目标的TCP超出工作空间")
        if np.linalg.norm(target[:3] - current[:3]) > self.safety["max_position_step_m"]:
            raise ValueError("模型目标相对当前TCP位移过大")
        angle = (Rotation.from_euler("xyz", current[3:]).inv() * Rotation.from_euler("xyz", target[3:])).magnitude()
        if angle > self.safety["max_rotation_step_rad"]:
            raise ValueError("模型目标相对当前TCP旋转过大")
        return action

    def prepare_motion(self, cancel, speed):
        deadline = time.monotonic() + self.safety["enable_timeout_s"]
        while not cancel.is_set():
            with self.lock:
                if cancel.is_set():
                    raise InterruptedError("使能已取消")
                self.enabled = True
                if self.arm.enable():
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("机器人使能超时")
            cancel.wait(.05)
        if cancel.is_set():
            raise InterruptedError("运动准备已取消")
        with self.lock:
            self.arm.set_speed_percent(speed)
            self.arm.set_motion_mode("j")
            sent_at = time.time()
        deadline = time.monotonic() + self.safety["control_ready_timeout_s"]
        while not cancel.is_set():
            with self.lock:
                self.check_status()
                feedback = self.arm.get_arm_status()
                enabled = list(self.arm.get_joints_enable_status_list())
                if (feedback.timestamp > sent_at
                        and feedback.msg.ctrl_mode == self.arm.ARM_STATUS.CtrlMode.CAN_CTRL
                        and feedback.msg.mode_feedback == self.arm.ARM_STATUS.ModeFeedback.MOVE_J
                        and len(enabled) == 6 and all(enabled)):
                    self.holding = False
                    self.hold_target = None
                    self.policy_started = False
                    self.heartbeat = time.monotonic()
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError("机器人尚未确认CAN关节控制模式与六关节使能")
            cancel.wait(self.safety["control_ready_poll_s"])
        raise InterruptedError("运动模式准备已取消")

    def send_action(self, action, cancel):
        with self.lock:
            if cancel.is_set() or self.holding or not self.enabled:
                raise InterruptedError("运动已停止，拒绝新目标")
            target = self.check_target(action)
            self.arm.move_j(target[:6].tolist())
            self.sent_commands += 1
            self.gripper.move_gripper_m(value=float(target[6]), force=self.safety["gripper_force_n"])
            self.sent_commands += 1
            self.heartbeat = time.monotonic()
            self.policy_started = True

    def go_home(self, cancel):
        target = np.asarray(self.home["joint_positions_rad"], float)
        current = self.read_joints()
        pose, target_pose = self.read_pose(), self.target_pose(target)
        bounds = np.asarray(self.home["start_workspace_m"])
        target_bounds = np.asarray(self.safety["workspace_m"])
        if (not self._within_limits(current, self.home["initial_joint_tolerance_rad"])
                or not self._within_limits(target)
                or np.max(np.abs(current - target)) > self.home["max_joint_delta_rad"]
                or np.any(pose[:3] < bounds[:, 0]) or np.any(pose[:3] > bounds[:, 1])
                or np.any(target_pose[:3] < target_bounds[:, 0]) or np.any(target_pose[:3] > target_bounds[:, 1])
                or np.linalg.norm(pose[:3] - target_pose[:3]) > self.home["max_distance_m"]
                or (Rotation.from_euler("xyz", pose[3:]).inv() * Rotation.from_euler("xyz", target_pose[3:])).magnitude() > self.home["max_rotation_rad"]):
            raise ValueError("归位起点或目标超出OpenPI现场保护范围")
        self.prepare_motion(cancel, self.home["speed_percent"])
        with self.lock:
            if cancel.is_set() or self.holding:
                raise InterruptedError("归位已取消")
            self.arm.move_j(target.tolist())
            self.sent_commands += 1
        deadline = time.monotonic() + self.home["timeout_s"]
        while not cancel.is_set():
            self.check_status()
            if np.max(np.abs(self.read_joints() - target)) <= self.home["joint_tolerance_rad"]:
                self.hold()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("机器人关节归位超时")
            cancel.wait(.02)
        raise InterruptedError("归位已停止")

    def hold(self):
        with self.lock:
            was_holding = self.holding
            self.holding = True
            if self.arm is None or not self.enabled or (was_holding and self.hold_target is not None):
                return
            try:
                self.hold_target = self.read_joints()
            except RuntimeError:
                self.arm.electronic_emergency_stop()
                raise RuntimeError("反馈失效，已请求阻尼急停")
            self.arm.move_j(self.hold_target.tolist())
            self.sent_commands += 1
            try:
                width = float(self.read_state()[6])
            except RuntimeError as error:
                self.diagnostics["hold_gripper_error"] = str(error)
                return
            lo, hi = self.safety["gripper_range_m"]
            if lo <= width <= hi:
                self.gripper.move_gripper_m(value=width, force=self.safety["gripper_force_n"])
                self.sent_commands += 1

    def close(self):
        if self.arm is not None:
            try:
                self.hold()
            finally:
                self.arm.disconnect()
                self.arm = None
                self.enabled = False
