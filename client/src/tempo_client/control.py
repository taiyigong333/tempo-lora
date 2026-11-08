"""三种推理模式共用30Hz关节目标调度与反馈保护。"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

from tempo_client.cli import report


@dataclass
class Plan:
    start: int
    actions: np.ndarray
    observed: float


class Plans:
    def __init__(self, mode, horizon, decay):
        self.mode, self.horizon, self.decay = mode, horizon, decay
        self.lock = threading.RLock()
        self.plans = []

    def add(self, plan):
        actions = np.asarray(plan.actions, np.float32)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("动作块形状或数值无效")
        with self.lock:
            if self.mode != "temporal_ensemble":
                self.plans.clear()
            self.plans = [value for value in self.plans if value.start + self.horizon > plan.start]
            self.plans.append(Plan(plan.start, actions.copy(), plan.observed))

    def action(self, tick, now, max_age):
        with self.lock:
            self.plans = [value for value in self.plans if value.start + self.horizon > tick]
            choices = [value for value in self.plans if value.start <= tick < value.start + self.horizon
                       and 0 <= now - value.observed <= max_age]
            if not choices:
                return None
            if self.mode != "temporal_ensemble":
                plan = choices[-1]
                return plan.actions[tick - plan.start].copy(), plan.observed
            weights = np.exp(-self.decay * np.asarray([tick - value.start for value in choices]))
            weights /= weights.sum()
            targets = np.stack([value.actions[tick - value.start] for value in choices])
            # 六维是关节角，按训练的绝对关节坐标加权，夹爪同样采用绝对米制。
            return (targets * weights[:, None]).sum(0).astype(np.float32), min(value.observed for value in choices)

    def prefix(self, tick):
        with self.lock:
            if not self.plans:
                return None
            plan = self.plans[-1]
            offset = tick - plan.start
            return plan.actions[offset:].copy() if 0 <= offset < self.horizon else None

    def latest_start(self):
        with self.lock:
            return None if not self.plans else self.plans[-1].start


class InferenceRun:
    def __init__(self, cfg, client, source, robot, cancel, events, prediction_stream, summary, execute):
        self.cfg, self.client, self.source, self.robot = cfg, client, source, robot
        self.cancel, self.events = cancel, events
        self.predictions, self.summary, self.execute = prediction_stream, summary, execute
        self.inference, self.safety = cfg["inference"], cfg["safety"]
        self.mode = self.inference["mode"]
        self.fps = cfg["model"]["fps"]
        self.horizon = cfg["model"]["action_horizon"]
        self.plans = Plans(self.mode, self.horizon, self.inference["ensemble_decay"])
        self.stop = threading.Event()
        self.errors = queue.Queue()
        self.first_action_at = None
        self.first_deadline = time.monotonic() + self.inference["first_action_timeout_s"]
        self.origin_wall = source.origin_wall
        self.origin_mono = time.monotonic() - (time.time() - self.origin_wall)
        self.summary.update(planned_targets=0, safety_violations=0, rtc_guided_requests=0)

    def tick(self):
        return math.floor((time.monotonic() - self.origin_mono) * self.fps)

    def _active(self):
        return not self.stop.is_set() and not self.cancel.is_set()

    def _fail(self, error):
        self.errors.put(error)
        self.stop.set()
        self.client.abort()
        if self.execute:
            try:
                self.robot.hold()
            except Exception as hold_error:
                self.events.put(("status", "停止保持失败 " + str(hold_error)))

    def _guard(self):
        while self._active():
            try:
                self.robot.check_status()
                if self.first_action_at is None and time.monotonic() > self.first_deadline:
                    raise TimeoutError("等待首个模型目标超时")
            except Exception as error:
                self._fail(error)
                return
            self.stop.wait(.02)

    def _request(self, prefix=None, delay=0, frames=None):
        frames = self.source.take(max_frames=self.cfg["live"]["max_pending_frames"]) if frames is None else frames
        self.events.put(("frame", frames[-1]))
        result = self.client.infer(frames, mode=self.mode, prompt=self.cfg["task"]["prompt"],
                                   rtc_prefix=prefix, rtc_delay=delay)
        if not self._active():
            raise InterruptedError("推理已停止")
        record = report(result, frames)
        record.update(state=frames[-1]["state"].tolist(), mode=self.mode,
                      task_id=self.cfg["task"]["id"], rtc_prefix_length=result.get("rtc_prefix_length", 0),
                      rtc_delay=delay, robot_commands=self.robot.sent_commands)
        self.predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.predictions.flush()
        self.summary["batches"] += 1
        self.summary["feedback_frames"] += len(frames)
        self.summary["rtc_guided_requests"] += int(result.get("rtc_prefix_length", 0) > 0)
        self.events.put(("prediction", record))
        return result, frames[-1]["timestamp"]

    def _async(self):
        latency = 0.
        try:
            while self._active():
                previous_start = self.plans.latest_start()
                if self.mode == "rtc" and previous_start is not None:
                    while self.tick() < previous_start + self.inference["rtc_min_execute_steps"] and self._active():
                        self.stop.wait(.002)
                if not self._active():
                    return
                frames = self.source.take(max_frames=self.cfg["live"]["max_pending_frames"])
                anchor = round((frames[-1]["timestamp"] - self.origin_wall) * self.fps) + 1
                prefix, delay = None, 0
                if self.mode == "rtc" and previous_start is not None:
                    prefix = self.plans.prefix(anchor)
                    delay = max(0, self.tick() - anchor) + math.ceil(latency * self.fps) + self.inference["rtc_delay_margin_steps"]
                    if prefix is None or len(prefix) < delay:
                        raise RuntimeError("RTC剩余前缀无法覆盖实测推理延迟，请检查GPU负载与推理参数")
                result, observed = self._request(prefix, delay, frames)
                latency = max(result["round_trip_ms"] / 1000, latency * .95)
                start = self.tick() + 1 if self.mode == "rtc" and previous_start is None else anchor
                if self.tick() >= start + self.horizon:
                    raise RuntimeError("异步结果已超出模型动作窗口")
                self.plans.add(Plan(start, result["actions"], observed))
        except Exception as error:
            if self._active():
                self._fail(error)

    def _apply(self, action, observed, tick, stream):
        if not self._active():
            raise InterruptedError("目标调度已停止")
        age = time.time() - observed
        if not 0 <= age <= self.safety["action_max_age_s"]:
            raise RuntimeError(f"动作来源观测过期，帧龄{age:.3f}秒")
        current = self.robot.read_state()
        violation = None
        try:
            self.robot.check_target(action)
        except ValueError as error:
            if self.execute:
                raise
            violation = str(error)
            self.summary["safety_violations"] += 1
        if self.execute:
            self.robot.send_action(action, self.cancel)
        if self.first_action_at is None:
            self.first_action_at = time.monotonic()
        self.summary["planned_targets"] += 1
        self.summary["robot_commands"] = self.robot.sent_commands
        record = {"tick": tick, "observed": observed, "time": time.time(), "age_s": age,
                  "action": np.asarray(action).tolist(), "state": current.tolist(), "executed": self.execute,
                  "mode": self.mode, "violation": violation, "robot_commands": self.robot.sent_commands}
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        self.events.put(("control", record))

    def run(self, duration):
        duration = math.inf if duration == 0 else duration
        guard = threading.Thread(target=self._guard, name="tempo-feedback-guard", daemon=True)
        network = None
        guard.start()
        if self.mode != "tempo":
            network = threading.Thread(target=self._async, name="tempo-async-inference", daemon=True)
            network.start()
        try:
            path = Path(self.summary["log_dir"]) / "controls.jsonl"
            with path.open("w") as stream:
                next_tick = time.monotonic()
                while self._active():
                    if self.first_action_at is not None and time.monotonic() - self.first_action_at >= duration:
                        break
                    if self.mode == "tempo":
                        result, observed = self._request()
                        next_tick = time.monotonic()
                        for action in result["actions"][:self.inference["execute_steps"]]:
                            if not self._active():
                                break
                            if self.first_action_at is not None and time.monotonic() - self.first_action_at >= duration:
                                break
                            self._apply(action, observed, self.tick(), stream)
                            next_tick += 1 / self.fps
                            self.stop.wait(max(0, next_tick - time.monotonic()))
                    else:
                        selected = self.plans.action(self.tick(), time.time(), self.safety["action_max_age_s"])
                        if selected is None:
                            if self.first_action_at is not None:
                                raise RuntimeError("异步动作缓冲耗尽或观测过期")
                            self.stop.wait(.002)
                            next_tick = time.monotonic()
                            continue
                        self._apply(*selected, self.tick(), stream)
                        next_tick += 1 / self.fps
                        remaining = next_tick - time.monotonic()
                        if remaining < -self.safety["max_control_lag_s"]:
                            raise RuntimeError("控制线程超过调度延迟上限")
                        self.stop.wait(max(0, remaining))
            if not self.errors.empty():
                raise self.errors.get()
        except Exception as error:
            if not self.errors.empty():
                raise self.errors.get() from error
            raise
        finally:
            self.stop.set()
            self.client.abort()
            self.source.stop.set()
            if self.execute:
                self.robot.hold()
            guard.join(2)
            if network is not None:
                network.join(5)
                if network.is_alive():
                    raise RuntimeError("推理线程尚未退出，禁止启动下一轮")
            self.summary["robot_commands"] = self.robot.sent_commands
