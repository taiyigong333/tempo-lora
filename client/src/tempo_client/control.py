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
from tempo_client.delay import DelayEstimator


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

    def action(self, tick, now, max_age, *, include_sources=False):
        with self.lock:
            self.plans = [value for value in self.plans if value.start + self.horizon > tick]
            choices = [value for value in self.plans if value.start <= tick < value.start + self.horizon
                       and 0 <= now - value.observed <= max_age]
            if not choices:
                return None
            if self.mode != "temporal_ensemble":
                choices = choices[-1:]
            ages = np.asarray([tick - value.start for value in choices])
            weights = np.exp(-self.decay * (ages - ages.min()))
            weights /= weights.sum()
            targets = np.stack([value.actions[tick - value.start] for value in choices])
            # 六维是关节角，按训练的绝对关节坐标加权，夹爪同样采用绝对米制。
            result = ((targets * weights[:, None]).sum(0).astype(np.float32),
                      min(value.observed for value in choices))
            if include_sources:
                sources = [{"plan_start_tick": value.start, "observed": value.observed,
                            "prediction_step": tick - value.start + 1, "weight": float(weight)}
                           for value, weight in zip(choices, weights)]
                return result[0], result[1], sources
            return result

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
        self.delay_estimator = DelayEstimator(self.inference, self.fps)
        self.plans = Plans(self.mode, self.horizon, self.inference["ensemble_decay"])
        self.stop = threading.Event()
        self.errors = queue.Queue()
        self.first_action_at = None
        self.first_deadline = time.monotonic() + self.inference["first_action_timeout_s"]
        self.origin_wall = source.origin_wall
        self.origin_mono = time.monotonic() - (time.time() - self.origin_wall)
        self.summary.update(planned_targets=0, safety_violations=0, rtc_guided_requests=0,
                            prediction_step_weight_sum={})

    def tick(self):
        return math.floor((time.monotonic() - self.origin_mono) * self.fps)

    def action_start(self, observed):
        # action[t]已经对应state[t+1]，首目标固定在观测后一帧，所有模式共享这个锚点。
        return round((observed - self.origin_wall) * self.fps) + 1

    def _wait_for_tick(self, tick):
        self.stop.wait(max(0, self.origin_mono + tick / self.fps - time.monotonic()))

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
                      rtc_delay=delay, robot_commands=self.robot.sent_commands,
                      rtc_delay_estimator=self.delay_estimator.method,
                      rtc_latency_estimate_ms=self.delay_estimator.seconds * 1000)
        anchor = self.action_start(frames[-1]["timestamp"])
        record.update(action_start_tick=anchor, received_timestamp=time.time(),
                      first_future_prediction_step=max(1, self.tick() + 2 - anchor),
                      action_target_timestamps=[self.origin_wall + (anchor + k) / self.fps
                                                for k in range(self.horizon)])
        self.predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.predictions.flush()
        self.summary["batches"] += 1
        self.summary["feedback_frames"] += len(frames)
        self.summary["rtc_guided_requests"] += int(result.get("rtc_prefix_length", 0) > 0)
        self.events.put(("prediction", record))
        return result, frames[-1]["timestamp"]

    def _async(self):
        try:
            while self._active():
                previous_start = self.plans.latest_start()
                if self.mode == "rtc" and previous_start is not None:
                    while self.tick() < previous_start + self.inference["rtc_min_execute_steps"] and self._active():
                        self.stop.wait(.002)
                if not self._active():
                    return
                frames = self.source.take(max_frames=self.cfg["live"]["max_pending_frames"])
                anchor = self.action_start(frames[-1]["timestamp"])
                prefix, delay = None, 0
                if self.mode == "rtc" and previous_start is not None:
                    prefix = self.plans.prefix(anchor)
                    delay = max(0, self.tick() + 1 - anchor) + math.ceil(self.delay_estimator.seconds * self.fps) + self.inference["rtc_delay_margin_steps"]
                    if prefix is None or len(prefix) < delay:
                        raise RuntimeError("RTC剩余前缀无法覆盖实测推理延迟，请检查GPU负载与推理参数")
                result, observed = self._request(prefix, delay, frames)
                self.delay_estimator.update(result["round_trip_ms"] / 1000)
                if self.tick() + 1 >= anchor + self.horizon:
                    raise RuntimeError("异步结果已超出模型动作窗口")
                self.plans.add(Plan(anchor, result["actions"], observed))
        except Exception as error:
            if self._active():
                self._fail(error)

    def _apply(self, action, observed, tick, stream, *, sources=()):
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
        # 读取反馈和FK检查也消耗时间，下发前重新核验时效，日志记录实际调度误差。
        dispatched = time.time()
        age = dispatched - observed
        if not 0 <= age <= self.safety["action_max_age_s"]:
            raise RuntimeError(f"动作来源观测过期，帧龄{age:.3f}秒")
        lag = time.monotonic() - (self.origin_mono + tick / self.fps)
        if lag > self.safety["max_control_lag_s"]:
            raise RuntimeError("控制线程超过调度延迟上限")
        if self.execute:
            self.robot.send_action(action, self.cancel)
        dispatch_end = time.time()
        if self.first_action_at is None:
            self.first_action_at = time.monotonic()
        self.summary["planned_targets"] += 1
        self.summary["robot_commands"] = self.robot.sent_commands
        for source in sources:
            step = str(source["prediction_step"])
            counts = self.summary["prediction_step_weight_sum"]
            counts[step] = counts.get(step, 0.) + source["weight"]
        record = {"tick": tick, "observed": observed, "time": dispatched, "age_s": age,
                  "action": np.asarray(action).tolist(), "state": current.tolist(), "executed": self.execute,
                  "mode": self.mode, "violation": violation, "robot_commands": self.robot.sent_commands,
                  "target_timestamp": self.origin_wall + tick / self.fps,
                  "schedule_lag_s": lag, "dispatch_end_timestamp": dispatch_end, "sources": sources}
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
                next_tick = self.tick() + 1
                while self._active():
                    if self.first_action_at is not None and time.monotonic() - self.first_action_at >= duration:
                        break
                    if self.mode == "tempo":
                        result, observed = self._request()
                        anchor = self.action_start(observed)
                        self.plans.add(Plan(anchor, result["actions"], observed))
                        next_tick = self.tick() + 1
                        if next_tick >= anchor + self.horizon:
                            raise RuntimeError("原生结果已超出模型动作窗口，动作来源观测过期")
                        for executed in range(self.inference["execute_steps"]):
                            if not self._active():
                                break
                            if self.first_action_at is not None and time.monotonic() - self.first_action_at >= duration:
                                break
                            next_tick = max(next_tick, self.tick())
                            self._wait_for_tick(next_tick)
                            if not self._active():
                                break
                            selected = self.plans.action(next_tick, time.time(), self.safety["action_max_age_s"],
                                                         include_sources=True)
                            if selected is None:
                                if executed:
                                    # 当前块有效尾部已耗尽，提前重算，不能把过期目标继续播放。
                                    break
                                raise RuntimeError("原生动作来源观测过期或动作窗口耗尽")
                            action, observed, sources = selected
                            self._apply(action, observed, next_tick, stream, sources=sources)
                            next_tick += 1
                    else:
                        next_tick = max(next_tick, self.tick())
                        self._wait_for_tick(next_tick)
                        if not self._active():
                            break
                        selected = self.plans.action(next_tick, time.time(), self.safety["action_max_age_s"],
                                                     include_sources=True)
                        if selected is None:
                            if self.first_action_at is not None:
                                raise RuntimeError("异步动作缓冲耗尽或观测过期")
                            self.stop.wait(.002)
                            next_tick = self.tick() + 1
                            continue
                        action, observed, sources = selected
                        self._apply(action, observed, next_tick, stream, sources=sources)
                        next_tick += 1
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
