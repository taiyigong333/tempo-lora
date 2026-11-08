"""任务、模式和运动参数校验，配置快照在每轮开始前固定。"""

import copy

import numpy as np

MODE_LABELS = {"tempo": "TEMPO 原生同步", "temporal_ensemble": "时间集成 TE", "rtc": "RTC 异步前缀引导"}


def select_runtime(cfg, task_id=None, mode=None):
    cfg = copy.deepcopy(cfg)
    catalog = cfg["task_catalog"]
    tasks = catalog["tasks"]
    ids = [task["id"] for task in tasks]
    if not tasks or len(ids) != len(set(ids)):
        raise ValueError("任务目录为空或ID重复")
    selected = task_id or catalog["default_task"]
    task = next((value for value in tasks if value["id"] == selected), None)
    if task is None or not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
        raise ValueError("所选任务不存在或语言指令为空")
    cfg["task"] = task
    inference = cfg["inference"]
    inference["mode"] = mode or inference["mode"]
    if inference["mode"] not in MODE_LABELS:
        raise ValueError("推理模式不支持")
    horizon = cfg["model"]["action_horizon"]
    for key, maximum in (("execute_steps", horizon), ("rtc_min_execute_steps", horizon - 1)):
        if type(inference[key]) is not int or not 1 <= inference[key] <= maximum:
            raise ValueError(f"{key}必须为1至{maximum}整数")
    if type(inference["rtc_delay_margin_steps"]) is not int or not 0 <= inference["rtc_delay_margin_steps"] < horizon:
        raise ValueError("RTC延迟余量不合法")
    if not np.isfinite(inference["ensemble_decay"]) or inference["ensemble_decay"] < 0:
        raise ValueError("时间集成衰减必须为有限非负数")
    if cfg["model"]["fps"] != 30:
        raise ValueError("当前控制与反馈统一使用30Hz")
    safety, home = cfg["safety"], cfg["home"]
    arrays = {"joint_limits_rad": (home["joint_limits_rad"], (6, 2)),
              "joint_positions_rad": (home["joint_positions_rad"], (6,)),
              "workspace_m": (safety["workspace_m"], (3, 2)),
              "start_workspace_m": (home["start_workspace_m"], (3, 2))}
    for key, (value, shape) in arrays.items():
        value = np.asarray(value, float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{key}维度或数值不合法")
        if len(shape) == 2 and np.any(value[:, 0] >= value[:, 1]):
            raise ValueError(f"{key}上下界不合法")
    limits, target = np.asarray(home["joint_limits_rad"]), np.asarray(home["joint_positions_rad"])
    if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
        raise ValueError("归位目标超出关节限位")
    for section in (safety, home):
        for key, value in section.items():
            if isinstance(value, (int, float)) and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key}必须为有限正数")
        if type(section["speed_percent"]) is not int or not 1 <= section["speed_percent"] <= 100:
            raise ValueError("SDK速度百分比必须为1至100整数")
    lo, hi = safety["gripper_range_m"]
    if not 0 <= lo < hi <= .10:
        raise ValueError("夹爪目标范围必须位于0至0.10米")
    return cfg
