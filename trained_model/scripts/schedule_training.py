"""服务器 tmux 内的单次定时训练，锁与不可变计划避免重复启动。"""

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback
from zoneinfo import ZoneInfo

from openpi.tempo_lora.config import ROOT, configure_environment, file_digest, read_config
from openpi.tempo_lora.data import json_write
from openpi.tempo_lora.train import check_caches


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def check_ready(plan):
    config_path = Path(plan["config"])
    if file_digest(config_path) != plan["config_sha256"]:
        raise ValueError("正式训练快照发生变化，请重新登记计划")
    cfg = read_config(config_path)
    configure_environment(cfg)
    report_path = Path(plan["verification"])
    if file_digest(report_path) != plan["verification_sha256"]:
        raise ValueError("完整验证报告发生变化")
    report = json.loads(report_path.read_text())
    if not (report.get("normal_policy_entry") and report.get("online_episode_reset")
            and report["reload_output_max_error"] == 0 and report["resume_update_max_error"] == 0):
        raise ValueError("完整模型集成验证未通过")
    check_caches(cfg)
    if cfg["training"].get("not_before") != plan["start_at"]:
        raise ValueError("配置的最早启动时间与计划不一致")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != plan["code_commit"]:
        raise ValueError("源码提交已变化，请核对并重新登记计划")
    import wandb
    viewer = wandb.Api(timeout=20).viewer
    if viewer.entity != cfg["wandb"]["entity"]:
        raise ValueError("W&B 登录的 entity 与正式配置不一致")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    cfg = check_ready(plan)
    if args.check_only:
        print(json.dumps({"ready": True, "start_at": plan["start_at"], "run": cfg["paths"]["run"]}), flush=True)
        return
    status_path = plan_path.parent / "schedule_status.json"
    with (plan_path.parent / "schedule.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = datetime.fromisoformat(plan["start_at"]).timestamp()
        status = {"state": "waiting", "start_at": plan["start_at"], "plan": str(plan_path), "checked_at": now()}
        json_write(status_path, status)
        print(f"完整验证与 W&B 检查通过，等待 {plan['start_at']}", flush=True)
        while time.time() < target:
            if (plan_path.parent / "cancel_requested").exists():
                status.update(state="cancelled", finished_at=now())
                json_write(status_path, status)
                return
            time.sleep(min(30, max(0, target - time.time())))
        try:
            check_ready(plan)
            free_memory = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            if free_memory < plan["minimum_free_memory_mib"]:
                raise RuntimeError(f"GPU 可用显存不足，当前 {free_memory} MiB")
            status.update(state="running", actual_start_at=now())
            json_write(status_path, status)
            print(f"启动正式流水线 {status['actual_start_at']}", flush=True)
            subprocess.run(["bash", str(ROOT / "trained_model/scripts/run.sh"), "run", "--config", plan["config"]],
                           cwd=ROOT, check=True)
            status.update(state="complete", finished_at=now())
            json_write(status_path, status)
        except Exception as error:
            status.update(state="failed", error=str(error), traceback=traceback.format_exc(), failed_at=now())
            json_write(status_path, status)
            raise


if __name__ == "__main__":
    main()
