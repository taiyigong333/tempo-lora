"""按依赖顺序运行，阶段完成后持久化状态，异常保留现场。"""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from openpi.tempo_lora.config import ROOT, artifact_root
from openpi.tempo_lora.data import json_write


def run(cfg):
    runtime = ROOT / "trained_model/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg_path = runtime / "effective_config.json"
        json_write(cfg_path, cfg)
        status_path = runtime / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"completed": []}
        for key in ("error", "traceback", "finished"):
            status.pop(key, None)
        status.update(config=str(cfg_path), started=time.time(), state="running")
        cache = artifact_root(cfg)
        stages = ["validate", "convert", "norm", "sam2", "mvae", "encode", "decoder", "verify", "train", "evaluate"]
        if Path(cfg["paths"]["base_pytorch"]).exists() and "convert" not in status["completed"]:
            status["completed"].append("convert")
        try:
            for stage in stages:
                if stage in status["completed"]:
                    continue
                status.update(stage=stage, updated=time.time())
                json_write(status_path, status)
                command = [sys.executable, str(ROOT / "tempo/scripts/piperx.py"), stage, "--config", str(cfg_path)]
                if stage in ("mvae", "decoder") and (cache / stage / "last.pt").exists():
                    command.append("--resume")
                if stage == "train" and (Path(cfg["paths"]["run"]) / "latest.json").exists():
                    command.append("--resume")
                with (ROOT / "trained_model/logs" / f"{stage}_alltrain.log").open("a") as log:
                    subprocess.run(command, check=True, cwd=ROOT / "tempo", stdout=log, stderr=subprocess.STDOUT)
                status["completed"].append(stage)
                json_write(status_path, status)
            status.update(state="complete", finished=time.time())
            json_write(status_path, status)
        except Exception as error:
            status.update(state="failed", error=str(error), traceback=traceback.format_exc(), updated=time.time())
            json_write(status_path, status)
            raise
