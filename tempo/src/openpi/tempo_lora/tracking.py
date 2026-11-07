"""辅助模型与策略分别记录独立的在线 W&B run，断点恢复保持 run id。"""

import json
from pathlib import Path

from openpi.tempo_lora.config import ROOT
from openpi.tempo_lora.data import json_write


class Tracker:
    def __init__(self, cfg, stage, output, *, resume=False):
        self.run = None
        settings = cfg.get("wandb", {})
        if not settings.get("enabled", False):
            return
        import wandb
        path = Path(output) / "wandb_run.json"
        previous = json.loads(path.read_text()) if resume and path.exists() else None
        directory = ROOT / "trained_model/wandb"
        directory.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(project=settings["project"], entity=settings.get("entity"), mode=settings["mode"],
                              name=f"{Path(cfg['paths']['run']).name}_{stage}", group=Path(cfg["paths"]["run"]).name,
                              job_type=stage, config=cfg, dir=str(directory),
                              id=previous["id"] if previous else None, resume="must" if previous else None)
        json_write(path, {"id": self.run.id, "url": self.run.url, "stage": stage})
        print(f"W&B {self.run.url}", flush=True)

    def log(self, values, step):
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
