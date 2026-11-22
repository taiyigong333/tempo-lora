"""原子发布阶段状态，供tmux监督进程与人工巡检读取。"""
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2))
    temporary.replace(path)


def stage_status(output, stage, state, **fields):
    write_json(Path(output)/'latest_status.json', {
        'stage': stage, 'state': state, 'updated_at': timestamp(), **fields,
    })
