"""握手、数据回放及实机只读推理入口。"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from tempo_client.transport import Client, validate_frames

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/piperx.yaml"


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["live"]["camera_parameters"] = str(path.parent / cfg["live"]["camera_parameters"])
    return cfg


def replay_frames(path):
    # 回放文件只接受数值数组，避免加载带任意 Python 对象的文件。
    with np.load(path, allow_pickle=False) as data:
        timestamps, states = data["timestamps"], data["states"]
        head, wrist = data["head"], data["left_wrist"]
        count = len(timestamps)
        if states.shape != (count, 7) or head.shape != (count, 224, 224, 3) or wrist.shape != head.shape:
            raise ValueError("回放数组长度或形状不匹配")
        frames = [{"timestamp": float(timestamps[i]), "state": states[i].copy(),
                   "images": {"head": head[i].copy(), "left_wrist": wrist[i].copy()}} for i in range(count)]
    validate_frames(frames, 30)
    return frames


def report(result, frames):
    return {"feedback_frames": len(frames), "timestamp": frames[-1]["timestamp"],
            "actions": result["actions"].tolist(), "round_trip_ms": result["round_trip_ms"],
            "end_to_end_ms": result.get("end_to_end_ms"), "robot_commands": 0}


def main():
    parser = argparse.ArgumentParser(description="Tempo PiperX 有线只读客户端")
    parser.add_argument("command", choices=["probe", "replay", "live"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("batch-size 与 duration 必须为正")
    if args.command == "replay" and args.input is None:
        parser.error("replay 需要 --input")
    cfg = load_config(args.config)
    frames = replay_frames(args.input) if args.command == "replay" else None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with Client(cfg) as client:
        print(json.dumps({"local_ip": client.local_ip, "metadata": client.metadata}, ensure_ascii=False), flush=True)
        if args.command == "probe":
            return
        records = []
        if frames is not None:
            for start in range(0, len(frames), args.batch_size):
                batch = frames[start:start + args.batch_size]
                item = report(client.infer(batch), batch)
                records.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
                if args.output:
                    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
            return
        from tempo_client.live import LiveFeedback
        with LiveFeedback(cfg) as source:
            until = time.monotonic() + args.duration
            while time.monotonic() < until:
                batch = source.take()
                item = report(client.infer(batch), batch)
                records.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
                if args.output:
                    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
