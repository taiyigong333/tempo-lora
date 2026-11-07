"""从数据准备到训练/恢复/服务的统一入口。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpi.tempo_lora.config import DEFAULT_CONFIG, configure_environment, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "split", "norm", "convert", "sam2", "mvae", "encode", "decoder",
                                          "train", "verify", "evaluate", "serve", "run"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    cfg = read_config(args.config)
    configure_environment(cfg)
    if args.stage == "validate":
        from openpi.tempo_lora.data import validate_data
        validate_data(cfg)
    elif args.stage == "split":
        from openpi.tempo_lora.data import split_data
        split = split_data(cfg)
        print(json.dumps({key: len(split[key]) for key in ("train_episodes", "val_episodes")}))
    elif args.stage == "norm":
        from openpi.tempo_lora.data import norm_stats
        norm_stats(cfg)
    elif args.stage == "convert":
        from examples.convert_jax_model_to_pytorch import convert_pi0_checkpoint
        from openpi.tempo_lora.config import model_config
        destination = Path(cfg["paths"]["base_pytorch"])
        if destination.exists():
            raise FileExistsError(f"已有基座 {destination}，保留现有权重")
        convert_pi0_checkpoint(cfg["paths"]["base_jax"], "float32", str(destination.parent), model_config(cfg, base=True))
    elif args.stage == "sam2":
        from openpi.tempo_lora.sam import prepare_sam
        prepare_sam(cfg, limit=args.limit)
    elif args.stage in ("mvae", "decoder"):
        from openpi.tempo_lora.auxiliary import train_auxiliary
        train_auxiliary(cfg, args.stage, resume=args.resume, epochs=args.epochs)
    elif args.stage == "encode":
        from openpi.tempo_lora.auxiliary import encode_ut
        encode_ut(cfg)
    elif args.stage == "train":
        from openpi.tempo_lora.train import train_policy
        train_policy(cfg, resume=args.resume, steps=args.steps)
    elif args.stage == "verify":
        from openpi.tempo_lora.verify import verify
        verify(cfg)
    elif args.stage == "evaluate":
        from openpi.tempo_lora.inference import evaluate
        evaluate(cfg, args.checkpoint, limit=args.limit)
    elif args.stage == "serve":
        from openpi.tempo_lora.inference import serve
        serve(cfg, args.checkpoint, port=args.port)
    else:
        from openpi.tempo_lora.pipeline import run
        run(cfg)


if __name__ == "__main__":
    main()
