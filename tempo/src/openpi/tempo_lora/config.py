"""训练、缓存和服务共用同一个配置快照。"""

import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "trained_model/config_use/pi05_piperx_tempo_ut_lora.json"
CONVERSION_COMMIT = "45d9f8275316c455c3aeb2c375add8b37d1bf661"
UPSTREAM_COMMIT = "dd8342a1234e7166b856ffec2d7917b51cce05e4"
ACTION_SPACE = "piperx_next_feedback_delta_joints_abs_gripper_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    return validate_config(cfg)


def validate_config(cfg):
    cfg = copy.deepcopy(cfg)
    data, model, aux = cfg["data"], cfg["model"], cfg["auxiliary"]
    if data["action_dim"] != model["action_history_dim"] or data["action_dim"] not in (7, 14):
        raise ValueError("原生动作和历史动作维度必须一致，支持 7D 或 14D")
    if not data["use_delta_joint_actions"] or data["undistort_crop"]:
        raise ValueError("当前反馈代理契约要求关节相对化，并关闭 YAM 鱼眼裁剪")
    if data["fps"] != data["action_history_fps"]:
        raise ValueError("历史反馈桶与数据帧率必须一致")
    if model["action_dim"] != 32:
        raise ValueError("π0.5 原生动作维度保持 32")
    if (model["sam2_token_dim"], model["sam2_num_tokens"]) != (256, 64):
        raise ValueError("当前 SAM2 tiny 特征契约为 64 个 256D tokens")
    if not (aux["window"] == model["action_horizon"] == model["ut_window"]):
        raise ValueError("MVAE、解码器与策略窗口必须一致")
    if aux["latent_dim"] != model["ut_dim"]:
        raise ValueError("MVAE 和 flow 的 u_t 维度必须一致")
    if cfg["training"]["ema_enabled"]:
        raise ValueError("首版 EMA 未启用，请显式使用 false")
    if model["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("仅支持 BF16 混合精度与全 FP32 诊断")
    for key in cfg["paths"]:
        cfg["paths"][key] = str((ROOT / cfg["paths"][key]).resolve())
    cfg["data"]["root"] = str((ROOT / data["root"]).resolve())
    return cfg


def artifact_root(cfg):
    return Path(cfg["paths"]["artifacts"])


def configure_environment(cfg):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["SAM2_REPO"] = cfg["paths"]["sam2_repo"]
    os.environ["SAM2_CKPT"] = cfg["paths"]["sam2_checkpoint"]
    os.environ["OPENPI_TOKENIZER_PATH"] = cfg["paths"]["tokenizer"]
    proxy = cfg.get("wandb", {}).get("proxy")
    if proxy:
        # 仅影响本项目进程，复用服务器已有 W&B 专用出口。
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ[name] = proxy


def model_config(cfg, *, base=False):
    from openpi.models.pi0_config import Pi0Config

    values = dict(cfg["model"])
    if base:
        values.update(predict_ut=False, lora_enabled=False, use_sam2_fusion=False,
                      use_action_history=False, obs_history=1)
    else:
        values["ut_decoder_path"] = str(artifact_root(cfg) / "decoder/best.pt")
    return Pi0Config(**values)


def make_train_config(cfg=None):
    from openpi.training.config import AssetsConfig, LeRobotYamDataConfig, TrainConfig

    cfg = read_config() if cfg is None else cfg
    configure_environment(cfg)
    cache = artifact_root(cfg)
    return TrainConfig(
        name=cfg["name"], model=model_config(cfg),
        exp_name=Path(cfg["paths"]["run"]).name,
        checkpoint_base_dir=str(Path(cfg["paths"]["run"]).parents[1]),
        batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"],
        num_train_steps=cfg["training"]["steps"], grad_accum_steps=cfg["training"]["grad_accum_steps"],
        log_interval=cfg["training"]["log_interval"], save_interval=cfg["training"]["save_interval"],
        pytorch_weight_path=str(Path(cfg["paths"]["base_pytorch"]).parent),
        assets_base_dir=str(cache / "assets"), ema_decay=None, wandb_enabled=False,
        data=LeRobotYamDataConfig(
            repo_id=cfg["data"]["repo_id"], action_dim=cfg["data"]["action_dim"],
            cameras=tuple(cfg["data"]["cameras"]), use_delta_joint_actions=True,
            undistort_crop=False, default_prompt=cfg["data"]["prompt"],
            assets=AssetsConfig(assets_dir=str(cache), asset_id="norm"),
            sam2_cache_dir=str(cache / "sam2"), action_history_cache_dir=str(cache / "sam2"),
            action_history_fps=cfg["data"]["fps"],
            action_history_seconds=cfg["data"]["action_history_seconds"],
            action_history_steps=cfg["model"]["action_history_steps"],
            ut_cache_dir=str(cache / "ut") if cfg["model"]["predict_ut"] else None,
        ),
    )
