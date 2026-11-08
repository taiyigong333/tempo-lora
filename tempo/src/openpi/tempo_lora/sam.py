"""离线和在线共享连续视频状态，逐帧上传避免整段 1024px 视频占用显存。"""

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from openpi.tempo_lora.config import ROOT, artifact_root, file_digest
from openpi.tempo_lora.data import contract, episode_path, metadata, read_episode, write_manifest

sys.path.insert(0, str(ROOT / "tempo"))
from tools.sam2_encoder import build_sam2_tiny, _full_frame_mask
from tools.precompute_sam2_tokens import read_video_rgb


class SAM2Stream:
    def __init__(self, model, device="cuda", precision="bfloat16", *, compile_model=False):
        self.model, self.device, self.precision = model, device, precision
        self.model.requires_grad_(False)
        self.executor = None
        self.cuda_stream = None
        if compile_model:
            # 图像骨干与记忆编码形状固定，时序记忆选择继续逐帧按原逻辑执行。
            self.model.forward_image = torch.compile(self.model.forward_image, dynamic=False, mode="default")
            self.model._encode_new_memory = torch.compile(self.model._encode_new_memory, dynamic=False, mode="default")
        self.reset()

    def reset(self):
        self.frame = 0
        self.outputs = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    @torch.no_grad()
    def step(self, rgb, *, return_tensor=False):
        import cv2

        x = torch.from_numpy(cv2.resize(rgb, (1024, 1024))).permute(2, 0, 1)[None]
        mixed = str(self.device).startswith("cuda") and self.precision == "bfloat16"
        dtype = torch.bfloat16 if mixed else torch.float32
        x = x.to(device=self.device, dtype=dtype) / 255.0
        context = torch.autocast("cuda", dtype=torch.bfloat16) if mixed else nullcontext()
        with context:
            features = self.model.forward_image(x)
            _, vf, vp, sizes = self.model._prepare_backbone_features(features)
            pix = self.model._prepare_memory_conditioned_features(
                frame_idx=self.frame, is_init_cond_frame=self.frame == 0,
                current_vision_feats=vf[-1:], current_vision_pos_embeds=vp[-1:], feat_sizes=sizes[-1:],
                output_dict=self.outputs, num_frames=self.frame + 1, track_in_reverse=False)
            latent = F.adaptive_avg_pool2d(pix, (8, 8))[0].float()
            memory, position = self.model._encode_new_memory(
                current_vision_feats=vf, feat_sizes=sizes,
                pred_masks_high_res=_full_frame_mask(self.model.image_size, self.device, dtype),
                object_score_logits=torch.ones(1, 1, device=self.device, dtype=dtype), is_mask_from_pts=False)
            slot = "cond_frame_outputs" if self.frame == 0 else "non_cond_frame_outputs"
            self.outputs[slot][self.frame] = {
                "maskmem_features": memory, "maskmem_pos_enc": position,
                "obj_ptr": torch.zeros(1, self.model.hidden_dim, device=self.device, dtype=dtype),
                "object_score_logits": torch.ones(1, 1, device=self.device, dtype=dtype)}
        # 保留首帧条件和模型可能访问的最近记忆，在线长度不随 episode 无界增长。
        keep = max(self.model.num_maskmem * self.model.memory_temporal_stride_for_eval,
                   self.model.max_obj_ptrs_in_encoder) + 2
        for old in list(self.outputs["non_cond_frame_outputs"]):
            if old < self.frame - keep:
                del self.outputs["non_cond_frame_outputs"][old]
        self.frame += 1
        return latent if return_tensor else latent.cpu().numpy()

    def encode_async(self, images):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tempo-sam2")
            self.cuda_stream = torch.cuda.Stream(device=self.device)
        # 权重初始化和上轮融合均先于本轮SAM2输入，双向依赖由CUDA事件串接。
        self.cuda_stream.wait_stream(torch.cuda.current_stream(self.device))

        def encode():
            with torch.cuda.stream(self.cuda_stream):
                for rgb in images:
                    latent = self.step(rgb, return_tensor=True)
                tokens = latent.permute(1, 2, 0).reshape(1, 64, 256)
                ready = torch.cuda.Event()
                ready.record(self.cuda_stream)
            return tokens, ready

        future = self.executor.submit(encode)

        def resolve():
            tokens, ready = future.result()
            torch.cuda.current_stream(self.device).wait_event(ready)
            return tokens

        return resolve


def prepare_sam(cfg, *, limit=None):
    cache = artifact_root(cfg)
    source = cfg["paths"].get("feature_reuse_source")
    if source and (Path(source) / "contract.json").exists():
        previous = json.loads((Path(source) / "contract.json").read_text())
        current = contract(cfg)
        for key in ("data_metadata_sha256", "action_space", "action_dim", "units"):
            if previous[key] != current[key]:
                raise ValueError(f"复用 SAM2 特征的数据契约不匹配 {key}")
        for kind in ("sam2", "rgb"):
            (cache / kind).mkdir(parents=True, exist_ok=True)
            for path in (Path(source) / kind).glob("episode_*"):
                destination = cache / kind / path.name
                if ".tmp." not in path.name and not destination.exists():
                    # 原始冻结特征与训练划分无关，硬链接复用只读数组，统计和标签重新构建。
                    os.link(path, destination)
    directory = cache / "sam2"
    manifest = write_manifest(cfg, directory, "sam2", checkpoint_sha256=file_digest(cfg["paths"]["sam2_checkpoint"]),
                              encoder="tempo_full_frame_memory_v1", grid=8, rgb_range="0_1")
    rgb_dir = cache / "rgb"
    write_manifest(cfg, rgb_dir, "rgb")
    model = build_sam2_tiny("cuda")
    stream = SAM2Stream(model, precision=cfg["model"]["dtype"])
    _, episodes = metadata(cfg)
    start = time.monotonic()
    for ep in episodes[:limit]:
        index, length = ep["episode_index"], ep["length"]
        destination = directory / f"episode_{index:06d}_latent.npy"
        for camera in cfg["data"]["cameras"]:
            output = rgb_dir / f"episode_{index:06d}_{camera}.npy"
            if not output.exists():
                frames = read_video_rgb(episode_path(cfg, index, video_key=f"observation.images.{camera}"))
                if frames.shape != (length, 224, 224, 3):
                    raise ValueError(f"视频形状不匹配 episode={index} camera={camera}")
                temporary = output.with_suffix(".tmp.npy")
                np.save(temporary, frames)
                temporary.replace(output)
        if destination.exists():
            if np.load(destination, mmap_mode="r").shape != (length, 256, 8, 8):
                raise ValueError(f"缓存长度不匹配 {destination}")
            continue
        video = np.load(rgb_dir / f"episode_{index:06d}_head.npy", mmap_mode="r")
        stream.reset()
        latent = np.stack([stream.step(frame) for frame in video])
        temporary = destination.with_suffix(".tmp.npy")
        np.save(temporary, latent)
        temporary.replace(destination)
        row = read_episode(cfg, index)
        # 同时写原有 TEMPO 工具可读的 npz，保留原接口和原始动作。
        temporary = directory / f"episode_{index:06d}.tmp.npz"
        np.savez_compressed(temporary, latent=latent, action=row["action"].astype(np.float32),
                            state=row["observation.state"].astype(np.float32), ep_idx=index, ep_len=length,
                            contract=manifest["contract"])
        temporary.replace(directory / f"episode_{index:06d}.npz")
        print(f"SAM2 episode={index} frames={length} elapsed={time.monotonic() - start:.1f}s", flush=True)
