"""确定性验证flow误差，复用正式数据入口并恢复训练随机状态。"""
import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from data import PiperDataset


@torch.no_grad()
def evaluate_flow(model, dataset, device, samples=128):
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    selected = np.random.default_rng(42).permutation(len(dataset))[:samples]
    previous = model.training
    model.eval()
    totals = torch.zeros(2, device=device)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(42+rank)
        for index in selected[rank::world]:
            batch = default_collate([dataset[int(index)]])
            batch.pop('sample_weight')
            batch = jax.tree.map(lambda x: x.to(device), batch)
            observation = Observation.from_dict(batch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(observation, batch['actions']).float().mean()
            valid = (~observation.ut_target_is_pad).sum()
            totals[0] += loss
            totals[1] += valid
    if distributed:
        dist.all_reduce(totals)
    model.train(previous)
    if totals[1] == 0:
        raise ValueError('验证样本中没有完整的30帧窗口')
    return {'val_flow_mse': float(totals[0]/totals[1]), 'val_valid_samples': int(totals[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--decoder', type=Path, required=True)
    args = ap.parse_args()
    project = args.project.resolve()
    os.environ['PALIGEMMA_TOKENIZER_PATH'] = str(project.parent/'base_model/pi0.5_base/paligemma_tokenizer.model')
    device = torch.device('cuda:0')
    dataset = PiperDataset(project, project/'experiments/preflight_20261114/ut', subset='val')
    cfg = Pi0Config(pi05=True, dtype='bfloat16', action_horizon=30, obs_history=3,
                    use_sam2_fusion=True, use_action_history=True, action_history_dim=7,
                    predict_ut=True, ut_dim=64, ut_window=30, ut_decoder_path=str(args.decoder.resolve()),
                    checkpoint_joint_forward=False)
    model = PI0Pytorch(cfg)
    safetensors.torch.load_model(model, args.checkpoint/'model.safetensors')
    model = model.to(device).eval()
    result = evaluate_flow(model, dataset, device, samples=8)
    # 实际走项目采样和解码入口，检查30帧7D输出契约。
    batch = default_collate([dataset[50]])
    batch.pop('sample_weight')
    batch = jax.tree.map(lambda x: x.to(device), batch)
    observation = Observation.from_dict(batch)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = PI0Pytorch.sample_actions(model, device, observation, num_steps=2)
    assert tuple(prediction.shape) == (1, 30, 32) and torch.isfinite(prediction).all()
    result.update(sampled_action_shape=list(prediction[:, :, :7].shape), test_only=True,
                  note='测试策略与测试辅助模型仅验证流程，数值不代表正式任务精度')
    (project/'experiments/preflight_20261114/evaluation_smoke.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
