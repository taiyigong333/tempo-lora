"""PiperX 多卡全量微调，保持官方模型与原生 AdamW，使用精确全局 batch。"""
import argparse
from contextlib import nullcontext
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import wandb

from openpi.models.pi0_config import Pi0Config
from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from data import PiperDataset, ExactBatchSampler
from evaluate import evaluate_flow
from runtime_status import stage_status


def cpu_tree(value):
    return jax.tree.map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, value)


def permanent_checkpoint(step, settings):
    schedule = settings.get('save_steps')
    return step in schedule if schedule is not None else (step % settings['save_interval'] == 0 or step == settings['steps'])


def save_checkpoint(model, optimizer, output, step, settings, rank):
    """每个 rank 独立落盘优化器与随机状态，完成后才发布完整目录。"""
    final = output / f'step_{step:08d}'
    temp = output / f'.step_{step:08d}.incomplete'
    if rank == 0:
        temp.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    start = time.perf_counter()
    previous = None
    if rank == 0 and (output / 'latest.json').is_file():
        previous = json.loads((output / 'latest.json').read_text())
    # DDP各rank持有相同的标准AdamW状态，只保存一份以降低周期性落盘成本。
    # 发布用配置保留恢复训练的参数，运行路径另由实验控制器本地保存。
    checkpoint_settings = settings
    if settings.get('direct_actions'):
        checkpoint_settings = {key: value for key, value in settings.items()
                               if key not in {'project', 'output', 'split_path', 'norm_dir', 'resume'}}
    state = {'checkpoint_format_version': 2, 'optimizer_state_rank': 0, 'step': step,
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
             'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(), 'settings': checkpoint_settings}
    if rank == 0:
        state['optimizer'] = cpu_tree(optimizer.state_dict())
    torch.save(state, temp / f'rank_{rank}.pt')
    if rank == 0:
        safetensors.torch.save_model(model, temp / 'model.safetensors')
        (temp / 'config.json').write_text(json.dumps(checkpoint_settings, indent=2))
        (temp / 'model_config.json').write_text(json.dumps(dataclasses.asdict(model.config), indent=2))
        project = Path(settings['project'])
        norm_source = Path(settings.get('norm_dir') or project/'configs/norm')/'norm_stats.json'
        split_source = Path(settings.get('split_path') or project/'configs/split.json')
        for source in (norm_source, split_source):
            shutil.copyfile(source, temp/source.name)
    dist.barrier()
    if rank == 0:
        temp.rename(final)
        pointer = output / '.latest.json.tmp'
        pointer.write_text(json.dumps({'step': step, 'path': str(final)}))
        pointer.replace(output / 'latest.json')
        # 新快照完整发布后才清理上一个滚动版本，千步检查点永久保留。
        if previous is not None:
            old = Path(previous['path'])
            keep = permanent_checkpoint(previous['step'], settings)
            if not keep and old.parent == output and old.name == f"step_{previous['step']:08d}":
                try:
                    shutil.rmtree(old)
                except OSError as error:
                    print(f'旧滚动检查点清理失败，保留供后续检查 {old}: {error}', flush=True)
    dist.barrier()
    return time.perf_counter()-start, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--ut-dir', type=Path)
    ap.add_argument('--direct-actions', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split-path', type=Path)
    ap.add_argument('--norm-dir', type=Path)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--micro-batch', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--loader-timeout', type=int, default=0)
    ap.add_argument('--eval-interval', type=int, default=500)
    ap.add_argument('--eval-samples', type=int, default=128)
    ap.add_argument('--verify-action-shape', action='store_true')
    ap.add_argument('--decoder', type=Path)
    ap.add_argument('--save-interval', type=int, default=1000)
    ap.add_argument('--latest-save-interval', type=int, default=100)
    ap.add_argument('--status-interval', type=int, default=100)
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--mode', choices=['preflight', 'formal'], default='preflight')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--history-frames', type=int, default=300)
    ap.add_argument('--history-steps', type=int, default=10)
    ap.add_argument('--history-source', choices=['action', 'state'], default='action')
    ap.add_argument('--save-steps', type=int, nargs='+')
    ap.add_argument('--wandb-mode', choices=['online', 'offline'], default='online')
    args = ap.parse_args()
    if args.save_interval <= 0 or args.latest_save_interval <= 0:
        raise ValueError('检查点保存间隔必须大于0')
    if args.history_frames <= 0 or args.history_steps <= 0 or args.history_frames % args.history_steps:
        raise ValueError('历史帧数与位置数不兼容')
    if args.save_steps is not None and any(step <= 0 or step > args.steps for step in args.save_steps):
        raise ValueError('固定保存步数需位于本次训练范围内')
    # 测试入口限定更新数，正式运行要求显式选择 formal。
    if args.mode == 'preflight' and args.steps > 20:
        raise ValueError('preflight 最多20次更新')
    if args.mode == 'formal' and args.no_save:
        raise ValueError('正式训练需要保存checkpoint')
    project = args.project.resolve()
    if args.direct_actions and (args.ut_dir is not None or args.decoder is not None):
        ap.error('直接动作模式不接受 u_t 标签或解码器')
    if not args.direct_actions and args.ut_dir is None:
        ap.error('u_t 模式需要标签目录')
    if args.mode == 'formal' and not args.direct_actions:
        manifest = json.loads((args.ut_dir / 'manifest.json').read_text())
        if manifest.get('test_only') or manifest.get('window') != 30:
            raise ValueError('正式训练需要正式30帧MVAE标签')
        if args.decoder is None or not args.decoder.is_file():
            raise ValueError('正式训练需要正式decoder检查点')
    os.environ.setdefault('PALIGEMMA_TOKENIZER_PATH', str(project.parent / 'base_model/pi0.5_base/paligemma_tokenizer.model'))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'metrics.jsonl').exists() and not args.resume:
            raise FileExistsError('新实验需要独立输出目录')
    dist.barrier()
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(world_size=world, window=30, precision='official mixed parameter dtypes + BF16 autocast',
                    optimizer='torch.optim.AdamW (official native parameter/state dtypes)', peak_lr=5e-5, end_lr=5e-6, warmup=500,
                    decay_steps=5000, betas=[.9, .95], eps=1e-8, weight_decay=1e-10,
                    clip_norm=1., seed=42, action_history_seconds=args.history_frames/30, history_bucket_frames=args.history_frames//args.history_steps)
    run = None
    if rank == 0:
        id_path = output / 'wandb_id.txt'
        run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'tempo-piperx-ma800'), name=output.name, mode=args.wandb_mode,
                         group=output.parent.name, config=settings,
                         id=id_path.read_text().strip() if args.resume and id_path.exists() else None,
                         resume='must' if args.resume and id_path.exists() else None, dir=str(output))
        id_path.write_text(run.id)
        (output / 'config.json').write_text(json.dumps(settings, indent=2))
        stage_status(output, 'policy', 'initializing', total_steps=args.steps, world_size=world, wandb_url=run.url)
        print('WANDB', run.url, flush=True)
    config = Pi0Config(pi05=True, dtype='bfloat16', action_horizon=30, obs_history=3,
                       obs_history_stride_s=.333, temporal_attn_period=4,
                       use_sam2_fusion=True, sam2_token_dim=256, sam2_num_tokens=64,
                       use_action_history=True, action_history_steps=args.history_steps, action_history_dim=7,
                       predict_ut=not args.direct_actions, ut_dim=64, ut_window=30,
                       ut_decoder_path=str(args.decoder.resolve()) if args.decoder else None)
    # 显式核对复现约定，避免数据适配意外改动模型输出空间。
    assert (config.action_dim, config.flow_dim, config.flow_positions, config.action_horizon, config.ut_window) == ((32, 32, 30, 30, 30) if args.direct_actions else (32, 64, 1, 30, 30))
    model = PI0Pytorch(config)
    if rank == 0:
        (output / 'model_config.json').write_text(json.dumps(dataclasses.asdict(config), indent=2))
        print(f'MODEL_CONTRACT predict_ut={config.predict_ut} action_dim={config.action_dim} '
              f'flow_dim={config.flow_dim} flow_positions={config.flow_positions} robot_action_dim=7', flush=True)
    if args.resume:
        safetensors.torch.load_model(model, args.resume / 'model.safetensors')
    else:
        weights = safetensors.torch.load_file(Path(os.environ.get('TEMPO_PRETRAINED_DIR', project/'caches/pi05_base_pytorch')) / 'model.safetensors')
        own = model.state_dict()
        resized = [k for k, v in weights.items() if k in own and v.shape != own[k].shape]
        for key in resized:
            del weights[key]
        missing, unexpected = model.load_state_dict(weights, strict=False)
        pointers = {v.data_ptr() for k, v in model.state_dict().items() if k in weights}
        illegal = [k for k in missing if not k.startswith(('sam2_fusion.', 'action_history_', 'action_in_proj.', 'action_out_proj.')) and model.state_dict()[k].data_ptr() not in pointers]
        if illegal or unexpected:
            raise ValueError({'missing': illegal, 'unexpected': unexpected})
        del weights, own
    model = model.to(device)
    model.gradient_checkpointing_enable()
    # 直接使用官方 AdamW 与模型原始参数精度，训练状态驻留 GPU。
    # Expert的词表输出头不参与策略前向，冻结后可省去无用DDP梯度桶。
    model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(.9, .95), eps=1e-8, weight_decay=1e-10,
    )
    ddp = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                  gradient_as_bucket_view=True, broadcast_buffers=False)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume / f'rank_{rank}.pt', map_location='cpu', weights_only=False)
        if state['settings']['batch'] != args.batch or state['settings']['world_size'] != world:
            raise ValueError('恢复时保持全局 batch 与 rank 数一致')
        # 兼容历史逐rank完整状态，以及新格式的共享优化器快照。
        optimizer_state = state.get('optimizer')
        if optimizer_state is None:
            optimizer_state = torch.load(args.resume / 'rank_0.pt', map_location='cpu', weights_only=False)['optimizer']
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state
        start_step = state['step']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
    else:
        torch.manual_seed(42+rank)
    history_options = dict(history_frames=args.history_frames, history_steps=args.history_steps, history_source=args.history_source)
    dataset = PiperDataset(project, args.ut_dir, split_path=args.split_path, norm_dir=args.norm_dir, **history_options)
    validation = PiperDataset(project, args.ut_dir, subset='val', split_path=args.split_path, norm_dir=args.norm_dir, **history_options) if args.eval_interval > 0 else None
    sampler = ExactBatchSampler(len(dataset), args.batch, args.micro_batch, rank, world, start_step, args.steps)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, timeout=args.loader_timeout if args.workers > 0 else 0, **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if args.workers else {}))
    iterator = iter(loader)
    model.train()
    rows = []
    if rank == 0:
        stage_status(output, 'policy', 'running', step=start_step, total_steps=args.steps,
                     batch=args.batch, micro_batch=args.micro_batch, accumulation=sampler.accum, wandb_url=run.url)
    for step in range(start_step, args.steps):
        torch.cuda.synchronize()
        dist.barrier()
        begin = time.perf_counter()
        lr = 5e-5/501 + (5e-5-5e-5/501)*step/500 if step < 500 else 5e-6+(5e-5-5e-6)*.5*(1+math.cos(math.pi*min(1.,(step-500)/4500)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=False)
        total = torch.zeros((), device=device)
        for micro in range(sampler.accum):
            batch = next(iterator)
            batch = jax.tree.map(lambda x: x.to(device, non_blocking=True), batch)
            observation = Observation.from_dict(batch)
            count = int(batch.pop('sample_weight').sum())
            # 首个micro先建立梯度桶视图，此后保留视图，避免no_sync首次复制整份梯度。
            synchronize = micro == sampler.accum-1 or (step == start_step and micro == 0)
            context = nullcontext() if synchronize else ddp.no_sync()
            with context, torch.autocast('cuda', dtype=torch.bfloat16):
                loss = ddp(observation, batch['actions']).float().mean()
                # DDP 对 rank 取均值，此处按样本数加权恢复精确全局均值。
                scaled = loss * (world*count/args.batch)
            scaled.backward()
            total += loss.detach()*count
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        gate_grad = model.sam2_fusion.gate.grad.detach().abs().max().item()
        optimizer.step()
        signatures = [None]*world
        signature = [float(model.action_out_proj.weight.mean()), float(model.sam2_fusion.gate.mean())]
        dist.all_gather_object(signatures, signature)
        if not np.allclose(signatures, signatures[0], rtol=0, atol=1e-7):
            raise ValueError("rank参数同步不一致")
        torch.cuda.synchronize()
        duration = torch.tensor(time.perf_counter()-begin, device=device)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        dist.all_reduce(total)
        memories = [None]*world
        dist.all_gather_object(memories, torch.cuda.max_memory_allocated()/2**30)
        row = dict(step=step+1, loss=float(total/args.batch), lr=lr, grad_norm=float(norm),
                   seconds=float(duration), samples_per_second=args.batch/float(duration),
                   peak_gib=memories, sam2_gate_grad=gate_grad)
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
            with (output/'metrics.jsonl').open('a') as log:
                log.write(json.dumps(row)+'\n')
            wandb.log(row, step=step+1)
            if (step+1)%args.status_interval == 0 or step+1 == args.steps:
                stage_status(output, 'policy', 'running', **row, total_steps=args.steps,
                             batch=args.batch, micro_batch=args.micro_batch, accumulation=sampler.accum, wandb_url=run.url)
        if validation is not None and ((step+1)%args.eval_interval == 0 or step+1 == args.steps):
            validation_metrics = evaluate_flow(model, validation, device, args.eval_samples)
            if rank == 0:
                wandb.log(validation_metrics, step=step+1)
                with (output/'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': step+1, **validation_metrics})+'\n')
        if args.verify_action_shape and step+1 == args.steps:
            if rank == 0:
                from torch.utils.data import default_collate
                sample = default_collate([(validation if validation is not None else dataset)[50]])
                sample.pop('sample_weight')
                sample = jax.tree.map(lambda x: x.to(device), sample)
                model.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    actions = PI0Pytorch.sample_actions(model, device, Observation.from_dict(sample), num_steps=2)
                assert tuple(actions.shape) == (1, 30, 32) and torch.isfinite(actions).all()
                (output/'action_contract.json').write_text(json.dumps({'shape': list(actions[:, :, :7].shape), 'finite': True, 'test_only': True}))
                model.train()
            dist.barrier()
        if not args.no_save and ((step+1)%args.latest_save_interval == 0
                                 or permanent_checkpoint(step+1, settings) or step+1 == args.steps):
            elapsed, checkpoint = save_checkpoint(model, optimizer, output, step+1, settings, rank)
            if rank == 0:
                wandb.log({'checkpoint_seconds': elapsed}, step=step+1)
                (output/'checkpoint_timing.json').write_text(json.dumps({'seconds': elapsed, 'path': str(checkpoint)}))
    if rank == 0:
        measured = [r['seconds'] for r in rows[1:]] or [r['seconds'] for r in rows]
        steady_seconds = float(np.median(measured)) if measured else None
        report = {'test_only': args.mode == 'preflight', 'steady_step_seconds': steady_seconds,
                  'steps_in_this_run': len(rows), 'world_size': world, 'batch': args.batch,
                  'rank_samples': sampler.rank_counts, 'accumulation': sampler.accum,
                  'estimated_5000_step_hours': steady_seconds*5000/3600 if steady_seconds is not None else None, 'wandb_url': run.url}
        (output/'summary.json').write_text(json.dumps(report, indent=2))
        stage_status(output, 'policy', 'finishing', step=args.steps, total_steps=args.steps, wandb_url=run.url)
        run.finish()
        stage_status(output, 'policy', 'completed', step=args.steps, total_steps=args.steps, **report)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
