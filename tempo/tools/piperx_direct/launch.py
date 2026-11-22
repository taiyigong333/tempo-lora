"""先验证真实训练和推理，再自动启动正式训练与发布。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def status(state, stage, **extra):
    record = dict(state=state, stage=stage, experiment=name, gpus=gpus,
                  updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **extra)
    for path in (root / 'status.json', runtime / 'latest_status.json'):
        write_json(path, record)
    print(json.dumps(record), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='项目数据、配置和实验根目录')
    parser.add_argument('--baseline', type=Path, required=True, help='同一顺时针训练数据及缓存配置')
    parser.add_argument('--tempo-repo', type=Path, required=True, help='带 .venv 的已验证 TEMPO 源码目录')
    parser.add_argument('--name', required=True, help='新实验目录名，同时用于发布前缀')
    parser.add_argument('--gpus', default='1,2', help='逗号分隔的 GPU 编号')
    parser.add_argument('--micro-batch', type=int, default=16)
    parser.add_argument('--publisher-python', type=Path, required=True, help='安装 ModelScope 的独立 Python')
    args = parser.parse_args()
    project = args.project.resolve()
    repo = args.tempo_repo.resolve()
    name = args.name
    source = Path(__file__).resolve().parent
    if not name.replace('-', '').replace('_', '').isalnum():
        parser.error('实验名只能包含字母、数字、短横线和下划线')
    gpus = [int(index) for index in args.gpus.split(',')]
    micro_batch = args.micro_batch
    if len(set(gpus)) != len(gpus) or min(gpus) < 0 or micro_batch < 1:
        parser.error('GPU 编号应为不同的非负整数，micro-batch 必须大于零')
    baseline = json.loads(args.baseline.read_text())
    runtime = project / 'runtime' / name
    root = project / 'experiments' / name
    config_path = project / 'configs' / f'{name}.json'
    if root.exists() or runtime.exists() or config_path.exists():
        parser.error('实验目录或配置已存在，请为新的训练指定新名称')
    runtime.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir(parents=True, exist_ok=False)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {key: baseline[key] for key in (
        'dataset', 'split_path', 'norm_dir', 'history', 'dataset_root', 'rgb_cache',
        'sam2_cache', 'pretrained_dir', 'tokenizer', 'remove_local_after_upload', 'upload_retry_seconds')}
    per_rank, remainder = divmod(128, len(gpus))
    rank_samples = [per_rank + int(rank < remainder) for rank in range(len(gpus))]
    config.update(project=str(project), experiment_root=str(root), runtime_directory=str(runtime),
                  tmux_session=name, gpus=gpus,
                  model_contract=dict(predict_ut=False, flow_dim=32, flow_positions=30,
                                      action_horizon=30, robot_action_dim=7),
                  policy={**baseline['policy'], 'micro_batch': micro_batch, 'rank_samples': rank_samples,
                          'accumulation_steps': (max(rank_samples) + micro_batch - 1) // micro_batch},
                  wandb={**baseline['wandb'], 'project': name},
                  modelscope={**baseline['modelscope'], 'path_prefix': f'ma800/{name}'})

    def status(state, stage, **extra):
        record = dict(state=state, stage=stage, experiment=name, gpus=gpus,
                      updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **extra)
        for path in (root / 'status.json', runtime / 'latest_status.json'):
            write_json(path, record)
        print(json.dumps(record), flush=True)

    write_json(config_path, config)
    write_json(root / 'config.json', config)
    temp = project / '.t90'
    temp.mkdir(mode=0o700, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='4', WANDB_ENTITY=config['wandb']['entity'],
               WANDB_PROJECT=config['wandb']['project'], WANDB_INIT_TIMEOUT='120',
               WANDB_CACHE_DIR=str(runtime / 'wandb_cache'), TMPDIR=str(temp),
               PALIGEMMA_TOKENIZER_PATH=config['tokenizer'], TEMPO_PRETRAINED_DIR=config['pretrained_dir'],
               TEMPO_RGB_CACHE=config['rgb_cache'], TEMPO_SAM2_CACHE=config['sam2_cache'],
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PYTHONUNBUFFERED='1', PYTHONPATH=str(repo / 'src'))
    if proxy := config['wandb'].get('fallback_proxy'):
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, NO_PROXY='localhost,127.0.0.1')
    torchrun = [str(repo / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node', str(len(gpus)),
                str(source / 'train.py'), '--project', str(project), '--direct-actions',
                '--split-path', config['split_path'], '--norm-dir', config['norm_dir'],
                '--history-frames', '90', '--history-steps', '10', '--history-source', 'state',
                '--loader-timeout', '120', '--batch', '128', '--micro-batch', str(micro_batch),
                '--workers', '2', '--eval-interval', '0', '--latest-save-interval', '100']

    def execute(stage, stage_args):
        log = root / 'logs' / f'{stage}.log'
        status('running', stage, log=str(log), micro_batch=micro_batch)
        with log.open('a') as stream:
            subprocess.run(torchrun + stage_args, env=env, cwd=repo, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    preflight = root / 'preflight'
    try:
        execute('preflight', ['--mode', 'preflight', '--output', str(preflight), '--steps', '3',
                              '--status-interval', '1', '--verify-action-shape', '--no-save',
                              '--wandb-mode', 'offline'])
        contract = json.loads((preflight / 'action_contract.json').read_text())
        if contract['shape'] != [1, 30, 7] or not contract['finite']:
            raise ValueError('直接动作试跑输出不符合 [1, 30, 7] 约定')
        modelscope = config['modelscope']
        publisher = ['exec', str(args.publisher_python.resolve()), '-u', str(source / 'upload_results.py'),
                     '--config', str(config_path), '--repo-id', modelscope['repo_id'],
                     '--token-file', str(Path(modelscope['token_file']).expanduser())]
        command = shlex.join(publisher) + ' >> ' + shlex.quote(str(root / 'logs/modelscope.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', name + '_publish', '-c', str(repo), command], check=True)
        # 正式训练重新从同一份 π0.5 base 初始化。
        execute('policy', ['--mode', 'formal', '--output', str(root / 'policy'), '--steps', '5000',
                           '--save-interval', '2000', '--save-steps', '2000', '4000', '5000',
                           '--status-interval', '100'])
        write_json(runtime / 'policy.complete.json', dict(state='completed', steps=5000))
        status('completed', 'policy', steps=5000)
    except Exception as error:
        status('failed', 'controller', error=str(error))
        raise


if __name__ == '__main__':
    main()
