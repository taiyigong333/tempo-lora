"""在4090等待固定检查点发布，从ModelScope下载并逐文件核验。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from zoneinfo import ZoneInfo

from modelscope_hub import HubApi

REPO = 'taiyigong3333/tempo'
NAME = 'ma800_clockwise_b234_hist90_direct_20261121'
PREFIX = f'ma800/{NAME}'
STEPS = (2000, 4000, 5000)
ASSETS = ('model.safetensors', 'model_config.json', 'norm_stats.json', 'config.json', 'split.json')
RUNTIME = Path(__file__).resolve().parent
TARGET = RUNTIME.parents[1] / 'tempo_deploy'
ROOT = TARGET / PREFIX
TARGET.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)

def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def fetch(endpoint, relative, sha256, size):
    client = HubApi(endpoint=endpoint)
    path = Path(client.download_file(REPO, repo_type='dataset', file_path=f'{PREFIX}/{relative}',
                                    local_dir=TARGET, revision='master', file_size=size,
                                    expected_sha256=sha256))
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != sha256 or path.stat().st_size != size:
        raise ValueError(f'下载文件校验失败 {relative}')
    return dict(path=relative, size=size, sha256=actual)

def verify_contract(path):
    config = json.loads((path / 'model_config.json').read_text())
    settings = json.loads((path / 'config.json').read_text())
    assert config['predict_ut'] is False and config['ut_decoder_path'] is None
    assert (config['action_horizon'], config['action_dim'], config['action_history_steps']) == (30, 32, 10)
    assert settings['direct_actions'] and settings['history_frames'] == 90 and settings['history_source'] == 'state'
    with (path / 'model.safetensors').open('rb') as stream:
        header_bytes, = struct.unpack('<Q', stream.read(8))
        assert 0 < header_bytes < 100_000_000
        header = json.loads(stream.read(header_bytes))
    assert header['action_out_proj.weight']['shape'][0] == 32
    assert header['action_in_proj.weight']['shape'][1] == 32
    return len(header) - int('__metadata__' in header)

def receive_step(endpoint, remote, step):
    checkpoint = f'policy/step_{step:08d}'
    manifest_key = f'{checkpoint}/manifest.json'
    item = remote.get(f'{PREFIX}/{manifest_key}')
    if item is None or not item.sha256:
        return None
    manifest = fetch(endpoint, manifest_key, item.sha256, item.size)
    path = ROOT / checkpoint
    release = json.loads((path / 'manifest.json').read_text())
    assert release['experiment'] == NAME and release['path_in_repo'] == f'{PREFIX}/{checkpoint}'
    rows = {row['path']: row for row in release['files']}
    for asset in ASSETS:
        remote_asset = remote.get(f'{PREFIX}/{checkpoint}/{asset}')
        if remote_asset is None or remote_asset.size != rows[asset]['size'] or remote_asset.sha256 != rows[asset]['sha256']:
            return None

    def download(asset):
        row = rows[asset]
        return fetch(endpoint, f'{checkpoint}/{asset}', row['sha256'], row['size'])

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, ASSETS))
    record = dict(step=step, verified_at=timestamp(), tensors=verify_contract(path), files=[manifest, *files])
    write_json(path / 'download_verification.json', record)
    return record

def main():
    # 4090 的默认代理会让魔塔 API 超时，沿用已验证的国内直连。
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    completed = {}
    status_path = RUNTIME / 'latest_status.json'
    while len(completed) < len(STEPS):
        error = None
        try:
            endpoint = HubApi().resolve_endpoint_for_read(REPO, repo_type='dataset')
            client = HubApi(endpoint=endpoint)
            remote = {row.path.lstrip('/'): row for row in client.list_repo_files(REPO, 'dataset', revision='master')}
            for step in STEPS:
                if step in completed:
                    continue
                received = receive_step(endpoint, remote, step)
                if received is not None:
                    completed[step] = received
                    print(json.dumps(received), flush=True)
        except Exception as exc:
            # 只记录异常类型，避免将短期下载链接写入日志。
            error = type(exc).__name__
            print(f'{timestamp()} 下载将重试 {error}', flush=True)
        pending = [step for step in STEPS if step not in completed]
        status = dict(state='waiting' if pending else 'completed', updated_at=timestamp(),
                      completed_steps=list(completed), pending_steps=pending, target=str(ROOT),
                      last_error=error)
        write_json(status_path, status)
        if pending:
            time.sleep(120)
    write_json(ROOT / 'download_verification.json', dict(repository=REPO, path_in_repo=PREFIX,
                                                        checkpoints=list(completed.values())))

if __name__ == '__main__':
    main()
