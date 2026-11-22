"""客户端统一模型管理，串行切换自有服务并保护正在使用的连接。"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=False)


def services(project):
    full = project.parent / 'tempo_full'
    shared = dict(root=full, session='tempo_full_inference', port=8002, module='openpi.tempo_full.inference')
    return {
        'lora10000': dict(root=project, session='tempo_lora_inference', port=8000,
                         module='scripts/piperx.py serve', config=None),
        'all3935000': dict(**shared, config='deployment_all393.json'),
        'clockwise4000': dict(**shared, config='deployment_step4000.json'),
        'clockwise5000': dict(**shared, config='deployment_rtc_paper.json'),
        'clockwise_hist90_5000': dict(**shared, config='deployment_hist90_step5000.json'),
        'clockwise_hist90_direct_2000': dict(**shared, config='deployment_hist90_direct_step2000.json'),
    }


def process(spec):
    if command('tmux', 'has-session', '-t', '=' + spec['session']).returncode:
        return None
    pane = command('tmux', 'display-message', '-p', '-t', spec['session'] + ':0.0', '#{pane_pid}')
    pid = int(pane.stdout.strip())
    args = command('ps', '-p', str(pid), '-o', 'args=').stdout.strip()
    if not args:
        return None
    if spec['module'] not in args or str(spec['root']) + '/' not in args:
        raise RuntimeError('服务会话中的进程身份不匹配，请检查 ' + spec['session'])
    return {'pid': pid, 'args': args}


def connections(spec):
    result = command('ss', '-Htn', 'state', 'established', f'( sport = :{spec["port"]} )')
    if result.returncode:
        raise RuntimeError('无法检查服务连接状态 ' + result.stderr)
    return len(result.stdout.splitlines())


def loaded_id(spec, active, catalog):
    if spec['port'] == 8000:
        return 'lora10000'
    # serve_wired保留默认--config，实际配置以最后一个参数为准。
    arguments = shlex.split(active['args'])
    configs = [arguments[i + 1] for i, value in enumerate(arguments[:-1]) if value == '--config']
    filename = Path(configs[-1]).name if configs else None
    for model_id, entry in catalog.items():
        if entry['port'] == 8002 and filename == entry['config']:
            return model_id
    if filename == 'deployment.json':
        return 'clockwise5000'
    return 'unknown'


def status(model_id, catalog):
    running = []
    seen = set()
    for spec in catalog.values():
        if spec['session'] in seen:
            continue
        seen.add(spec['session'])
        active = process(spec)
        if active:
            ready = bool(command('ss', '-Hltn', f'( sport = :{spec["port"]} )').stdout.strip())
            running.append({'model_id': loaded_id(spec, active, catalog), 'pid': active['pid'],
                            'ready': ready, 'connections': connections(spec), 'port': spec['port']})
    return {'selected_model_id': model_id, 'running_models': running,
            'selected_running': any(row['model_id'] == model_id for row in running)}


def validate_assets(spec):
    root = spec['root']
    if not (root/'trained_model/scripts/serve_wired.sh').is_file():
        raise FileNotFoundError('模型服务入口缺失')
    if spec['config']:
        path = root/'trained_model'/spec['config']
        cfg = json.loads(path.read_text())
        checkpoint = path.parent / cfg['checkpoint']
        required = [checkpoint/'model.safetensors', checkpoint/'model_config.json',
                    checkpoint/'norm_stats.json',
                    path.parent/cfg['tokenizer'], path.parent/cfg['sam2_checkpoint']]
        if not (path.parent/cfg['sam2_repo']).is_dir():
            raise FileNotFoundError('SAM2源码目录缺失')
    else:
        required = [root/'trained_model/checkpoints/pi05_piperx_tempo_ut_lora/formal_20261108_b16_g1/step_00010000.pt']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('模型资产缺失 ' + str(path))
    if spec['config']:
        # 切换前先校验模型与解码器配套关系，避免停掉可用服务才发现配置错误。
        model = json.loads((checkpoint/'model_config.json').read_text())
        decoder = cfg.get('decoder')
        if model.get('predict_ut') is True:
            if not decoder or not (root / 'trained_model' / decoder).is_file():
                raise FileNotFoundError('潜变量模型的解码器资产缺失')
        elif model.get('predict_ut') is not False or decoder is not None:
            raise ValueError('模型与解码器配置不匹配')


def stop(spec):
    active = process(spec)
    if active is None:
        return
    if connections(spec):
        raise RuntimeError('模型仍有客户端连接，请先停止推理再管理模型服务')
    # 精确核验PID与部署目录后直接终止，避免终端按键被tmux吞掉。
    os.kill(active['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if command('tmux', 'has-session', '-t', '=' + spec['session']).returncode:
            if command('ss', '-Hltn', f'( sport = :{spec["port"]} )').stdout.strip():
                raise RuntimeError('模型进程已退出，端口仍被其他进程占用')
            return
        time.sleep(.1)
    raise TimeoutError('模型服务退出超时，请检查远端日志')


def manage(action, model_id, project):
    catalog = services(project)
    if model_id not in catalog or action not in ('start', 'stop', 'status'):
        raise ValueError('不支持的模型或服务操作')
    runtime = project/'trained_model/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # 与既有全量启动脚本共享锁，兼容原入口的并发管理。
        for path in (runtime/'model_service.lock', project.parent/'tempo_full/trained_model/runtime/service_switch.lock'):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open('a'))
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('其他客户端正在管理模型，请稍后重试')
                    time.sleep(.05)
        before = status(model_id, catalog)
        spec = catalog[model_id]
        if action == 'start' and not before['selected_running']:
            validate_assets(spec)
            # 先核查全部连接再退出任何模型，避免检查中途造成半切换状态。
            if any(row['connections'] for row in before['running_models']):
                raise RuntimeError('模型仍有客户端连接，请先停止推理再切换模型')
            occupied = bool(command('ss', '-Hltn', f'( sport = :{spec["port"]} )').stdout.strip())
            managed_port = any(row['port'] == spec['port'] for row in before['running_models'])
            if occupied and not managed_port:
                raise RuntimeError('目标端口被其他进程占用')
            for row in before['running_models']:
                current = next(entry for entry in catalog.values() if entry['port'] == row['port'])
                stop(current)
            if command('ss', '-Hltn', f'( sport = :{spec["port"]} )').stdout.strip():
                raise RuntimeError('目标端口被其他进程占用')
            log = spec['root']/'trained_model/logs'/f'client_{model_id}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            arguments = ['bash', str(spec['root']/'trained_model/scripts/serve_wired.sh')]
            if spec['config'] is None:
                # 直接exec模型Python进程，tmux的PID可精确核验和正常终止。
                arguments = ['env', 'JAX_PLATFORMS=cpu', 'OMP_NUM_THREADS=8', 'MKL_NUM_THREADS=8',
                             'OPENPI_DATA_HOME=' + str(project/'trained_model/cache/openpi'),
                             'HF_HOME=' + str(project/'trained_model/cache/huggingface'),
                             'TORCHINDUCTOR_CACHE_DIR=' + str(project/'trained_model/cache/inductor_inference'),
                             'PYTHONUNBUFFERED=1', str(project/'tempo/.venv/bin/python'),
                             str(project/'tempo/scripts/piperx.py'), 'serve', '--checkpoint',
                             str(project/'trained_model/checkpoints/pi05_piperx_tempo_ut_lora/formal_20261108_b16_g1/step_00010000.pt'),
                             '--host', '192.168.50.2', '--port', '8000']
            if spec['config']:
                arguments += ['--config', str(spec['root']/'trained_model'/spec['config'])]
            launch = 'exec ' + shlex.join(arguments) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'
            result = command('tmux', 'new-session', '-d', '-s', spec['session'], '-c', str(spec['root']), launch)
            if result.returncode:
                raise RuntimeError('模型启动失败 ' + result.stderr)
            # tmux创建pane到exec存在短暂间隙，先等待进程身份固定再返回状态。
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    if process(spec):
                        break
                except RuntimeError:
                    pass
                time.sleep(.05)
        elif action == 'stop':
            stop(spec)
        return {'action': action, **status(model_id, catalog)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    root = Path(__file__).resolve().parents[2]
    project = root.parent/'tempo' if root.name == 'tempo_full' else root
    parser.add_argument('model_id', choices=services(project))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = root.parent/'tempo' if root.name == 'tempo_full' else root
    try:
        result = manage(args.action, args.model_id, project)
    except Exception as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
