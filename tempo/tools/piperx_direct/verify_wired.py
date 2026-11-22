"""通过 PiperX 正式客户端入口测试模型管理与录制反馈，不访问相机和 CAN。"""
import argparse
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np

from tempo_client.cli import load_config, replay_frames
from tempo_client.control import InferenceRun
from tempo_client.session import SessionWorker
from tempo_client.settings import select_runtime, service_model_id
from tempo_client.transport import Client

modes = ('tempo', 'temporal_ensemble', 'rtc', 'dynamicvla', 'vlash')


class RecordedSource:
    def __init__(self, frames, fps):
        self.frames = frames
        self.fps = fps
        self.origin_wall = time.time()
        self.origin_mono = time.monotonic()
        self.next_frame = 0
        self.stop = threading.Event()

    def take(self, max_frames):
        while not self.stop.is_set():
            last = int((time.monotonic() - self.origin_mono) * self.fps)
            if last >= self.next_frame:
                break
            self.stop.wait(.001)
        if self.stop.is_set():
            raise InterruptedError('回放已停止')
        if last + 1 - self.next_frame > max_frames:
            raise RuntimeError('反馈积压超过限制')
        batch = [{**self.frames[i % len(self.frames)], 'timestamp': self.origin_wall + i / self.fps}
                 for i in range(self.next_frame, last + 1)]
        self.next_frame = last + 1
        return batch


class SimulatedRobot:
    sent_commands = 0

    def __init__(self, state):
        self.state = state

    def check_status(self):
        pass

    def read_state(self):
        return self.state.copy()

    def check_target(self, action):
        assert np.isfinite(action).all()


def verify_realtime(cfg, output, duration):
    frames = replay_frames('client/logs/feedback_replay.npz')
    reports = {}
    for mode in modes:
        selected = select_runtime(cfg, mode=mode)
        directory = output / 'realtime' / mode
        directory.mkdir(parents=True, exist_ok=True)
        summary = {'log_dir': str(directory), 'batches': 0, 'feedback_frames': 0}
        log = directory / 'predictions.jsonl'
        with Client(selected) as client:
            client.validate_mode(mode, selected['task']['prompt'])
            client.reserve()
            source = RecordedSource(frames, selected['model']['fps'])
            robot = SimulatedRobot(frames[0]['state'])
            with log.open('w') as stream:
                run = InferenceRun(selected, client, source, robot, threading.Event(),
                                   queue.SimpleQueue(), stream, summary, False)
                try:
                    run.run(duration)
                    summary['outcome'] = 'completed'
                except Exception as error:
                    summary.update(outcome='stopped', error=str(error))
            summary['robot_commands'] = robot.sent_commands
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        if rows:
            summary['wired_rtt_p95_ms'] = float(np.quantile([row['round_trip_ms'] for row in rows], .95))
        print(json.dumps({'mode': mode, **summary}, ensure_ascii=False), flush=True)
        reports[mode] = summary
    return reports


def write_replay(replay):
    with np.load('client/logs/feedback_replay.npz', allow_pickle=False) as capture:
        count = len(capture['states'])
        indices = np.resize(np.arange(count), 104)
        # 借用录制图像和反馈值延长回放，覆盖历史窗口积满及滑动。
        np.savez_compressed(replay, timestamps=np.arange(len(indices)) / 30,
                            **{key: capture[key][indices] for key in ('states', 'head', 'left_wrist')})


def run(worker, operation, timeout, *, config_path, output, **kwargs):
    worker.start(operation, config_path, log_root=output, **kwargs)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kind, message = worker.events.get(timeout=.5)
        except queue.Empty:
            continue
        if kind == 'status':
            print(message, flush=True)
        if kind == 'done':
            worker.thread.join(2)
            if message['outcome'] != 'completed':
                raise RuntimeError(message)
            return message
    worker.stop()
    worker.thread.join(8)
    raise TimeoutError(f'客户端测试超时 {operation}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--realtime-seconds', type=int, default=15)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_id = service_model_id(cfg)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    replay = output / 'feedback_104_frames.npz'
    write_replay(replay)
    options = dict(config_path=args.config, output=output)
    admin = SessionWorker()
    before = admin._manage_service(cfg, 'status')
    if before['running_models']:
        raise RuntimeError(f'已有受管模型服务运行，取消本次验证 {before}')
    report = {'model_id': model_id, 'initial': before, 'modes': {}, 'robot_commands': 0}
    worker = SessionWorker()
    try:
        started = run(worker, 'service', 640, **options)
        metadata = json.loads((Path(started['log_dir']) / 'metadata.json').read_text())
        server = metadata.get('server_metadata', metadata.get('server', metadata))
        # 握手已经核对配置及权重路径，再记录直接动作与引导方式。
        direct = cfg['model'].get('predict_ut') is False
        assert server['predict_ut'] is (not direct) and server['step'] == cfg['model']['step']
        expected_rtc = 'direct_action_vjp_soft_prefix' if direct else 'decoded_action_vjp_soft_prefix'
        assert server['rtc_algorithm'] == expected_rtc
        assert server['sam2_skipped'] is True
        assert (server['action_history_source'], server['action_history_frames'],
                server['action_history_bucket_frames']) == ('state', 90, 9)
        report['metadata'] = server
        running = admin._manage_service(cfg, 'status')
        rows = running['running_models']
        if len(rows) != 1 or rows[0]['model_id'] != model_id or not rows[0]['ready']:
            raise RuntimeError(f'客户端启动后的模型状态不匹配 {running}')
        report['started'] = running
        for mode in modes:
            summary = run(worker, 'replay', 240, replay_path=replay, batch_size=8, inference_mode=mode, **options)
            records_path = Path(summary['log_dir']) / 'predictions.jsonl'
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            assert len(records) == 13 and sum(row['feedback_frames'] for row in records) == 104
            assert all(row['robot_commands'] == 0 for row in records)
            assert all(np.asarray(row['actions']).shape == (30, 7)
                       and np.isfinite(np.asarray(row['actions'])).all() for row in records)
            assert all(row['sampling_parameters'] == {'denoise_steps': 5, 'rtc_max_guidance': 5.0}
                       for row in records)
            if mode == 'vlash':
                assert all(row['action_start_offset_steps'] > 0 for row in records[1:])
            report['modes'][mode] = summary
            print(f'{mode} 已通过 104 帧回放', flush=True)
        sample = {'denoise_steps': 3, 'rtc_max_guidance': 2.5}
        report['runtime_sampling'] = run(worker, 'replay', 240, replay_path=replay, batch_size=8,
                                         inference_mode='rtc', sampling=sample, **options)
        records_path = Path(report['runtime_sampling']['log_dir']) / 'predictions.jsonl'
        assert all(json.loads(line)['sampling_parameters'] == sample for line in records_path.read_text().splitlines())
        report['verification'] = 'passed'
        if args.realtime_seconds > 0:
            report['realtime'] = verify_realtime(cfg, output, args.realtime_seconds)
    finally:
        try:
            current = admin._manage_service(cfg, 'status')
            selected = [row for row in current['running_models'] if row['model_id'] == model_id]
            if selected:
                expected = report.get('started', {}).get('running_models', selected)[0]['pid']
                if selected[0]['pid'] != expected:
                    raise RuntimeError('模型 PID 已变化，需核查服务归属')
                report['stop'] = admin._manage_service(cfg, 'stop')
            report['final'] = admin._manage_service(cfg, 'status')
            if report['final']['selected_running']:
                raise RuntimeError('本次测试模型仍在运行')
        finally:
            (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'verification': report['verification'], 'report': str(output / 'verification.json'),
                      'modes': list(report['modes']), 'robot_commands': 0, 'service_running': False},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
