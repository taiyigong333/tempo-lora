"""以30Hz重复录制帧，走真实控制调度与模型网络，全程使用模拟机器人。"""
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
from tempo_client.cli import load_config, replay_frames
from tempo_client.control import InferenceRun
from tempo_client.settings import select_runtime
from tempo_client.transport import Client

frames = replay_frames('client/logs/feedback_replay.npz')


class RecordedSource:
    def __init__(self):
        self.origin_wall = time.time()
        self.origin_mono = time.monotonic()
        self.next_frame = 0
        self.stop = threading.Event()

    def take(self, max_frames):
        while not self.stop.is_set():
            latest = int((time.monotonic() - self.origin_mono) * 30)
            if latest >= self.next_frame:
                break
            self.stop.wait(.001)
        if self.stop.is_set():
            raise InterruptedError('回放已停止')
        count = latest + 1 - self.next_frame
        if count > max_frames:
            raise RuntimeError(f'反馈积压{count}帧超过{max_frames}')
        batch = []
        for i in range(self.next_frame, latest+1):
            batch.append({**frames[i % len(frames)], 'timestamp': self.origin_wall+i/30})
        self.next_frame = latest+1
        return batch


class FakeRobot:
    sent_commands = 0

    def check_status(self):
        pass

    def read_state(self):
        return frames[0]['state'].copy()

    def check_target(self, action):
        assert np.isfinite(action).all()


reports = {}
for mode in ('rtc',):
    cfg = select_runtime(load_config('client/configs/piperx_full_rtc_paper.yaml'), mode=mode)
    directory = Path('client/logs/tempo_full/rtc_paper/realtime') / mode
    directory.mkdir(parents=True, exist_ok=True)
    summary = {'mode': mode, 'log_dir': str(directory), 'batches': 0, 'feedback_frames': 0}
    with Client(cfg) as client:
        summary['metadata'] = client.metadata
        client.reserve()
        with (directory/'predictions.jsonl').open('w') as stream:
            run = InferenceRun(cfg, client, RecordedSource(), FakeRobot(), threading.Event(),
                               queue.SimpleQueue(), stream, summary, False)
            try:
                run.run(60)
                summary['outcome'] = 'completed'
            except Exception as error:
                summary.update(outcome='stopped', error=str(error))
    records = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    timings = [r['round_trip_ms'] for r in records]
    if timings:
        summary.update(p50_ms=float(np.median(timings)), p95_ms=float(np.quantile(timings,.95)))
    reports[mode] = summary
    print(json.dumps(summary, ensure_ascii=False), flush=True)
Path('client/logs/tempo_full/rtc_paper/realtime_report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2)+'\n')
