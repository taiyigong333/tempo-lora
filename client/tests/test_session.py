"""后台会话回归覆盖完成、失败、停止和任务互斥。"""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from tempo_client.session import SessionWorker
from tempo_client.live import LiveFeedback


def frames():
    return [{"timestamp": index / 30, "state": np.zeros(7, np.float32),
             "images": {name: np.zeros((224, 224, 3), np.uint8) for name in ("head", "left_wrist")}}
            for index in range(3)]


class FakeClient:
    metadata = {"step": 10000, "checkpoint": "step_00010000.pt"}
    local_ip = "192.168.50.1"

    def __init__(self, cfg):
        self.closed = False

    def infer(self, batch):
        return {"actions": np.zeros((16, 7), np.float32), "round_trip_ms": 1}

    def close(self):
        self.closed = True

    def abort(self):
        pass


class BlockingClient(FakeClient):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.entered = threading.Event()
        self.stopped = threading.Event()

    def infer(self, batch):
        self.entered.set()
        if not self.stopped.wait(5):
            raise TimeoutError("测试未及时停止")
        raise OSError("连接已中断")

    def abort(self):
        self.stopped.set()


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker = SessionWorker()
        self.config_patch = patch("tempo_client.session.load_config", return_value={})
        self.config_patch.start()

    def tearDown(self):
        self.worker.stop()
        if self.worker.thread is not None:
            self.worker.thread.join(5)
        self.config_patch.stop()
        self.temporary.cleanup()

    def start(self, mode):
        self.worker.start(mode, self.root / "config.yaml", replay_path="sample.npz", batch_size=2,
                          log_root=self.root / "logs")

    def finish(self):
        self.worker.thread.join(5)
        self.assertFalse(self.worker.thread.is_alive())
        events = list(self.worker.events.queue)
        summary = next(value for kind, value in events if kind == "done")
        self.assertTrue((Path(summary["log_dir"]) / "summary.json").exists())
        return events, summary

    @patch("tempo_client.session.Client", FakeClient)
    def test_probe_records_model_identity(self):
        self.start("probe")
        events, summary = self.finish()
        self.assertEqual(summary["outcome"], "completed")
        self.assertEqual(summary["batches"], 0)
        self.assertTrue(any(kind == "metadata" for kind, _ in events))

    @patch("tempo_client.session.Client", FakeClient)
    @patch("tempo_client.session.replay_frames", side_effect=lambda path: frames())
    def test_replay_records_all_frames(self, _):
        self.start("replay")
        events, summary = self.finish()
        self.assertEqual(summary["outcome"], "completed")
        self.assertEqual(summary["feedback_frames"], 3)
        self.assertEqual(summary["batches"], 2)
        path = Path(summary["log_dir"]) / "predictions.jsonl"
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(predictions), 2)
        self.assertEqual(sum(x["robot_commands"] for x in predictions), 0)

    @patch("tempo_client.session.Client", side_effect=ValueError("错误模型"))
    def test_failed_connection_emits_done(self, _):
        self.start("probe")
        _, summary = self.finish()
        self.assertEqual(summary["outcome"], "error")
        self.assertIn("错误模型", summary["error"])

    @patch("tempo_client.session.replay_frames", side_effect=lambda path: frames())
    def test_stop_interrupts_rpc_and_rejects_parallel_task(self, _):
        client = BlockingClient({})
        with patch("tempo_client.session.Client", return_value=client):
            self.start("replay")
            self.assertTrue(client.entered.wait(3))
            with self.assertRaises(RuntimeError):
                self.start("probe")
            self.worker.stop()
            _, summary = self.finish()
        self.assertEqual(summary["outcome"], "cancelled")
        self.assertEqual(summary["batches"], 0)
        self.assertTrue(client.closed)

    def test_hardware_startup_reports_original_error(self):
        source = LiveFeedback({"live": {"max_pending_frames": 1, "cameras": {}, "startup_timeout_s": 1},
                               "model": {"fps": 30}})
        def fail_start(*args):
            source.error = RuntimeError("相机已被其他程序占用")
            source.stop.set()
        with patch.object(source, "_connect_arm"), patch.object(source, "_start", side_effect=fail_start):
            with self.assertRaisesRegex(RuntimeError, "相机已被其他程序占用"):
                source.__enter__()

    def test_hardware_startup_accepts_cancellation(self):
        source = LiveFeedback({"live": {"max_pending_frames": 1, "cameras": {}, "startup_timeout_s": 1},
                               "model": {"fps": 30}})
        source.stop.set()
        with patch.object(source, "_connect_arm"), patch.object(source, "_start"):
            with self.assertRaises(InterruptedError):
                source.__enter__()


if __name__ == "__main__":
    unittest.main()
