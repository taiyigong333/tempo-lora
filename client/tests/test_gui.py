"""在真实Tk窗口核验控件状态与结果展示，使用合成反馈隔离现场硬件。"""

import os
import tkinter as tk
import unittest
from unittest.mock import Mock

import numpy as np

from tempo_client.gui import TempoApp


@unittest.skipUnless(os.environ.get("DISPLAY"), "需要X11显示环境")
class GuiTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = TempoApp(self.root)
        self.root.update()

    def tearDown(self):
        if not self.app.closed:
            self.app.worker.stop()
            if self.app.worker.thread is not None:
                self.app.worker.thread.join(5)
            self.app.worker.close_cameras()
            self.app.closed = True
            self.root.destroy()

    def test_starts_and_stops_without_blocking_ui(self):
        self.app.worker.start = Mock()
        self.app.worker.stop = Mock()
        self.app.start("probe")
        self.assertTrue(self.app.busy)
        self.assertEqual(str(self.app.stop_button["state"]), "normal")
        self.app.stop()
        self.app.worker.stop.assert_called_once()
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")
        self.app._handle("done", {"outcome": "cancelled", "log_dir": "/tmp/tempo-gui-test"})
        self.assertFalse(self.app.busy)

    def test_renders_metadata_images_and_native_actions(self):
        self.app.batch_count = self.app.frame_count = 0
        self.app._handle("metadata", {"step": 10000, "checkpoint": "step_00010000.pt",
                                     "action_horizon": 16, "native_action_dim": 7})
        self.app._handle("frame", {"images": {camera: np.zeros((224, 224, 3), np.uint8)
                                              for camera in ("head", "left_wrist")}})
        self.app._handle("prediction", {"feedback_frames": 4, "round_trip_ms": 280,
                                        "state": [0.] * 7, "actions": [[1.] * 7] * 16})
        self.root.update()
        self.assertIn("10000", self.app.model.get())
        self.assertEqual(len(self.app.images), 2)
        self.assertEqual(len(self.app.prediction_table.get_children()), 16)
        self.assertIn("4 反馈帧", self.app.progress.get())

    def test_close_during_task_waits_for_cancellation(self):
        self.app.busy = True
        self.app.worker.stop = Mock()
        self.app.worker.start = Mock()
        self.app.close()
        self.assertTrue(self.app.closing)
        self.assertFalse(self.app.closed)
        self.app.worker.stop.assert_called_once()
        self.app._handle("done", {"outcome": "cancelled", "log_dir": "/tmp/tempo-gui-test"})
        self.app._release_for_exit()
        self.assertFalse(self.app.closed)
        self.app._handle("done", {"mode": "release", "outcome": "completed", "log_dir": "/tmp/tempo-gui-test"})
        self.assertTrue(self.app.closed)

    def test_release_does_not_require_valid_configuration(self):
        self.app.config_path.set("/missing/tempo-config.yaml")
        self.app.worker.start = Mock()
        self.app.start("release")
        self.app.worker.start.assert_called_once()
        self.assertEqual(self.app.worker.start.call_args.args[0], "release")


if __name__ == "__main__":
    unittest.main()
