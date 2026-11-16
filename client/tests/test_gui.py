"""在真实Tk窗口核验控件状态与结果展示，使用合成反馈隔离现场硬件。"""

import os
import gc
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tempo_client.gui import TempoApp
from tempo_client.settings import MODE_LABELS


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
            self.app._destroy()
        self.app = self.root = None
        # 连续创建Tk解释器的测试必须在创建线程回收，后台任务只处理普通数据。
        gc.collect()

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

    def test_start_task_and_three_modes_are_visible(self):
        self.assertIn("开始推理", self.app.start_button["text"])
        self.assertEqual(tuple(self.app.mode_box["values"]), tuple(MODE_LABELS.values()))
        self.assertTrue(self.app.task_ids)
        self.assertEqual(str(self.app.task_box["state"]), "readonly")

    @patch("tempo_client.gui.messagebox.askyesno", return_value=False)
    def test_declined_motion_confirmation_does_not_start(self, confirm):
        self.app.worker.start = Mock()
        self.app.execute.set(True)
        self.app.start("live")
        confirm.assert_called_once()
        self.app.worker.start.assert_not_called()

    @patch("tempo_client.gui.messagebox.askyesno", return_value=True)
    def test_selected_rtc_and_motion_confirmation_reach_worker(self, confirm):
        self.app.worker.start = Mock()
        self.app.mode_label.set(MODE_LABELS["rtc"])
        self.app.execute.set(True)
        self.app.start("live")
        confirm.assert_called_once()
        options = self.app.worker.start.call_args.kwargs
        self.assertTrue(options["execute"] and options["motion_confirmed"])
        self.assertEqual(options["inference_mode"], "rtc")
        self.assertEqual(options["task_id"], "rotating_red_square")
        self.assertEqual(options["duration"], 0)
        self.assertEqual(str(self.app.mode_box["state"]), "disabled")

    @patch("tempo_client.gui.messagebox.showerror")
    def test_missing_task_blocks_start(self, error):
        self.app.worker.start = Mock()
        self.app.task_box.set("不存在的任务")
        self.app.start("live")
        error.assert_called_once()
        self.app.worker.start.assert_not_called()

    def test_model_selection_changes_checkpoint_and_resets_motion(self):
        selected = next(row for row in self.app.model_choices if row["id"] == "clockwise4000")
        self.app.execute.set(True)
        self.app.model_label.set(selected["label"])
        self.app._select_model()
        self.assertTrue(self.app.config_path.get().endswith("piperx_full_step4000.yaml"))
        self.assertEqual(self.app.denoise_steps.get(), "5")
        self.assertFalse(self.app.execute.get())
        self.assertEqual(int(float(self.app.steps_entry["to"])), 30)

    def test_sampling_and_delay_parameters_reach_background_worker(self):
        self.app.worker.start = Mock()
        self.app.denoise_steps.set("3")
        self.app.rtc_guidance.set("2.5")
        self.app.delay_method.set("window_max")
        self.app.delay_buffer.set("7")
        self.app.initial_delay.set("2")
        self.app.delay_margin.set("3")
        self.app.start("live")
        kwargs = self.app.worker.start.call_args.kwargs
        self.assertEqual(kwargs["sampling"], {"denoise_steps": 3, "rtc_max_guidance": 2.5})
        self.assertEqual(kwargs["parameters"]["rtc_delay_buffer_size"], 7)
        self.assertEqual(kwargs["parameters"]["rtc_initial_delay_steps"], 2)
        self.assertEqual(kwargs["parameters"]["rtc_delay_margin_steps"], 3)

    def test_service_stop_and_status_are_available_with_invalid_inference_duration(self):
        self.app.worker.start = Mock()
        self.app.duration.set("invalid")
        self.app.start("service_stop")
        self.assertEqual(self.app.worker.start.call_args.args[0], "service_stop")
        self.app._set_busy(False)
        self.app.start("service_status")
        self.assertEqual(self.app.worker.start.call_args.args[0], "service_status")

    def test_service_state_reports_actual_running_model(self):
        self.app._handle("service_state", {"running_models": [{"model_id": "clockwise4000", "ready": True}]})
        self.assertIn("step4000", self.app.model.get())
        self.app._handle("service_state", {"running_models": []})
        self.assertEqual(self.app.model.get(), "模型服务已停止")


if __name__ == "__main__":
    unittest.main()
