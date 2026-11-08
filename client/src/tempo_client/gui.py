"""PiperX现场图形界面，所有网络与硬件任务在后台运行。"""

import argparse
from datetime import datetime
import fcntl
from pathlib import Path
import queue
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from tempo_client.cli import DEFAULT_CONFIG, load_config
from tempo_client.session import SessionWorker
from tempo_client.settings import MODE_LABELS, select_runtime


class TempoApp:
    def __init__(self, root, config_path=DEFAULT_CONFIG):
        self.root = root
        self.worker = SessionWorker()
        self.busy = False
        self.closing = False
        self.closed = False
        self.log_dir = DEFAULT_CONFIG.parents[1] / "logs/gui"
        self.config_path = tk.StringVar(value=str(Path(config_path).resolve()))
        self.replay_path = tk.StringVar(value=str(Path(config_path).resolve().parents[1] / "logs/feedback_replay.npz"))
        self.duration = tk.StringVar(value="0")
        self.batch_size = tk.StringVar(value="4")
        self.task_label = tk.StringVar()
        self.mode_label = tk.StringVar(value=MODE_LABELS["tempo"])
        self.execute = tk.BooleanVar(value=False)
        self.home_before = tk.BooleanVar(value=True)
        self.execution_label = tk.StringVar(value="只读推理")
        self.execute_steps = tk.StringVar(value="1")
        self.ensemble_decay = tk.StringVar(value="0.05")
        self.rtc_min_steps = tk.StringVar(value="1")
        self.selection_info = tk.StringVar()
        self.violation = tk.StringVar(value="目标检查信息将在推理时显示")
        self.task_ids = []
        self.active_mode = None
        self.status = tk.StringVar(value="就绪，先检查模型连接")
        self.camera_status = tk.StringVar(value="双相机尚未开启；首次开启后持续采集")
        self.endpoint = tk.StringVar(value="有线连接待检查")
        self.model = tk.StringVar(value="模型身份待确认")
        self.latency = tk.StringVar(value="— ms")
        self.progress = tk.StringVar(value="0 批次 · 0 反馈帧 · 0 机器人指令")
        self.images = {}
        self.action_buttons = []
        self.root.title("Tempo · PiperX 有线推理")
        self.root.geometry("1280x980")
        self.root.minsize(1180, 900)
        self.root.configure(bg="#eef2f6")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", foreground="#172b46", font=("sans-serif", 11))
        style.configure("Title.TLabel", font=("sans-serif", 23, "bold"))
        style.configure("Muted.TLabel", foreground="#586b82", font=("sans-serif", 10))
        style.configure("TButton", padding=(12, 8), font=("sans-serif", 10))
        style.configure("Start.TButton", background="#1e7254", foreground="white", font=("sans-serif", 12, "bold"), padding=(20, 10))
        style.configure("Treeview", rowheight=25, font=("sans-serif", 10))
        style.configure("Treeview.Heading", font=("sans-serif", 10, "bold"))
        self._build()
        self._load_choices()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F5>", lambda event: self.start("live"))
        self.root.bind("<Escape>", lambda event: self.stop())
        self.root.after(50, self._drain)

    def _button(self, parent, text, command, **grid):
        button = ttk.Button(parent, text=text, command=command)
        button.grid(**grid)
        self.action_buttons.append(button)
        return button

    def _build(self):
        page = ttk.Frame(self.root, padding=22)
        page.pack(fill="both", expand=True)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(5, weight=1)
        title = ttk.Frame(page)
        title.grid(row=0, column=0, sticky="ew")
        ttk.Label(title, text="Tempo / PiperX", style="Title.TLabel").pack(side="left")
        ttk.Label(title, textvariable=self.execution_label, foreground="#167554").pack(side="right")
        ttk.Label(page, text="有线模型服务 · 双相机反馈 · 六关节与夹爪预测", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 14))

        cards = ttk.Frame(page)
        cards.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        for column, (label, variable) in enumerate((("网络", self.endpoint), ("模型", self.model), ("最近往返", self.latency))):
            cards.columnconfigure(column, weight=1)
            card = ttk.LabelFrame(cards, text=label, padding=12)
            card.grid(row=0, column=column, sticky="ew", padx=(0, 10 if column < 2 else 0))
            ttk.Label(card, textvariable=variable).pack(anchor="w")

        controls = ttk.LabelFrame(page, text="任务、推理方式与开始", padding=12)
        controls.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="配置文件").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.config_entry = ttk.Entry(controls, textvariable=self.config_path)
        self.config_entry.grid(row=0, column=1, columnspan=5, sticky="ew")
        self._button(controls, "选择配置", self._select_config, row=0, column=6, padx=(10, 0))
        selection = ttk.Frame(controls)
        selection.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(10, 6))
        ttk.Label(selection, text="任务").pack(side="left", padx=(0, 8))
        self.task_box = ttk.Combobox(selection, textvariable=self.task_label, state="readonly", width=24)
        self.task_box.pack(side="left", padx=(0, 16))
        self.task_box.bind("<<ComboboxSelected>>", self._show_selection)
        ttk.Label(selection, text="推理方式").pack(side="left", padx=(0, 8))
        self.mode_box = ttk.Combobox(selection, textvariable=self.mode_label, state="readonly", width=25,
                                     values=list(MODE_LABELS.values()))
        self.mode_box.pack(side="left", padx=(0, 16))
        self.mode_box.bind("<<ComboboxSelected>>", self._show_selection)
        self.execute_check = ttk.Checkbutton(selection, text="允许实机运动", variable=self.execute, command=self._execution_changed)
        self.execute_check.pack(side="left", padx=(0, 12))
        self.home_check = ttk.Checkbutton(selection, text="开始前低速归位", variable=self.home_before, state="disabled")
        self.home_check.pack(side="left")
        ttk.Label(controls, textvariable=self.selection_info, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=7, sticky="w", pady=(0, 8))
        toolbar = ttk.Frame(controls)
        toolbar.grid(row=3, column=0, columnspan=7, sticky="w", pady=(0, 10))
        self.start_button = ttk.Button(toolbar, text="开始推理 F5", style="Start.TButton", command=lambda: self.start("live"))
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        self.action_buttons.append(self.start_button)
        self.stop_button = ttk.Button(toolbar, text="停止并保持 Esc", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 12))
        for index, (text, mode) in enumerate((("启动模型服务", "service"), ("检查连接", "probe"),
                                             ("录制数据回放", "replay"), ("开启相机", "camera"), ("释放相机", "release"))):
            self._button(toolbar, text, lambda mode=mode: self.start(mode), row=0, column=index+2, padx=(0, 6))
        ttk.Button(toolbar, text="打开日志", command=self.open_logs).grid(row=0, column=7)
        ttk.Label(controls, text="回放数据").grid(row=4, column=0, sticky="w", padx=(0, 10))
        self.replay_entry = ttk.Entry(controls, textvariable=self.replay_path)
        self.replay_entry.grid(row=4, column=1, columnspan=5, sticky="ew")
        self._button(controls, "选择NPZ", self._select_replay, row=4, column=6, padx=(10, 0))
        options = ttk.Frame(controls)
        options.grid(row=5, column=0, columnspan=7, sticky="w", pady=(10, 0))
        entries = []
        for label, variable, low, high, width in (
                ("运行秒数（0持续）", self.duration, 0, 3600, 6),
                ("原生每块步数", self.execute_steps, 1, 16, 4),
                ("TE衰减", self.ensemble_decay, 0, 10, 5),
                ("RTC最少推进", self.rtc_min_steps, 1, 15, 4),
                ("回放批帧数", self.batch_size, 1, 60, 4)):
            ttk.Label(options, text=label).pack(side="left")
            entry = ttk.Spinbox(options, from_=low, to=high, width=width, textvariable=variable)
            entry.pack(side="left", padx=(6, 14))
            entries.append(entry)
        self.duration_entry, self.steps_entry, self.decay_entry, self.rtc_entry, self.batch_entry = entries
        ttk.Label(controls, textvariable=self.camera_status, style="Muted.TLabel").grid(
            row=6, column=0, columnspan=7, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.violation, style="Muted.TLabel").grid(
            row=7, column=0, columnspan=7, sticky="w", pady=(4, 0))

        views = ttk.Frame(page)
        views.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        self.preview_labels = {}
        for column, (name, title) in enumerate((("head", "前视 / head"), ("left_wrist", "腕视 / left_wrist"))):
            views.columnconfigure(column, weight=1)
            box = ttk.LabelFrame(views, text=title, padding=8)
            box.grid(row=0, column=column, sticky="nsew", padx=(0, 10))
            label = tk.Label(box, bg="#172b46", fg="#d1dceb", text="等待推理反馈图像", height=12)
            label.pack(fill="both", expand=True)
            self.preview_labels[name] = label
        state_box = ttk.LabelFrame(views, text="当前反馈 → 调度目标", padding=8)
        state_box.grid(row=0, column=2, sticky="nsew")
        self.state_table = ttk.Treeview(state_box, columns=("name", "state", "target"), show="headings", height=7)
        for key, title, width in (("name", "维度", 72), ("state", "实测", 82), ("target", "预测", 82)):
            self.state_table.heading(key, text=title)
            self.state_table.column(key, width=width, anchor="e" if key != "name" else "w", stretch=False)
        self.state_table.pack(fill="both", expand=True)
        for index in range(7):
            name = f"J{index + 1} rad" if index < 6 else "夹爪 m"
            self.state_table.insert("", "end", iid=str(index), values=(name, "—", "—"))

        tabs = ttk.Notebook(page)
        tabs.grid(row=5, column=0, sticky="nsew")
        log_box = ttk.Frame(tabs)
        tabs.add(log_box, text="事件日志")
        self.log = tk.Text(log_box, height=6, wrap="word", bg="white", fg="#24354b", relief="flat", font=("sans-serif", 10))
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")
        predictions = ttk.Frame(tabs)
        tabs.add(predictions, text="完整16步预测")
        columns = ["step", *[f"j{i}" for i in range(1, 7)], "gripper"]
        self.prediction_table = ttk.Treeview(predictions, columns=columns, show="headings", height=6)
        for key in columns:
            self.prediction_table.heading(key, text="步" if key == "step" else "夹爪 m" if key == "gripper" else key.upper() + " rad")
            self.prediction_table.column(key, width=110, anchor="e")
        self.prediction_table.pack(fill="both", expand=True)
        footer = ttk.Frame(page)
        footer.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        ttk.Label(footer, textvariable=self.progress, style="Muted.TLabel").pack(side="right")
        self.inputs = [self.config_entry, self.replay_entry, *entries, self.execute_check]
        self.choice_inputs = [self.task_box, self.mode_box]

    def _select_config(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择Tempo客户端配置", filetypes=[("YAML配置", "*.yaml *.yml")])
        if path:
            self.config_path.set(path)
            self._load_choices()

    def _load_choices(self):
        try:
            cfg = load_config(self.config_path.get())
            tasks = cfg["task_catalog"]["tasks"]
            self.task_ids = [task["id"] for task in tasks]
            self.task_box.configure(values=[task["label"] for task in tasks])
            self.task_box.current(self.task_ids.index(cfg["task_catalog"]["default_task"]))
            self.mode_label.set(MODE_LABELS[cfg["inference"]["mode"]])
            self.execute_steps.set(str(cfg["inference"]["execute_steps"]))
            self.ensemble_decay.set(str(cfg["inference"]["ensemble_decay"]))
            self.rtc_min_steps.set(str(cfg["inference"]["rtc_min_execute_steps"]))
            self._show_selection()
        except Exception as error:
            self.status.set("配置加载失败 " + str(error))
            self._append(self.status.get())

    def _selected_task(self):
        index = self.task_box.current()
        if index == "" or not 0 <= int(index) < len(self.task_ids):
            raise ValueError("请先选择任务")
        return self.task_ids[int(index)]

    def _selected_mode(self):
        for mode, label in MODE_LABELS.items():
            if self.mode_label.get() == label:
                return mode
        raise ValueError("请先选择推理方式")

    def _show_selection(self, event=None):
        try:
            cfg = load_config(self.config_path.get())
            task = next(task for task in cfg["task_catalog"]["tasks"] if task["id"] == self._selected_task())
            mode = self._selected_mode()
            detail = "RTC保持10次去噪，延迟超限停止" if mode == "rtc" else "10次去噪 · 16×7绝对关节与夹爪目标"
            self.selection_info.set(task["prompt"] + "  ·  " + detail)
        except Exception:
            self.selection_info.set("请选择配置中的有效任务和推理方式")

    def _execution_changed(self):
        self.execution_label.set("实机执行待现场确认" if self.execute.get() else "只读推理")
        self.home_check.configure(state="normal" if self.execute.get() and not self.busy else "disabled")

    def _select_replay(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择录制反馈", filetypes=[("NumPy反馈", "*.npz")])
        if path:
            self.replay_path.set(path)

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", f"{datetime.now():%H:%M:%S}  {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy):
        self.busy = busy
        for widget in [*self.action_buttons, *self.inputs]:
            widget.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        for widget in self.choice_inputs:
            widget.configure(state="disabled" if busy else "readonly")
        self._execution_changed()

    def start(self, mode):
        if self.busy:
            return
        try:
            duration, batch = 10, 4
            runtime = {}
            if mode != "release":
                cfg = load_config(self.config_path.get())
                duration, batch = float(self.duration.get()), int(self.batch_size.get())
                if not np.isfinite(duration) or not 0 <= duration <= 3600 or not 1 <= batch <= 60:
                    raise ValueError("时长需为0至3600秒，0表示持续运行；每批反馈需为1至60帧")
                if mode == "replay" and not Path(self.replay_path.get()).is_file():
                    raise ValueError("请选择有效的NPZ录制反馈文件")
                self.endpoint.set(f"{cfg['connection']['local_ip']} → {cfg['connection']['host']}:{cfg['connection']['port']}")
                if mode in ("live", "replay"):
                    parameters = {"execute_steps": int(self.execute_steps.get()),
                                  "ensemble_decay": float(self.ensemble_decay.get()),
                                  "rtc_min_execute_steps": int(self.rtc_min_steps.get())}
                    cfg["inference"].update(parameters)
                    cfg = select_runtime(cfg, self._selected_task(), self._selected_mode())
                    execute = mode == "live" and self.execute.get()
                    if execute:
                        home = ("开始前按配置关节角低速归位，速度10%" if self.home_before.get()
                                else "从现场当前关节位置开始")
                        runtime_text = "持续运行，使用停止并保持按钮结束" if duration == 0 else f"运行{duration:g}秒后停止保持"
                        text = (f"任务 {cfg['task']['label']}\n模式 {MODE_LABELS[cfg['inference']['mode']]}\n"
                                f"{home}\n策略速度 {cfg['safety']['speed_percent']}%\n{runtime_text}\n\n"
                                "确认现场工作空间与机器人状态后，启动真实机器人运动？")
                        if not messagebox.askyesno("本轮实机运动确认", text, parent=self.root):
                            return
                    runtime = {"task_id": cfg["task"]["id"], "inference_mode": cfg["inference"]["mode"],
                               "execute": execute, "motion_confirmed": execute,
                               "home_before": self.home_before.get(), "parameters": parameters}
            self.worker.start(mode, self.config_path.get(), replay_path=self.replay_path.get(),
                              duration=duration, batch_size=batch, **runtime)
        except Exception as error:
            messagebox.showerror("无法启动任务", str(error), parent=self.root)
            return
        self._set_busy(True)
        if mode in ("probe", "service", "replay", "live"):
            self.model.set("模型身份待确认")
        self.latency.set("— ms")
        self.progress.set("0 批次 · 0 反馈帧 · 0 机器人指令")
        self.batch_count = self.frame_count = 0
        self.target_count = self.command_count = 0
        self.active_mode = mode
        self.violation.set("等待模型目标与保护检查")
        if mode == "replay":
            self._clear_previews("等待回放图像")
        self.status.set("正在启动任务")
        self._append(f"开始任务 {mode}")
        if runtime:
            self._append(f"任务 {self.task_label.get()} · {self.mode_label.get()} · " + ("现场确认实机执行" if runtime["execute"] else "只读推理"))

    def stop(self):
        if self.busy:
            self.status.set("正在停止当前任务")
            self.worker.stop()
            self.stop_button.configure(state="disabled")

    def open_logs(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(self.log_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _clear_previews(self, text):
        self.images.clear()
        for label in self.preview_labels.values():
            label.configure(image="", text=text, height=12)

    def _handle(self, kind, value):
        if kind == "status":
            self.status.set(value)
            self._append(value)
        elif kind == "camera_status":
            self.camera_status.set(value)
            self._append(value)
            if value == "相机已释放":
                self._clear_previews(value)
        elif kind == "metadata":
            self.model.set(f"TEMPO · {value['step']} 步 · {value['action_horizon']}×{value['native_action_dim']}")
            self._append(f"有线握手通过，模型 {value['checkpoint']}")
        elif kind == "log_dir":
            self.log_dir = Path(value)
        elif kind == "frame":
            for name, rgb in value["images"].items():
                image = Image.fromarray(rgb).resize((210, 210), Image.Resampling.BILINEAR)
                self.images[name] = ImageTk.PhotoImage(image, master=self.root)
                self.preview_labels[name].configure(image=self.images[name], text="", height=210)
        elif kind == "prediction":
            self.batch_count += 1
            self.frame_count += value["feedback_frames"]
            self.latency.set(f"{value['round_trip_ms']:.1f} ms")
            self.progress.set(f"{self.batch_count} 批次 · {self.frame_count} 反馈帧 · {getattr(self, 'target_count', 0)} 目标 · {value.get('robot_commands', 0)} 运动指令")
            if self.active_mode != "live":
                for index in range(7):
                    name = f"J{index + 1} rad" if index < 6 else "夹爪 m"
                    self.state_table.item(str(index), values=(name, f"{value['state'][index]:.4f}", f"{value['actions'][0][index]:.4f}"))
            self.prediction_table.delete(*self.prediction_table.get_children())
            for index, target in enumerate(value["actions"]):
                self.prediction_table.insert("", "end", values=(index + 1, *[f"{x:.5f}" for x in target]))
            self._append(f"批次 {self.batch_count}，{value['feedback_frames']} 帧反馈，往返 {value['round_trip_ms']:.1f} ms")
        elif kind == "control":
            self.target_count += 1
            self.command_count = value["robot_commands"]
            self.progress.set(f"{self.batch_count} 批次 · {self.frame_count} 反馈帧 · {self.target_count} 目标 · {self.command_count} 运动指令")
            for index in range(7):
                name = f"J{index + 1} rad" if index < 6 else "夹爪 m"
                self.state_table.item(str(index), values=(name, f"{value['state'][index]:.4f}", f"{value['action'][index]:.4f}"))
            self.violation.set(("只读目标检查 " + value["violation"]) if value["violation"] else
                               f"目标检查通过 · 观测帧龄 {value['age_s']:.3f} 秒 · " + ("实机下发" if value["executed"] else "只读规划"))
        elif kind == "done":
            self._set_busy(False)
            status = {"completed": "任务完成", "cancelled": "任务已停止", "error": "任务失败"}[value["outcome"]]
            self.status.set(status)
            self._append(status + (" · " + value["error"] if "error" in value else ""))
            self._append(f"日志保存于 {value['log_dir']}")
            if self.active_mode == "live":
                self.progress.set(f"{value.get('batches', 0)} 批次 · {value.get('feedback_frames', 0)} 反馈帧 · {value.get('planned_targets', 0)} 目标 · {value.get('robot_commands', 0)} 运动指令")
            if value.get("robot_close_error"):
                self._append("机器人停止清理异常 " + value["robot_close_error"])
            if self.closing:
                if value.get("mode") != "release":
                    self.root.after(20, self._release_for_exit)
                elif value["outcome"] == "completed":
                    self._destroy()
                else:
                    self.closing = False
                    self.status.set("释放相机失败，请检查日志后重试退出")

    def _drain(self):
        try:
            while True:
                kind, value = self.worker.events.get_nowait()
                self._handle(kind, value)
                if self.closed:
                    return
        except queue.Empty:
            pass
        if not self.closed:
            frame = self.worker.take_preview()
            if frame is not None and self.worker.preview_enabled.is_set():
                self._handle("frame", frame)
            self.root.after(50, self._drain)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.busy:
            self.stop()
        else:
            self._release_for_exit()

    def _release_for_exit(self):
        if self.worker.thread is not None and self.worker.thread.is_alive():
            self.root.after(20, self._release_for_exit)
            return
        self.worker.start("release", self.config_path.get())
        self._set_busy(True)
        self.status.set("正在关闭相机并退出")

    def _destroy(self):
        # Tk定时回调和图像归主线程清理，避免退出后回调或跨线程回收解释器。
        for timer in self.root.tk.splitlist(self.root.tk.call("after", "info")):
            self.root.after_cancel(timer)
        self.images.clear()
        self.closed = True
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Tempo PiperX图形客户端")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    root = tk.Tk()
    lock_path = DEFAULT_CONFIG.parents[1] / "logs/gui.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            # 桌面重复点击只保留一个GUI，避免多个窗口争用在线模型历史。
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            root.withdraw()
            messagebox.showinfo("Tempo已运行", "Tempo图形界面已经打开，请切换到现有窗口。", parent=root)
            root.destroy()
            return
        TempoApp(root, args.config)
        root.mainloop()


if __name__ == "__main__":
    main()
