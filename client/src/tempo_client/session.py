"""GUI后台任务只通过事件队列返回结果，Tk主线程负责全部界面更新。"""

from datetime import datetime
import json
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
import traceback

from tempo_client.cli import DEFAULT_CONFIG, load_config, replay_frames, report
from tempo_client.live import LiveFeedback
from tempo_client.transport import Client


class SessionWorker:
    def __init__(self):
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.thread = None
        self.client = None
        self.source = None

    def start(self, mode, config_path, *, replay_path=None, duration=10, batch_size=4, log_root=None):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("请等待当前任务结束")
        self.cancel.clear()
        self.thread = threading.Thread(target=self._run, args=(mode, config_path, replay_path,
            duration, batch_size, log_root), daemon=True, name="tempo-gui-session")
        self.thread.start()

    def stop(self):
        self.cancel.set()
        if self.source is not None:
            self.source.stop.set()
        if self.client is not None:
            self.client.abort()

    def _check_cancel(self):
        if self.cancel.is_set():
            raise InterruptedError("任务已停止")

    def _connect(self, cfg):
        self._check_cancel()
        self.client = Client(cfg)
        self._check_cancel()
        self.events.put(("metadata", {"local_ip": self.client.local_ip, **self.client.metadata}))

    def _start_service(self, cfg):
        service = cfg["service"]
        project = shlex.quote(service["project_dir"])
        session = shlex.quote(service["tmux_session"])
        launch = shlex.quote("exec bash trained_model/scripts/serve_wired.sh >> trained_model/logs/serve_wired.log 2>&1")
        script = (f"set -eu\ncd {project}\n"
                  f"if ! tmux has-session -t ={session} 2>/dev/null; then\n"
                  f"  tmux new-session -d -s {session} -c {project} {launch}\nfi\n")
        self.events.put(("status", "正在通过有线SSH检查并启动模型服务"))
        # SSH仅管理Tempo自己的tmux会话，模型加载在远端后台继续。
        with subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                               service["ssh_host"], "bash -s"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            process.stdin.write(script)
            process.stdin.close()
            deadline = time.monotonic() + 20
            while process.poll() is None:
                if self.cancel.wait(.1) or time.monotonic() > deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    self._check_cancel()
                    raise TimeoutError("远端服务管理命令超时")
            message = process.stdout.read()
            if process.returncode:
                raise RuntimeError("远端服务启动失败 " + message.strip())
        deadline = time.monotonic() + service["startup_timeout_s"]
        self.events.put(("status", "等待模型服务就绪"))
        while time.monotonic() < deadline:
            self._check_cancel()
            try:
                self._connect(cfg)
                return
            except OSError:
                self.cancel.wait(2)
        raise TimeoutError("模型服务启动超过配置时限，请查看4090服务日志")

    def _predict(self, frames, stream, summary):
        self._check_cancel()
        self.events.put(("frame", frames[-1]))
        result = self.client.infer(frames)
        self._check_cancel()
        record = report(result, frames)
        record["state"] = frames[-1]["state"].tolist()
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        summary["batches"] += 1
        summary["feedback_frames"] += len(frames)
        self.events.put(("prediction", record))

    def _run(self, mode, config_path, replay_path, duration, batch_size, log_root):
        log_root = Path(log_root or DEFAULT_CONFIG.parents[1] / "logs/gui")
        directory = log_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        summary = {"mode": mode, "batches": 0, "feedback_frames": 0, "robot_commands": 0,
                   "outcome": "error", "log_dir": str(directory)}
        try:
            directory.mkdir(parents=True)
            self.events.put(("log_dir", str(directory)))
            cfg = load_config(config_path)
            (directory / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
            self.events.put(("status", "正在连接有线推理服务"))
            if mode == "service":
                self._start_service(cfg)
            else:
                self._connect(cfg)
            (directory / "metadata.json").write_text(json.dumps(self.client.metadata, ensure_ascii=False, indent=2) + "\n")
            with (directory / "predictions.jsonl").open("w") as stream:
                if mode == "replay":
                    self.events.put(("status", "正在回放录制反馈"))
                    frames = replay_frames(replay_path)
                    for begin in range(0, len(frames), batch_size):
                        self._predict(frames[begin:begin + batch_size], stream, summary)
                elif mode == "live":
                    self.events.put(("status", "正在启动双相机与CAN只读采样"))
                    self.source = LiveFeedback(cfg)
                    self._check_cancel()
                    with self.source:
                        self.events.put(("status", "现场只读推理中"))
                        until = time.monotonic() + duration
                        while time.monotonic() < until:
                            self._check_cancel()
                            self._predict(self.source.take(), stream, summary)
                elif mode not in ("probe", "service"):
                    raise ValueError("未知的GUI任务")
            self._check_cancel()
            summary["outcome"] = "completed"
        except Exception as error:
            summary["outcome"] = "cancelled" if self.cancel.is_set() else "error"
            if not self.cancel.is_set():
                summary["error"] = str(error)
                if directory.is_dir():
                    (directory / "error.log").write_text(traceback.format_exc())
        finally:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception as error:
                    summary["close_error"] = str(error)
            self.client = self.source = None
            if directory.is_dir():
                (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            self.events.put(("done", summary))
