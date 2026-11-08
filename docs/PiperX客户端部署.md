# PiperX Tempo 客户端部署

## 目标与范围

2026-11-08 将轻量客户端部署到 PiperX 的 `/home/cv/gcj/project/baseline_fast/tempo-lora`，使用 `https://github.com/taiyigong333/tempo-lora.git` 管理同一项目。4090 的模型及依赖留在 `/data/gcj_data/fast/baseline/tempo`。

客户端提供中文图形界面与命令行入口，支持模型服务启动、握手校验、录制数据回放、硬件只读推理、双图预览、实测关节与动作块展示、停止任务及日志查看。实机运动执行待用户进一步指定。

## 有线网络

| 设备 | 网卡 | 地址 |
| --- | --- | --- |
| PiperX | enp0s31f6 | 192.168.50.1/24 |
| 4090 | enp4s0 | 192.168.50.2/24 |

两端已有静态有线连接。双向 ping 无丢包，初检平均约 0.2ms。客户端绑定本机 `192.168.50.1` 创建连接，并校验实际出口。模型服务绑定 `192.168.50.2:8000`。

## 图形界面启动

在 PiperX 桌面双击 **Tempo LoRA · PiperX**，或在应用菜单打开同名应用。也可运行以下入口。

```bash
cd /home/cv/gcj/project/baseline_fast/tempo-lora
./run_gui.sh
```

1. 点击“检查连接”，确认有线地址、10000步模型与16×7动作块。
2. 服务尚未启动时点击“启动模型服务”，界面通过现有有线SSH检查Tempo自己的tmux会话并等待模型就绪。
3. 点击“录制数据回放”可用部署样例查看双相机图像、实测关节和模型预测。默认每批4帧。
4. 现场释放其他程序占用的相机后，设置时长并点击“现场只读推理”。界面显示关节rad与夹爪m，输出保存至每轮日志。
5. 点击“停止当前任务”中断网络接收和本轮采样，下一轮使用新episode。关闭窗口会先结束当前任务。模型服务保留其独立远端会话。
6. 点击“打开日志”查看本轮config.json、metadata.json、predictions.jsonl与summary.json。失败时另存error.log。

图形界面以后台线程执行网络与采样，事件队列驱动Tk更新。桌面重复启动通过文件锁限制为一个GUI进程。应用菜单入口位于 `/home/cv/.local/share/applications/tempo-lora.desktop`，桌面入口位于 `/home/cv/桌面/Tempo-LoRA.desktop`，仓库内保留 `client/tempo-lora.desktop`。

GUI已在独立虚拟屏幕上完成真实有线模型联调，回放12帧、3批，双图与16步预测显示通过。模型推理期间停止和关闭窗口通过，客户端共16项回归通过。截图与详细报告保存在 `client/logs/gui_validation/`。现场真实CAN/双相机持续联调仍待验证。

## 命令行启动

```bash
cd /home/cv/gcj/project/baseline_fast/tempo-lora
client/run.sh probe

# 使用部署时保存的真实训练反馈片段验证推理，不占用现场相机或CAN。
client/run.sh replay --input client/logs/feedback_replay.npz \
  --batch-size 4 --output client/logs/replay.json

# 现场相机和CAN可用时执行，只读取状态并保存模型预测。
client/run.sh live --duration 10 --output client/logs/live.json
```

运行 live 前先退出其他占用双相机的程序。部署现场检测到 realtime_expo_ft GUI 正在运行，因此部署验证采用录制数据回放，硬件只读入口留待现场验证。机器人命令计数保存在输出的 `robot_commands` 中。

配置位于 `client/configs/piperx.yaml`。相机参数复用现有采集配置快照，映射 `camera_front → head` 与 `camera_wrist_right → left_wrist`。SDK 路径指向现有 `dynamic_vla/pyAgxArm`。环境位于 `client/.venv`，Python 3.11，通过 `client/uv.lock` 固定版本。

```bash
/home/cv/.local/bin/uv sync --project client --frozen --python 3.11
client/.venv/bin/python -m unittest discover -s client/tests -v
```

## 4090 推理服务

在 4090 项目根目录执行以下命令。部署已创建 `tempo_lora_inference` 会话。

```bash
cd /data/gcj_data/fast/baseline/tempo
tmux new-session -d -s tempo_lora_inference \
  'bash trained_model/scripts/serve_wired.sh 2>&1 | tee trained_model/logs/serve_wired.log'
curl --noproxy '*' http://192.168.50.2:8000/healthz
tail -f trained_model/logs/serve_wired.log
```

服务固定加载 `formal_20261108_b16_g1/step_00010000.pt`。新增 `--host` 允许显式指定服务地址，默认行为继续读取检查点配置。握手增加 checkpoint 文件名与实际模型步数，客户端核验配置名、步数、动作语义、维度、帧率与动作块长度。该在线服务按单个推理客户端使用。

## 数据契约

- 输入包含独立 `episode_id` 与连续 30Hz `feedback`，每帧是 float32[7] 六关节弧度和夹爪米，以及两路 224×224 RGB uint8 等比补黑边图像。
- 输出 float32[16,7]，六个绝对关节角与绝对夹爪开口。客户端直接记录输出。
- live 在固定 30Hz 时间网格对真实 SDK 反馈插值，各相机选取网格时刻之前的最新有效帧，记录原始图像时间戳。缺反馈、陈旧图像、超时和积压超限结束当前会话。
- 网络接收失败后关闭当前连接。新连接使用全新 episode ID 重建 SAM2、图像和关节历史。
- 推理同时处理自上次请求以来全部反馈帧，负载取决于积压长度与共享 GPU 状态。回放延迟不能直接代表真机持续吞吐。

回放 NPZ 包含 `timestamps[N]`、`states[N,7]`、`head[N,224,224,3]` 与 `left_wrist[N,224,224,3]`。仅接受数值数组，数据类型与连续性在发送前核验。

## Git 维护

GitHub 初始为空仓库，已确认可见性为 public。用户已明确授权推送包含内部路径、相机序列号和原有Git历史的完整项目，初次完整推送5b80738已成功。GUI更新继续通过同一main分支维护。PiperX 从 4090 的完整 main 历史创建工作副本，保留原项目历史。origin fetch 使用用户给定 HTTPS URL，push 使用 PiperX 已有 GitHub SSH 身份。`deployment-source` 指向 4090 的有线 SSH 路径，可用于两端同步。

```bash
git fetch origin
git pull --ff-only origin main
git push origin main
```

`client/logs/`、虚拟环境和打包元数据排除在 Git 外。部署验证结果见 `docs/PiperX客户端验证.json`，当前接续事项见 `docs/交接.md`。
