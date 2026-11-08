# PiperX Tempo 客户端部署

## 目标与范围

2026-11-08 将轻量客户端部署到 PiperX 的 `/home/cv/gcj/project/baseline_fast/tempo-lora`，使用 `https://github.com/taiyigong333/tempo-lora.git` 管理同一项目。4090 的模型及依赖留在 `/data/gcj_data/fast/baseline/tempo`。

客户端提供中文图形界面与命令行入口，支持模型服务启动、握手校验、录制数据回放、硬件只读推理、双图预览、实测关节与动作块展示、停止任务及日志查看。2026-11-09已补齐任务选择、三种模式和经每轮现场确认的实机执行。最新验收与限制见本文末尾及 `推理控制补全.md`。

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
4. 现场释放其他程序占用的相机后，可先点击“开启相机”预热，选择任务与推理方式，再设置时长并点击“开始推理 F5”。首次只读任务也会开启相机，此后跨轮复用持续视频流。界面显示关节rad与夹爪m，输出保存至每轮日志。
5. 点击“停止并保持 Esc”结束网络接收和本轮CAN反馈，相机继续采集和预览，未使用的旧帧自动覆盖。下一轮使用新episode与新反馈历史。空闲时点击“释放相机”，或关闭窗口，统一关闭相机进程。模型服务保留其独立远端会话。
6. 点击“打开日志”查看本轮config.json、metadata.json、predictions.jsonl与summary.json。失败时另存error.log。

图形界面以后台线程执行网络与采样，事件队列驱动Tk更新。桌面重复启动通过文件锁限制为一个GUI进程。应用菜单入口位于 `/home/cv/.local/share/applications/tempo-lora.desktop`，桌面入口位于 `/home/cv/桌面/Tempo-LoRA.desktop`，仓库内保留 `client/tempo-lora.desktop`。

GUI已在独立虚拟屏幕上完成真实有线模型联调，回放12帧、3批，双图与16步预测显示通过。模型推理期间停止和关闭窗口通过，客户端共16项回归通过。截图与详细报告保存在 `client/logs/gui_validation/`。现场真实CAN/双相机持续联调仍待验证。

## 持续相机流

参考 `/home/cv/gcj/project/baseline_fast/openpi/piper_openpi/controller.py` 中的相机复用生命周期，2026-11-08将相机所有权从单轮LiveFeedback移到SessionWorker应用会话。

- 相机首次通过“开启相机”或“开始推理 F5”打开后持续运行，停止推理、结束一轮、网络断开和连接检查均保留视频流。空闲时界面预览继续。
- 独立相机进程持续排空设备帧，使用有界共享缓存覆盖旧图。当前配置每路保留最近9帧以支持固定30Hz网格的时间对齐，GUI只保留一份最新预览。
- 每轮重新建立CAN历史和模型episode，等待本轮反馈就绪之后拍摄的两路新图。上一轮反馈、闲置期间的画面和动作预测不会进入新一轮模型历史。
- “释放相机”与退出GUI统一关闭相机，关闭过程位于后台。参数内容变化时重建一次，相机故障时报告错误并等待显式释放重试。
- 相机模块为 `client/src/tempo_client/cameras.py`，图像处理沿用原RGB224等比补黑边，LiveFeedback通过可选cameras参数复用视频流，命令行单次live仍在进程结束时关闭其自有相机。

24项客户端回归通过。独立Tk窗口、合成相机/CAN和真实有线10000步模型的联调通过，同一相机进程632372跨两轮及中途停止复用，空闲时计数与图像时间戳继续增长。显式释放后重新开启，以及推理中退出GUI时回收相机均通过。测试报告位于 `client/logs/continuous_cameras/integration.json`，回归日志为同目录 `tests.log`。

现场双相机当前由realtime_expo_ft客户端进程4119924占用，本轮使用合成设备验证生命周期，真实USB双相机跨轮联调留待现场释放设备后进行。联调时原Tempo推理端口未监听，已通过GUI服务启动入口恢复推理服务。

## 相机开启故障处理与实际恢复

2026-11-08现场相机开启失败的原因为旧realtime_expo_ft客户端4119924仍持有两路设备。确认旧客户端处于停止状态、机器人保持任务空闲、远端任务停止后，正常退出旧客户端并释放相机。

真实开启又发现预热阶段出现短暂旧帧时会提前失败。当前wait_ready会在既有启动时限内等待新鲜同步帧，CameraFrameError在实际采样时仍按原帧龄和双图同步阈值报错。设备忙会明确显示相机名称和序列号，并提示在占用程序中释放设备后，再点击Tempo的“释放相机”和“开启相机”。切换客户端时，先在当前程序释放相机，再打开另一程序。

修复后29项回归通过，真实桌面GUI首次开启两路相机成功，进程637357持续开流，空闲预览更新和重复开启复用通过，有线模型握手通过。验证后保留视频流运行。实际报告位于 `client/logs/camera_busy_fix/real_camera_verification.json`，回归日志位于同目录 `tests.log`。

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

## 2026-11-09 任务与控制升级

现场正式界面和192.168.50.2的8000服务已切换新版，候选8001服务已退出。首次运行保持只读，实机执行功能的每轮确认由操作者完成。

1. 选择“抓取旋转红色方块”，当前10000步模型仅支持这一训练任务。
2. 选择“TEMPO 原生同步”“时间集成 TE”或“RTC 异步前缀引导”。
3. 默认只读，点击“开始推理 F5”或按F5。运行秒数0表示持续至停止。
4. 实机执行时勾选“允许实机运动”，按现场需要选择“开始前低速归位”。开始后弹窗列出任务、模式、归位和速度，需要每轮现场确认。默认沿用OpenPI的归位速度10%、策略速度90%，配置在client/configs/safety.yaml。
5. 点击“停止并保持 Esc”或按Esc。停止任务保留相机视频流和空闲预览，实机执行中同时保持当前位置。
6. 每轮日志新增controls.jsonl，记录实际调度目标、来源帧龄、保护检查与真实运动指令计数。predictions.jsonl仍保留完整模型动作块。

任务、模式和保护分别配置于client/configs/tasks.yaml、inference.yaml和safety.yaml。原生每块步数默认1，TE衰减默认0.05，RTC最少推进默认1步。所有模式继续10次去噪，RTC延迟超限停止。

真实双相机/CAN原生只读3秒任务完成，8批91帧反馈、7个规划目标、0运动指令。F5/Esc两次只读开始和取消通过，同一相机进程跨三种模式复用，退出推理后双路预览继续更新。TE在15个规划目标后、RTC在8个规划目标后触发缓冲耗尽或观测过期保护。RTC真实GPU引导另经回放验证，预热后3帧反馈约465至476ms。当前未达到持续异步运行条件。

一次只读预测出现夹爪目标超出0至0.10m范围，已作为保护提示记录。实机执行路径会拒绝该目标。全部真实联调运动指令为0，实机归位、执行、停止保持的流程使用合成SDK验证，真实抓取尚待现场确认后评测。

最终回归为客户端53项、服务端28项通过。现场报告在client/logs/mode_deployment/desktop_verification.json，合成执行报告在同目录gui_integration.json，服务端报告在trained_model/runtime/mode_deployment/。真实桌面的Xwayland截图接口不可用，独立Tk截图已检查，现场控件与实时预览通过Tk控件接口核验。


## 软件限位与停止保持的当前设置

2026-11-09经用户确认，软件TCP范围更新为X [-0.07, 1.05]、Y [-1.05, 1.05]、Z [0, 1.05]米，归位起点使用同一范围。单目标相对当前TCP位移上限0.35米、转角上限0.65弧度，配置仍为client/configs/safety.yaml。

Esc/停止按钮、目标越界、延迟超限和任务结束均使用位置保持，软件不发送失能命令。停止日志中的robot_diagnostics.stop_mode为position_hold_keep_enabled，hold_joint_enabled记录发送保持目标后读取的六关节使能反馈。CAN反馈失效保留阻尼故障急停，物理急停和控制器故障保护继续生效。

当前客户端59项回归通过，真实只读F5/Esc、配置加载和持续双相机通过，实机运动保持效果使用合成SDK验证。原生模式也观察到0.505秒帧龄超限，记录在client/logs/hold_limits/desktop_verification.json。完整变更和限制见推理控制补全.md末节。


## 当前连续推理修复

2026-11-09正式8000服务启用SAM2与双相机视觉编码并行，并编译去噪、动作解码和SAM2固定形状部分。真实CAN/双相机原生只读20秒、TE只读60秒通过。TE往返中位193.5ms、P95为201.2ms。客户端SDK反馈缓存读取竞争已修复，客户端63项和服务端37项回归通过。

当前界面保持TE与只读，F5开始、Esc停止保持，相机持续预览。RTC仍在10次去噪下触发时效保护。完整结果、浮点数值差异和日志入口见推理延迟故障修复.md。
