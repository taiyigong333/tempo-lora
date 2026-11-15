# RTC论文参数部署

更新于2026-11-15，北京时间。

## 配置与适配范围

新增配置rtc_paper采用RTC论文的5次去噪、引导裁剪beta5、最近10次推理延迟最大值估计。当前模型输出30步、控制30Hz，预测范围1秒。

| 参数 | rtc_paper部署 | 论文真机实验 |
|---|---|---|
| 去噪步数 | 5 | 5 |
| 引导上限beta | 5 | 5 |
| 延迟估计窗口b | 10次RTT，窗口最大值 | 10次，窗口最大值 |
| 预测窗口 | 30步，30Hz | 50步，50Hz |
| 最小推进 | 1步，33.3ms | 25步，500ms |
| 初始延迟估计 | 8步，266.7ms | 算法参数d_init |
| 延迟裕量 | 额外1步，33.3ms | 表4中单独裕量未列出 |

最小推进1步依据当前0.5秒动作来源时效预算选择，初始8步依据此前RTC延迟与预热路径开销估计。实际推进受模型推理耗时约束。论文的25步推进需要结合硬件和本地时效条件换算。

本项目在64维u_t上采样，经配套动作decoder计算动作前缀误差及VJP。数学对应关系和来源见 `TEMPO与RTC实现和参数核查.md`。新采样步数与beta会改变模型动作输出，现场抓取效果需独立验收。

来源为RTC论文算法1、附录A.5表4。https://arxiv.org/html/2506.07339v2#alg1 与 https://arxiv.org/html/2506.07339v2#A1.T4

## 4090启动

```bash
cd /data/gcj_data/fast/baseline/tempo_full
bash trained_model/scripts/start_service.sh rtc_paper
tail -f trained_model/logs/serve_rtc_paper.log
```

服务在192.168.50.2:8002，tmux会话为tempo_full_inference。切换脚本串行管理本项目的单个完整模型进程，核验当前进程入口后发送正常退出信号，等待旧进程退出再启动新配置。当前GPU还有其他任务，单模型切换可以复用部署显存。

服务会完成原生与RTC预热后开放端口。检查配置文件与模型资产可以执行

```bash
bash trained_model/scripts/serve_wired.sh \
  --config "$PWD/trained_model/deployment_rtc_paper.json" --check-files
curl --noproxy '*' http://192.168.50.2:8002/healthz
```

修改JSON参数后使用以下命令重新加载。更改denoise_steps或rtc_max_guidance时，同步客户端配置model中的相应握手字段。

```bash
bash trained_model/scripts/start_service.sh rtc_paper --restart
```

## PiperX图形入口

```bash
cd /home/cv/gcj/project/baseline_fast/tempo-lora
bash run_gui_full_rtc_paper.sh
```

界面配置为client/configs/piperx_full_rtc_paper.yaml，推理配置为client/configs/inference_rtc_paper.yaml。默认选择RTC及只读，显示5次去噪。

切换客户端配置后先点击“启动模型服务”，再点击“检查连接”。启动按钮通过有线SSH调用同一服务切换脚本，握手核验inference_profile=rtc_paper、denoise_steps=5、rtc_max_guidance=5以及模型身份。之后F5开始，Esc停止。实机运动沿用GUI逐轮现场确认流程。

## 切回原配置

4090执行

```bash
cd /data/gcj_data/fast/baseline/tempo_full
bash trained_model/scripts/start_service.sh default
```

PiperX执行

```bash
cd /home/cv/gcj/project/baseline_fast/tempo-lora
bash run_gui_full.sh
```

原配置对应deployment.json，10次去噪、beta10，客户端使用原有衰减峰值延迟估计。新旧客户端都校验推理参数，以便识别同权重的不同采样配置。

## 实现与日志

- 服务端入口显式将rtc_max_guidance传入OnlinePolicy，补齐参数接线，并将inference_profile发布到握手。
- DelayEstimator的window_max模式保存有限10次延迟，decayed_peak模式兼容原有配置。
- 每批predictions.jsonl新增rtc_delay_estimator及rtc_latency_estimate_ms，rtc_delay记录实际传入的冻结步数。
- 原有观测锚点、完整反馈接收、SAM2逐帧记忆、mask缺失相机省算与0.5秒时效条件继续用于此部署。

## 验证状态

客户端73项在真实Tk环境通过，含6项新增窗口估计、配置和握手测试。服务端专项13项通过，含beta配置接线与参数校验。新配置GUI默认RTC、5次去噪和只读已核验。真实GPU原生与RTC预热通过，握手显示rtc_paper、5次去噪与beta5，GUI“启动模型服务”的正常入口通过。

论文参数RTC完成连续60秒30Hz录制反馈调度，388批1800帧、1802个规划目标、387次RTC前缀引导，输出为float32[30,7]。往返P50为155.22ms，P95为162.72ms，最大动作来源年龄0.36742秒，机器人指令0。

逐批回放审计388个延迟估计值全部符合最近10次窗口最大值。实际冻结步数分布为首批0步、9次9步、324次6步、54次7步。结果位于PiperX client/logs/tempo_full/rtc_paper/realtime_report.json，审计位于同目录window_audit.json。

该结果使用录制反馈与模拟机器人。当前5次去噪与beta5的现场动态抓取效果仍待验收。
