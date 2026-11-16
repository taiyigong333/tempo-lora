# 顺时针step4000与新增推理模式

客户端已提供统一模型选择、启停与参数调整，操作说明见[客户端模型与参数管理](client_model_control_20261116.md)。

## 快速使用

在PiperX桌面终端进入项目目录，选择需要的模型入口。

```bash
cd /home/cv/gcj/project/baseline_fast/tempo-lora
# 顺时针step4000，默认RTC
bash run_gui_full_step4000.sh
# 顺时针step5000，默认DynamicVLA
bash run_gui_full_dynamicvla.sh
# 顺时针step5000，默认VLASH
bash run_gui_full_vlash.sh
```

1. 点击“启用／切换模型”，等待预热完成，再点击“检查连接”，确认界面显示所选step4000或step5000。
2. 在“推理方式”下拉框选择需要的模式。切换推理方式后开始新一轮即可，切换检查点需先停止本轮并使用对应配置重新启动模型服务。
3. 首次运行使用默认只读方式，点击“开始推理 F5”，查看双相机、当前反馈、预测和事件日志。
4. 实机执行时勾选“允许实机运动”，按需选择“开始前低速归位”，点击开始后完成本轮现场确认。
5. 点击“停止并保持 Esc”结束本轮。停止推理后相机继续开流，可使用界面的释放相机操作关闭。模型服务独立运行。

step4000也可直接以两种新模式打开，操作步骤相同。

```bash
bash run_gui.sh --config client/configs/piperx_full_step4000_dynamicvla.yaml
bash run_gui.sh --config client/configs/piperx_full_step4000_vlash.yaml
```

当前8002模型服务已按用户要求关闭，下次使用需要先点击“启用／切换模型”。连接时会校验检查点步数、动作表示、帧率及采样参数。运行日志保存在`client/logs/gui`。

## 模型来源和配置

实验为gpu_clockwise_b234_chunk30_4gpu_20261116，仅使用batch2、3、4顺时针，289条轨迹、30285帧。step4000来自ModelScope的taiyigong3333/tempo，同实验decoder与归一化配套。全部12个下载文件的大小和SHA256均通过核验。

4090模型目录为`/data/gcj_data/fast/baseline/tempo_full/trained_model/tempo_deploy/gpu/gpu_clockwise_b234_chunk30_4gpu_20261116/policy/step_00004000`。

模型SHA256为`1d565bbde3d1027574229f57cfa68f7108fdcfb7864f347dec5b25da66dff8fe`。

配套decoder SHA256为`0d7e280260da90cd20c5d8cbdf04f61369a012fa07a01eae8612ff0c34351222`。

部署文件为独立目录的`trained_model/deployment_step4000.json`，可恢复模板为Git工程的`trained_model/config_use/deployment_full_step4000.json`。文件检查确认837个张量、30步输出、图像历史偏移[-20,-10,0]、5次去噪、beta5。

## 入口

PiperX项目目录为`/home/cv/gcj/project/baseline_fast/tempo-lora`。step4000图形入口如下，打开后使用“启用／切换模型”切换到所选模型，再开始本轮推理。

```bash
cd /home/cv/gcj/project/baseline_fast/tempo-lora
bash run_gui_full_step4000.sh
```

该入口默认RTC，模式下拉框支持原生、时间集成、RTC、DynamicVLA及VLASH。step4000两种新模式的专用配置为`client/configs/piperx_full_step4000_dynamicvla.yaml`与`client/configs/piperx_full_step4000_vlash.yaml`，可通过通用GUI入口的`--config`选择。

step5000专用图形入口为`run_gui_full_dynamicvla.sh`与`run_gui_full_vlash.sh`。

4090切换命令如下。启动脚本对8002已有客户端连接执行检查，并使用锁串行化服务切换。

```bash
cd /data/gcj_data/fast/baseline/tempo_full
bash trained_model/scripts/start_service.sh step4000
# 返回step5000
bash trained_model/scripts/start_service.sh rtc_paper
```

同一GPU只能容纳一个完整模型，切换由同一tempo_full_inference服务会话完成。当前运行模型以握手中的step及checkpoint完整路径核验。每轮真实机器人执行仍由现场GUI确认。

## 推理定义

DynamicVLA使用持续异步请求，每轮返回后立即开始下一轮。动作与反馈使用30Hz统一时间轴，同一目标时刻使用最新计划，已经过期的动作前缀由时钟丢弃。参考DynamicVLA论文3.3、3.4节。https://arxiv.org/html/2601.22153

VLASH从已有绝对动作计划估计未来接管状态，根据反馈已过去的时间及实测网络推理延迟选择严格大于延迟的整数偏移。策略使用当前图像与预计接管状态，新动作窗口从接管状态后一帧开始。旧计划在接管前保留，未来查询不会提前清除待执行动作。参考VLASH论文IV-A。https://arxiv.org/html/2512.01031

未来状态仅替换本次策略的state条件，实测last_state、反馈历史和SAM2历史继续使用实际输入。相对六关节动作以本次策略state恢复绝对目标，夹爪继续使用绝对开口。原始训练action采用下一实测状态代理。

当前step4000及step5000没有VLASH论文IV-B中的时间偏移训练增强，本次属于TEMPO权重上的VLASH推理适配。metadata中的vlash_offset_training_verified为false，GUI同步展示该限制。实际抓取效果尚待现场验证。

## 验证记录

下载12文件均通过SHA256核验。step4000部署配置通过837张量检查。新增模式代码已部署到客户端、服务器Git工程和独立tempo_full源码目录。切换服务会加载新源码。

客户端及隐藏GUI回归76项通过，覆盖五模式模拟机器人执行与只读会话、新增模式后台回放、未来状态查询和未来计划接管。服务端38项通过。

用户随后允许切换验证，并要求完成后关闭相关进程。step4000与step5000均完成严格加载和预热，并分别通过五模式录制反馈验证，每模式3批12帧，累计30批120帧。动作均为有限float32[30,7]，模型握手路径对应本次顺时针实验。VLASH后续请求确实使用正数时间偏移和预计未来状态。

两份模型的DynamicVLA与VLASH均分别完成10秒连续模拟调度，每轮79批，规划301至302个30Hz目标，全部完成，安全违规和真实机器人指令均为0。原有0.5秒动作时效阈值保留。此次为录制图像重复回放与模拟机器人调度，真实抓取效果待现场验收。

用户要求验证后关闭服务。当前完整模型服务已关闭，tempo_full_inference会话已退出，8002端口已释放，未残留该部署模型进程。后续使用时按上文入口重新启动。

PiperX报告分别位于`client/logs/tempo_full/inference_modes_20261116/step4000/verification.json`和`step5000/verification.json`。4090汇总及关闭记录位于`trained_model/runtime/inference_modes_20261116`。

差异部署保留原文件备份于两端Git工程的`trained_model/runtime/inference_modes_20261116/before`。本地完整工作交接位于`work/inference_modes_20261116/handoff.md`。
