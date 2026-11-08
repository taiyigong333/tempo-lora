# 项目协作约定

- 用户沟通使用中文，重要代码注释使用中文解释设计原因。
- 先读 `need/need.md`、`docs/实现设计.md`、`docs/交接.md` 及 `trained_model/runtime/status.json`。
- 当前用户要求全部 393 episodes 用于训练，启用 W&B online。统一指令为 `grab the rotating red square`。
- 用户已授权尝试完整训练流程，按 SAM2、norm、MVAE、u_t、decoder、完整验证、策略训练的依赖顺序推进。
- 用户指定正式策略训练于 2026-11-07 北京时间 23:30 在 tmux 启动。当前计划与配置快照位于 trained_model/runtime/schedule_20261107_2330，正式训练入口带 not_before 时间保护。
- 当前用户指定 batch16、梯度累积1次、有效batch16，10000个优化器更新。每3000步定期保存，最后10000步保存，并维护latest.pt/latest.json。后续恢复使用当前正式快照参数。
- 原batch32任务因无界mmap缓存触及worker的1024文件句柄上限停在20步，已保留原记录。当前任务使用有界128数组LRU缓存和120秒加载超时，独立运行目录formal_20261108_b16_g1。
- uv 独立环境位于 `tempo/.venv`，优先通过 `trained_model/scripts/run.sh` 执行。保留 Transformers 4.53.2 与补丁。
- Git 根目录为本目录，源码位于 tempo 子目录。原始数据、基础权重和历史 openpi 目录均属于输入资产。
- `data/` 原件只读。缓存动作语义、维度、窗口或归一化变化时，新建版本目录。
- 反馈代理 action[t] 已对应下一实测状态。窗口起点不可再次移位，历史只取 i<t，在线历史由实测反馈重建。
- 实施小而聚焦的修改，修改后运行基础回归及可行的完整模型验证。区分代码验证、短程诊断、正式训练和真机评测。
- 每次重要变化同步维护交接，包括命令、配置、状态、W&B 地址和实际结果。
- 执行提速已从完整step500安全生效，先读 docs/训练性能优化.md 和 trained_model/runtime/performance_20261108/transition.json。正式执行compact_prefix=true、checkpoint_joint_forward=false、joint_gradient_checkpointing=true，约3.18秒/步；同一run与W&B继续。实际配置trained_model/runtime/performance_20261108/config.json，切换程序已完成，勿重跑transition.py或从头训练。
- 正式10000步训练与32样本离线评估已完成，先读 docs/训练完成报告.md 和 trained_model/runtime/final_verification_20261108.json。最终模型/优化器/scheduler均10000、W&B finished，禁止例行巡检重新启动训练；后续新实验需按用户新要求执行。
- PiperX有线客户端部署在 `/home/cv/gcj/project/baseline_fast/tempo-lora`，先读 `docs/PiperX客户端部署.md` 和验证JSON。客户端独立环境在client/.venv，现场已有其他GUI，访问相机前核对占用。
- Tempo在线服务会话tempo_lora_inference，192.168.50.2:8000，最终10000步模型。客户端默认只读，输出绝对关节rad与夹爪m。用户已要求同时提供只读和实机执行，真实运动每轮由GUI现场确认，自动验证保持只读。
- PiperX图形入口为根目录run_gui.sh，GUI后台任务位于client/src/tempo_client/session.py。用户已明确授权完整项目推送至公开tempo-lora仓库，后续相关提交可继续同步；GUI测试应使用独立显示环境和录制反馈。
- 相机由SessionWorker跨轮持有，首次开启后持续采集与覆盖旧帧；停止推理保留视频流，退出或显式释放才关闭。参考OpenPI持续相机生命周期，真实设备被占用时使用合成相机验证，不抢占其他客户端。
- 相机首次开启允许CameraFrameError在原启动时限内等待预热，运行期snapshot继续立即拒绝陈旧/不同步帧。设备忙先核查占用程序状态，切换前正常释放；真实相机恢复记录见docs/交接.md的最新段落。

- 任务、三模式和运动保护已部署，先读docs/推理控制补全.md。F5开始、Esc停止保持，相机跨轮常驻。用户确认RTC保持10次去噪、延迟超限停止。真实原生只读通过，TE/RTC目前会因缓冲耗尽或观测过期停止，不宣称持续异步或真机抓取验证成功。客户端53项、服务端28项回归通过。

- 软件限位已按用户明确确认小幅放宽，具体值见client/configs/safety.yaml和docs/推理控制补全.md末节。软件Esc、越界与延迟停止均位置保持且不发送失能。反馈失效保留故障急停，禁止对物理急停或控制器外部失能自动恢复。客户端59项回归通过，原生也可能因0.5秒动作帧龄偶发超时。

- 最新延迟修复见docs/推理延迟故障修复.md。当前正式8000启用compile_inference与parallel_sam，SAM2独立CUDA流和双相机视觉编码并行，融合等待当前批特征。真实原生只读20秒、TE只读60秒通过，RTC仍触发0.5秒时效保护。CAN可变反馈缓存竞争已修复，仍保留0.1秒反馈保护。当前回归客户端63项、服务端37项。编译后的数值差异已记录，真实运动每轮由GUI现场确认。
