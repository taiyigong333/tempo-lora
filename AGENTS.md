# 项目协作约定

- 用户沟通使用中文，重要代码注释使用中文解释设计原因。
- 先读 `need/need.md`、`docs/实现设计.md`、`docs/交接.md` 及 `trained_model/runtime/status.json`。
- 当前用户要求全部 393 episodes 用于训练，启用 W&B online。统一指令为 `grab the rotating red square`。
- 用户已授权尝试完整训练流程，按 SAM2、norm、MVAE、u_t、decoder、完整验证、策略训练的依赖顺序推进。
- uv 独立环境位于 `tempo/.venv`，优先通过 `trained_model/scripts/run.sh` 执行。保留 Transformers 4.53.2 与补丁。
- Git 根目录为本目录，源码位于 tempo 子目录。原始数据、基础权重和历史 openpi 目录均属于输入资产。
- `data/` 原件只读。缓存动作语义、维度、窗口或归一化变化时，新建版本目录。
- 反馈代理 action[t] 已对应下一实测状态。窗口起点不可再次移位，历史只取 i<t，在线历史由实测反馈重建。
- 实施小而聚焦的修改，修改后运行基础回归及可行的完整模型验证。区分代码验证、短程诊断、正式训练和真机评测。
- 每次重要变化同步维护交接，包括命令、配置、状态、W&B 地址和实际结果。
