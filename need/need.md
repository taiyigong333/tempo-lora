# LoRA-TEMPO 实施说明

## 任务目标

请以 https://github.com/tempo-robot/TEMPO 的 PyTorch 实现为基础，实现支持自有数据的完整 LoRA-TEMPO。交付可运行的代码、配置、训练入口、断点恢复和推理入口。

完整模型包含 TEMPO-MOT、TEMPO-ACT、MVAE、u_t 标签构建和 u_t 动作解码器。默认完整配置使用 `predict_ut=True`。保留原有直接动作块配置的兼容性。

自有数据使用 https://github.com/taiyigong333/piper_x_data_tranfer/tree/piperx_to_tempo_vla 的转换产物。文末“Piper-X 自有数据接入要求”是本任务的具体约束，优先于本文中的通用数据示例，尤其是历史动作和在线控制语义。

直接完成完整方案，以完整模型做集成验证。不要将任务改成分阶段消融研究，不要等待逐阶段审批。辅助模型训练及缓存生成按必要依赖顺序执行。

本说明是基于源码审查形成的实现设计，尚未经过训练验证。参数值属于建议起点。不得将“FP32 更新即可与 JAX 等效”作为已证实结论。

## 工作方式与边界

- 开始前读取当前仓库 AGENTS.md，检查分支、提交号和已有修改。
- 本任务底座为 TEMPO 原仓库。复用其现有模型、数据变换、工具和接口。
- 先核查实际代码，再确定修改位置。本文中的符号名称可能随提交变化。
- 优先局部改动，默认关闭新增 LoRA 功能，使原有调用继续工作。
- 使用中文注释解释关键设计原因。
- 在 docs/ 维护实现设计和交接记录，包含提交基线、修改范围、命令、实际验证结果和未完成事项。
- 缺少数据路径、GPU、控制定义等必要信息时，向用户明确询问。可同时完成不依赖这些信息的代码工作。
- 不编造结果，不擅自启动长时间或付费训练。用户提供运行资源并授权训练后，再执行相应训练命令。

## 模型结构要求

### TEMPO-MOT

保留仓库现有历史图像采样、SigLIP 因果时间注意力及 SAM2 特征融合。

- SAM2 编码器冻结，训练时复用离线缓存。
- SAM2 特征融合到主相机的视觉 tokens，保持现有语义。
- SigLIP 时间注意力复用空间注意力投影，给共享投影添加 LoRA 将同时影响两类计算。
- 核查 Transformers 替换文件的安装与运行路径，防止实际执行未修改的依赖。
- 在线 SAM2 按连续视频维护时序状态，每个 episode 重置。不要把 SigLIP 的三帧采样设置误当成 SAM2 仅处理三帧。

### TEMPO-ACT

保留历史动作的两条注入路径。

1. 历史动作桶均值经过投影和位置编码，作为 VLM 前缀 tokens。
2. 历史动作展平后经过 MLP，与 flow 时间条件相加，注入 Action Expert 的 adaRMS。

保留 padding mask。历史分支继续使用现有原始动作单位，避免误套策略目标动作的归一化或相对化。

### u_t 预测

保留 MVAE 和视频编码均值 mu_v 作为 u_t 标签的机制。默认示例为 64 维 u_t、16 帧窗口，但这些值必须配置化并保持相关组件一致。

策略通过 Flow Matching 预测 u_t，冻结的动作解码器将其恢复为动作块。未来窗口只能用于构造训练监督。

## 参数训练范围

| 对象 | 默认策略 |
|---|---|
| PaliGemma 语言主干 | 基础权重冻结，注意力与 MLP 使用 LoRA |
| Gemma Action Expert | 基础权重冻结，注意力与 MLP 使用 LoRA |
| SigLIP | 基础权重冻结，仅启用时间注意力的层中共享注意力投影使用 LoRA |
| SAM2 | 冻结 |
| sam2_fusion | 全量训练 |
| action_history_vlm_tokens | 全量训练，包含位置参数 |
| action_history_cond_path | 全量训练 |
| u_t 输入、输出投影 | 全量训练 |
| flow 时间条件投影 | 首版冻结，允许配置显式解冻 |
| MVAE、u_t 解码器 | 独立训练，策略训练时冻结 |

以上范围为明确的首版选择。实现时打印完整可训练参数名单和各模块数量，避免后续冻结操作覆盖 TEMPO 模块。视觉位置参数等未列出的基础参数默认冻结。

## LoRA 实现

优先使用与当前依赖兼容的 PEFT 定向注入。核查依赖约束，避免升级破坏 TEMPO 的 Transformers 补丁。若选择轻量自定义实现，必须在设计文档说明兼容性原因。

- 使用完整模块路径或精确规则，打印命中名单，命中为空时立即报错。
- Gemma 注意力通常为 q_proj/k_proj/v_proj/o_proj，SigLIP 通常为 q_proj/k_proj/v_proj/out_proj。以实际模块树为准。
- 核查自定义联合前向中对 `.weight`、内部层和 dtype 的访问，确保包装后仍兼容。
- 初始 adapter 应保持基础层输出，采用可解释的初始化策略。
- 不引入 QLoRA 或基础模型量化。

默认建议配置如下，全部允许覆盖。

| 项目 | 建议起点 |
|---|---|
| VLM rank | 16 |
| Action Expert rank | 32 |
| SigLIP rank | 16 |
| alpha | 分别等于各自 rank |
| dropout | 0 |
| LoRA 学习率 | 5e-5 |
| TEMPO 新增模块学习率 | 1e-4 |
| u_t 输入输出投影学习率 | 5e-5 |

## 精度和优化器

目标是选择性混合精度，接近 openpi JAX 的参数更新机制。

- 冻结主干参数使用 BF16，保留现有必要的 FP32 层。
- LoRA、TEMPO 新增模块和其他可训练参数以 FP32 保存及更新。
- 主干前向以 BF16 为主，明确模块间类型转换。
- 核查归一化、注意力 logits、Softmax 的高精度运算。不要仅依据 autocast 开关判断已对齐 JAX。
- Flow Matching 目标、速度误差及损失归约保持 FP32。
- FP32 可训练参数对应的梯度和 AdamW 一阶、二阶动量保持 FP32。
- 优化器只接收可训练参数，转换精度必须在创建优化器之前完成。
- 保留全 FP32 诊断模式，覆盖全部相关模块，不能只转换主干。

推荐初始化顺序为构建模型、加载基础权重、冻结基础参数、注入 LoRA、解冻指定模块、设置参数精度、创建优化器、分布式包装。恢复训练按当前保存格式重建结构后加载模型和优化器状态。

冻结主干仍需保留必要的反向传播图，不能对整段主干使用 no_grad。通过若干步梯度及更新检查确认新增模块可学习，考虑零初始化造成的首步零梯度。

沿用 AdamW、warmup 和余弦退火，梯度裁剪建议为 1.0。参数组保留学习率倍率，修正现有循环可能将所有参数组设成相同学习率的问题。

检查 EMA 是否真实实现。配置中的 ema_decay 不等于已执行 EMA。若本任务选择实现，至少覆盖全部可训练参数，并明确验证及部署权重的选择；否则明确标记未启用。

## 动作语义与 u_t 解码器修正

实施前确认自有数据的 state/action 顺序、单位、绝对或相对含义以及控制模式。默认 YAM 接口按每臂 6 个关节位置加 1 个夹爪处理，其他机器人需要显式适配。

默认统一以下语义。

```text
关节目标 Δq[t,k] = 原始绝对关节目标[t+k] - 当前关节状态[t]
夹爪目标保留绝对值
所有窗口步都减去同一个窗口起点状态
```

当前源码审查发现，u_t 动作解码器训练直接归一化缓存原始 action，而默认输出变换会加回当前状态。请重新核实并修正这一不一致。

- MVAE 动作分支与独立 u_t 动作解码器采用一致的动作语义。
- 每个窗口先相对化，再使用对应归一化统计。相对化依赖窗口起点，不能用一份逐帧相对化数组替代所有窗口目标。
- 策略归一化统计只从训练集计算，且匹配上述动作变换。
- MVAE 如保留独立归一化统计，保存并明确其作用范围。独立部署解码器必须输出策略归一化空间。
- 部署执行反归一化、加回当前关节状态及原生维度裁剪，夹爪跳过加法。
- 缓存和检查点记录 action_space、维度、单位、窗口和归一化版本，拒绝不兼容产物并提示重建。
- 不直接复用动作语义变更前的 MVAE、u_t 标签及解码器，除非完成明确兼容性检查。

复用或提取现有动作变换，避免在多个脚本中维护不同公式。用确定性小样例检查整个动作往返。

## 数据和缓存

- 保持视频、状态、动作和时间戳同步，按 episode 或场景划分训练验证集。
- 检查 SAM2 预处理脚本对旧版 LeRobot 目录和 chunk-000 的假设，按实际数据兼容读取。
- 核查 MVAE 工具中硬编码的 14 维动作，按数据或配置推导并验证维度。
- episode、frame_index、视频帧数和缓存长度必须一致，缺失缓存应明确报错。
- 固定历史图像顺序和采样间隔，同一窗口内采用一致随机几何增强。
- 历史动作桶宽、帧率、padding 与在线实现完全一致。
- 明确在线历史缓存与示教 action 字段的对应关系。Piper-X 本次数据使用反馈代理，具体规则见文末，不得直接替换为实际下发命令历史。
- 区分策略预测长度和每次闭环实际执行长度。

## 保存、恢复与推理

交付统一模型构建及加载逻辑，由训练、恢复和 serve_policy 复用。

检查点包含基础模型标识、LoRA、所有额外可训练参数、完整配置、归一化统计、辅助模型引用及版本。恢复训练另需优化器、调度、训练步数和随机状态。

先支持未合并 adapter 的完整推理。若提供 merge 功能，验证 FP32 合并及降精度后的误差，禁止合并后重复应用 adapter。

在线策略维护历史图像、SAM2 状态和历史动作，episode 切换统一重置。输出为可供机器人控制接口使用的原生单位动作块。

## 验收与交付

围绕完整模型做最小必要验证，覆盖以下内容。

1. 语法、导入、原有非 LoRA 配置兼容性。
2. 完整模型单批次前向、反向和优化器更新，检查 dtype、有限值和参数组覆盖。
3. 冻结权重保持一致，LoRA 和 TEMPO 可训练模块能产生有效更新。
4. 动作相对化、归一化、解码器输出和绝对化的确定性样例。
5. 保存重载及断点恢复。固定观测、采样噪声和设置，对比输出与后续训练更新。
6. 完整输入经过正常推理入口，覆盖历史启动 padding、episode 重置及单臂/双臂实际支持范围。
7. 数据和硬件具备时，执行已授权的完整训练与闭环评测，记录成功率、端到端延迟和峰值显存。

补充必要的回归测试，尤其覆盖动作空间修正、冻结规则和保存加载。无数据或 GPU 时完成可行的检查，逐项说明未运行项和运行条件。

最终交付应包括代码、完整示例配置、数据准备到部署的准确命令、依赖及补丁安装说明、实际验证记录和 docs/ 交接文档。不得将未执行训练描述为复现成功。

## 主要参考

- TEMPO 仓库 https://github.com/tempo-robot/TEMPO
- TEMPO 训练入口 https://github.com/tempo-robot/TEMPO/blob/main/scripts/train_pytorch.py
- TEMPO 模型 https://github.com/tempo-robot/TEMPO/blob/main/src/openpi/models_pytorch/pi0_pytorch.py
- TEMPO 解码器训练 https://github.com/tempo-robot/TEMPO/blob/main/tools/ut/train_ut_decoder.py
- openpi JAX 训练 https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train.py
- openpi JAX Gemma https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py
- openpi JAX LoRA https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/lora.py

参考主分支会变化。实施时固定实际提交，重新核查相关结论。

## Piper-X 自有数据接入要求

### 固定来源与证据边界

- 数据转换仓库 https://github.com/taiyigong333/piper_x_data_tranfer
- 分支 `piperx_to_tempo_vla`
- 本次实际读取提交 `45d9f8275316c455c3aeb2c375add8b37d1bf661`
- TEMPO 本次审查提交 `dd8342a1234e7166b856ffec2d7917b51cce05e4`
- 优先阅读 README.md、docs/TEMPO训练参数与数据表示.md、docs/handoff.md、piperx_to_tempo_vla/convert.py 和 tempo_entry.py。

上述仓库描述了已转换数据，但本次未取得真实 HDF5、Parquet 和视频。393 个 episodes、44,398 帧等数字来自其交接记录，执行 agent 应回读实际产物确认。服务器路径仅是来源记录，必须配置或由用户确认，不能假设当前机器存在。

### 数据契约

| 字段 | 本次具体语义 |
|---|---|
| 格式 | LeRobot v2.1 |
| repo_id 默认值 | piperx_rotating_red_square |
| observation.state | 6 个绝对关节角 rad，加绝对夹爪开口 m，float32[7] |
| action[t] | state[min(t+1,T-1)]，下一 30Hz 实测状态代理，末帧保持 |
| 时间 | 30Hz，timestamp=t/30 |
| 图像 | RGB，224×224 等比缩放并黑边填充 |
| head | 原 camera_front |
| left_wrist | 原 camera_wrist_right，仅接口槽位映射，不镜像 |
| right_wrist | 缺失，由模型补零并 mask=False |
| 当前统一指令 | grab the rotating red square |

磁盘 action 保持绝对表示，禁止转换器提前写入 delta，避免训练时重复相减。禁止把数据扩为虚假双臂 14D。关节与夹爪单位已有转换项目文档依据，仍应从实际元信息核查范围和顺序。

配置必须显式设置 `data.action_dim=7`、`model.action_history_dim=7`、`data.cameras=("head", "left_wrist")`、`action_history_fps=30`、`use_delta_joint_actions=True`、`undistort_crop=False`。

π0.5 原生模型动作维度保持 32，通过现有变换完成 7→32 padding 和输出裁剪。u_t flow 维度为独立的 64，解码器输出原生 7D，再按现有接口补齐。禁止把模型 action_dim、原生数据维度和 u_t 维度混为一谈。

保留原有颜色处理及 resize-with-padding，避免重复红蓝交换、重复裁剪或施加 YAM 鱼眼标定。

### 反馈代理标签与因果性

这批数据没有记录机器人目标 command。完整复现应明确命名为“基于 Piper-X 反馈代理数据的 LoRA-TEMPO 适配”，在记录中注明它与论文命令监督的差异。

动作窗口起点为 t，k 从 0 开始时，关节监督实际为：

```text
delta[t,k] = state[min(t+k+1,T-1), :6] - state[t, :6]
gripper[t,k] = state[min(t+k+1,T-1), 6]
```

因此首个目标对应下一帧，不能在窗口构造时再次向后偏移一帧。MVAE 与独立解码器共享该窗口语义。对于 16 步预测，目标对应后续约 0.533 秒。

离线历史只允许使用索引 i<t 的 action。由于 action[i]=state[i+1]，这些代理在时刻 t 已可观测。不得读取 action[t] 作为当前可用历史，它含有下一帧反馈。

在线历史应由带时间戳的实测反馈重建，按离线同一索引和桶边界放置，最晚代理为 action[t-1]=state[t]。不要直接把预测动作块或下发命令塞入该历史分支。训练标签来自反馈代理，在线改成命令历史会改变输入分布。

这种历史摘要更接近测量轨迹记忆。若以后补采真实 command，需要显式更换数据语义并重建相关产物。完整真机部署前确认目标位置接口、单位、限位及跟踪延迟，并确认 agent 获得执行机器人动作的授权。

### 训练配置与跨仓库衔接

当前转换项目的 tempo_entry.py 动态注册 `pi05_piperx`、`pi05_piperx_tempo_mot`、`pi05_piperx_tempo`，并统一强制 `predict_ut=False`。它没有完整 u_t LoRA 配置，不能只加一个 LoRA 开关就宣称完成本任务。

建议在 TEMPO 内新增独立配置 `pi05_piperx_tempo_ut_lora`，复用已有 Piper-X 数据契约。保留转换项目旧配置原有行为。若使用动态注册入口，应明确新增配置的注册位置及可发现性，覆盖 norm/train/serve 三个入口，避免三个进程采用不同配置。

初始化从匹配当前 TEMPO 的 π0.5 PyTorch 基座开始，禁止继承其他机器人任务的中间 checkpoint 路径。配置全部缓存和基础权重路径，提供完整命令，明确训练环境与数据转换环境相互独立。

MVAE、encode_ut_windows、train_ut_decoder 中动作维度必须由 7D 数据或元信息推导。核查所有张量形状、checkpoint model_cfg、归一化及 padding，不能只把一个硬编码的 14 改成 7。

### 数据划分与缓存

转换器当前将全部 episodes 记为 train。实施时建立固定、持久化的 episode 划分，策略训练、MVAE、解码器、归一化统计共享同一训练集合。允许为验证集生成特征和标签，但禁止用其训练辅助模型或统计归一化。

按 `meta/source_episodes.jsonl` 核查采集批次、相近录制及来源关系，避免相邻片段跨集合泄漏。划分后保留全局 episode_index，不重编号，以保持缓存对应。

交接记录中的 393 条轨迹共 44,398 帧，平均约 3.77 秒。默认 ACT 历史为 10 秒，很多窗口会有大量 padding。首版保留该配置并输出有效历史比例，不得静默缩短历史或删除短轨迹。

SAM2 预处理应从实际 episode 元数据确定覆盖范围。若使用 --end，注意它是排他上界，当前记录的 0～392 需要覆盖到 393；不要沿用工具默认 60 或 README 示例 90。处理所有实际 chunk，验证缺失缓存和长度。

缓存元信息记录转换提交、数据标识、划分、动作语义、配置和归一化版本。动作空间或转换结果改变时，新建产物目录。未经核查不得覆盖现有转换数据。

### 任务文本核查

来源包含 rotating_red_square_blue_plate 的 batch_2/3/4，以及 rotating_red_square 的 batch_5，目前统一指令为 `grab the rotating red square`。请结合 source_episodes 和样本核查任务终止目标是否一致。若确有“抓起”与“放入蓝盘”两类任务，应向用户确认语言与任务划分，不要根据目录名自行重新标注。

### 本次额外交付

- 完整 Piper-X u_t LoRA 配置与数据契约检查。
- 转换产物验证、划分、SAM2、norm、MVAE、u_t 编码、解码器训练、策略训练、恢复和服务的准确命令。
- 单臂 7D、相机缺失掩码、下一状态代理偏移、历史因果性和动作往返的必要回归验证。
- 说明哪些结果由本次实际运行得到，哪些来自转换项目既有记录。
