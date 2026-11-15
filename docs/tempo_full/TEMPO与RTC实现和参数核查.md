# TEMPO与RTC实现和参数核查

2026-11-15，北京时间。联网获取论文HTML、官方仓库当前提交，并回读4090与PiperX现场配置。

## TEMPO公开说明

TEMPO论文第4.3节写明主方法搭建在VLASH异步推理管线上，第4.4节说明运动编码器运行在独立后台线程，按传感器速率更新最新运动token。论文给出RTX PRO 6000上运动编码约11ms/帧、TEMPO模型前向中位35.3ms、RTC基线127.7ms。比较本项目延迟时需要同时核对硬件、采样步数、模型头和计时范围。

TEMPO附录B.3写明作者依据RTC论文自行重实现RTC。第6节也报告TEMPO接入RTC的实验。RTC原论文附录A.6提供原作者Kinetix仿真实验代码链接。

本次检查TEMPO官方公开HEAD为dd8342a1234e7166b856ffec2d7917b51cce05e4，共127个文件条目。README提供模型服务与action_chunk接口，公开pi0_pytorch.py的sample_actions默认num_steps=10。仓库内检索未找到论文实验使用的完整RTC/VLASH真机控制管线。

来源如下。

- TEMPO第4.3、4.4节 https://arxiv.org/html/2609.16864#S4.SS3 与 https://arxiv.org/html/2609.16864#S4.SS4
- TEMPO附录B.3 https://arxiv.org/html/2609.16864#A2.SS3
- 官方README https://github.com/tempo-robot/TEMPO/blob/dd8342a1234e7166b856ffec2d7917b51cce05e4/README.md
- 公开采样接口 https://github.com/tempo-robot/TEMPO/blob/dd8342a1234e7166b856ffec2d7917b51cce05e4/src/openpi/models_pytorch/pi0_pytorch.py#L629

## 当前RTC实现对应关系

本项目在64维u_t flow上采样，通过冻结decoder将干净潜变量估计解码为动作，再在归一化动作空间对旧动作前缀误差计算VJP。该路径属于潜变量策略的RTC适配。

代码prefix_weights与RTC论文公式5对应，overlap=H-s。使用指数软衰减，前d步权重1，超过重叠范围权重0。

代码的时间变量t从1递减到0，论文变量tau从0递增到1，满足t=1-tau。当前引导系数

$$
g(t)=\min\left(\beta,\frac{t^2+(1-t)^2}{t(1-t)}\right)
$$

由论文公式2～4可换元得到，速度校正符号也随积分方向反转。该对应关系支持当前软掩码与时间权重公式的来源。经过decoder的VJP包含额外雅可比，其尺度与直接动作flow不同，因此引导上限需在当前模型上评估。

来源为RTC第3节 https://arxiv.org/html/2506.07339v2#S3 。

## 超参数对照

| 参数 | 当前现场 | RTC论文真机实验 | 核查判断 |
|---|---|---|---|
| 去噪次数n | 10 | 5 | 10与TEMPO公开采样默认一致，5属于值得独立验证的速度/动作效果对照 |
| 引导裁剪beta | 10 | 5 | 当前上限较高，优先把5作为有论文依据的对照起点 |
| chunk H | 30 | 50 | 当前30Hz、论文50Hz，二者均覆盖1秒 |
| 最小执行步数s_min | 1 | 25 | 当前33ms下限，实际推进受约200ms推理限制；论文25步为500ms，应结合本地时效条件换算 |
| 延迟历史窗口b | 衰减峰值max(L,0.95*旧峰值) | 最近10次最大值 | 当前为工程近似，论文的窗口10最大值便于直接复核保守性 |
| 延迟裕量 | 1步，约33ms | 单独裕量未列在表4 | 需结合RTT长尾及观测采样年龄验证 |
| 最大观测帧龄 | 0.5秒 | 原文以预测窗口与延迟约束描述算法 | 本项目控制条件，需与吞吐和抖动联合评估 |

RTC论文附录A.5表4明确列出n=5、H=50、s_min=25、beta=5、b=10。附录A.2讨论引导过强会产生动作发散或更大加速度，并给出选择beta=5的依据。原作者仿真实现eval_flow.py的max_guidance_weight默认也为5.0。

- 超参数表 https://arxiv.org/html/2506.07339v2#A1.T4
- 引导上限消融 https://arxiv.org/html/2506.07339v2#A1.SS2
- 原作者参考代码 https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L32

核查时beta=10来自OnlinePolicy构造默认值。后续部署已在全量模型main入口显式接入rtc_max_guidance，新增deployment_rtc_paper.json设置beta5，操作见RTC论文参数部署.md。

## 对超时的判断

此前正式RTC两轮记录中，往返中位约200～203ms、P95约210～224ms，两次均约11～12秒触发观测过期或动作缓冲耗尽。数值来自前次验证日志，当前重新查看GPU时另有约36.8GB显存进程，应按测试时段分别评估负载。

连续推理期间，上一块预测需要支撑下一次结果返回。若上一块由时刻t_n的观测生成，其自身耗时为L_n，下一轮耗时为L_{n+1}，则旧块末端年龄约为

$$
L_n+L_{n+1}+\epsilon,
$$

其中epsilon包含采样时间网格、网络调度与本地等待。因此当前时效条件近似要求

$$
L_n+L_{n+1}+\epsilon<0.5\ \mathrm{s}.
$$

两次200ms请求已经占用约400ms预算，单次延迟尖峰可能耗尽剩余余量。beta调整主要改变引导幅度，现有实现仍在每个去噪步计算一次VJP，延迟改善需要实测。把s_min设到论文25步或按比例设15步时，也需要同时核验该时效预算。

## 有依据的后续对照

先在当前10次去噪条件下评估beta取2、5、10，统计前缀动作误差、块切换跳变、远期动作误差与延迟分位数。将beta5作为论文参考值。

延迟估计使用最近10次RTC引导请求的窗口最大值对照当前衰减峰值，初始延迟依据完成预热后的RTC请求测量。最小推进1步可作为反应速度优先的起点。

5次去噪可单独列为速度与动作误差实验，配套真机抓取效果评估。当前研究只完成公式、参数和日志核查，尚未运行以上扫参或给出新参数的真机效果结论。
