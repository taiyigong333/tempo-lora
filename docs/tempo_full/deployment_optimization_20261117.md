# TEMPO 部署选项与延迟优化

更新于 2026-11-17。

## 部署与参考关系

本次优化 TEMPO。参考服务为 gh4090piper 上 `/data/gcj_data/fast/baseline/openpi/trained_model/deployment`，主要借鉴模型目录管理、串行推理、普通与 RTC 双路径预热、模型握手及客户端启停管理。TEMPO 已具备模型管理与连接占用保护，本次补足代表性预热、运行参数覆盖和实际耗时统计。

服务端 Git 工程为 `/data/gcj_data/fast/baseline/tempo`，独立完整模型部署为 `/data/gcj_data/fast/baseline/tempo_full`。客户端为 PiperX 的 `/home/cv/gcj/project/baseline_fast/tempo-lora`，原 GUI 模型选择与启停入口继续使用。

本次真实 GPU 与有线基准使用顺时针 batch2/3/4 全量 step5000、30 步输出、7 维原生动作。测试开始时 GPU 空闲。参数扫描及有线测试阶段，同卡另一 OpenPI 服务占用约 36 GB 显存，当前 TEMPO 占用约 9 GB。以下结果包含该共享环境的影响。

## 实现

- 去噪迭代次数改由 CPU 整数控制，消除每次 GPU 标量循环条件造成的同步。保留 FP32 时间递推。固定噪声验证包含 3、5、10 次去噪及普通、RTC 路径，优化前后动作逐元素一致。
- 启动时对普通与 RTC 分别执行 1、4、8 帧反馈批次，各重复 3 轮，覆盖连续 SAM2 记忆与 RTC 反向图。预热成功后开放端口，结束或失败均清空预热历史。此次完整启动约 50 秒。
- 部署文件和命令行可配置 CPU 线程、编译开关、去噪次数、RTC 引导、SAM2 并行、预热批次与重复次数。运行握手回传实际线程数与编译、预热配置。
- `policy_timing.infer_ms` 在输出回传 CPU 后计时，包含等待 GPU 完成和设备传输。增加 `input_transform_ms`、`output_transform_ms`、`total_ms`。服务端 `end_to_end_ms` 和客户端 `round_trip_ms` 分别记录在线整段处理与网络往返。

## 使用

PiperX 打开现有 GUI，选择顺时针 step5000，点击“启用／切换模型”。采样次数和引导强度仍可在 GUI 参数区设置，下一轮生效。当前rtc_paper配置使用2个CPU线程、5次去噪、引导强度5。default配置沿用10次去噪和引导10，两份配置均已写入2线程与新增预热参数。

服务器独立部署的常规启动命令如下。

```bash
cd /data/gcj_data/fast/baseline/tempo_full
bash trained_model/scripts/start_service.sh rtc_paper
```

自定义部署参数时，在对应部署 JSON 中增加以下字段。当前 step5000 的 `deployment.json` 和 `deployment_rtc_paper.json` 已写入本次实测配置，其他模型可按需单独调整。

```json
{
  "cpu_threads": 2,
  "denoise_compile_mode": "default",
  "warmup_batches": [1, 4, 8],
  "warmup_repeats": 3
}
```

命令行临时覆盖示例如下。该入口前台运行，退出后参数覆盖结束。单独启动前先通过客户端停止当前受管模型服务。

```bash
cd /data/gcj_data/fast/baseline/tempo_full
bash trained_model/scripts/serve_wired.sh \
  --config "$PWD/trained_model/deployment_rtc_paper.json" \
  --cpu-threads 2 \
  --denoise-steps 5 \
  --rtc-max-guidance 5 \
  --compile-inference \
  --parallel-sam \
  --warmup-batches 1 4 8 \
  --warmup-repeats 3
```

`--no-compile-inference`、`--no-parallel-sam` 可用于诊断。`--compile-mode` 支持 `default`、`reduce-overhead`、`max-autotune-no-cudagraphs`，作用于去噪器与动作解码器。三种模式均通过普通与RTC真实GPU功能验证。小样本下，`reduce-overhead` 与默认输出一致，`max-autotune-no-cudagraphs` 的关节最大差异约0.0647 rad、夹爪约0.00120 m。当前默认使用 `default`。SAM2 编译继续沿用现有默认模式。`--check-files` 验证配置、模型文件、维度与归一化契约。`--warmup-only` 完成真实加载及双路径预热后退出，便于部署前自检。

旧配置缺少新增字段时使用 8 个 CPU 线程、默认编译模式、1/4/8 帧预热和 3 次重复。命令行明确传入的值优先于部署 JSON。CPU 线程数影响当前推理进程，脚本同时允许覆盖 `OMP_NUM_THREADS` 与 `MKL_NUM_THREADS`。

## 配对基准

同一模型、相同录制反馈和噪声，交替执行优化前后各 3 轮。每种模式每个版本 24 个样本，均匀包含 1、4、8、16 帧反馈。两组均为 8 线程、5 次去噪，GPU 完成后读取墙钟耗时。

| 服务端处理耗时 | 原实现 P50 / P95 | 优化后 P50 / P95 |
| --- | --- | --- |
| 普通推理 | 109.91 / 168.98 ms | 108.31 / 156.13 ms |
| RTC | 126.40 / 182.03 ms | 124.65 / 177.72 ms |

循环优化的 P50 改善约 1.4%，普通推理 P95 改善约 7.6%，RTC P95 改善约 2.4%。该优化属于小幅收益。

线程扫描采用 1、2、4、8 线程的正反顺序，每种模式每个线程数 16 个测量样本。2 线程的普通与 RTC P95 合计最低，据此选择当前 step5000 部署的 2 线程。样本量较小，扫描结果不代表其他 GPU 或并发负载下的最优值。

| CPU 线程 | 普通 P50 / P95 | RTC P50 / P95 |
| --- | --- | --- |
| 1 | 108.09 / 174.14 ms | 128.89 / 195.15 ms |
| 2 | 107.71 / 162.25 ms | 129.45 / 181.28 ms |
| 4 | 108.91 / 175.91 ms | 131.55 / 190.05 ms |
| 8 | 112.95 / 160.03 ms | 131.42 / 200.12 ms |

## 有线持续调度

使用 PiperX 正常 Client 和 InferenceRun 入口，192.168.50.1 到 192.168.50.2，30Hz 录制反馈与模拟机器人。保留现有动作时效及控制调度检查。

| 模式 | 去噪次数 | 时长 | 批次 | 往返 P50 / P95 |
| --- | --- | --- | --- | --- |
| 原生 | 5 | 20 s | 133 | 128.88 / 188.29 ms |
| TE | 5 | 20 s | 155 | 122.84 / 165.48 ms |
| RTC | 5 | 60 s | 366 | 154.98 / 213.45 ms |
| DynamicVLA | 5 | 20 s | 154 | 123.75 / 166.56 ms |
| VLASH | 5 | 20 s | 152 | 124.34 / 175.30 ms |
| RTC | 3 | 30 s | 209 | 133.50 / 198.91 ms |

全部完成，安全违规与实机指令数均为 0。RTC 5 次去噪包含 365 次有前缀引导请求。VLASH 沿用现有未来状态推理适配，该权重缺少时间偏移训练增强。

## 采样参数选择

扫描 3、5、10 次去噪，分别验证原实现与新循环动作一致。3 次相对 5 次的固定输入输出差异，关节最大约 0.327 rad、平均约 0.015 rad，夹爪最大约 0.0302 m。10 次相对 5 次也会改变输出。当前保留 5 次默认采样，3 次作为可选低延迟实验参数。抓取成功率与实机运动效果待现场验证。

## 证据与回退

服务器证据位于 `tempo/trained_model/runtime/deployment_20261117`，包含基准脚本、配对原始耗时、参数扫描、回归日志及修改前备份。客户端证据位于 `client/logs/deployment_20261117`，包含每批预测、目标调度与完整有线验证报告。

原始文件备份在上述服务器证据目录的 `before`。其中 `independent` 对应独立 `tempo_full` 部署原文件。回退时先通过客户端停止模型服务，再按对应相对路径恢复所需源码或 JSON，随后重新启动并等待预热。

依赖链为完整部署入口、FullPolicy、OnlinePolicy、PI0Pytorch 及 SAM2Stream。共享的去噪循环与计时修改也会被 LoRA 入口使用。模型资产与训练过程沿用原工程。

## 最终验证记录

全部98项服务端回归通过，包含真实CUDA流测试。三种编译选项均完成普通与RTC功能验证，默认编译模式执行了完整有线持续调度。详细编译结果位于服务器证据目录的 `compile_modes.json`。

自动审批限制了完整有线报告跨主机归档，报告保存在PiperX的原始日志目录。项目文档记录了汇总数值。

用户已在2026-11-17明确授权将本次13个相关文件提交并推送至公开GitHub的tempo-lora仓库main分支。修改已在服务器工程和独立部署落地。
