# 全量TEMPO与RTC部署

2026-11-16更新。本仓库同步了全量微调模型部署、30步客户端、动作时间锚点修复、两项逐元素一致的省算，以及RTC论文参数部署。

## 2026-11-17 部署与延迟更新

新增可选CPU线程、编译模式、SAM2并行和多批次预热参数，修复GPU去噪循环同步与计时边界。当前顺时针step5000配置为2线程、5次去噪、引导5。有线五模式持续调度完成，RTC连续60秒的往返P50约155ms、P95约213ms。详细选项、测试环境、固定噪声对照和回退见[部署选项与延迟优化](deployment_optimization_20261117.md)。

## 现有服务器

4090运行目录为 `/data/gcj_data/fast/baseline/tempo_full`，服务为 `192.168.50.2:8002`。当前配置rtc_paper使用5次去噪、beta5、10次RTT窗口最大值估计。PiperX入口为仓库根目录run_gui_full_rtc_paper.sh。

操作与回退命令见 [RTC论文参数部署](RTC论文参数部署.md)，整体输入输出见 [全量模型部署与推理数据流](全量模型部署与推理数据流.md)，接续状态见 [交接](交接.md)。

## 从仓库恢复独立部署目录

在4090的本仓库根目录执行

```bash
bash trained_model/scripts/install_tempo_full.sh /data/gcj_data/fast/baseline/tempo_full
cd /data/gcj_data/fast/baseline/tempo_full
bash trained_model/scripts/serve_wired.sh \
  --config "$PWD/trained_model/deployment_rtc_paper.json" --check-files
bash trained_model/scripts/start_service.sh rtc_paper
```

安装入口复制源码、文档和部署配置到目标目录。配置指向trained_model/tempo_deploy中的ModelScope模型文件，Python环境及SAM2/tokenizer路径见部署JSON。迁移主机时需准备对应资产并调整配置。当前服务器已安装并完成验收，修改配置后按部署说明重载服务。

仓库中可维护的配置为trained_model/config_use/deployment_full.json和deployment_full_rtc_paper.json。完整模型启动脚本在仓库中分别命名为serve_tempo_full.sh和start_tempo_full_service.sh，安装后对应独立部署目录中的serve_wired.sh和start_service.sh。

## 客户端

```bash
bash run_gui_full_rtc_paper.sh
```

界面选中RTC与只读后，先点击“启动模型服务”，再检查连接。该入口支持在同一8002服务切换论文参数配置与默认10步配置，握手核验配置名、去噪步数和beta。

## 已执行验证

- 本次仓库检出中客户端73项、服务端52项全部通过，含旧训练目录配置发现验证。
- 两项省算对照27组固定输入，输出和SAM2特征逐元素一致，序列末端记忆一致。
- 论文参数RTC连续60秒录制反馈调度完成，388批1800帧，往返P50约155ms、P95约163ms。

摘要结果在 [validation](validation/)。自动调度验证使用录制反馈与模拟机器人，真实动态抓取效果需要现场验收。5步采样与beta5会改变相对原10步配置的预测输出。
