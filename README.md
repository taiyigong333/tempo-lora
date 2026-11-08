# Piper-X LoRA-TEMPO

基于 Piper-X 反馈代理数据的 LoRA-TEMPO 适配。完整模型使用 TEMPO-MOT、TEMPO-ACT 和 64D u_t flow，通过冻结解码器输出 16×7 原生动作块。基础动作接口维度为 32。

## 项目目录

```text
tempo/                         TEMPO 固定基线与局部修改
data/                          只读 LeRobot 转换产物
need/need.md                   原始需求
origin_checkpoint/             SAM2 与转换后的 π0.5 基础权重
trained_model/config_use/      完整 JSON 配置
trained_model/scripts/         uv 安装与统一入口
trained_model/cache/           SAM2、RGB、norm、MVAE、u_t、decoder
trained_model/checkpoints/     策略 adapter 与完整恢复状态
trained_model/logs/            各阶段实时日志
trained_model/runtime/         流水线状态、配置快照和进程锁
trained_model/testing/         独立验证产物
docs/                         实现设计、使用说明和交接
```

Git 根目录为本目录，origin 指向 `https://github.com/taiyigong333/tempo-lora.git`，upstream 指向 TEMPO。源码基线和实际验证详见 docs/交接.md。

用户要求全部 393 条轨迹参与训练，W&B 在线观察。归一化和辅助模型使用相同的全量训练集合，评估文件明确记录 `data_split=train`。最佳检查点依据训练集监测指标保存。

## 使用

在服务器项目根目录运行。

```bash
bash trained_model/scripts/setup.sh
bash trained_model/scripts/run.sh run
cat trained_model/runtime/status.json
tail -f trained_model/logs/train_alltrain.log
```

完整命令、恢复和服务输入见 docs/使用说明.md。

## PiperX 有线客户端

PiperX 路径 `/home/cv/gcj/project/baseline_fast/tempo-lora`，桌面双击“Tempo LoRA · PiperX”或在项目根目录运行 `./run_gui.sh` 打开中文图形界面，命令行连接检查为 `client/run.sh probe`。客户端独立环境位于 `client/.venv`，连接 `192.168.50.2:8000` 的最终10000步模型。支持任务选择、TEMPO 原生同步、时间集成 TE 和 RTC 异步前缀引导。点击“开始推理 F5”启动，“停止并保持 Esc”结束，默认只读。勾选“允许实机运动”后每轮需现场确认，可选择开始前低速归位。当前模型登记任务为“抓取旋转红色方块”。部署、启动、数据契约与验证记录见 `docs/PiperX客户端部署.md`。

相机首次打开后保持持续开流，停止推理后预览继续，未使用的旧帧自动覆盖。空闲时可点击“释放相机”，退出GUI也会统一关闭。相机生命周期与验证详见部署文档。

三种模式保持10次去噪。真实双相机/CAN原生只读推理已验证，TE与RTC在当前延迟下会触发缓冲耗尽或观测过期保护。真实运动与抓取成功率尚待现场验证，详细限制见 `docs/推理控制补全.md`。
