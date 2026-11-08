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

PiperX 路径 `/home/cv/gcj/project/baseline_fast/tempo-lora`，在项目根目录运行 `client/run.sh probe`。客户端独立环境位于 `client/.venv`，连接 `192.168.50.2:8000` 的最终10000步模型。支持反馈回放与实机只读推理。部署、启动、数据契约与验证记录见 `docs/PiperX客户端部署.md`。
