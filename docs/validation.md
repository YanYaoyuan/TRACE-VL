# 验证记录

2026-11-05 更新：已完成真实缓存 Qwen3-1.7B / SigLIP2 的单卡四帧训练试验，并进行保留视频问答验证与视觉错配诊断。实际模型为 2,104,675,891 参数，详见 [真实数据实验报告](experiments/activitynetqa_baseline_2026-11-05.md)。以下保留早期 tiny 软件验证记录，不将其结果混作真实模型成绩。

后续真实双卡 DDP 已从 450 步恢复并完成至 2250 步，双 rank 梯度训练、保留 checkpoint 和资源收集通过。最新沙箱测试 43 passed、2 CUDA tests skipped；实际宿主双卡训练与推理另外完成。结果及视觉依赖局限见 [延长训练报告](experiments/ddp_extension_2026-11-05.md)。

高显存调参随后完成：两个学习率各进行32帧/224 RGB tokens/每卡batch=2的450步DDP训练，每卡峰值分配约30.2 GB，所有loss和梯度有限；三组各200题的标准评测与视频错配控制完成。新增记忆归一化、独立权重初始化与状态复用回归后，宿主CPU测试46 passed、2 CUDA tests skipped，ruff通过。结果见 [高显存调参报告](experiments/high_memory_tuning_2026-11-05.md)。

最终状态：**锁定 uv 环境下的 CPU/HF/GPU 测试与单双卡端到端流水线全部通过**。以下结果均为 tiny 软件测试，不能作为真实监控或约 2B 模型的论文成绩。机器可读记录见 [validation_summary.json](validation_summary.json)。

## 最终 uv 环境

- Python 3.10.12，uv 0.9.21，PyTorch 2.8.0+cu128，CUDA 12.8；2×RTX 5090，compute capability 12.0。
- `uv sync --locked --extra dev --extra hf --extra data` 完成；`uv pip check` 确认 79 个包兼容。
- CUDA 大包因连接限速采用官方分段下载，70 个下载 wheel 均逐个核对 uv.lock SHA256，再通过 uv wheelhouse 安装，最后执行标准 locked sync。依赖版本没有绕过锁文件。
- `env -u PYTHONPATH uv run --locked pytest -q`：**30 passed**，无 skip。包含 CPU 因果性/状态容量/训练恢复、真实 GPU FP32/BF16 前后向和生成、使用本地微型 SigLIP/Qwen3 权重的真实 HF 接口测试。
- 锁定 Ruff 规则检查通过。宿主 ROS 会通过 PYTHONPATH 注入 pytest 插件，项目脚本只在子进程内清除此变量，不修改系统环境。

## 端到端结果

`bash scripts/validate_gpu.sh` 已完整成功执行。所有训练均使用 517,803 参数的 tiny 模型及合成软件 fixture；不是正式 2B 训练。

| 模式 | 优化步数 | 首步 / 末步总 loss | PyTorch 峰值 allocated（rank0） |
|---|---:|---:|---:|
| 单卡 RTX 5090 BF16 | 10 | 44.7602 / 35.4450 | 32,869,888 bytes |
| 双卡 RTX 5090 DDP BF16 | 10 | 48.4995 / 34.1285 | 34,942,976 bytes |

单卡和双卡 checkpoint 均完成 4 条测试 fixture 的逐帧观测、问答、告警和指标导出。独立 `score` 入口也完成复算。路径分别为 `outputs/5090-single[-eval]/`、`outputs/5090-ddp[-eval]/`，含 checkpoint、完整配置、数据 hash、环境、loss 日志、预测及资源记录；outputs 被 Git 忽略。十步仅验证优化和工程链路，不代表收敛或学会真实场景。

最终 uv 配置流长度分析：16 / 64 / 256 帧的逻辑状态均为 **50,085 bytes**，每层缓存 4 帧，事件 bank 8 条，PyTorch 峰值 allocated 均为 12,319,232 bytes。随机输入观测约 1.98–2.03 ms/帧。这是 32×32 tiny 随机模型的计算测量，**不能外推 384×384 正式 2B 吞吐**，也不含解码、网络或 QA。张量逻辑字节不去重别名；allocator 指标不含 CUDA context、NCCL 等外部分配和其他进程显存。

## 数据与尚未完成的验证

- Video-MME 官方 parquet 405,167 字节已下载并转换：**2700 条问题、900 个来源视频**，全部通过统一 schema / source-group 检查。没有下载视频，因此没有执行其真实媒体评测。
- 早期 HF 接口测试使用微型本地权重；后续已完成真实骨干四帧试验，见上方实验报告。新增参数修正为 67,817,011，包含旧预算漏计的 1,536 个根级 token_types 参数。
- StreamingBench / OVO / XD-Violence 仍是官方来源登记状态，尚未完成适配和官方成绩复现。论文级对照、置信区间、真实负例时长以及 codec-native 输入仍待后续研究。

## 首轮宿主环境

- 2026-11-05，2×RTX 5090 32GB，驱动 570.153.02，CUDA 12.8。
- 系统已有 PyTorch 2.12.0.dev20260317+cu128；该版本不是仓库 uv 锁定的 PyTorch 2.8.0，因此只作预检查。
- CPU 单元/集成测试通过；GPU FP32/BF16 前后向及生成接口通过。
- 单卡 BF16 三步训练与推理、双卡 BF16 DDP 三步训练与 checkpoint 保存通过。
- 随机权重 tiny 流式观测 16/64/256 帧：逻辑状态均为 50,085 字节；每层缓存 4 帧，事件 bank 8 条。该值不含权重、激活、CUDA runtime 与分配器。
- 已下载 Video-MME 官方 405,167 字节 parquet 标注，尚未下载其视频。来源和 SHA256 保存在被 Git 忽略的 raw 目录。
