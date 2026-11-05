# GPU 与存储预算

现有资源是本地 2×RTX 5090 32GB，云端 4×A100（40GB/80GB 待区分）。本仓库配置采用 DDP，每卡复制模型；两张卡的显存不是一个 64GB 地址空间。目前未实现 FSDP/ZeRO、activation checkpointing 或分片优化器。

2026-11-05 已在单张 RTX 5090 实测四帧、384×384、冻结骨干、BF16、batch=1/累积4：100 步训练循环含保存约 66.7 秒，峰值 allocated 6.18 GiB、reserved 6.92 GiB，完整 checkpoint 约 5.2 GiB。只适用于该起步配置；准确效果与完整一轮对照见 [实验报告](experiments/activitynetqa_baseline_2026-11-05.md)。

真实双卡继续训练新增 1800 步至累计 2250 步，每卡 batch=1/累积2、全局 batch=4；两个 rank 峰值 allocated 分别约 6.44/6.43 GiB，reserved 约 6.55/6.56 GiB。作业约 18.9 分钟，期间有共享 GPU 的只读评测，不能将此时间直接作为单双卡吞吐比较。详见 [双卡报告](experiments/ddp_extension_2026-11-05.md)。

高显存配置 `configs/tuning/high_memory_base.yaml` 使用最多32帧、224 RGB+16 motion tokens、每卡batch=2、累积1、完整序列反向，开启无参数记忆归一化。首组450步实测：rank0 allocated 28.16 GiB（30.23 GB）、reserved 28.79 GiB；rank1 allocated 28.14 GiB，约12分钟。两骨干保持冻结；该测量不代表全参数训练的显存需求。

| 阶段 | 配置 | 用途 |
|---|---|---|
| 软件验证 | tiny / 2x5090_smoke | 不下载骨干，验证 loss/backward、DDP、保存恢复、因果性与指标 |
| 约 2B 对齐 | 2x5090_2b | 冻结两骨干，单卡 batch=1，梯度累积，短 TBPTT；先实测峰值再扩帧 |
| 中等训练 | 4xa100_40gb / 80gb | 提升有效 batch 与帧数；配置是起点，尚不构成显存保证 |

总参数约 2B 的 BF16 权重本身约 4GB；CUDA、冻结 LLM 反向到 prefix 所需激活、视觉计算、新模块梯度与 Adam 状态都需额外空间。冻结骨干省去其优化器状态，并不意味着整个骨干前向都能 no_grad：语言损失仍需经 LLM 回传到视觉前缀。显存由帧数、分辨率、TBPTT、batch、文本长度共同决定。新模块当前以 FP32 参数配合 BF16 autocast，不能简单按 2 字节/参数估计总训练成本。

当前新增模块与根级 token_types 的结构计数为 67,817,011，FP32 参数、梯度和两份 Adam 状态合计约 1.01 GiB，尚不含激活与临时缓冲。HF 语言骨干用存储 dtype 自动加载；当前视觉骨干加载默认是 FP32，因此整体权重未必全部 BF16。这些结构预算与上述特定四帧配置的显存实测应分别理解。

复算无需下载骨干：`uv run --locked python scripts/parameter_budget.py`。该脚本只统计新模块，用 meta device 构造替身后明确排除两个替身骨干的参数。

存储建议分阶段预留：环境/CUDA 约数 GB，骨干缓存约数 GB；视频试验集 100–300GB，大规模视频训练通常需要 1–5TB 或更多，取决于小时数与码率；处理后帧 cache 往往比原始码流大。数据量未确定前不能承诺精确存储。当前机器创建环境前磁盘可用约 85GB，适合原型和小数据，不宜直接下载 TB 级训练集。

checkpoint 默认保存完整模型与优化器。tiny 很小；正式 2B 全量 checkpoint 即使冻结骨干也会保存其权重，建议评估后实现仅保存可训练模块及锁定骨干 revision，以降低多实验占用。目前默认只保留每个 run 的 last.pt，防止无界累积。

先用实际视频计时 100 个优化步，记录有效样本数、采样帧数与显存，才可估算正式训练 GPU 小时。现有双卡适合验证架构与小规模对齐，4×A100 可承担进一步实验；论文质量更依赖真实对照、有效数据和完整分析，无法单凭 GPU 数保证。

uv/PyTorch 配置参考官方文档：[Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)。仓库锁定 CUDA 12.8 wheel；驱动在宿主机提供。沙箱看不到 /dev/nvidia 时须在可访问宿主设备的运行上下文执行，不能仅通过安装包解决。
