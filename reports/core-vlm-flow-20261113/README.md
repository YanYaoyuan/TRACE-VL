# 核心视频 QA 真实模型训练检查（2026-11-13）

本轮训练路径是 **真实视频 → Qwen3.5-4B 冻结视觉编码器 → 全视觉 token / 固定空间池化 →
Qwen 语言模型 LoRA → 答案 token 交叉熵（含 EOS）**。没有告警头、响应时间标签、teacher KD 或保留评分器。
这是主模型训练链路验证，不是外部 benchmark、泛化验证或新架构贡献。

## 输入和设置

- 使用现存 LLaVA-Video ActivityNet-QA 派生清单的 `train` split，按 ID 排序选择前两条。
  两条为不同真实视频的运动识别问题，答案为 `badminton` / `tennis`。
  保留源视频 SHA256、实际帧 PTS 和帧索引；完整与池化 8 帧臂的帧清单相同。
  ActivityNet 数据仅用于这次训练流程检查，不授权将其加入以同源 benchmark 为目标的正式训练。
- Qwen3.5-4B 本地预训练权重，bf16，SDPA，冻结原始参数，LM attention/Gated DeltaNet 投影 LoRA。
  rank=8，alpha=16，dropout=0，AdamW lr=1e-4、weight_decay=0，非重入梯度检查点，seed=7。
- 全部可训练参数为 LM LoRA，共 6,193,152 个。每步 1 个问题，交替使用两个问题。
- torch 2.8.0+cu128，transformers 5.20.2，peft 0.21.2；每臂单张 RTX 5090。
  8 帧 grid 用物理 GPU 0，8 帧 full 用物理 GPU 1，32 帧 full 用物理 GPU 0；不是 DDP 测试。
- 使用 TRACE 显式 embedding/position 输入协议，不是官方 Qwen native video-chat 复现。
  视觉特征预先计算并缓存；以下显存含模型及本轮缓存，不代表在线端到端吞吐。

## 实测

| 路径 | 实际帧数 | 每帧目标像素预算 | 输入视觉 tokens | 更新数 | 训练样本平均 CE：前 → 后 | PyTorch 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| grid，4×5 空间区域 | 8 | 256² | 80 | 4 | 0.13120 → 0.00421 | 8.77 |
| full | 8 | 256² | 264 | 4 | 0.09726 → 0.00209 | 8.89 |
| full，较长输入检查 | 32 | 448² | 3040 | 2 | 0.12006 → 0.04325 | 10.75 |

像素预算按原始宽高比及 32 倍数取整，实际尺寸见 `inputs.json`；并非每帧都是正方形。
三臂的 LoRA 梯度有限且非零，参数有实际更新；保存后的 adapter 逐 tensor 重载完全相同，
重载前后的答案 CE 差为 0，贪心生成也相同。

**两个样本训练前已经答对。** 因此表中损失下降只说明已知答案的概率提高，不能写成理解精度提升。
把视觉特征置零会改变答案 CE，说明路径使用了视觉输入；这也不证明可靠时序理解。
没有开发集、最终测试集、配对涨点区间或多 seed 结论。
32 帧配置只提供本机显存参考，不能保证长答案、大 batch、长视频或 A100 的相同性能。

## 文件和复现

- `grid/`、`full/`、`full32/`：原始 `inputs.json`、`metrics.jsonl`、`report.json`。
- `executed-source/`：这三次实际执行的 QA、数据和 flow 源码。
- `summary.json`：摘要及上述文件 SHA256。
- 权重保存在用户指定的大文件目录：
  `/media/user/50D2BD30D2BD1AE4/WorkSpace/TRACE-VL/core-vlm-runs/qa-flow-4b-*-20261113-001/adapter/`。
  仓库只保存小型报告，不提交预训练权重和 adapter。

在已安装项目 VLM 依赖及 ffmpeg/ffprobe 的环境中，
从工程目录执行；输出目录必须是新目录：

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
  python -m trace_vl.core_vlm.flow \
  --model /path/to/Qwen3.5-4B \
  --manifest /path/to/llava-pilot-cfr.jsonl \
  --out /path/to/new-flow-run \
  --mode grid --steps 4 --pixels 65536 --examples 2 --updates 4
```

`--mode full` 为完整视觉 token；`--steps 16 --pixels 200704` 为 32 帧检查。
完成标记 `VERIFIED core video-QA training and reload` 只认证此处列出的工程检查。

下一步仍是原生 Qwen 对照、正式训练数据排重、分布式 QA 训练与开发集评测；
随后才冻结架构改动和对照实验，并对齐 Mage-VL / MOSS-VL 的评测协议。
