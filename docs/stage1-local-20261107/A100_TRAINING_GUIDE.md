# TRACE-VL：本机验证后的 A100 训练步骤（2026-11-07）

推荐代码版本：`d8b260133005cd1ecb629c5f182c2c11f545962b`。本机200步主实验固定在 `a9ab454`；两版训练、模型、读取器、UV锁文件的等价性已核对。d8修复了真实选择题标注中的问题ID冲突，并在本机通过90项CPU/CUDA测试。配套云端启动脚本已在双5090上端到端演练；A100硬件本身尚未实测，先执行下面的硬件检查和短程校准。

本机实验及曲线见同目录 `LOCAL_TRAINING_REPORT.md`、`training_curves.png`。A100先做Stage 1；Stage 2长流训练器尚未实现。

## 1. 使用验证过的代码与UV环境

代码包 `TRACE-VL-A100-d8b2601-20261107.tar.gz` 内含上游源码、`tools/stage1/`辅助脚本和验证报告。将代码包上传至A100机器，然后执行：

```bash
tar -xzf TRACE-VL-A100-d8b2601-20261107.tar.gz
cd TRACE-VL
uv sync --python 3.10 --locked --extra dev --extra vlm --extra data
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
nvidia-smi
uv run --locked --no-sync python - <<'PY'
import torch, transformers, peft
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 4
print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i,p.name,p.total_memory/1e9,'GB',torch.cuda.get_device_capability(i))
PY
uv run --locked --no-sync pytest -q tests/test_v3.py tests/test_v3_gpu.py tools/stage1/test_stage1_data.py
```

也可以直接克隆已经配齐辅助脚本的实验分支：

```bash
git clone --branch experiment/stage1-local-a100-20261107 --single-branch https://github.com/YanYaoyuan/TRACE-VL.git
cd TRACE-VL
git rev-parse HEAD
```

实验分支基于上述d8上游版本，附带 `tools/stage1/` 和本次报告。先使用锁定环境和默认GDN回退实现；优化核可在后续单独环境验证。这样不会在首次迁移时同时改变代码、Torch版本和内核实现。本机5090使用过的禁用P2P/IB设置不是A100性能配置，A100先使用平台默认NCCL设置。

## 2. 指定模型与视频路径

```bash
export TRACE_MODEL_DIR=/data/models/Qwen3.5-2B
export TRACE_LLAVA_ROOT=/data/llava-video
export TRACE_VIDEO_ROOT=/data/llava-video
export TRACE_DATA_DIR=/data/tracevl-data/stage1-30s-v1
```

`TRACE_MODEL_DIR`应包含Qwen3.5-2B的完整多模态权重、config及tokenizer，约4.6GB；`TRACE_VIDEO_ROOT / 标注中的video字段`应直接指向可读取的视频。解压目录可能与上例不同，先用一个实际文件确认。

如需下载模型，使用本机验证过的固定版本：

```bash
uv run --locked --no-sync python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.5-2B',
    revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
    local_dir=os.environ['TRACE_MODEL_DIR'],
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],
)
PY
```

下载依赖A100机器能访问Hugging Face。需要代理时使用该机器可达的HTTP(S)代理地址；本机的 `127.0.0.1:7890` 地址不能直接作为云端代理。

## 3. 准备真实多选题训练数据

先从已下载的0–30秒YouTube MC子集开始。官方固定数据版本为 `lmms-lab/LLaVA-Video-178K@6d8c562dc26d70042a0d9704d1cae58c94b89098`，标注路径为：

`0_30_s_youtube_v0_1/0_30_s_youtube_mc_v0_1_qa_processed.json`

标注和对应视频均需存在。已有的 `*_oe_*` 是开放式问答，当前Stage 1入口使用选择题。本机的二选一转换用于代码诊断；正式云端训练使用原生多选题。用于验证转换器的NExT-QA标注没有进入本轮训练；准备论文训练集时应检查与最终benchmark的上游视频重叠。

若还没有该子集，可先下载标注和首个视频包；首包压缩大小约5.30GB，解压视频另占空间。Hugging Face下载器会利用本地缓存，中断后重跑同一命令：

```bash
uv run --locked --no-sync python - <<'PY'
import os
from huggingface_hub import hf_hub_download
bucket='0_30_s_youtube_v0_1'
for name in ['0_30_s_youtube_mc_v0_1_qa_processed.json','0_30_s_youtube_v0_1_videos_1.tar.gz']:
    hf_hub_download(repo_id='lmms-lab/LLaVA-Video-178K',repo_type='dataset',
        revision='6d8c562dc26d70042a0d9704d1cae58c94b89098',
        filename=bucket+'/'+name,local_dir=os.environ['TRACE_LLAVA_ROOT'])
PY
mkdir -p "$TRACE_VIDEO_ROOT"
tar -xzf "$TRACE_LLAVA_ROOT/0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz" -C "$TRACE_VIDEO_ROOT"
```

解压后按标注中的相对路径核对根目录；如果首包对应的可用问题不足1,000，再下载后续 `videos_2.tar.gz` 等包。准备脚本会只保留已有媒体对应的问题，并报告缺失数量。

```bash
uv run --locked --no-sync python tools/stage1/stage1_data.py \
  --annotations "$TRACE_LLAVA_ROOT/0_30_s_youtube_v0_1/0_30_s_youtube_mc_v0_1_qa_processed.json" \
  --video-root "$TRACE_VIDEO_ROOT" \
  --output "$TRACE_DATA_DIR" --max-seconds 30 --min-records 1000 \
  --max-train-questions 5000 --max-val-questions 500
```

输出目录必须是新目录，内容为：

- `train.jsonl`：训练使用。
- `val.jsonl`：固定留出验证使用。
- `all.jsonl`：两者合并，仅供审计。
- `audit.json`：唯一问题ID、视频隔离、标注SHA256、样本数、标签分布、缺失/超长/解码失败情况。

转换器按视频划分train/val，修复上游重复问题ID，删除精确重复和冲突答案组。辅助脚本再检查ID、媒体可读性及长度，并以固定种子按完整视频组选择不超过5,000条训练/500条验证问题，避免首次校准就评测完整大子集。数据缺失或被排除过多会导致最少样本数检查失败；修正路径或补齐视频后，用新输出目录重跑。只把 `train.jsonl` 交给训练，评测传 `val.jsonl`。

本机已对上述完整YouTube MC标注做格式审计：解析167,927道唯一问题、9,977个视频，删除21条精确重复，未发现解析失败。标注中的实际视频路径示例为 `liwei_youtube_videos/videos/youtube_video_2024/ytb_....mp4`，因此视频根目录应指向该路径前缀的上一级，而不是仅凭分桶目录名猜测。该审计不含YouTube视频下载或训练。

对该标注全部9,977个上游视频ID与Video-MME的900个视频ID做了文件名规范化精确匹配，未发现交集；这不等同于重剪辑或重上传内容的感知去重。首次仍按上面的固定小子集进行数据和硬件校准。

## 4. 先用4卡、同本机有效batch跑100步

本机有效batch为 `2卡 × 累积2 = 4`。A100初次使用 `4卡 × 累积1 = 4`，保留同样的学习率。模型、CST维度、视觉分辨率保持本机配置。

```bash
uv run --locked --no-sync python tools/stage1/configure_stage1.py \
  --model "$TRACE_MODEL_DIR" --output configs/v3/a100-pilot-local.yaml \
  --steps 100 --accum 1 --warmup 10 --save-every 50

export TRACE_CONFIG="$PWD/configs/v3/a100-pilot-local.yaml"
export TRACE_RECORDS="$TRACE_DATA_DIR/train.jsonl"
export TRACE_RUN=/data/tracevl-runs/a100-pilot-100
export TRACE_GPUS=4 TRACE_CLIP_SECONDS=30 TRACE_STEPS=100
```

先用4步小型输入做A100 FP32等价性检查，保持这一步的显存开销可控。真实视频读取和训练在随后100步中校准：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-sync python -m trace_vl v3 check \
  --config "$TRACE_CONFIG" --steps 4 --dtype float32 --device cuda:0
```

要求 `tokens_equal=true`、`pass=true`。然后训练：

```bash
bash tools/stage1/cloud_train.sh
```

可在tmux会话内运行以保留SSH断开后的进程。检查 `TRACE_RUN.log`、每个 `rank-*-resources.json`：梯度和loss有限、所有rank完成、每卡峰值有余量。记录100步真实耗时后再估算GPU小时。本机30秒视频约7.4GB与180秒合成样例约14.45GB，不能替代真实A100长视频校准。

## 5. 重载检查点验证

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-sync python -m trace_vl v3 eval \
  --config "$TRACE_CONFIG" --records "$TRACE_DATA_DIR/val.jsonl" \
  --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_RUN/val-step100" --mode v3 \
  --checkpoint "$TRACE_RUN/checkpoints/step-00100/modules.pt" \
  --adapter "$TRACE_RUN/checkpoints/step-00100/lora" --device cuda:0

uv run --locked --no-sync python tools/stage1/validate_and_score.py \
  --records "$TRACE_DATA_DIR/val.jsonl" --predictions "$TRACE_RUN/val-step100" \
  --output "$TRACE_RUN/val-step100/score.json"
```

核对 `errors.0of1.json` 为空、预测条数等于 `audit.json` 的val_questions，并检查唯一ID覆盖。Stage 1验证保持full记忆，先隔离token压缩影响；长期KV策略属于下一阶段实验。

## 6. 再开一轮正式训练

100步确认A100环境和数据正常后，用新目录从Qwen3.5-2B骨干初始化新模块与LoRA，启动第一轮1000步实验：

```bash
uv run --locked --no-sync python tools/stage1/configure_stage1.py \
  --model "$TRACE_MODEL_DIR" --output configs/v3/a100-stage1-local.yaml \
  --steps 1000 --accum 4 --warmup 50 --save-every 250

export TRACE_CONFIG="$PWD/configs/v3/a100-stage1-local.yaml"
export TRACE_RUN=/data/tracevl-runs/a100-stage1-1000
export TRACE_GPUS=4 TRACE_CLIP_SECONDS=30 TRACE_STEPS=1000
bash tools/stage1/cloud_train.sh
```

有效batch为16，共16,000次随机抽样，训练器采用有放回抽样，不是完整遍历的固定epoch。当前实现每卡顺序处理一个clip，grad_accum改变有效batch，不会自动变成批量视频前向。根据实际训练样本数和验证曲线决定下一轮步数。每个保存点保留成对的 `modules.pt + lora/ + checkpoint.json`，并保留step 0初始化状态。本机每对约324MiB；保存间隔和训练轮数会影响存储用量。

扩大到60/120/180秒时，重新准备对应 `--max-seconds` 的数据清单；先在最长的真实样例上做一两步校准，再多卡训练。对整段视频的问题，输入窗口应覆盖整段标注视频，避免截掉答案依据。

当前保存的是模块与LoRA权重，**不含优化器/RNG的完整续训状态**。现有 `v3 train --init ... --adapter ...` 可做权重热启动，但学习率、优化器和随机状态会重置，应记录为新实验。实例重启后的恢复不能当作精确断点续训。

每轮固定记录代码SHA、数据SHA、模型版本、随机种子、有效batch、视觉tokens、KV、训练时长和每卡显存。正式效果对比至少包括全token、同预算池化和TRACE-VL；原生模型对照与训练后模型使用一致的数据和评测方法。本机小样本实验尚未证明架构精度优势。
