# 本机第一次真实视频训练

建议选择 LLaVA-Video-178K 的 0–30 秒 ActivityNetQA 子集：一个视频归档和一份开放问答标注，下载合计 **928,035,564 bytes，约 0.93 GB**。这是用于训练的视频问答来源，适合先验证真实媒体解码、问答损失、参数更新和 checkpoint 推理。

官方文件目录：https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main/0_30_s_activitynetqa 。本仓库已绑定下载版本，容量依据本地来源审计。官方卡片同时声明学术研究/教育用途限制，原视频仍遵循来源条款；本次建议用于论文研究试验。

2026-11-05 已完成下载、解压和现有约 2B 架构的真实训练试验：原始归档有 **295 个视频、2,950 条问答**，SHA256 与官方记录一致，解压文件内容合计 **942,899,451 bytes，约 0.943 GB**。实际选择 200 视频，划分 1,800/200 条训练/验证问答；7 个 MKV 已生成 CFR H.264 副本并通过解码检查。结果见 [实验报告](experiments/activitynetqa_baseline_2026-11-05.md)。

当前本机直接复用处理后的数据和缓存模型，使用 configs/local_2b_pilot_cfr.yaml（100 步）或 configs/local_2b_epoch1.yaml（450 步完整一轮）。新建输出目录执行，不要把不同实验追加到同一 metrics.jsonl：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked trace-vl train \
  --config configs/local_2b_epoch1.yaml --output outputs/epoch1-new-run
```

这轮先做约 2B 模型的短视频对齐试验。保留 Qwen3-1.7B、SigLIP2 与当前时空/记忆/读取模块，冻结视觉和语言骨干，训练新增模块。默认每个视频最多 4 帧、每卡 batch=1、梯度累积 4、BF16、100 个优化器步。单卡有效 batch=4，双卡 DDP 有效 batch=8；每张卡各持完整模型，显存不能简单相加。该配置已通过 5 步检查；100 步实测峰值 allocated 6.18 GiB、reserved 6.92 GiB，训练循环含保存约 66.7 秒，不含模型加载。

标注转换最多选择 200 个本地可用视频，约 90%/10% 划分 train/val；问答条数根据实际标注统计，多轮对话按 turn 展开，保留先前对话上下文。按解析后的同一视频路径分组，避免其不同问答跨集合。这里的 val 是本地管线验证集，不是论文 benchmark；未来正式比较还须核对原始长视频身份、近重复与 benchmark 来源重叠。

没有告警事件时间标注，因此不生成 events=[]，也不训练告警、写入和 utility 辅助头。四帧短视频试验不检验长视频记忆或告警效果。100 步用于检查训练是否正常，不保证问答已经收敛。

## 存储与硬件预算

本次检查时工作盘可用约 67 GiB。当前 uv 环境已安装，建议为此次试验保留 **40–50 GB**，现有空间可以开始，无需先删除项目文件。

| 项目 | 新增空间预算 | 依据 |
|---|---:|---|
| 压缩视频与标注 | 0.93 GB | 官方文件字节数 |
| 解压视频 | 文件内容约 0.943 GB，另留文件系统余量与 CFR 副本 | 已解压并读取当前试验使用的视频 |
| 预训练模型缓存 | 原始仓库约 7.65 GB，可预留 8–12 GB | 本机已缓存并成功离线加载 |
| checkpoint 与原子替换临时文件 | 预留 12–16 GB | 包含冻结骨干与优化器；写入时可能同时存在 last.pt 和临时文件 |
| 日志与其他余量 | 预留 5–10 GB | 估算；多次试验会继续占用空间 |

两张 32 GB RTX 5090 可作为冻结骨干试验的硬件起点。先单卡检查实际峰值显存，再决定双卡；不承诺未经测量的完整模型显存或训练耗时。若 OOM，先把帧数改为 2、text_limit 改为 128，保留模型宽度；减小梯度累积不会降低单个样本的峰值激活。首次性能结论以 outputs 下 resources.json 和运行时 nvidia-smi 为准。

## 按顺序运行

以下均从 TRACE-VL 仓库根目录执行。数据下载、解压和本机真实训练已经执行；以下原始数据步骤保留供其他机器准备数据。MKV 可靠定位需要 CFR 转换，处理命令与哈希见 data/processed/llava-pilot-cfr.preprocessing.json，本机应优先使用上方已处理清单。

环境已经配置过时可以跳过同步：

```bash
uv sync --locked --extra dev --extra hf --extra data
```

先离线查看下载清单；不访问网络、不创建数据目录：

```bash
env -u PYTHONPATH uv run --locked python scripts/download_datasets.py llava-activitynet-pilot --show-files
```

准备开始时，显式加 --download；仅获取上述两个文件，下载总量上限 2 GB，并至少留 30 GB 空闲。这个检查只约束数据传输，不覆盖后续模型缓存和解压开销：

```bash
env -u PYTHONPATH uv run --locked python scripts/download_datasets.py llava-activitynet-pilot --download --max-gb 2 --reserve-gb 30
```

本机代理环境下，若出现不支持 socks://、Xet 停滞或 Python TLS 中断，可改用已加入脚本的 curl 通道。下面只在当前进程忽略 ALL_PROXY，仍保留 HTTP_PROXY/HTTPS_PROXY；不修改全局设置。curl 通道用于公开文件，支持断点续传，完成后同样校验预期字节数和哈希：

```bash
env -u PYTHONPATH -u ALL_PROXY -u all_proxy uv run --locked python scripts/download_datasets.py \
  llava-activitynet-pilot --download --transport curl --max-gb 2 --reserve-gb 30
```

解压到独立媒体目录。已确认归档内包含 ActivityNet-QA/... 相对路径，下面的 media 可直接作为标注转换器的 video-root；逐个视频是否可解码仍需后续检查。若单独解压失败，停止排查，不自动追加下载其他数据：

```bash
archive=data/raw/llava-video-178k/0_30_s_activitynetqa/0_30_s_activitynetqa_videos_1.tar.gz
media=data/raw/llava-video-178k/activitynet-media
mkdir -p "$media"
tar -xzf "$archive" -C "$media"
```

转换训练清单。--video-root 要指向能与标注中 video 相对路径直接拼接的目录；如果解压产生额外顶层文件夹，请调整此参数。转换器只保留本地存在的视频，少于两个视频时直接报错，不生成空训练集；完整数量见旁边的 .report.json：

```bash
env -u PYTHONPATH uv run --locked trace-vl data import-llava \
  data/raw/llava-video-178k/0_30_s_activitynetqa/0_30_s_activitynetqa_oe_qa_processed.json \
  --video-root "$media" \
  --max-videos 200 --val-fraction 0.1 --seed 7 \
  --output data/processed/llava-pilot.jsonl
env -u PYTHONPATH uv run --locked trace-vl data validate data/processed/llava-pilot.jsonl
```

validate 只验证标注结构与集合隔离，不逐一解码全部视频。现有视频读取器假设恒定帧率；若遇到 VFR 媒体，应先转为 CFR，再进行时间相关实验。

先单卡 5 步。首次构造 HF 模型会额外下载骨干，不包含在上面的 0.93 GB 中：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl train \
  --config configs/local_2b_pilot.yaml --max-steps 5 --output outputs/llava-preflight
```

确认损失有限、显存正常、last.pt 保存成功后，再单卡继续到第 100 步：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl train \
  --config configs/local_2b_pilot.yaml --resume outputs/llava-preflight/last.pt \
  --output outputs/llava-preflight
```

也可以用双卡重新开始一次试验；它会产生另一份输出。两张卡并不意味着自动有更低的每卡显存：

```bash
bash scripts/train_2x5090.sh configs/local_2b_pilot.yaml outputs/llava-ddp-pilot
```

对单卡试验的本地验证集生成答案：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl evaluate \
  --config configs/local_2b_pilot.yaml --checkpoint outputs/llava-preflight/last.pt \
  --split val --output outputs/llava-pilot-val
```

查看 metrics.jsonl、resources.json 与 predictions.jsonl，确认优化器完成预定步数、loss/grad_norm 有限，并人工检查答案。开放回答的词面匹配分数仅用于联调，不能代替官方论文评测。OVO、StreamingBench 和 Video-MME 继续作为保留评测集，不加入本次训练。

后续要验证监控告警，可以再考虑约 1.89 GB 的 CCTV-Anomaly 单个训练分片；它需要先获得 HF 访问权限，标注转换和告警监督协议也需要另行适配。当前不作为第一次试验的前置条件。
