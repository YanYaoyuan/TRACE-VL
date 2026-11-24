# 固定 CausalPatchPool 检查点的机制验证

主线仍是核心视频模型架构。本轮不训练新模型；它回答刚完成的 40K 比较里，微小收益是否依赖压缩器读取上一时刻的区域信息。不能用模块关闭后的表现代替独立训练的静态可学习池化基线。

源实验：`/data/yyy/tracevl-runs/core-pool-2efc7cc-001`。两组训练均已完成 2,500 更新。NExT-QA 时序题的 CausalPatchPool−Plain 为 +0.721 pp，视频配对 95% 区间 [0.172, 1.479]；OVO 回溯为 +0.344 pp，区间 [−0.699, 1.674]。两项都未达到事先设定的 +2 pp，本轮诊断不改变这个决定。

## 三种设置

| 设置 | 压缩器 | 语言模型 LoRA | 模型可见帧 |
|---|---|---|---|
| normal | 恢复原训练参数及历史条件 | 原 causal_pool 检查点 | 同一批 16 帧 |
| history_off | 仅将压缩器的上一时刻条件项置零 | 同一个检查点 | 同一批 16 帧 |
| module_off | 关闭压缩器修正，恢复固定网格池化 | 同一个检查点 | 同一批 16 帧 |

history_off 不是清空语言模型的历史帧。module_off 保留与压缩器共同训练的 LoRA。历史条件贡献、整个压缩器路径贡献、与独立 Plain 的差异必须分别解释。报告复用上轮 Plain 的原始预测作参照，所有控制设置都重新计算。

## 一次执行

在 A100 的 `/data/yyy/TRACE-VL` 中 fetch 分支后，从明确的提交导出 `scripts/launch_core_pool_probe_a100.sh`，并设置 `TRACE_REV`。启动器自动完成 CPU 测试 → 原始实验及检查点核验 → 本地视频/模型与精确抽帧准备 → 四卡烟测 → 四卡完整三设置评测 → 配对报告 → SwanLab 记录。

默认目录：

- 源实验：`/data/yyy/tracevl-runs/core-pool-2efc7cc-001`。
- 本地数据及模型：`/data/yyy/tracevl-data/core-pool-probe-2efc7cc`，可用 `TRACE_DATA` 覆盖。
- 新结果：`/data/yyy/tracevl-runs/core-pool-probe-<提交前七位>-001`，可用 `TRACE_OUT` 覆盖。

视频先按源评测记录的 SHA256 搬到本地，复用原始抽帧索引，不再逐题 ffprobe。抽到的 RGB 必须逐字节符合旧评测的哈希。模型文件也按原始身份搬到本地。相同视觉观察的解码帧及冻结视觉特征共享，每题每种设置使用独立、全新的语言模型状态。缓存不保存 KV 或答案状态。第一次使用的视觉特征缓存还必须与现场重新编码逐元素相等。

默认 CPU 16 路并行（`TRACE_WORKERS`），GPU 0/1/2/3（`TRACE_GPUS`），每卡最低可用 12 GiB（`TRACE_MIN_FREE_GIB`）。磁盘额外保留 30 GiB，并在复制前估计源视频、模型、RGB 和特征缓存空间。`TRACE_PREPARE_ONLY=1` 只准备数据，`TRACE_SMOKE_ONLY=1` 在烟测通过后停止。两种模式使用同一输出目录，重跑完整命令继续。

## 拒绝静默漂移

固定源 manifest、依赖版本、底座文件及最终检查点身份。normal 必须逐题复现旧评测的选项、生成 token、生成文本、帧信息和视觉 token 数；不匹配即停止。烟测还会在切换两个控制设置后再跑 normal，验证开关恢复和会话隔离。所有分组须覆盖相同问题和同样的观察，任何缺题、重复问题或改变帧的情况都不形成完整报告。

输出目录单独保存，已有源结果和权重保持可审计。重启按逐题结果跳过已完成项目；对保存的结果仍执行身份及完整覆盖检查。只有明确的 `VERIFIED POOL MECHANISM DIAGNOSTIC` 才表示本轮全部完成。

## 输出与解释

- `preparation.json`：本地准备清单、唯一视频/观察数。
- `progress-0.json` 等：每张卡当前阶段、完成题数和目标题数。
- `smoke-ok.json`：四卡实际模型烟测及精确回放验证。
- `report.json`、`digest.txt`：各设置得分、视频配对差值及 4,000 次 bootstrap 区间。
- `eval/<数据集>/<设置>/predictions.jsonl`：逐题预测。
- `reports.tar.gz`：报告、身份记录和日志；不打包权重与视觉缓存。
- `swanlab-upload.log`、`.swanlab-upload/receipt.json`：SwanLab SDK 执行及交付模式记录。

配对主对照为 normal−history_off 和 normal−module_off；history_off−module_off 描述静态加权部分。还报告 module_off−上轮 Plain，但它仍混合了 LoRA 训练差异，不能当作独立的训练控制。报告采用当前两个开发集的全部题目，不挑选有利案例。多个探索对照的区间未做多重比较校正，也不能改写上轮预设门槛。

`answer_changes` 对总体、主指标和 no-harm 子集分别记录选项改变数、生成 token 改变数、双向正确性变化及改变的问题 ID。准确率差为零可能来自收益和损失互相抵消，不能据此认定历史输入未影响答案。源实验、输出和暂存数据目录必须互不包含；代码身份覆盖推理、评分及报告依赖，续跑拒绝代码漂移。

若去掉历史条件后收益没有可检测变化，下一轮不直接加深或扩训这版历史条件结构；先把它作为静态池化参考。若能观察到稳定的历史依赖，再设计显式前后证据交互，并加入独立训练、同数据同预算的静态可学习池化对照。任何诊断都不能单独证明新的论文贡献成立。
