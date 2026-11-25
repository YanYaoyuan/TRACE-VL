# 独立训练的静态可学习池化对照

目标是判断当前 CausalPatchPool 的收益，是否超过“只学习区域内空间权重”。本轮只增加静态组，不扩大数据、底座或模块深宽。已有独立训练的 Plain / CausalPatchPool 最终权重及逐题预测保留并复用。

## 固定协议

- 原实验：`/data/yyy/tracevl-runs/core-pool-2efc7cc-001`。
- 已完成的观察/机制消融：`/data/yyy/tracevl-runs/core-pool-probe-a68217e-001`。
- 模型：原 Qwen3.5-4B 的逐文件相同副本，读取 `/data/yyy/tracevl-data/core-pool-probe-2efc7cc/model`。
- 训练：原本地 manifest 的 39,997 题，seed 17，4 卡×每卡 microbatch 8，2,500 updates；学习率、保持损失、LoRA 初始值和实际样本顺序与两组参考逐项核对。
- 预算：16 帧、8 个两帧块、每块 4×4 区域，合计 128 个视觉 token。视觉冻结，静态区域打分器和 LM LoRA 联合训练。
- 训练特征复用原 `/data/yyy/tracevl-runs/core-pool-2efc7cc-001/vision-cache`，原生视觉、预处理及缓存代码的 hash 必须相同；运行库也必须与原评测一致。
- 评测：原 OVO 674 题和 NExT-QA 1,744 题，复用 `/data` 上已经逐字节核验的 RGB 与冻结视觉特征，不从 NAS 重复解码视频；每题独立 LM 状态。

## 静态组实际改变什么

`StaticPatchPool` 复用 CausalPatchPool 构造过程、参数布局与空间打分公式，但从初始化起就把上一块的条件置零。静态组的 LoRA 也从原始初始值重新训练，不使用 causal 组的 LoRA。因此它与旧的同 checkpoint `history_off` 消融是两种不同实验。

为了保持随机数消耗和初始空间权重一致，保留了原 predictor 矩阵。该矩阵始终只接收零输入、梯度为零，AdamW 不使用 weight decay，并在 checkpoint 审计中要求它逐位保持初始值。报告单独列出这 9,216 个不活跃参数；不能把它们宣称为有效容量，也不宣称各组部署成本相同。

## 烟测、恢复与自动接续

启动器导出指定 commit，执行 CPU 测试后运行完整流程：

1. 核查两组参考 checkpoint、原 manifest、模型副本、预处理/缓存身份、可用显存与磁盘。
2. 使用完整 manifest 和 **2,500 步的原学习率计划**执行前两步，保存 `checkpoint-000002`。不是另起一个“两步训练配方”，也不缩小为 256 题。
3. 对这两步核对初始 LoRA、两组参考的实际问题 ID、学习率、答案 token 数、teacher 使用数，以及静态模块收到有效梯度、历史始终关闭、predictor 未改变、训练缓存命中。
4. 四卡完成 8 道实际问答烟测，核对旧观察和 128-token 预算，首次缓存与重新编码特征逐元素相等，重复答案以及跨问题后重放一致。烟测不解释与最终参考组的分数差。
5. 烟测通过才从第 2 步的参数、优化器及 RNG 状态恢复到总计 2,500 步，随后完整评测及配对报告。

`TRACE_SMOKE_ONLY=1` 在真实烟测后停止；使用同 commit、同输出目录去掉该变量可继续。完整流程不自动新增训练种子、不扩训到 5,000 步、不打开冻结最终测试，也不改变原来的联合 +2 pp 门槛。

## 启动

在原 A100 仓库 fetch 现有研发分支，从确定提交导出 `scripts/launch_core_static_pool_a100.sh`，设置 `TRACE_REV` 并执行。默认新输出目录为 `/data/yyy/tracevl-runs/core-static-pool-<提交前七位>-001`，可以用 `TRACE_OUT` 指定一个独立目录。源码、日志、checkpoint 和新评测均保存在该目录。

## 报告

- `train-smoke-audit.json` / `smoke-ok.json`：前两步训练及真实问答烟测的凭证。
- `train/static_pool/metrics.jsonl`：实际训练进度，包含每步 `temporal_history_enabled=false`。
- `train-audit.json`：完整 2,500 步与两个参考的配方/顺序/初始化及 checkpoint 核验。
- `eval/<dataset>/static_pool/predictions.jsonl`：完整逐题预测。
- `report.json` / `digest.txt`：三组分数，以及 causal−static、static−plain、causal−plain 的按视频聚类 4,000 次 bootstrap 区间和答案改变情况。
- `reports.tar.gz`：轻量报告与日志包，不打包权重和共享缓存。

单种子开发集结果用于归因。静态组如果解释了多数收益，下一步应停止扩大现有历史打分式；若出现可信的额外历史收益，再决定是否值得复制第二个匹配种子。当前脚本只完成本轮静态对照，不自动做这个后续选择。
