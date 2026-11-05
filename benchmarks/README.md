# 可复现的评测协议

当前论文横向比较方案：**OVO-Bench + StreamingBench + Video-MME 主表，CUVA 专项补充**，暂不依赖自建物业集。参见 [公开论文比较方案](paper_comparison_plan.md) 和 [协议规格](paper_suite.yaml)。下文保留本地原型指标定义，它们不自动等价于官方 benchmark scorer。

当前实现本地统一指标与 Video-MME 标注导入，不把软件 fixture 当成正式 benchmark，也不把登记了来源称为已经复现官方分数。

| 任务 | 已实现指标 | 边界 |
|---|---|---|
| 长视频 QA | text exact match、词元 F1、受约束的选项字母准确率、按任务与时长分组 | 英文 whitespace token-F1 不能代表中文语义或开放式回答质量 |
| 流式告警 | 一对一 event precision/recall/F1、漏报率、误报次数/规则小时、告警延迟 P50/P95 | 依赖人工 known_at、deadline；未完成窗口被删失，不加入 FN |
| 帧级分数 | 所采样帧 AUROC/AP，正确处理 ties 和单类别不可定义情形 | 不是 XD-Violence 官方 dense-frame AP/AUC |
| 成本 | observation compute、query latency、CUDA 峰值显存（训练 rank0） | 不含解码/网络；不是 RTSP 端到端延迟，无时钟加速保证 |

预测文件必须完整覆盖评测样本，禁止缺样本和重复 id。采样点、告警不得晚于观察截止，预测文件不得决定自身真值标签。对历史时刻提问必须使用该时刻的状态快照，不能拿已经看过未来的 working state 查询过去。

完成窗口内相同类别的第一条有效告警匹配事件；提前或重复告警记 FP；持续高分只发一次 rising-edge 告警，可配置 cooldown。对尚未到 deadline 的事件，窗口内告警暂不计 TP/FP，其余告警仍计 FP。这是显式的本地协议，正式研究需要额外分析删失比例及不同 deadline。

```bash
uv run --locked trace-vl evaluate --config configs/tiny.yaml --checkpoint outputs/tiny-train/last.pt --output outputs/test
uv run --locked trace-vl score data/synthetic/manifest.jsonl outputs/test/predictions.jsonl --split test --output outputs/test/rescored.json
```

阈值和 cooldown 只能在验证集确定并冻结；测试集不能用于选阈值。论文至少报告三种子、按 source_group bootstrap 置信区间（脚本待补）、相同采样预算下基线、真实负例观测小时及显存/吞吐。没有上述实验时不能宣称部署可靠性。

资源曲线可执行 `uv run --locked trace-vl profile --config configs/hardware/2x5090_smoke.yaml --lengths 16 64 256 --output outputs/state-profile.json`。该命令使用随机权重/合成输入，仅检查流长度增加时缓存和状态是否保持有界、记录观测计算成本；不包含 QA、视频解码或模型效果。logical_state_bytes 是张量元素字节之和，不是 allocator 实际占用。
