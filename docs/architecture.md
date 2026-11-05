# 从架构设想到可训练实现

TRACE-VL 研究的是：在固定视频状态预算下，用包含变化前后信息的证据胶囊，兼顾流式告警和较晚出现的问题。约 2B 是复用骨干后的总参数规模目标；`inspect` 输出才是本实现实例化后的实际参数量。

```text
RGB frame -> vision tower -> spatial coverage + saliency selection -> RGB bridge
RGB difference -> motion CNN -------------------------------> merge tokens
    -> [same-frame spatial attention -> causal temporal attention -> FFN] x L
    -> gated working state + transition capsules -> bounded event bank
rule embedding + working state + event summary -> alert head at each observation
question embedding -> retrieve past capsules -> evidence resampler -> LLM prefix -> answer
```

## 已实现

- 帧内空间 attention、过去帧的 projected K/V cache 和跨帧时间窗口。位置偏置输入真实 patch 坐标差及 `log1p(时间差)`。新状态只能使用当前帧、过去缓存和上一步状态。
- RGB patch 覆盖采样与特征离差排序；先运行完整 ViT 再选 token，因此没有减少 ViT 自身计算。
- 门控工作状态；每个事件 4 个潜在 token，分别读取之前状态、变化上下文、之后状态和当前视觉细节。`before_valid` 标记是否存在之前观测；它不是对象轨迹或文字事件标注。
- 可选 `normalize_memory: true` 在门控工作状态和新胶囊上应用无参数 LayerNorm，抑制长展开时状态放大；默认 false 保持旧 checkpoint 行为。高显存调参开启此开关，参数量不变。
- FIFO 或预测 utility 的优先保留；固定容量；默认每帧写入是可复现基线。learned-write 有二值 gate、训练时 straight-through surrogate 和写入辅助损失。priority 的离散替换本身不可微，utility head 需真实 `utility_targets` 监督。
- 规则条件告警。未来问答不参加观测与写入。问题仅在读出阶段选择历史事件，附带 `observed_start`、`known_at` 和 before-valid 的时间编码，然后转换为语言模型视觉前缀。
- 文本、告警、写入和可选 utility 损失；TBPTT detach；冻结骨干和可选 LoRA；单机 DDP。

## 尚未实现或未验证

| 设计概念 | 当前边界 |
|---|---|
| codec-native residual / motion vectors | 当前输入是 RGB 差分，尚无码流残差提取与编码器 |
| scout 与自适应采样 | 当前是预先定义截止时刻的采样回放，未实现实时 RTSP 调度 |
| 对象交互、身份关联 | 当前为空间/时空 attention，没有检测、跟踪或专门图网络 |
| 事件语义解释 | capsule 是潜在表征，不能保证每个 token 已学会所命名的语义 |
| 长期因果证据 | 有界状态无法无损保留无限事件；检索到的是压缩状态，必须评估遗忘 |
| utility 蒸馏、反事实训练 | 有监督接口，没有教师生成、对比配对和完整训练配方 |
| 真实 2B 效果与显存 | 已完成缓存骨干上的四帧真实训练与本地验证；更长输入、监控能力和论文 benchmark 仍待验证，见本机实验报告 |
| 论文创新性 | 当前是研究假设，须与 streaming-memory / codec-VLM 最近工作对比并完成消融 |

## 预算与消融

参考尺寸 `d=768, L=4, N=96 RGB+16 motion, K=8 frames, S=32 state slots, C=128 capsules×4 tokens`。BF16 主状态中：事件潜变量约 0.75 MiB；单层 K/V 约 2.625 MiB，4 层约 10.5 MiB；这不含 working state、元数据、张量临时缓冲、反向图、骨干、优化器和显存分配器。训练不是这些推理状态字节数的简单倍数。

新模块及根级 token_types 参数合计 **67,817,011**，明细见 [implementation_budget.json](implementation_budget.json)。修正了旧统计只累加子模块、漏计 token_types 的 1,536 个参数的问题，没有改变模型结构。本机真实加载缓存 HF 权重后，总参数为 **2,104,675,891**：视觉骨干 316,283,904，语言骨干 1,720,574,976，新增参数 67,817,011。HFVision 包含 pooling head；早期提案中的 2.084B 是设计估算，不是当前实测值。真实数据试验见 [实验报告](experiments/activitynetqa_baseline_2026-11-05.md)。

消融提供去除新增 ST blocks、recent-only、summary 与 capsule 等配置。`recent_only` 仍保留短期时序 K/V，移除跨窗口工作状态传递与事件写入；它不是单帧模型。`summary_equal_slots` 控制事件条目数；`summary_equal_tokens` 控制潜在 token 数与读取 token 数，元数据、attention 开销和参数量不完全一致。正式论文须同时报告参数、显存、读取 token、每秒计算时间，而不是只比较槽位数量。

主张的核心可检验问题：胶囊是否在相同资源预算下提高变化证据问答和低误报告警，尤其是在长间隔后仍能区分“曾出现”与“状态已经改变”。不能仅凭多加一个时空 Transformer 声称创新。

TBPTT 会截断跨 chunk 的梯度，后出现的 QA 不能直接监督已经 detach 的旧胶囊写入。正式长视频训练需要中间问题/告警监督、随机截止采样或专门的分段训练策略；不能把支持长期前向状态保留等同于已经解决长期信用分配。
