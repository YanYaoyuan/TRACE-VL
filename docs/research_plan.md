# 研究路线与论文实验

核心假设：把状态改变作为显式记忆单位，在固定缓存和历史 token 预算下，可改善“变化前后”的可追溯问答与流式告警。高质量论文需要证明这个假设，而不是仅给模块取名或增加 Transformer。

2026-11-05 当前阶段调整：**优先公开论文横向比较，暂不以自建物业集为前置条件**。主表为 OVO、StreamingBench、Video-MME，CUVA 作为监控理解专项。具体基线、因果约束与官方代码版本以 [公开评测方案](../benchmarks/paper_comparison_plan.md) 为准。下列自建场景工作保留为后续扩展。

1. 数据与基线：按摄像头/session 划分真实监控；冻结测试集；先跑复用预训练 VLM 的均匀帧、recent-window、单向量摘要和 FIFO；long-QA 与 streaming-alert 分开汇报。
2. 对齐：冻结视觉/语言骨干，用真实图文/视频 QA 训练 bridge、ST、memory、reader/projector。分阶段选数据与损失权重；当前训练器提供各项 loss，但未实现自动多阶段调度。
3. 写入和证据：标注 known_at 和可接受告警窗口，比较 every-frame、supervised learned-write；加入丢弃后对答案的影响等 utility 教师目标后，再测试 priority 保留。
4. 控制变量：比较 `no_temporal`、`recent_only`、等条目摘要、等 token 摘要、完整 capsule；额外补充去掉 before/change/after/detail 的结构消融、参数匹配替代模块及随机检索基线。
5. 压力测试：间隔从 30 秒扩到 30 分钟；遮挡、离场再进入、多次相似事件、记忆溢出、摄像头域偏移；画性能随状态字节与延迟变化的曲线。
6. 公共评测：Video-MME 视觉长 QA；完成 OVO/StreamingBench 官方任务适配后再报其分数；XD-Violence 须单独实现官方 dense-frame pipeline，不能把采样帧 AP 写成其官方成绩。

所有表格保留 manifest/config/hash、seed、checkpoint、实际参数量、训练时长和 GPU 型号；保留失败案例。不以 synthetic 的 QA 提升或 tiny 的速度推断 2B 模型效果。不保证论文录用；创新性需要与上级 DOC 中现有流式记忆、token compression、主动响应研究逐项核对。

最先做的有价值实验：相同 32 个历史 latent tokens 下，4×8 capsule 对比 1×32 summary；问题只在长间隔后给定，包含“拿着→放下→离开”及看似相同但人仍在的困难负例。控制视觉采样、训练数据、语言骨干和读取预算，检验证据结构是否改善变化判断，而非单纯增加容量。
