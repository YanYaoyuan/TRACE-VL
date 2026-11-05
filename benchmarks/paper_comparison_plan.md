# 公开论文 benchmark 与对比方案

2026-11-05 更新：当前阶段以公开论文横向比较为目标，自建物业数据暂不纳入实验依赖。不下载数据或模型，不启动跑分。可机读方案见 [paper_suite.yaml](paper_suite.yaml)；它是协议规格，不是现有 train/evaluate CLI 可执行配置。

## 主表确定为三套公开 benchmark

| benchmark | 主张与报告方式 | 来源与当前接入状态 |
|---|---|---|
| **OVO-Bench** | 实时感知、历史回溯、等待证据后的主动响应；按官方当前版分别报告三类及其官方聚合 | [官方代码](https://github.com/JoeLeelyf/OVO-Bench)，[数据](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench)。适配器待实现 |
| **StreamingBench** | 流式理解与时机；视觉/上下文子任务及主动响应分别报告；音频能力未实现时不宣称 omni 总分 | [官方代码](https://github.com/THUNLP-MT/StreamingBench)，[数据](https://huggingface.co/datasets/mjuicem/StreamingBench)。适配器待实现 |
| **Video-MME** | 通用视频理解回归；主实验视觉无字幕/无音频；总准确率与 short/medium/long 分组 | [官方代码](https://github.com/MME-Benchmarks/Video-MME)，[数据](https://huggingface.co/datasets/lmms-eval/Video-MME)。已有标注导入和本地 MCQ 解析，官方输出/完整评测尚待接入 |

**专项扩展：CUVA。** 用于监控异常类型、原因、后果的语言理解比较，按[作者协议](https://github.com/fesvhtr/CUVA)与 judge 配置报告；不把解释分数当成连续告警误报率。

FastBench 后续补充高动态感知；若论文主张小时级记忆，增加 StreamArena 及其长距离回溯分组。UCF-Crime/XD-Violence 的 dense-frame 异常检测属于另一条评测管线，本轮不并入流式问答主表。

下载入口与容量仍见 [数据清单](../docs/dataset_download_plan.md)。选择完整论文基准，不意味着当前 67GB 空闲盘能够同时存放全量媒体；小分片仅用于接口联调，不生成可当作官方完整成绩的数字。

## 对比模型

| 组别 | 已核验的公开 checkpoint | 作用与限制 |
|---|---|---|
| 约 2B 同规模 | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 第一优先复现的通用视频 VLM 基线 |
| 约 2B 同规模 | [InternVL3.5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B) | 第二模型族，避免只与同一家族比较 |
| 约 2B 同规模 | [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | 小规模视频模型补充对照 |
| 更大参考 | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 展示准确率与资源的取舍，不能视为参数完全匹配 |
| 流式方法 | [Mage-VL](https://huggingface.co/microsoft/Mage-VL)，[推理说明](https://github.com/microsoft/Mage/blob/main/mage_vl/README.md) | 最初研究的直接方法参照；RGB 与 codec-native 路径分开记录，官方任务覆盖需复现核对 |
| 较大流式参考 | [JoyAI-VL-Interaction](https://huggingface.co/jdopensource/JoyAI-VL-Interaction) | 8B 级，评测主动响应；如启用后台教师/委派，必须报告其成本，不能只记前端参数 |

本轮只核验公开模型卡，没有下载权重、验证依赖兼容或运行这些基线。模型名中 2B/4B 为发布标签，最终表格应审计唯一参数总量，区分语言/视觉/可训练参数。当前 TRACE-VL 的完整约 2B 权重仍未训练，不能从 tiny 软件验证产生主表成绩。

Jev-Omni 可以作为原生选择题子任务的决策专项对照。其单次候选分类接口不能覆盖所有开放回答任务；不得从答案构造新候选项，再把改题后的分数混入原始官方榜单。原生开放任务仍需使用语言回答路径，或明确报告不支持/覆盖率。

## 主表和消融表如何区分

1. **同规模模型表：** TRACE-VL、Qwen3-VL-2B、InternVL3.5-2B、SmolVLM2-2.2B。在统一可见视频时间线与明确预算下重跑。它衡量整体方案，不能把预训练数据差异造成的提升全部归因于架构。
2. **流式模型表：** Mage、JoyAI 等已发表方法。分别列实际复现结果和原论文报告结果，保留协议、任务覆盖及输入模态。只复现部分任务时标记 N/A，不借其他协议数字补空。
3. **同骨干结构消融：** 相同训练数据、语言/视觉骨干与训练预算，比较新增 ST 去除、recent-window、等 latent token 摘要、capsule FIFO/learned-write，以及现有 MLP 与候选条件决策头。该表才用于检验架构贡献；尚未实现的变体必须先实现验证，不能把名字当作已有实验。

## 需要锁定的协议

- **OVO：** 作者强调当前代码与早期 arXiv 设置不同。使用本方案固定的代码和标注版本。Forward Active Responding 允许提问后继续接收视频，但每次决策只访问当时已经到达的帧，不能提前加载未来目标时刻的证据。按官方 per-task 脚本评分。
- **StreamingBench：** 官方 Main 为 query 前 60 秒；Full Context 是另一设置。60 秒实验必须在窗口起点重新初始化所有缓存和 latent memory，不能让工作状态暗中携带更早视频。全历史结果单列。音频和字幕只在对应的额外设置中启用。
- **Video-MME：** 本方案主实验无字幕、无音频；固定采样预算，按官方格式与答案解析器评分。已有本地 exact match/token-F1 不是所有官方任务的替代实现。
- **输入预算：** 可先预设 32/64/128 帧三档作为资源曲线，最终档位在外部开发数据上确定。共同 RGB 采样控制组记录帧时间、分辨率、crop、实际视觉 tokens；不同编码器的 native resolution 不等于相同像素/计算量。原生 codec、adaptive sampling 的作者设置另表，不能标成完全相同视觉输入。
- **因果性：** 当前时刻的问题使用对应状态快照；不得访问未来帧、参考回答、GT evidence、目标响应时刻或 focus timestamp 来安排默认采样。问题是否预先给定遵循各官方任务，不统一改造为 question-aware 预筛选。
- **评分与覆盖：** 使用同一官方 scorer、相同 judge/版本/提示与选项解析。保持原始题目、选项和测试划分；不能漏掉困难任务后仍报 overall。接口 pilot 只查路径、输出和 OOM，不依据其正确率选择最终超参数。
- **计算成本：** 相同 GPU 分别记录 ingest 累计计算、每次查询 P50/P95、峰值 allocated/reserved 显存、状态字节、实际 token 数；另测包含解码/I/O 的端到端延迟。多问题共享状态时累计整段视频成本，JEV 式无生成也必须计算视觉前向与输入 prefill 成本。
- **统计与调参：** 训练型 ours/消融使用独立训练种子，建议至少三次；不能把同一 checkpoint 的三次 greedy 推理当作训练方差。按原视频配对 bootstrap，避免把同视频多问题当独立样本。校准、阈值、超参数使用独立开发数据，测试集仅评测。

Mage 原论文中 recent-window 的 OVO 子集/聚合，不能直接与上述完整三模式 total 排序。只有输入模态、任务范围与评分设置一致的结果才放在同一列；不能确认的格子留空。

## 实施顺序与当前状态

1. 先对接 **Video-MME** 官方输出和解析，复用现有标注导入器；用 Qwen3-VL-2B 验证完整 baseline 链路。
2. 对接 **OVO** 官方类与三种模式，添加过去/未来可见性审计。
3. 对接 **StreamingBench** 的 60 秒与全历史两条路径，独立保留 proactive 评分。
4. 主协议锁定后扩大同规模模型和 Mage 对照，再接 CUVA；更大模型及长历史/高动态补充按资源安排。

当前仅完成选型、版本记录与实验规格；**没有新下载、没有训练、没有官方 benchmark 跑分**。paper_suite.yaml 中 results 均为空，不能用作结果表。
