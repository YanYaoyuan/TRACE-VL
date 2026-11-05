# 训练数据与 benchmark 下载清单

核查日期：2026-11-05。目标：约 2B 的物业摄像头模型，兼顾规则告警、事件解释与长视频回溯。

来源核查阶段只查询了官方网页及文件元数据。后续按用户指令，已单独下载 **0.928 GB 的 ActivityNetQA 本机训练子集**并完成哈希与归档检查，详见 [本机训练记录](local_training_pilot.md)；其他清单中的新视频尚未下载。先前已有的 Video-MME 2700 条标注及转换文件保留。脚本测试全部离线，只使用软件 fixture。

建议第一轮采用 **LLaVA 视频能力训练 + CCTV-Anomaly/UCA 监控适配 + 自有物业规则标注**。公开评测以 **CUVA、OVO、StreamingBench、Video-MME** 为主，**FastBench** 检查短暂动作，**StreamArena** 检查长历史。下载优先选择小分片；通过端到端验证后再扩大数据量。

最新评测优先级：当前阶段先做公开论文比较，**主表固定 OVO-Bench + StreamingBench + Video-MME，CUVA 为专项补充**；自建物业集暂不作为依赖。详见 [比较协议与基线模型](../benchmarks/paper_comparison_plan.md)。本文件的容量及训练数据来源仍保留供后续选择。

**只想在本机先跑一次真实训练：** 新增 llava-activitynet-pilot，0–30 秒 ActivityNetQA 的视频归档与标注合计 **0.928 GB**，无须先拉取下表的大组合。转换器、约 2B 四帧配置及完整命令见 [本机训练指南](local_training_pilot.md)。

## 1. 训练与领域适配

| 数据集与官方入口 | 适合训练什么 | 数据形态、规模与分割 | 存储与本轮优先级 |
|---|---|---|---|
| [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | 视觉到语言对齐、视频描述、开放/选项 QA | 约 178K 视频；短/长片段分组，conversations JSON 与视频压缩包分开 | HF 全库文件合计约 **1.281 TB**。优先取 0–30 秒 academic 的 1 个视频分片及三份标注，合计 **5.412 GB**；标注必须与实际媒体求交集 |
| [CCTV-Anomaly](https://huggingface.co/datasets/xiaozhang79/CCTV-Anomaly)，[作者代码](https://github.com/xiaozhang79/CamVLM) | 监控异常类别、事件描述，适合领域 SFT 和决策头监督 | 作者公布 14,133 视频、10 类、234 小时；有 train/test JSONL 与分片视频 | 全库约 **166.35 GB**。需 HF 访问申请；可先取 train_shard_14 和 train 标注，约 **1.892 GB**。必须保留 test 独立 |
| [UCA / UCF-Crime Annotation](https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet)，[实际标注目录](https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet/tree/main/UCF%20Annotation/json) | 监控事件描述、时间定位、证据记忆 | 1,854 个 UCF-Crime 视频，23,542 句带时间区间描述；公开 Train/Val/Test JSON | 三份非空 JSON 合计 **4.61 MB**；视频另取 UCF-Crime。空的 revised_en_UCFCrime_Train.json 不纳入计划。原论文项目页已失效，本清单给出可读数据目录 |
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) | 广泛监控异常与正常背景适配；给 UCA 提供原始视频 | 官方下载包含 anomaly/action 两套划分，不可混用；原始异常训练监督主要是视频级 | 原始 ZIP/分包由官网给出；本轮未核实整包字节数和完整可下载性，不写虚构容量。优先挑与 UCA train 对应的视频 |
| [XD-Violence](https://roc-ng.github.io/XD-Violence/) | 可选的打斗、事故等异常适配；需要时扩展音频 | 4,754 个非裁剪视频、217 小时；训练弱标签、测试时间标注。官网提供 OneDrive、阿里云与部分百度入口 | 未核验原始媒体包总字节。**官网已标记百度训练链接失效**，优先用当前 OneDrive/阿里云。不要把 I3D/VGGish 特征包当成原始视频 |
| [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | 跌倒任务的小规模验证与辅助监督 | 70 段序列：30 跌倒、40 日常活动；RGB/深度图片序列与同步时间 CSV | 只选 RGB 和时间文件即可，先不用深度/加速度；总大小未核验。样本很小且为受控环境，不能代表物业泛化或独立大 benchmark |
| [NExT-QA](https://github.com/doc-doc/NExT-QA) | 可选的时序、因果和对象交互问答训练 | 使用官方 train；QA CSV 在 dataset/nextqa，原始视频来自作者给出的 Google Drive；有 val/test | 原始视频体积未核验。不要下载旧 BERT/外观特征代替本项目的 RGB 媒体；与 LLaVA 中包含的 NExT-QA 来源去重 |
| [JoyAI-VL-Interaction 数据](https://huggingface.co/datasets/jdopensource/JoyAI-VL-Interaction) | 后续研究何时沉默、响应、委派的时间对齐监督 | 本次文件清单为 JSON 标注，作者介绍时间对齐交互样本；还需确认样本对应媒体的完整获取方式 | 全部已发布文件约 **2.647 GB**；**这个数不是完整训练视频容量**。列为扩展参考，不作首轮自包含训练集 |

这里的 SFT 是复用预训练骨干后训练对齐/时空/记忆/决策模块，并非从零训练视觉和语言骨干。第一轮没有必要全量拉取通用大视频库。

## 2. benchmark 选择与下载

| benchmark 与官方下载 | 主要测什么 | 容量与分片方案 | 评测注意 |
|---|---|---|---|
| [CUVA：数据](https://huggingface.co/datasets/fesvhtr/CUVA)，[官方评测](https://github.com/fesvhtr/CUVA) | 异常是什么、原因与后果；贴近事件解释 | HF 全库约 **25.65 GB**；只取两份 parquet 标注约 **0.34 MB**；1 个 group_0 视频包加标注约 **7.568 GB** | `full`/all.parquet 包含 test，**不能把 full 当作独立 train**；统一比较需遵循官方评测和 judge 配置 |
| [OVO-Bench：数据](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench)，[代码/标注](https://github.com/JoeLeelyf/OVO-Bench) | 回溯、实时感知、等待证据后主动响应 | 源视频 5 个 tar 分卷共 **46.342 GB**；已切片版本 15 卷共 **153.277 GB**，两条路线任选一条 | 标注 `data/ovo_bench_new.json` 在 GitHub；所有同名分卷须完整，单下 partaa 不能解压成完整数据；必须绑定当前官方协议 |
| [StreamingBench：数据](https://huggingface.co/datasets/mjuicem/StreamingBench)，[评测代码](https://github.com/THUNLP-MT/StreamingBench) | 流式感知、上下文、主动输出等 | 900 视频、4500 QA；本方案完整媒体/CSV 约 **202.550 GB**。实时感知小包 201–250 加 CSV 约 **2.232 GB**；主动输出 1–25 加 CSV 约 **4.477 GB** | 主实验 60 秒前文与全历史设置分开；音视频任务不能以视觉子集冒充全榜成绩；部分包只适合联调 |
| [Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME)，[官方代码](https://github.com/MME-Benchmarks/Video-MME) | 通用长视频离线理解，保证监控适配没有破坏基础能力 | 900 视频、2700 问题；原标注已在本机。视觉媒体 20 个 ZIP 加标注共 **100.995 GB**；首包加标注 **5.180 GB** | 无字幕/有字幕/音视频需区分；首包只产生部分评测，不是完整 Video-MME 成绩 |
| [FastBench](https://huggingface.co/datasets/Ashenone3/FastBench) | 高动态、短暂事件、快速时序变化 | **300 个 MP4、306 问题，8.893 GB**；无压缩包解压环节，适合首轮完整获取 | HF viewer 的 split 名为 train，**作者明确这是评测集，没有训练/验证划分**；不能因此拿去训练；按官方时机与 judge 规则评分 |
| [StreamArena](https://huggingface.co/datasets/hkuzxc/StreamArena) | 小时级历史回溯、延迟响应、工具使用 | 243 源视频、3646 问题；全库约 **321.45 GB**。可按 video_id 单独下载 tar；本计划最小视频包连标注约 **0.146 GB** | 中英文是同一问题的翻译，不能当两倍独立样本；视频能力与工具辅助子任务分别报告 |
| [ShanghaiTech Campus](https://svip-lab.github.io/dataset/campus_dataset.html) | 固定摄像头跨场景异常检测补充对照 | 官方 Google Drive / OneDrive；13 个场景，带像素标注；体积本轮未核验 | 常用 train 为正常事件建模协议，不能把测试异常当成监督训练正例；与 VLM QA 分数不同 |
| [CUHK Avenue](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) | 小规模异常检测与空间定位联调 | 官方原视频 ZIP 和 ground truth ZIP；16 train / 21 test；体积本轮未核验 | 训练为正常场景；测试有异常。官方位置评测与 sampled-frame AUC/AP 不等价 |

OVO 标注：[当前版 ovo_bench_new.json](https://github.com/JoeLeelyf/OVO-Bench/blob/main/data/ovo_bench_new.json)。官方 OVO 源视频还需按上游脚本切片；本仓库下载器不执行其预处理脚本。

## 3. 当前存储预算

所有精确下载大小来自当日 HF API 文件清单，单位 **GB=10^9 bytes**；GiB=2^30 bytes。数据集文件总量包括其标注等小文件，不代表解码后的帧缓存。审计记录保存每个所选来源的 revision、公开的文件字节与 LFS SHA256，见 [原始审计](dataset_source_audit_2026-11-05.json)。普通 Git 小文件未公布 LFS SHA256 时保留 revision 固定；UCA 额外核对 Git blob SHA1。

本机本次检查可用约 **67GB**。建议分阶段，不同时下载下面所有完整数据集。

| 阶段 | 下载选择 | 传输量 | 建议数据工作空间（估计，非官方解压大小） |
|---|---|---:|---:|
| 只研究格式 | UCA、LLaVA-short、CCTV、CUVA、StreamingBench、StreamArena 标注 | 约 0.1GB 量级 | 约 1GB；不包含训练媒体 |
| 第一轮软件/训练联调 | llava-short-pilot + cctv-train-pilot + fastbench + streamingbench-real-pilot + streamarena-pilot + UCA/CUVA 标注 | **约 18.6GB** | **约 40–50GB**，为压缩包、解压视频与临时文件预留；CCTV 需先获访问权限 |
| 完整公开评测 | OVO（二选一）、StreamingBench、Video-MME、FastBench、StreamArena、CUVA | 远超当前空闲空间 | 建议独立数据盘/云盘，按所选集合预算；1TB 仅是部分方案起点，不含全量 LLaVA |
| 自有物业数据 | 按摄像头/日期分组的正常长流、规则事件与困难负例 | 见码率公式 | 与公开集分开预算，不默认写满当前系统盘 |

完整保留 1.281TB LLaVA、CCTV 和上表几套大 benchmark，单下载文件就约 2TB 以上；压缩包、解压和缓存共存时要另做数 TB 预算。首轮 2B 原型不要求全部拥有。

ActivityNetQA 本机小试验已读取归档目录，文件内容解压后合计约 0.943 GB；其余媒体归档内部尚未检查，**没有官方实测解压大小**。profile 中的 suggested_data_space_gb_estimate 只是容量规划估计，脚本不保证解压后足够。OVO 若 `cat parts > combined.tar` 再解压会额外产生一整份 tar，可按作者格式使用管道减少中间副本；不要默认把五卷中的一卷当作独立视频子集。

物业码流的近似存储公式：`小时数 × 平均码率(Mbps) × 0.45 GB`。例如 100 小时、2Mbps 约 90GB，未计额外音轨和容器开销。初期优先保留压缩码流和标注，按需解码，避免把所有视频预展开成图片或浮点张量。

## 4. 下载脚本：默认仅预览

[scripts/download_datasets.py](../scripts/download_datasets.py) 读取 [下载配置](../configs/datasets/downloads.json) 与审计快照。**列清单和预览均完全离线，不创建数据文件、不请求网络。**

```bash
cd /home/user/work/bgyfw/VLM/TRACE-VL

# 列出所有方案及其字节数；不会下载
uv run --locked python scripts/download_datasets.py --list

# 预览第一轮组合；不会下载，约 18.6GB
uv run --locked python scripts/download_datasets.py \
  llava-short-pilot cctv-train-pilot fastbench streamingbench-real-pilot \
  streamarena-pilot uca-annotations cuva-annotations --max-gb 20

# 查看确切文件路径、上游 revision 和字节数；不会下载
uv run --locked python scripts/download_datasets.py llava-short-pilot --show-files
```

下面仅为**以后决定下载时**的命令示例，本轮没有执行：

```bash
# 需要先安装 data extra；当前工作区此前已经安装
uv sync --locked --extra dev --extra hf --extra data

# 未来只下载一个已选训练分片，不下载整个仓库；输出目录可改到数据盘
uv run --locked python scripts/download_datasets.py llava-short-pilot \
  --download --root /path/to/data/raw --max-gb 8
```

默认 --max-gb 10 限制所选去重后的文件总量；重叠 profile 不重复计费/下载。实际执行前核对已有文件 hash、剩余空间，并默认额外保留 5GB；可用 --reserve-gb 调整。HF 下载固定 commit，使用官方客户端的缓存/续传；UCA HTTP 下载先核对大小和 Git blob hash 再原子写入。下载后保存每文件本地 SHA256 和来源 receipt。已有文件不匹配时拒绝静默覆盖。**不自动解压、不转换标注、不启动训练。**

某些 gated 来源需要在 HF 页面自行申请/接受访问条件，再配置已有 HF 登录凭据；脚本不代替接受条款、不绕过 gated access、不记录 token。FineVideo 虽有分片但需访问条款，完整约 673GB；Video-R1 全库约 309GB 且混合图像和视频；LLaVA-OneVision-2 数据仓库含多个任务，元数据总量约 66.6TB。这些均不放入首轮自动下载组合。

UCF-Crime、XD-Violence、Avenue、ShanghaiTech、URFD 和 NExT-QA 的云盘/网页渠道保留在上表，尚未写成自动批量下载方案。部分旧渠道可能失效或需要登录，不能把“官网有入口”当成每个镜像都已实测可用。

## 5. 从下载数据到有效训练/评测

1. **训练与评测隔离。** UCA、CCTV 用各自官方 train；CUVA full 含 test；FastBench 的 train 名称不代表训练许可。来自同一原视频的 caption、QA、裁剪片段、重编码版本必须同组。混合 UCA 与 UCF-Crime 时还需按原视频核对两套划分的交叉，不能假定兼容。
2. **先对齐实际媒体。** 小分片并不覆盖全部 JSON。解压后按 video_id 求交集，保存未找到媒体的条目列表，不用随机其它视频顶替。大基准只取一部分时标记为 pilot，不能写成官方完整成绩。
3. **流式标签单独制作。** UCA 的事件起止、CUVA 的事后原因解释，都不自动等于最早可判定时刻 known_at。不能用“后来发生了什么”在较早前缀训练告警。消防通道占用/遗留物持续时长需要额外规则、ROI、人工 known_at/deadline 与证据可见性标注。
4. **JEV 式监督。** 可先把训练集事件/QA改写成规则、候选项、人工正确项，用 CE 训练；打乱选项顺序、加入困难负例。教师分布只能作辅助软标签，不能直接当物业场景的已校准真值。独立验证集用于温度校准、阈值与复核策略。
5. **自有物业集必需。** 公共数据可证明通用能力；最终需按摄像头/地点/日期留出未见测试集，覆盖日夜、遮挡、逆光和正常长时段，测事件召回、告警延迟、误报/摄像头小时以及多规则成本。跌倒受控数据不能替代真实场景验证。

当前已有 Video-MME 导入器和自有 CSV→统一 JSONL 导入器；本次新增的 LLaVA/UCA/CCTV 等**只有下载方案，训练转换与官方评测适配尚待实现**。下载完成不等于可直接送进现有 trainer；原始格式仍需转换并通过 data validate。

## 6. 许可与访问核查

| 来源 | 已看到的条款及冲突 | 使用决定 |
|---|---|---|
| CCTV-Anomaly | HF tag 是 Apache-2.0；卡片正文及作者仓库明确标注 CC BY-NC 4.0，限研究非商业 | 先作为研究候选；生产训练需向作者确认，不按 metadata 标签自动放行 |
| UCA | README 同时写 Apache-2.0 和 academic/research-only | 标注与原始 UCF 视频权限分开核实；生产用途需澄清 |
| OVO | HF tag CC BY-SA 4.0，GitHub README CC BY-NC-SA 4.0 | 存在矛盾，先保留研究评测用途，向作者确认商用范围 |
| CUVA / URFD / StreamArena | 分别可见 NC/SA 或 NC 的数据条款 | 不作为默认可商用训练池；具体按作者条款处理 |
| LLaVA / Video-R1 | 聚合了多种上游视频源 | 外层代码/标注协议不能替代原始媒体权利核查 |
| FastBench | 作者未声明统一媒体许可；评测代码 MIT 不覆盖视频 | 研究评测前确认来源条款，不直接分发视频或纳入商业训练 |
| 自有物业视频 | 本项目不提供现成物业原片与授权 | 以已获授权的素材建立可部署数据池 |

这些是对可见声明的记录，不是统一的许可保证。完整信息以链接来源与固定 revision 为准；本仓库 MIT 不覆盖外部数据。
