# 条件响应时机：第一轮架构实验

这是一条可恢复的实验流水线。它实现了新研究方案的 **P0 条件响应时机验证**，不是已完成的物业事件状态模型，也没有将待验证的设计写成论文贡献。先确认模型能利用视觉/条件信号，再扩展需要更强标注的持续状态、解除条件和组合规则。

## 本轮实际实现

冻结原始 Qwen3.5-4B 的视觉编码器、merger、语言模型。每秒在时刻 `t-.5` 和 `t` 采样两个已到达的画面；以解码时间戳检查不读未来帧。每步压缩为 8×8 网格。条件文本只编码一次，输入不含答案文本。

五组共享同一视频划分、视觉/条件缓存、时间信息和有效监督：

| 组 | 实现 | 目的 |
|---|---|---|
| clock | 仅经过时间的 MLP | 检查固定回应间隔捷径 |
| frame | 条件空间注意力 + 逐步 MLP | 检查不使用时间记忆的能力 |
| gru | 条件空间注意力 + 4 层 GRU | 循环时间建模基线 |
| temporal | 条件空间注意力 + 4 层因果时间 Transformer | 16 步局部时间窗口，参数匹配 |
| event_state | 条件空间注意力 + 16 个循环状态 token + 局部历史交叉注意力 + 写入/替换门 | 新机制候选 |

默认 hidden=512。temporal 约 2050 万参数，event_state 约 2100 万，差距小于 10%。GRU 和 frame 更小，报告参数量，不伪称所有组参数相同。所有组看到相同的经过时间；没有给新模块独占的辅助标签。状态门学到什么由实验检验，不把它称作已学会“事件解除”。

初轮每条轨迹最多等待 120 秒、最多到视频 180 秒。轨迹内完整反传，轨迹之间清空状态；没有跨视频拼接，没有训练 Qwen 的 recurrent cache。这里还没有更长流的 TBPTT、显式持续时间/未知状态监督、自由生成和 LoRA 联合训练。

## 数据准备

默认标注为已有 `JoyAI-VL-Interaction/event_grounding/event_grounding.json`，原视频从 `/nas-data/vlm/llava` 搜索。可以通过 `TRACE_EXTRA_MEDIA` 添加第二个原片/压缩包目录。

1. 从原始视频 ID 固定哈希划分 train/calibration/dev（75/10/15），在构造条件轨迹前划分。
2. 扫描 OVO、StreamingBench、Video-MME、NExT-QA 标注和媒体 ID，剔除重叠视频。对仍在 ZIP 内的视频只读中央目录中的真实文件名，不解压视频；StreamingBench 的官方 CSV 只有题号，没有源视频 ID，不能把题号当作视频号。缺失标注或解析失败会在使用 GPU 前报错；不悄悄跳过 benchmark 排重。此检查不是跨数据集感知去重。
3. 同一视频最多 8 条条件，优先复用已在磁盘上的原片，再按固定哈希选择归档候选，上限 1000 个原视频。原片同名但内容不一致时不猜测；跨划分完全相同内容剔除。
4. 只从现有 ZIP/tar/分卷包解压所需视频，保留原包。对选中的视频逐帧解码并计算 SHA256，检查标注时间是否在视频内。
5. 写到固定根目录 `/nas-data/vlm/yyy/tracevl-data/event-state-v1/<代码版本>-<配置哈希>/`。run/data 为指向该目录的链接；源文件不移动。
6. 输出数据计数、缺失/损坏/歧义清单、排除原因、来源和等待时间分布。数量不足会明确停止，不借用 benchmark 测试标签。

**JoyAI 的 response time 不是事件最早可判定时刻。** 本轮只支持一次条件提问和一个数值响应时刻；多轮、多响应时间串会统计并排除。监督为响应时刻前 WAIT，响应后最多 3 秒 READY，不推断物理事件结束。没有完整正常时段标注，不能报告物业误报/摄像头小时。

UCA/CUVA、CCTV-Anomaly、finevideo 暂不直接混进该损失：它们的标签语义需要独立适配。OVO、StreamingBench、Video-MME 在此只作训练排重，不运行它们的评测，也不加入训练。后续官方评测仍用原有独立启动器。

## CPU 准备吞吐与进度

逐帧核验使用有上限的多进程池，每个视频的解码器只用一个线程，逐帧处理而不将整段视频装入内存。默认取 CPU 配额、可用内存与 16 的较小值；资源充足的 A100 主机可设 `TRACE_DATA_WORKERS=32`。这只控制数据核验并发，不改变样本选择、划分、模型或训练参数。内存占用低本身不代表瓶颈。

目录扫描先筛选文件后缀，减少 NAS 元数据查询；归档匹配按文件名索引候选，不再为每个归档成员遍历全部待找视频。独立归档默认最多 8 路并行读取，可用 `TRACE_ARCHIVE_WORKERS` 调整（1–32）；每组 tar/分卷内部仍按顺序读取。并行结果按原来的归档顺序接纳，避免读取速度影响文件选择；重名歧义仍阻断。NAS 吞吐可能限制提速，增加并发并不保证线性加速。

准备阶段每 10 秒输出 `DATA PROGRESS`，包含当前阶段、文件/归档或已核验视频数，并写入 `data/preparation-progress.json`。阶段区分 benchmark 库存、媒体索引、原片解析、归档处理和全视频核验。归档阶段另报 `archive_workers`、`archives_finished` 及 `archives_active`（各包成员数和当前文件）；`TRACE_DATA_WORKERS` 控制的是后续解码核验，不代表解压并发。每个已完成视频立即保存核验记录；同版本中断后重跑仍核对内容 SHA256，再跳过已完成的逐帧解码。更换代码版本使用新的数据目录，旧记录保留。若旧任务停在数据准备阶段，停止它后可设 `TRACE_REUSE_PREPARATION=/旧run/data`，只复用已接纳的解压视频。程序要求标注、配置、benchmark 库存及媒体根目录完全一致，沿用新运行独立选出的名单，重做全解码与 SHA；不复用旧 manifest、特征或实验结果，来源记录在 `reused-preparation.json`。

## 训练与资源

单种子 7；每组最多 5 次数据遍历 / 2000 更新，先到为止。每个独立实验 microbatch=16（不累积梯度），AdamW lr=2e-4、wd=.01、梯度裁剪 1、5% warmup + cosine。选择最后 checkpoint，不挑开发集最高点。原模型权重另约 9.3 GB；特征缓存约 1.18 GB/视频小时（2560 维、64 token/s、bf16），额外解压空间由实际原片决定。

脚本先以 4 个独立进程分片提特征，再排队跑五组训练+评测：默认最多 4 组并行，每组独占一张卡，结束就填入下一组。`TRACE_PARALLEL_ARMS=1` 可改为串行。没有不等长流的 DDP collective 等待。不要按低显存占用判断吞吐；特征提取、每组耗时、峰值显存都写入报告。A100 的新架构吞吐要以本轮测量为准。

重启会核验代码、配置、数据和模型文件指纹；已验证完成的实验跳过。训练每 25 更新存模型、optimizer、RNG 和日志；中断后从最近 checkpoint 恢复。崩溃前不足 25 更新的工作会重算。已有 completion 但权重损坏会拒绝继续。

## 指标与输出

只在 calibration 集选择阈值，目标是每条监控请求出现提前响应的比例 ≤5%。开发集固定该阈值，报告响应后 3 秒内命中率、提前率、未响应率、匹配时延。它与真实 FAPH 是不同指标。置信区间按源视频聚类 bootstrap，多个比较为探索性、不作确认性优越结论。

新状态模块另有 `zero_video` 和 `shuffled_query` 两项诊断。必须结合 clock 基线、数据等待时间分布判断是否只是计时捷径；不允许看到高命中率就自动扩大训练。

主要输出：`data/preparation-progress.json`、`data/data-audit.json`、`data/manifest.jsonl`、`protocol.json`、`model-identity.json`、`features/profile-*.json`、`train/*/metrics.jsonl`、`eval/*/predictions.jsonl`、`report.json`、`REPORT.md`、`training-curves.svg`、`reports.tar.gz`。汇总包不含权重、视频和特征缓存。

入口：`scripts/launch_event_state_a100.sh`。本地验证记录见 `validation.md`。尚无这套新模块的 A100 精度结果。

## A100 一次启动

在 `/data/yyy/TRACE-VL` 获取 PR #3 的分支，从该分支提取启动器到 `/tmp`。启动器内部固定代码提交，并从 Git 对象生成独立 source 快照；不要求切换或清理当前工作目录。

```bash
cd /data/yyy/TRACE-VL
git fetch origin design/v3-transition-streaming
git show origin/design/v3-transition-streaming:scripts/launch_event_state_a100.sh > /tmp/launch_event_state_a100.sh
nohup setsid env TRACE_DATA_WORKERS=32 TRACE_ARCHIVE_WORKERS=8 bash /tmp/launch_event_state_a100.sh </dev/null > event-state.log 2>&1 &
tail -f event-state.log
```

默认 run 根目录为 `/nas-data/vlm/yyy/tracevl-runs/event-state-fd11ff0-001`。数据、模型和排重 benchmark 根目录见上述说明；必要时用 `TRACE_JOY`、`TRACE_MEDIA`、`TRACE_EXTRA_MEDIA`、`TRACE_MODEL` 指向已有文件。没有 4B 模型时下载配置中固定 revision，下载失败后可重跑，已完成文件复用；已有 2B checkpoint 不能冒充 4B。

`TRACE_PARALLEL_ARMS=1` 串行，默认 4 卡并行队列。`progress.json` 与 `logs/*.log` 查看阶段和各组进度。`launcher.log` 保存在实验目录。正常结束打印 `VERIFIED` 和 `REPORT BUNDLE`，提交 `reports.tar.gz` 即可检查数据审计、配置、训练曲线、校准、开发集结果和失败案例。

再次运行同一条启动命令会核验身份后恢复，已有完成记录会验证后跳过。不要一边运行一边修改其脚本/source/config；另一个配置使用新的 `TRACE_OUT`。
