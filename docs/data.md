# 数据与标注约定

官方入口、训练/评测用途、存储预算及获取优先级见 [数据集下载清单](dataset_download_plan.md)。下载脚本默认离线预览，本轮没有下载新数据。

每行一个 JSON 对象，`schema_version=1`。相对视频路径以 manifest 所在目录为基准；配置 `data.media_root` 可覆盖。读取器支持带真实时间戳的 NPZ 和 OpenCV CFR 视频。VFR 文件须先归一化 CFR；不能把 `frame_index/fps` 冒充任意 VFR 的真实 PTS。当前解码器一次加载所选前缀帧，模型逐帧更新；尚非实时媒体服务。

## 字段

| 字段 | 含义 |
|---|---|
| id | 样本唯一标识 |
| source_group | 原始视频、摄像头或拍摄 session 的来源组；同组禁止跨 split |
| split | train / val / test |
| video | 视频或 NPZ 路径 |
| protocol | streaming 必须显式给时间截止；offline 允许 video_end |
| observation.start / end | 允许观察的时间范围，秒；模型不得观察 end 之后 |
| question / answer | 问答与训练/评测答案；答案不会传入推理模型 |
| answer_type / options | text 或 choice；choice 的答案是大写选项字母 |
| rule / monitor_category | 推理前给定的告警规则及输出类别，不从正确答案或 events 推导 |
| events | 有完整监控标注才提供；[] 表示已确认阴性，缺字段表示没有监督 |
| events[].start / end | 事件物理起止时间 |
| events[].known_at | 仅看过去时，最早能判定该事件的时刻，必须人工定义 |
| events[].deadline | 最晚可接受告警时间；不小于 known_at |
| utility_targets | 可选 time/value(0..1)，训练保留价值头；缺失时不计算 utility loss |

一个监控样本对应一个规则类别。多类别需求拆成独立规则 episode；报告多规则误报率时，单位是 rule-hours，不能冒充独立摄像头小时。相同区间、同规则的多条 QA 不能重复计入监控时长，评测器会拒绝这种重复。

训练中告警标签在 `known_at <= time <= end` 为 1；写入标签在窗口内第一个不早于 known_at 的采样点为 1。若 known_at 在事件结束后（需要事后确认），当前 alert-presence 标签将没有正例，必须按具体任务补充规则/标签策略后再训练，不能无声套用默认规则。utility 无标签时 head 不学习，不能宣称已有有效 utility 排序。

`max_frames` 超过时使用均匀子采样；采样计划依赖预先给定的可见区间。它适合前缀回放软件验证；真实低延迟在线告警需固定时钟持续采样、不按未来问题时刻重选过去帧。长视频 QA 建议先增加 max_frames 做资源扫描，而不是用 tiny 的 8 帧估计真实性能。

## 自有监控数据

参照 `data/examples/alerts.csv`，人工填写事件，保留阴性长片段和遮挡/停留等困难负例。不同划分按拍摄地点/日期/摄像头分组。不要从视频文件名推测 known_at 或从弱标签推测精确时间。

```bash
uv run --locked trace-vl data import-alerts data/examples/alerts.csv --video-root /data/cfr --output data/processed/trainval.jsonl
uv run --locked trace-vl data validate data/processed/trainval.jsonl
```

示例引用的媒体不随仓库提供。正式训练 manifest 应包含 train 和 val，测试数据分开冻结。synthetic 的来源与生成种子自动保存在 provenance.json，禁止混入真实论文统计。

## 公共数据

`datasets list` 给出官方来源和当前接入状态。[Video-MME](https://github.com/MME-Benchmarks/Video-MME) 的 HF 标注通过 [lmms-eval/Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME) 获取；转换器保留 duration 的 short/medium/long 类别，用媒体解析截止时间，不把类别当成秒数。默认视听配置是无字幕、无音频，仅视觉。选项字母准确率是本地生成式解析，若要提交官方榜单须遵循上游完整协议。

本机已下载官方 `videomme/test-00000-of-00001.parquet`（405,167 字节）至 `data/raw/videomme/test.parquet`，并保存来源与 SHA256；raw 默认被 Git 忽略，不随代码重分发。安装 data extra 后可直接导入这个文件。`configs/eval_videomme.yaml` 为独立评测数据配置，须用同结构的已训练 checkpoint，并准备媒体；仅有标注不能跑真实视频指标。

[StreamingBench](https://streamingbench.github.io/)、[OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench) 和 [XD-Violence](https://roc-ng.github.io/XD-Violence/) 的媒体和评测适配尚未打通。XD-Violence 的弱监督训练标注不能直接生成本项目的 known_at/deadline。遵守各上游访问及数据使用条款；仓库 MIT 不覆盖第三方数据和权重。
