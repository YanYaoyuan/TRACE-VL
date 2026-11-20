# A100 pilot 一键评测

已完成的首轮实验见 [100 步 A100 pilot 记录与结果](v3/results/a100-pilot-001.md)。该结果对应旧版模型；新旧代码评测需要分别使用匹配的配置与输出目录。

完成 transition 和 B4 两组 pilot 训练后，在已有 UV 环境中执行一个 Shell 入口：

```bash
export TRACE_RUNS=/path/to/tracevl-runs/a100-pair-001
export TRACE_DATA_DIR=/path/to/tracevl-data/stage1-30s-001
export TRACE_VIDEO_ROOT=/path/to/extracted-videos
bash scripts/evaluate_a100_pilot.sh --check-only
nohup bash scripts/evaluate_a100_pilot.sh > pilot-eval.log 2>&1 &
tail -f pilot-eval.log
```

脚本会自动切回工程目录，默认使用物理 GPU 0 和 `step-00100` 配对检查点。
可通过 `TRACE_EVAL_GPU`、`TRACE_EVAL_STEP`、`TRACE_EVAL_OUTPUT` 覆盖。
配置默认是 `configs/v3/a100-pilot-transition.yaml` 与 `a100-pilot-b4.yaml`，
自定义配置可设置 `TRACE_TRANSITION_CONFIG` 和 `TRACE_B4_CONFIG`。

依次完成：

1. 检查训练完成状态、训练清单 SHA、验证集视频隔离、配对检查点和一致的两组配置。
2. 汇总 rank 0 已记录样本的 P-token / gist-only 数量。该统计不代表所有 rank 的逐帧活动率。
3. 在训练清单中按视频键排序的前 8 个视频上审计实际预算（每段最多 30 秒）。
4. 串行评测 native、训练后的 TRACE、训练后的 B4，复用原评测器的预测断点续接。
5. 校验完整的预测 ID 覆盖、标签、视频分组、运行指纹及错误记录，再生成分数。
6. 以视频为独立单位进行 4,000 次配对 bootstrap，汇总 TRACE−B4、TRACE−native 的准确率差及95%区间。
   少于两个独立视频时不输出置信区间。

结果位于 `$TRACE_RUNS/pilot-eval/`：

- `pilot-gate-summary.json`：门控统计。
- `budget.json`：预算审计。
- `native/score.json`、`transition/score.json`、`b4/score.json`：三个条件的留出集结果。
- `summary.json`：完整汇总及配对区间，只有所有阶段通过完整性校验才生成。
- `workflow.json`：输入指纹。

重复执行相同命令会保留已有预测并续接缺失的问题；同一输出目录只允许一个脚本进程运行。
检查点、adapter、数据、配置或模型代码改变后，必须指定新的 `TRACE_EVAL_OUTPUT`，避免混合实验。
发生解码、推理或覆盖检查错误时退出，保留已有预测供排查和重试，不生成新的成功汇总。
`--check-only` 只检查输入，不启动 GPU 评测或创建评测结果。

预算不匹配时仍完成准确率评测，但汇总中的 `budget_matched` 为 false。
这类结果不能支持等预算架构优势的结论；区间包含0时也不能据此宣称准确率提升。
脚本不会更改或重训模型，不会自动调整门控阈值。
