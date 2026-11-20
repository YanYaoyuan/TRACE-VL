# Order probing and pre-query evidence binding

## 40K result

The 40K experiment remains on its original configuration. The final
`report.json` contains the 5,000-update results under `results` and the
2,500-update results under `learning_curve["2500"]`. Both compare Temporal
with Plain on identical development questions. Keep the question-level
predictions and evaluation manifests for independent recomputation.

Read the NExT temporal aggregate and TN, TC, TP separately. Read OVO backward
and ASI, EPM, HLD separately. Use NExT descriptive QA and OVO real-time
perception to check retention. Interpret intervals as video-cluster paired
intervals. Use the final checkpoint for the existing architecture gate.
A gain on the pooled temporal category alone cannot identify use of action
order. Use the held-out order-pair intervention below to test that mechanism.
The 20K versus 40K comparison includes changes in data mix, global batch and
updates. The within-40K checkpoint comparison addresses late training gains.

## Frozen-Plain order experiment

Each input JSONL row specifies two reviewed action spans in one source video.

```json
{"id":"scene-001","video":"scene-001.mp4","source_group":"scene-001","split":"train","ask":"first","spans":[[0.4,1.8],[2.4,4.0]],"actions":["tapping the table","waving a hand"]}
```

Use English action descriptions and put all crops from a source or recording
scene in one split. Each span contributes eight distinct sampled frames. The
paired views use the exact same 16 RGB frames, a shared 4 Hz presentation grid,
identical question text and identical option order. Only the span order changes.
Both span interiors retain forward playback. Labels refer to the presented
sequence. Use independent actions where possible and avoid edits with clocks,
subtitles or narration that reveal the original chronology.

```bash
export PYTHONPATH="$PWD/src"
PY=.venv/bin/python
"$PY" -m trace_vl.core_vlm.order_probe preview \
  --pairs pairs.jsonl --out order-review
```

Open `order-review/index.html` and verify both events in the sampled frames
of each pair. Mark `approved: true` and fill `reviewer` in
`order-review/review.jsonl`. The fingerprint binds the reviewed labels, media,
source group, split and sampled RGB. To reject a candidate, remove it from the
input manifest and keep only the corresponding reviewed rows. Rerunning preview
on the same population preserves the approvals.

```bash
"$PY" -m trace_vl.core_vlm.order_probe run \
  --pairs pairs.jsonl --review order-review/review.jsonl \
  --model /nas-data/vlm/models/qwen3.5_4b \
  --plain /nas-data/vlm/yyy/tracevl-runs/core-scale40k-54e161a-001/train/plain/checkpoint-005000 \
  --out order-probe-001 --updates 100
```

This is a separate one-device job. It uses the completed Plain checkpoint,
checks its base weights, checks held-out source aliases and media hashes against
Plain training, and trains only the two-layer temporal adapter. The frozen
language model remains in the gradient path from answer CE to the adapter.
Training maintains a matched presentation of both orders. The held-out
controls are frozen Plain at 16 frames, one distinct frame repeated over the
same observation grid, and the same formatted question as text alone. The
primary metric is both orders correct for the same pair. Report question
accuracy and flip rate as diagnostics. A high single-frame pair accuracy
indicates that the sampled frame can disclose the order in this population.

Treat the first few dozen pairs as an engineering check. Use several hundred
source-group-disjoint reviewed pairs for a retention and accuracy decision.
Keep the held-out set and final update fixed before comparing depths. If the
adapter gains over both frozen Plain and the single-frame control, test
Plain / 2 layers / 4 layers with the same trainable LoRA policy, inputs, answer
objective and sample budget. Report the parameter and compute differences.
On new data, distinguish gains from more independent pairs and more presentations
of the same pairs.

## T1 memory experiment

Freeze the same visual backbone and Plain LoRA for all three memory arms.
Process the stream once. Before the question arrives, independently record
before/after evidence linked by entity or spatial track. Store enough
provenance to verify that both observations are earlier than the query.
At query time, retrieve stored entries with a fixed read budget.

Compare a recent window, unbound feature retrieval and pre-query bound
retrieval on identical streams. Match the visual token budget, memory bytes
including metadata, maximum entries read, and query-time encoder calls.
Include a link-shuffled control with identical stored content and retrieval
limits to isolate the value of the binding. Use primarily natural, annotated
questions about both states. Vary evidence delay and include confusing
intervening events and questions answerable from the recent window alone.

Use paired analysis by source stream for questions whose annotated evidence
falls outside the recent window. Report accuracy, evidence-hit rate for both
states, storage and read bytes, query latency, and recent-only QA retention.
Promote T1 only if it consistently beats unbound retrieval at matched budgets
on held-out streams. This experiment has its own data and results.
