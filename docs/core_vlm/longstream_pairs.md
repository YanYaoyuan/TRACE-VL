# Controlled scene-order retention diagnostic

The first longstream pilot finished32 development queries on16 original videos. The delayed caption cohort had text-only6/8 at every delay. Exact-input contact-sheet inspection also found scene persistence, compound captions and boundary-sharing steps. Its scores remain unchanged. This follow-up changes the question and uses synthetic paired streams to isolate dependence on stored historical order.

The frozen annotation file is `longstream_pairs_v1.json`. All eight original delayed sources are included. Each scene contributes two already inspected two-frame source steps. Source frames within the two scenes must be disjoint. The assistant supplied descriptive scene labels after viewing the sheets; this is not human certification or an unseen benchmark.

For each source create AB and BA. Each event contributes four original RGB frames. Present those at2fps, followed by304 seconds of the same natural real-video distractor from a different staged OVO source. Both variants share every distractor pixel. Materialize lossless FFV1 files, decode every output pixel, and compare the full RGB stream hash to the source sequence. Record original source frame indices/times separately from the new monotonically increasing presentation clock. The swap is synthetic: it must not be described as the original video's chronology.

Questions arrive after15/60/300 distractor steps plus the four-step recent window, at22.5/67.5/307.5 seconds. The last historical event step ends at3.5 seconds, so the corresponding evidence gaps are19/64/304 seconds. The source streams are308 seconds long. Each view is encoded once, and intermediate queries cannot change writer state. AB and BA have the same writer random seed, question, options and query times, with opposite labels. Text/recent4/recent8 predictions must be exactly equal across variants; their both-correct rate must therefore be zero. Any violation aborts the report.

Freeze the same Plain LoRA2500 and base model as the first pilot. Reuse the existing bounded writer and native packing. Compare recent4, recent8, text-only, FIFO-uniform and reservoir-uniform against separately marked privileged evidence/stored-routing oracles. Use at most128 read tokens,32 historical slots per deployable policy and no question-time re-encoding. The evidence oracle stores all four event steps and reads them with the four recent steps; it has privileged write annotations and is not a deployable method. Retained feature payload and logical fixed metadata are bounded; this is not a claim about exact process RSS.

Primary descriptive metrics are both questions correct within a pair, individual question accuracy, and choice changes, separately at each delay. Count8 independent event sources,16 materialized views and48 queries. The three delays and two variants are dependent observations. Preserve all sources and all arms regardless of oracle or model success. Any later uncertainty intervals must cluster by original event source, not48 independent queries. No performance promotion threshold or statistical generalization claim is set for this diagnostic. It locates storage, selection or readout failures before further training.

Run the CPU tests and native tiny-model regression, then a realGPU AB/BA smoke on the first source covering all three query times and all seven arms. Full inference starts only if the smoke report passes identical-input checks, causal reads, token budget, encoding counts and bounded capacities. Full inference reuses completed smoke views. All media/model inputs remain on `/data`; old results are retained. No model training or external download is required.

```sh
TRACE_REV=<tested-revision> bash scripts/launch_core_longstream_pairs_a100.sh
```
