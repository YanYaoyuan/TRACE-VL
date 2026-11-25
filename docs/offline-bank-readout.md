# Offline stored-bank readout diagnostic

Following the negative capacity-matched writer controls, freeze all77 existing development sources and their original300-second queries. Keep Plain LoRA2500, actual stored32+4 bank states, question/options/gold, and all three Reservoir memory seeds. No training or new held-out sources.

Per source: text, protected snapshot, and latest8/uniform8/all stored pre-target slots for each seed:11 predictions,847 overall. Each selector sees only its own actual retained bank. The shared reconstruction union is an offline encoding cache; normal selectors cannot read another seed or snapshot. RGB hashes, original video bytes and full PTS identities must match. Re-encoding is explicitly allowed only for this offline diagnostic and its calls are recorded.

All five original controls per source must exactly replay, including candidate scores and read identities (385 predictions). A sparse and a widest bank target are selected without answer outcomes for native-layout/RGB/replay smoke. Failure holds full execution; accuracy never gates launch.

The primary paired comparison is source-level mean across three seeds of uniform8 minus latest8 (both at most128 read tokens). Bootstrap77 sources,4000 draws,seed7. Full bank reads at most32 steps/512 tokens and is a separately labeled larger-budget diagnostic, not equal-budget method evidence. It can show sensitivity to the read interface, but neither a positive nor negative result uniquely identifies the cause of earlier errors. No streaming latency or official benchmark claim follows from reconstruction.

Launch a tested immutable revision with `TRACE_REV=<sha> bash scripts/launch_core_bank_readout_a100.sh`. The launcher archives source separately, runs CPU checks and two-target GPU smoke, then all77. It does not change the server checkout or restart training. Reports include actual read tokens, reconstruction calls, paired CIs and all predictions.

## Completed development diagnostic (2026-11-25)

Revision27c35b0;77sources/847predictions,385complete original-control replays,3190offline RGB-pair encodes and5extra native-parity encodes. Full independent local review against the previous downloaded raw writer records passed. The immutable run completed and archived at15:47:41Beijing.

|Reader|Original seed correct/77|Seed29|Seed43|Mean accuracy|Paired delta vs latest8|Source-bootstrap95% interval|
|---|---:|---:|---:|---:|---:|---:|
|latest8|57|53|53|70.56%|—|—|
|uniform8|55|58|55|72.73%|+2.16pp|[-1.73,+6.06]pp|
|all_bank|54|55|53|70.13%|-0.43pp|[-3.90,+3.03]pp|

Uniform8 and latest8 have identical per-source token counts (cap128). All-bank actual max400, mean196.71visual tokens across three seeds; planned cap512. Text42/77, privileged snapshot64/77. Both paired intervals cross zero: no stable superiority demonstrated, no equivalence conclusion. No policy promotion or blind training scale-up. Wider reads did not deliver a stable recovery here; this does not uniquely identify writer, retrieval or representation as the bottleneck. Further module design should address event continuity/question-conditioned retrieval and define real history-dependent training before scaling.

Run `/data/yyy/tracevl-runs/core-bank-readout-27c35b0-002`. Input `7a68f9cbb4de35fade6c384a1bca090ab74d4f3daf1e53fd394126593f1afcd5`; report SHA256 `94f5c2ddc6c5668c5aeb89bcf5d1357bd7edce0ae1154d7dbaa8f5169bcbf5d8`; packet SHA256 `5b414843f21af8074bdc86dca7240d7c1bc3e437de2f50646a8a0f0d9945a123`. Full evidence is retained in the research workspace under `DOC/experiment-reviews/bank-readout-27c35b0-20261125`.
