# Offline stored-bank readout diagnostic

Following the negative capacity-matched writer controls, freeze all77 existing development sources and their original300-second queries. Keep Plain LoRA2500, actual stored32+4 bank states, question/options/gold, and all three Reservoir memory seeds. No training or new held-out sources.

Per source: text, protected snapshot, and latest8/uniform8/all stored pre-target slots for each seed:11 predictions,847 overall. Each selector sees only its own actual retained bank. The shared reconstruction union is an offline encoding cache; normal selectors cannot read another seed or snapshot. RGB hashes, original video bytes and full PTS identities must match. Re-encoding is explicitly allowed only for this offline diagnostic and its calls are recorded.

All five original controls per source must exactly replay, including candidate scores and read identities (385 predictions). A sparse and a widest bank target are selected without answer outcomes for native-layout/RGB/replay smoke. Failure holds full execution; accuracy never gates launch.

The primary paired comparison is source-level mean across three seeds of uniform8 minus latest8 (both at most128 read tokens). Bootstrap77 sources,4000 draws,seed7. Full bank reads at most32 steps/512 tokens and is a separately labeled larger-budget diagnostic, not equal-budget method evidence. It can show sensitivity to the read interface, but neither a positive nor negative result uniquely identifies the cause of earlier errors. No streaming latency or official benchmark claim follows from reconstruction.

Launch a tested immutable revision with `TRACE_REV=<sha> bash scripts/launch_core_bank_readout_a100.sh`. The launcher archives source separately, runs CPU checks and two-target GPU smoke, then all77. It does not change the server checkout or restart training. Reports include actual read tokens, reconstruction calls, paired CIs and all predictions.
