# Persistent writer controls: complete negative primary result

Run `core-writer-controls-60bd0df-001` finished2026-11-25 at15:05:26
Asia/Shanghai. All77sources/385queries/10arms/3850predictions completed.
Full packet and independent audit passed. Original6arms x385queries =2310
predictions exactly replayed, including both score precisions and input identities.

Same frozen Plain LoRA2500, same original data/queries/gold/sampling,32historical+
4recent slots and128read tokens per normal policy. Added deterministic temporal
coverage/spatial change writers and2prespecified extra reservoir memory streams.
One stream encode per source, no training, no new reserved sources.

|Arm (correct/77)|0s|15s|60s|180s|300s|
|---|---:|---:|---:|---:|---:|
|Original reservoir|63|55|55|54|57|
|Extra reservoir29|61|56|56|55|53|
|Extra reservoir43|65|58|58|54|53|
|Time coverage|65|56|55|56|52|
|Spatial change|63|54|55|49|50|
|Text|42|42|42|42|42|
|Protected snapshot8 (privileged)|64|64|64|64|64|

Predeclared300s primary: each new writer minus mean3reservoir correctness within
source, bootstrap77sources4000draws/seed7. Do not count231seed-video units or
select a convenient memory seed.

- Reservoir300 mean:70.56%, individual counts57/53/53.
- Time coverage52/77=67.53%, delta -3.03pp, exploratory95% interval[-9.96,3.90]pp.
- Spatial change50/77=64.94%, delta -5.63pp, interval[-12.56,1.30]pp.

Both point estimates are below all3reservoir runs. Intervals cross0: no established
superiority and no claim of confirmed inferiority. Do not promote either writer.
Time coverage's65/77 at0s or56/77 at180s cannot replace the fixed300s endpoint.
Descriptive300s Action: reservoir27/26/25 of39, new22/22. Object: reservoir30/27/28
of38, time30/spatial28. Keep all77primary; no post-hoc task filtering.

All new writers reach the same2,952,576B logical feature+slot-metadata bound.
Mean append wall ms/step: time0.601, spatial4.543, extra reservoirs0.373/0.242.
Includes synchronization/checks/logging and is not isolated end-to-end profiling.
Spatial change increases writing work without primary endpoint gain.

39,028 stream encoder calls plus6native-parity calls/proofs.50/3850score-precision
choices differ; BF16 sensitivity at300s is reservoir56/53/53,new52/50, same decision.
72,512admission trace events replayed independently with fullPTS, store/PRNG/read,
capacity and score audits. Spatial feature-origin costs were not recomputed by a
second visual encoding; only recorded scalar ranking/state evolution is replayed.

Reference snapshot steps retained/read at300s: reservoir35/38/43, time35,
spatial37 of616. All retained reference steps were read; snapshot8 is not certified
semantic evidence. These counts alone cannot locate a unique storage/readout cause.

## Decision

Keep full failures and reservoir as baseline. Investigate motion/context continuity
and bank readout before designing a learnable memory objective. A wider stored-bank
readout diagnostic would need explicit different-token-budget labeling and exact
8step replay, not an equal-budget method or streaming-latency claim. It has not
started yet. No training scale-up just to rescue these fixed heuristics; no new
architecture or CausalPatchPool promotion. Existing development sources and one
frozen training/model seed remain; final public benchmarks require official
protocols and explicit development/unused-source boundaries.

Input ID:`fffbd282b72fb0dff10e221ba94dbeb93f4a25c7f21b91f94d4cb6f87d5d594c`
Archive SHA256:`a3db3987a515f444d60aca4b060c9ea8a829cfc796dfdd9084f3cfd289d05847`

Workspace artifacts: `DOC/experiment-reviews/writer-controls-60bd0df-20261125/`
Full packet SHA256:`eb336324e31ddc3aa1439d379dcd3c91d5715c5c6837327762c5bc32aba98a83`
Report SHA256:`4b10112e4b187c6563b449de0bf2fd58a3ad26e04a8f120addc719f458499350`
