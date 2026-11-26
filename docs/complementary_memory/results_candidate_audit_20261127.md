# Candidate data audit: complete; current scope cannot supply150 sources

All250 outside-old-development candidates are now staged on A100 `/data`:
**85,643,895,263 bytes**. Candidate byte duplicates, overlap with the20,370
unique frozen training-media hashes, overlap with the250 current old-development
media, and invalid headers are all zero. This is byte-level data preparation,
not a model/QA evaluation or a certified independent test cohort.

The fixed task/cutoff/delay scope still falls below the proposed150-source
acquisition floor, **before** full PTS and semantic evidence checks:

| Scope | Header-eligible sources | Metadata rows | Sources with>=2 question IDs | Sources with>=30s cutoff span |
| --- | ---: | ---: | ---: | ---: |
| Concurrent Action/Object, cutoff>=60 and cutoff+300 within header duration | 87 | 98 | 11 | 10 |
| Above plus Attribute/Spatial | 139 | 197 | 50 | 42 |

The first two columns implement the frozen audit rules. Multiple-question/span
columns are explicitly post hoc metadata counts. Different IDs/timestamps do not
certify different semantic facts, memory dependence or trainability. Header
duration does not verify complete causal PTS. Additional data or an explicitly
revised task scope is needed before promising150 eligible fresh sources;150 was
an acquisition floor, not a statistical power guarantee. No cohort is frozen.

## A dependency correction for the old diagnostics

The old250 development media contain five exact-byte duplicate pairs:
**11/38,12/29,13/33,14/40,15/45**. Four pairs have both members in the frozen77-source
diagnostic (13/33 contributes only33). Thus the old cohort is **77 source IDs but
73 byte-distinct media clusters**. Every one of the77 newly measured file hashes
matches its historical sealed reference; this is existing dependency, not media
drift. The four double-included pairs have different question/options/gold tuples
and different target times, so they are not simply duplicate question records.

The frozen reports are preserved. Their source-ID bootstrap does not account for
these known video dependencies. A separate **post hoc** sensitivity resamples73
SHA256 clusters with replacement, retaining every source fact and averaging
memory seeds within source. A ratio of sampled cluster sums/counts preserves the
original source-ID-weighted point estimate. A second, clearly labelled analysis
averages facts within video before bootstrapping; this changes the estimand.
Both use10,000 draws/seed7, with pointwise intervals and no multiplicity correction.

| Old300s NLL contrast (first minus second) | Original mean | SHA-cluster CI, original point estimand | Equal-video mean [CI] |
| --- | ---: | --- | --- |
| Test A history-KV fork S1K1 minus S1K0 | -0.12833 | [-0.22721,-0.03175] | -0.12325 [-0.22624,-0.02333] |
| Test A S/K interaction (signed, not a benefit) | +0.00061 | [-0.06289,+0.06376] | +0.00015 [-0.06578,+0.06654] |
| Reservoir minus window40 | -0.12438 | [-0.22852,-0.02181] | -0.11698 [-0.22519,-0.01236] |
| Protected minus Reservoir | -0.07079 | [-0.15877,+0.02249] | -0.06817 [-0.15897,+0.02822] |
| Fresh R128 minus R32 | -0.06837 | [-0.13893,+0.00936] | -0.06279 [-0.13538,+0.01645] |

The main qualitative readings persist: historical KV value and the B0 NLL
comparison remain favorable; state interaction, target protection and expansion
remain inconclusive. B0 accuracy stays+5.63pp with cluster CI[-3.38,+14.72]pp;
capacity accuracy stays+3.90pp with[-0.89,+9.05]pp. These accuracy intervals do
not establish superiority. This post hoc correction does not create a new
confirmatory result or justify training. SHA clusters are only known dependency
groups; re-encoded/overlapping/common-origin footage can remain undetected.

All old point contrasts were recomputed from saved logits and matched the frozen
means to1e-12. The sensitivity covers all five delays,10 Test A contrasts,three
B0 contrasts and the protected/capacity primaries, for NLL and accuracy.15,000
explicit resample-expansion checks validate the vectorized cluster estimators.
[Complete sensitivity](results_candidate_audit_20261127/cluster_sensitivity.json)
retains the original intervals alongside both alternatives. Future uncertainty
analysis should use media-identity groups and disclose remaining provenance limits.

The [portable reproduction script](results_candidate_audit_20261127/reproduce_cluster_sensitivity.py)
loads only saved records and NumPy, with explicit input roots; it imports no model
code. Use the verified input/report/question trees from the four completed runs:

```bash
python reproduce_cluster_sensitivity.py --test-a-root TEST_A --b0-root B0 \
  --protected-root PROTECTED --capacity-root CAPACITY --out cluster_sensitivity.json
```

The two weighting schemes and all15,000 expansion checks are in that script.

## Execution and provenance

Data-audit inference/source revision: `556181c7a6c61737464004fbd911e12ddbc4868b`.
Input: `563398c855fa3df2e676cc96e29e87b3765dc3922c69ec8c42ae4eeffe98fcca`.
Root: `/data/yyy/tracevl-runs/candidate-media-audit-556181c-001`.

The parent ran2026-11-27 01:16:09–01:33:12 Asia/Shanghai,
**1,022.80 seconds /17.05 minutes**, within its hard3,600-second ceiling.
Two workers staged candidates with frozen-size/CRC/SHA checks; a third hashed
the46,258,083,176 bytes of existing old-development media. All three workers
and parent exited normally. GPUs were disabled and model calls were zero.
Completed staged files remain available for later validated use.

A100 CPU12 tests/Ruff and metadata plan/load roundtrip passed before execution.
Both exact-head hosted jobs passed578 tests/5 skipped, Ruff and repository smoke.
The authoring PC ran no tests/lint/models; later work there was saved-JSON/NumPy
artifact/accounting analysis only. Runtime/protocol and historical results were
not changed to produce these findings.

The846,147-byte metadata archive excludes video and question/option/gold content.
Independent saved-artifact verification checked input identity, source/protocol
hashes, all500 media records, bytes/overlaps/duplicates and both eligibility counts,
all77 baseline media hashes, phase duration and terminal process receipts. The
extraction-time SHA/CRC pass was not independently repeated over the staged85.64GB.

- Archive SHA256: `ed846b951341949311bf14e1af3e46e395f31384d4230cb14358cca71e100fb4`.
- Report SHA256: `86453c5a03e44cdb56586737f794f09cb6bc30cfaf550a1c24ac21db53ac2454`.
- [Report](results_candidate_audit_20261127/report.json),
  [verification](results_candidate_audit_20261127/verification.json),
  [execution](results_candidate_audit_20261127/execution.json).

No fresh-source model evaluation, new training, benchmark-subset freeze or
automatic larger experiment follows. Full PTS, evidence validity, source-family
independence and the next quality protocol remain open decisions.
