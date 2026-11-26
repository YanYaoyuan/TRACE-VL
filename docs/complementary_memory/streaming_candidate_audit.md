# Outside-development candidate byte/header audit

Prepare data for the next long-stream protocol without scoring questions or
selecting a confirmatory cohort. The original development manifest contains
sources1–250. All250 remaining Real-Time candidates (251–500), approximately
85.64GB uncompressed, already exist in five NAS ZIPs.

`scripts/audit_streaming_candidates.py` pins the annotation/development manifest
hashes, existing20,370 unique training-media hashes, code and archive byte/stat
identities. It accesses only source ID/task/cutoff/clue metadata; it never accesses
or exports question/options/answer/frames_required fields. No model is loaded.

Two CPU workers copy every candidate to a fixed `media/sample_N.mp4` name under
the run on `/data`, checking ZIP CRC, frozen size and SHA256 while copying. No ZIP
pathname is used as an extraction destination. A third worker hashes the existing
250 old-development media files, without copying those files. The report lists
training/old-development byte overlaps and candidate duplicate groups. All
members of duplicate groups are excluded from header-eligibility counts; no
representative is chosen using question labels.

FFprobe reads primary-video/container header duration and stream properties.
Invalid headers remain recorded as data findings and never pass eligibility.
Concurrent Action/Object and broader +Attribute/Spatial, cutoff>=60, cutoff+300
within header duration, are counted separately after byte exclusions. These are
**header-only upper bounds**: complete PTS availability, snapshot-local evidence
and underlying-source independence remain unverified. Disjoint byte hashes do
not establish distinct underlying videos or scientific test independence.

One parent enforces3600 seconds over planning, all three CPU jobs and reporting,
using the reviewed owned-process-group supervisor. GPUs are disabled. Planned
new media must be <=100GB and leave20GiB free. A failed run cannot reset the
deadline/resume under the same output directory. Completed staged files remain
available for later validated use; there is no automatic training/evaluation.

Before the full audit: A100 CPU integration checks cover copied-byte/hash/CRC
identity, overwrite refusal, corrupted ZIP rejection, invalid-header accounting,
and exclusion of train/dev overlaps plus entire duplicate groups. Run metadata
planning separately and review its complete250+250/85.64GB budget. Require green
exact-head hosted CI. Authoring PC performs syntax parsing only.

Run from an immutable source archive on A100 with `PYTHONPATH=source/src`:

```bash
python scripts/audit_streaming_candidates.py plan --out RUN/preflight
python scripts/audit_streaming_candidates.py run --out RUN
```

The planned data remain development candidates. This audit neither freezes a
benchmark subset nor makes an unseen/official-result claim.
