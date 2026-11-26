# Pin consumed FineVideo shards before training-data materialization

The NAS census establishes long-video availability, but a file's name, size and mtime do not establish its upstream release identity. This tool rereads exactly the selected local parquet shards and matches their bytes to a declared, pinned public repository file tree. It does not download anything, parse parquet, inspect question values or execute models.

The unauthenticated public HF file tree at revision `84c74091e1c6ee7a5dffabfafb5c9033e4718883` has 1,357 shard names/sizes matching the existing NAS inventory. Its `lfs.oid` values are masked. The ordinary Git pointer blob IDs remain public. Given an already-held local file's SHA256 and size, the [standard Git-LFS pointer](https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md) can be reconstructed locally and its Git blob ID compared with that public metadata. This fetches no gated content and accepts no terms. A predetermined 493,919,884-byte NAS shard passed this check; the remaining shard bytes have not yet been verified.

## Inputs and use on A100

Use the captured public tree JSON with `receipt` (completed, HTTP status, revision, file count) and `files` (path, size, Git blob `oid`, LFS size/pointerSize). Pin that JSON's SHA256 independently. The current captured metadata file is `public-file-tree.json`, SHA256 `f5462d6765ec1d4aadd5c40165110186b61c6caac38c80ccf19183a5b026aeef`. A later capture has its own receipt and hash; do not silently reuse this hash for different bytes. The source of trust is the captured public metadata and its declared pin, not a signature supplied by this verifier.

Create a JSON selection listing exact paths such as `["data/train-00000-of-01357.parquet"]`, selected from the metadata-defined source cohort. No glob or implicit full-dataset scan is performed. `--root` points to the dataset directory containing `data/`. A root-relative path must occur in the pinned tree and cannot be an escaping or symlinked shard path. The total selected size must fit the explicit read budget.

```bash
timeout --signal=TERM --kill-after=5s 600s \
  /data/yyy/TRACE-VL/.venv/bin/python scripts/verify_finevideo_shards.py \
  --tree /path/to/public-file-tree.json \
  --tree-sha256 f5462d6765ec1d4aadd5c40165110186b61c6caac38c80ccf19183a5b026aeef \
  --revision 84c74091e1c6ee7a5dffabfafb5c9033e4718883 \
  --root /nas-data/vlm/data/finevideo \
  --selection /path/to/selected-shards.json \
  --max-bytes 2000000000 \
  --out /data/yyy/tracevl-runs/finevideo-selected-verification-UNIQUE
```

Review a separate invocation with `--plan-only` and a separate new output directory if needed. This checks selected path existence/sizes and writes a plan with `verified: false`; it never reads shard content or produces a successful verification report. Actual verification uses its own new output directory. Existing receipts are never overwritten or reused as a skip-hash cache.

The CLI is single-process and uses no subprocesses. The external timeout supplies the hard process limit; the Python verifier itself does not claim a hard wall-time guarantee. The example 2 GB/600 s budget is an invocation constraint, not evidence that every storage device can meet it. Read only the consumed shards rather than blindly hashing all 673 GB.

## Meaning of a successful receipt

Every selected file was fully reread and matched the expected size, standard pointer size and Git pointer blob ID. File identity/size/mtime/ctime are checked across the read. The report pins the plan, revision, metadata file and each observed local hash. Failure or interruption cannot publish a successful report; progress alone is not verification.

This is a check of observed selected bytes, not a permanent cache guarantee. The materializer must consume those same validated files; changes require revalidation. Unselected shard contents remain unverified. Valid parquet structure, valid video bytes/PTS, visual grounding, answer correctness, causal cutoffs and source independence require separate checks. Native FineVideo QA only has question/answer fields; do not invent earlier cutoffs or evidence labels from a successful shard receipt. This tool neither freezes a source split nor qualifies any memory-training label.

Validation status: no tests, lint or tool execution on the authoring PC. Hosted CI is the first execution of this reusable implementation. The prior A100 one-shard operational check is a separate observation and does not substitute for this implementation's tests or an actual selected-cohort run. Continues issue #18; no training or benchmark scores change.
