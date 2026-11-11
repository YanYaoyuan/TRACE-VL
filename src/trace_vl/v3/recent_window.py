"""Recent-window control on the frozen OVO transfer population (experiments.md §14.4). Evaluation only.

Arms per checkpoint (S1, S2):
  fifo           existing §14.3c arm, reused (recent 16 s + 1,024 older KV tokens; state persistent from stream start)
  stream_window  whole prefix streamed with memory.policy = window (recent 16 s of KV only; recurrent / adapter
                 temporal state persistent) - run with the ordinary Stage-2 evaluator
  fresh16        PRIMARY control: for EVERY question a new session sees only the last 16 steps before the query
  fresh4         secondary: the last 4 steps (4 s, EIGHT sampled frames; not a Mage-VL / SimpleStream four-frame
                 reproduction)

Fresh-state protocol (this module). Per question, independently of every other question:
- a NEW StreamSession: new LM KV cache (standard prompt prefix only), new recurrent state, new visual-adapter
  temporal caches (CST / TE), new transition tracker, new retention book, new decision memory, new position counter;
- frames: the stream's ORIGINAL global step grid (the §14.3 run's step times). k* is the last step whose two-frame
  window ends at or before the query time (the step §14.3 answered after); steps k*-W+1 .. k* are fed (fewer when the
  stream is shorter). The video is decoded sequentially from its start exactly as in the stream run and each step
  is preprocessed once, so the fed pixels are those of the stream run; nothing after the query time is read;
- prompt timestamps keep their absolute stream times;
- same question text, options and option-likelihood scoring; no eviction can occur.

What differs from `fifo` besides missing history (limitations, not removable in this protocol):
- the position counter restarts, so the window's tokens sit at small positions instead of late-stream positions;
- the first fed step is an I-step for the adapter (no previous step to difference against) even where the stream
  run had a P-step there;
- the recurrent state starts from its initial value.
So fifo vs stream_window vs fresh16 change the processing trajectory and the representations as well as the
retained state: they are DIAGNOSTIC CONTRASTS, not a clean isolation of one mechanism.

Isolation check (GPU flow stage): for the first questions of each stream the window is answered, then a different
window, then the first window again in another new session (A-B-A). The two A results must agree, which tests for
state leaking through the backbone or the modules between sessions; `session_steps == window_steps` alone would
not show that.
"""

from __future__ import annotations

import json
import random
import time as clock
from collections import defaultdict, deque
from pathlib import Path

from .data import read_jsonl, write_jsonl
from .ovo_transfer import BACKWARD, REALTIME

FRESH_WINDOWS = {"fresh16": 16, "fresh4": 4}
CONTROLS = ("stream_window", "fresh16", "fresh4")
CHECKPOINTS = ("s1", "s2")
NEW_ARMS = tuple(f"{c}_{a}" for c in CHECKPOINTS for a in CONTROLS)
ISOLATION_TOLERANCE = 1e-3  # option log-probabilities (fp32 letter scoring over a bf16 backbone)
SCOPE = ("diagnostic contrasts on one training seed and one external-dataset development population; the arms differ "
         "in processing trajectory and representations (fresh position counter, first-step I scheduling, initial "
         "recurrent state) as well as in retained state, so no contrast isolates a single mechanism; none of them "
         "changes the §14.3 decision")  # fmt: skip


def window_steps(k_star: int, window: int) -> list[int]:
    """Indices of the steps a fresh session is fed: the `window` steps ending at k* (fewer at the stream start)."""
    return list(range(max(0, k_star - window + 1), k_star + 1))


def answer_fresh(make_session, steps: list[tuple[float, object]], question: dict) -> dict:
    """One question on a brand-new session fed `steps` [(t, preprocessed pixels)] in order."""
    session = make_session()
    for t, pixels in steps:
        session.step(None, t, pixels=pixels)
    choice, scores = session.ask(question["question"], question["options"])
    memory = session.memory
    return {"choice": int(choice), "scores": [round(float(s), 5) for s in scores], "session_steps": session.step_index,
            "kv_tokens": len(memory), "evicted_tokens": memory.evicted_tokens}  # fmt: skip


def run_fresh_stream(times: list[float], due: dict[int, list[dict]], window: int, gap: float, read_step, make_session,
                     isolation: int = 0) -> list[dict]:  # fmt: skip
    """All questions of one stream. `read_step(t)` returns the preprocessed step and is called once per step in
    increasing time order (sequential decoding, as in the stream run), only up to the last step any question needs.
    `due` maps a step index to the questions answered after it. `isolation` > 0 runs the A-B-A check on that many
    questions of the stream."""
    rows, checked = [], 0
    if not due:
        return rows
    buffer: deque = deque(maxlen=window)
    for k in range(max(due) + 1):
        buffer.append((times[k], read_step(times[k])))
        for q in due.get(k, []):
            steps = list(buffer)
            result = answer_fresh(make_session, steps, q)
            row = {"id": q["id"], "t": q["time"], "delay": q.get("delay", 0.0), "evidence_age": None,
                   "correct": int(result["choice"] == q["answer"]), **result,
                   "t_step": float(times[k]), "step_window_end": round(float(times[k]) + gap / 2, 4),
                   "window_steps": len(steps), "first_step_t": float(steps[0][0]),
                   "kv_appended_tokens": result["kv_tokens"] + result["evicted_tokens"]}  # fmt: skip
            if checked < isolation:
                other = steps[: max(1, len(steps) // 2)]  # a different window in between (B)
                answer_fresh(make_session, other, q)
                again = answer_fresh(make_session, steps, q)  # A again, on another new session
                row["isolation_max_abs_diff"] = max(abs(a - b) for a, b in zip(result["scores"], again["scores"]))
                row["isolation_same_choice"] = again["choice"] == result["choice"]
                checked += 1
            rows.append(row)
    return rows


def evaluate_recent_window(cfg, streams_path: str, video_root: str, output: str, window: int, checkpoint: str,
                           adapter: str | None = None, device: str = "cuda", split: str = "dev", shard=(0, 1),
                           isolation: int = 0) -> dict:  # fmt: skip
    """Fresh-session recent-window evaluation of every question of every stream (see the module docstring)."""
    from .backbone import Backbone
    from .data import StepReader
    from .harness import _modules, clean_partial, read_unit, run_fingerprint, write_unit
    from .session import StreamSession
    from .train_stage2 import chunk_plan, stream_times, summarize_streams

    if window < 1:
        raise ValueError("window must be at least one step")
    records = [r for r in read_jsonl(streams_path) if r.get("split") == split and r.get("questions")]
    if not records:
        raise ValueError(f"no streams with split == {split}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(cfg, streams_path, f"recent-window-fresh-{window}-{split}-isolation-{isolation}",
                                  checkpoint, adapter, video_root, records)  # fmt: skip
    run_file = output / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()).get("id") != fingerprint["id"]:
        raise ValueError(f"{output} holds a different run; use a new --output")
    run_file.write_text(json.dumps(fingerprint, indent=2) + "\n")
    units = output / "streams"
    if units.exists():
        clean_partial(units)
    done = {r["id"] for r in records if read_unit(units, r["id"], fingerprint["id"]) is not None}
    todo = [i for i, r in enumerate(records) if r["id"] not in done and i % shard[1] == shard[0]]
    bb = modules = None
    if todo:
        bb = Backbone(cfg.backbone, cfg.stream, device)
        if adapter:
            bb.add_lora(0, 0, adapter=adapter)
            bb.model.eval()
        modules = _modules(cfg, bb, checkpoint, device)
    gap = cfg.stream.frame_gap

    def plan(record):
        times = stream_times(record, cfg.stream.step_seconds, gap, cfg.stage2.max_seconds)
        due = {steps[-1]: qs for steps, qs in chunk_plan(times, record["questions"], len(times), gap) if qs} if times else {}
        return times, due

    for index in todo:
        record = records[index]
        times, due = plan(record)
        if not times:
            continue
        reader = StepReader(record, video_root)
        try:
            started = clock.perf_counter()
            rows = run_fresh_stream(
                times, due, window, gap,
                read_step=lambda t: bb.preprocess(bb.resize_step(reader.step(t, gap))),
                make_session=lambda: StreamSession(bb, cfg, modules, mode="v3"),
                isolation=isolation,
            )  # fmt: skip
            elapsed = round(clock.perf_counter() - started, 2)
        finally:
            reader.close()
        for row in rows:
            row.update({"stream": record["id"], "stream_steps": len(times), "stream_seconds_elapsed": elapsed,
                        "window": window, "run": fingerprint["id"]})  # fmt: skip
        write_unit(units, record["id"], rows, fingerprint["id"])
    rows, pending = [], 0
    for record in records:
        if not plan(record)[0]:
            continue
        unit = read_unit(units, record["id"], fingerprint["id"])
        if unit is None:
            if shard[1] == 1:
                raise RuntimeError(f"stream {record['id']} has no complete result")
            pending += 1
            continue
        rows += unit
    if pending:
        return {"shard": list(shard), "pending_streams_other_shards": pending}
    write_jsonl(output / "predictions.jsonl", rows)
    summary = summarize_streams(rows)
    summary["window_steps"] = window
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------- checks and report
def _load(root: Path, arm: str) -> list[dict] | None:
    path = Path(root) / arm / "predictions.jsonl"
    return read_jsonl(path) if path.is_file() else None


def check_arms(preds: dict[str, list[dict] | None], questions: list[dict], arms: tuple[str, ...]) -> list[str]:
    """Coverage and protocol checks. `preds` holds the reused fifo arms and the new arms present in `arms`."""
    expected = {q["id"]: q for q in questions}
    problems = []
    for arm in arms:
        rows = preds.get(arm)
        if rows is None:
            problems.append(f"{arm}: no predictions")
            continue
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            problems.append(f"{arm}: {len(set(ids) & set(expected))}/{len(expected)} questions, {len(ids) - len(set(ids))} duplicates")
            continue
        fresh = arm.split("_", 1)[1] in FRESH_WINDOWS
        for r in rows:
            q = expected[r["id"]]
            if r["stream"] != q["stream"] or abs(float(r["t"]) - q["time"]) > 1e-6:
                problems.append(f"{arm}: {r['id']} stream/time differs from the manifest")
                break
            if r.get("step_window_end") is None or r["step_window_end"] > q["time"] + 1e-6:
                problems.append(f"{arm}: {r['id']} uses frames after the query time")
                break
            if fresh:
                w = FRESH_WINDOWS[arm.split("_", 1)[1]]
                if r["evicted_tokens"] != 0 or r["kv_appended_tokens"] != r["kv_tokens"]:
                    problems.append(f"{arm}: {r['id']} evicted tokens in a fresh session")
                    break
                if r["session_steps"] != r["window_steps"] or not 1 <= r["window_steps"] <= w:
                    problems.append(f"{arm}: {r['id']} session saw {r['session_steps']} steps for a window of {r['window_steps']}")
                    break
                if r["first_step_t"] > r["t_step"] + 1e-6:
                    problems.append(f"{arm}: {r['id']} window starts after the answering step")
                    break
    reference = preds.get("s1_fifo")
    if reference is not None:
        ref = {r["id"]: (r.get("t_step"), r.get("step_window_end")) for r in reference}
        for arm in arms:
            rows = preds.get(arm)
            if rows is not None and {r["id"]: (r.get("t_step"), r.get("step_window_end")) for r in rows} != ref:
                problems.append(f"{arm}: answering step differs from the §14.3c s1_fifo rows")
    for control in ("fifo",) + CONTROLS:
        a, b = preds.get(f"s1_{control}"), preds.get(f"s2_{control}")
        if a is None or b is None or f"s1_{control}" not in arms or f"s2_{control}" not in arms:
            continue
        keys = ("kv_tokens", "window_steps", "first_step_t") if control in FRESH_WINDOWS else ("kv_tokens",)
        ka, kb = ({r["id"]: tuple(r.get(k) for k in keys) for r in rows} for rows in (a, b))
        if ka != kb:
            problems.append(f"{control}: {keys} differ between S1 and S2 for {sum(ka.get(i) != kb.get(i) for i in ka)} questions")
    return problems


def contrast(correct: dict[str, dict[str, int]], meta: dict[str, dict], weights: dict[str, float],
             samples: int = 4000, seed: int = 7) -> dict:  # fmt: skip
    """Paired contrast sum_arm weight * accuracy (pp) with a video-cluster bootstrap (4,000 draws, seed 7, the §14
    percentile convention). All arms are resampled on the SAME videos, so a difference of differences (an
    interaction) is one statistic on one bootstrap sample."""
    tallies = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # video -> task -> [sum of weighted correct, n]
    for key, q in meta.items():
        cell = tallies[q["video"]][q["task"]]
        cell[0] += sum(w * correct[arm][key] for arm, w in weights.items())
        cell[1] += 1
    videos = sorted(tallies)

    def bootstrap(stat):
        rng = random.Random(seed)
        draws = sorted(stat(rng.choices(videos, k=len(videos))) for _ in range(samples))
        return [round(draws[int(0.025 * samples)], 3), round(draws[int(0.975 * samples) - 1], 3)]

    def pooled(tasks):
        def stat(sample):
            num = sum(tallies[v][t][0] for v in sample for t in tasks if t in tallies[v])
            den = sum(tallies[v][t][1] for v in sample for t in tasks if t in tallies[v])
            return 100 * num / den if den else 0.0

        n = sum(tallies[v][t][1] for v in videos for t in tasks if t in tallies[v])
        return {"n": n, "delta_pp": round(stat(videos), 3) if n else None, "ci95_pp": bootstrap(stat) if n else [None, None]}

    def macro(sample):
        deltas = []
        for t in BACKWARD:
            num = sum(tallies[v][t][0] for v in sample if t in tallies[v])
            den = sum(tallies[v][t][1] for v in sample if t in tallies[v])
            if den:
                deltas.append(100 * num / den)
        return sum(deltas) / len(deltas) if deltas else 0.0

    present = {q["task"] for q in meta.values()}
    return {"backward": pooled(BACKWARD), "realtime": pooled(REALTIME), "all": pooled(BACKWARD + REALTIME),
            "backward_task_macro": {"delta_pp": round(macro(videos), 3), "ci95_pp": bootstrap(macro)},
            "per_task": {t: pooled((t,)) for t in BACKWARD + REALTIME if t in present}}  # fmt: skip


def recent_window_report(eval_root: str | Path, fifo_root: str | Path, questions_path: str | Path,
                         samples: int = 4000, seed: int = 7) -> dict:  # fmt: skip
    questions = read_jsonl(questions_path)
    meta = {q["id"]: q for q in questions}
    arms = tuple(f"{c}_fifo" for c in CHECKPOINTS) + NEW_ARMS
    preds = {f"{c}_fifo": _load(Path(fifo_root), f"{c}_fifo") for c in CHECKPOINTS}
    preds.update({arm: _load(Path(eval_root), arm) for arm in NEW_ARMS})
    problems = check_arms(preds, questions, arms)
    out: dict = {"protocol": "ovo-transfer-dev-v2 / recent-window-control-v1", "questions": len(questions),
                 "coverage_problems": problems, "scope": SCOPE,
                 "fresh4_note": "4 steps = 8 sampled frames over 4 s; not a Mage-VL / SimpleStream four-frame reproduction"}  # fmt: skip
    if problems:
        out["status"] = "invalid_coverage"
        return out
    out["status"] = "complete"
    correct = {arm: {r["id"]: int(r["correct"]) for r in rows} for arm, rows in preds.items()}

    def accuracy(arm, tasks):
        ids = [k for k in meta if meta[k]["task"] in tasks]
        return round(100 * sum(correct[arm][k] for k in ids) / len(ids), 3) if ids else None

    out["accuracy_pct"] = {arm: {"backward": accuracy(arm, BACKWARD), "realtime": accuracy(arm, REALTIME),
                                 "all": accuracy(arm, BACKWARD + REALTIME),
                                 "per_task": {t: accuracy(arm, (t,)) for t in BACKWARD + REALTIME if accuracy(arm, (t,)) is not None}}
                           for arm in arms}  # fmt: skip
    out["kv_tokens_mean"] = {arm: round(sum(r["kv_tokens"] for r in preds[arm]) / len(preds[arm]), 2) for arm in arms}

    def c(weights):
        return contrast(correct, meta, weights, samples, seed)

    contrasts = {}
    for ck in CHECKPOINTS:
        contrasts[f"{ck}: fifo - fresh16 (reliance on persistent state; primary control)"] = c({f"{ck}_fifo": 1, f"{ck}_fresh16": -1})
        contrasts[f"{ck}: fifo - stream_window (diagnostic)"] = c({f"{ck}_fifo": 1, f"{ck}_stream_window": -1})
        contrasts[f"{ck}: stream_window - fresh16 (diagnostic)"] = c({f"{ck}_stream_window": 1, f"{ck}_fresh16": -1})
        contrasts[f"{ck}: fresh4 - fresh16 (descriptive)"] = c({f"{ck}_fresh4": 1, f"{ck}_fresh16": -1})
    for control in ("fifo",) + CONTROLS:
        contrasts[f"s2 - s1 under {control}"] = c({f"s2_{control}": 1, f"s1_{control}": -1})
    for control in CONTROLS:
        contrasts[f"interaction: (s2 - s1)_fifo - (s2 - s1)_{control}"] = c(
            {"s2_fifo": 1, "s1_fifo": -1, f"s2_{control}": -1, f"s1_{control}": 1})  # fmt: skip
    out["contrasts"] = contrasts
    inter = contrasts["interaction: (s2 - s1)_fifo - (s2 - s1)_fresh16"]["backward"]
    lo, hi = inter["ci95_pp"]
    out["reading"] = {
        "interaction_backward": inter,
        "differential_effect": ("the S2 - S1 difference differs between fifo and fresh16 (interaction CI excludes 0)"
                                if lo > 0 or hi < 0 else
                                "no detectable difference between the S2 - S1 gaps under fifo and fresh16 (interaction CI "
                                "contains 0); an S2 - S1 interval containing 0 under fresh16 is NOT evidence that the deficit "
                                "is tied to streaming state"),
        "note": "a CI containing 0 for fifo - fresh16 means no detectable reliance here, not that memory is unnecessary",
    }  # fmt: skip
    return out


def flow_check(eval_root: str | Path, streams_path: str | Path, fifo_flow_root: str | Path | None = None,
               arms=("s1_fresh16", "s2_fresh16")) -> dict:  # fmt: skip
    """GPU flow check on a few streams: coverage, protocol fields, S1/S2 identical inputs, and the A-B-A isolation
    result (every checked question: same choice, option log-probabilities within ISOLATION_TOLERANCE)."""
    streams = read_jsonl(streams_path)
    questions = [{**q, "stream": s["id"], "video": s["video"]} for s in streams for q in s["questions"]]
    preds = {a: _load(Path(eval_root), a) for a in arms}
    if fifo_flow_root is not None:
        preds["s1_fifo"] = _load(Path(fifo_flow_root), "s1_fifo")
    problems = check_arms(preds, questions, tuple(arms))
    isolation = {}
    for arm in arms:
        rows = [r for r in preds.get(arm) or [] if "isolation_max_abs_diff" in r]
        worst = max((r["isolation_max_abs_diff"] for r in rows), default=None)
        isolation[arm] = {"checked": len(rows), "max_abs_diff": worst, "same_choice": all(r["isolation_same_choice"] for r in rows)}
        if not rows:
            problems.append(f"{arm}: no A-B-A isolation check was run")
        elif worst > ISOLATION_TOLERANCE or not isolation[arm]["same_choice"]:
            problems.append(f"{arm}: A-B-A isolation failed (max diff {worst}); state leaks between fresh sessions")
    seconds = {}
    for arm in arms:
        rows = preds.get(arm) or []
        per_stream = {r["stream"]: r.get("stream_seconds_elapsed", 0.0) for r in rows}
        seconds[arm] = round(sum(per_stream.values()) / max(1, len(rows)), 3)
    return {"streams": len(streams), "questions": len(questions), "problems": problems, "passed": not problems,
            "isolation": isolation, "tolerance": ISOLATION_TOLERANCE, "wall_seconds_per_question": seconds}  # fmt: skip
