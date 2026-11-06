"""MEVA-derived long-gap benchmark for fixed-camera inspection (public data, CC BY 4.0).

MEVA object tracks only cover their own activity, and a put-down and the later pick-up of the same
object use different object IDs. Persistence is therefore reconstructed by linking each put-down to
the first later pick-up at the same place (box IoU or nearby centers). Links are heuristic: export
`audit.csv` and report the human-verified label accuracy with any result. Door activities are not
used because closings are almost never annotated.

Question types (answers are option letters; all stream-causal):
- placed_and_removed: did someone put down an object in the red box and pick it up again within the
  last W minutes? The current frame looks the same either way, so this needs memory.
- duration: how long has the object in the red box been there? (4 bins) Needs the onset time.
- count: how many times was an object put down in the red box in the last W minutes? (0/1/2/3+)
- still_there: control; is the object put down in the red box still there? The current frame suffices.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = "https://gitlab.kitware.com/meva/meva-data-repo/-/raw/master/annotation/DIVA-phase-2/MEVA/"
SUBSETS = ("kitware", "kitware-meva-training")
CLIP_LIST = REPO + "kitware/list-of-annotated-meva-clips.txt"
S3 = "https://mevadata-public-01.s3.amazonaws.com"
FPS = 30.0
FRAME_SIZE = (1920, 1080)
ACTIVITIES = {"person_puts_down_object": "put", "person_picks_up_object": "pick"}
CLIP = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d{2})-(\d{2})-(\d{2})\.(\d{2})-(\d{2})-(\d{2})\.([A-Za-z0-9-]+)\.(G\d+)$")
GEOM = re.compile(r"id1'?:\s*(\d+).*?ts0'?:\s*(\d+)|ts0'?:\s*(\d+).*?id1'?:\s*(\d+)")
BOX = re.compile(r"g0'?:\s*'?(\d+) (\d+) (\d+) (\d+)")

GAPS = (15, 60, 180, 480, 900)
WINDOWS = (120, 300, 600, 1200, 1800)
DURATIONS = (20, 40, 150, 240, 450, 700, 1100, 1500)
DURATION_OPTIONS = ["Less than 1 minute", "1 to 5 minutes", "5 to 15 minutes", "More than 15 minutes"]
COUNT_OPTIONS = ["0", "1", "2", "3 or more"]


def parse_clip(name):
    match = CLIP.match(name)
    if not match:
        return None
    date, h0, m0, s0, h1, m1, s1, site, camera = match.groups()
    start = int(h0) * 3600 + int(m0) * 60 + int(s0)
    end = int(h1) * 3600 + int(m1) * 60 + int(s1)
    if end <= start:
        end += 86400
    return {"name": name, "date": date, "hour": h0, "start": start, "end": end, "site": site, "camera": camera}


def _get(url, attempts=4):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    return None


def list_clips():
    text = _get(CLIP_LIST).decode()
    return [c for c in (parse_clip(line.strip()) for line in text.splitlines()) if c]


def _yaml_items(text):
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        try:
            value = yaml.safe_load(line)
        except yaml.YAMLError:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            yield value[0]


def _activity_events(activities_text, types_text):
    types = {}
    for item in _yaml_items(types_text):
        if "types" in item:
            types[item["types"]["id1"]] = next(iter(item["types"]["cset3"]))
    counts, events = Counter(), []
    for item in _yaml_items(activities_text):
        act = item.get("act")
        if not act:
            continue
        name = next(iter(act["act2"]))
        counts[name] += 1
        if name not in ACTIVITIES:
            continue
        span = act["timespan"][0]["tsr0"]
        actors = [(a["id1"], types.get(a["id1"], "unknown"), a["timespan"][0]["tsr0"]) for a in act["actors"]]
        objects = [a for a in actors if a[1] not in {"person", "vehicle"}]
        if len(objects) != 1:
            continue  # ambiguous or missing object actor
        events.append(
            {
                "kind": ACTIVITIES[name],
                "frames": [int(span[0]), int(span[1])],
                "object_id": objects[0][0],
                "object_type": objects[0][1],
                "object_frames": [int(objects[0][2][0]), int(objects[0][2][1])],
            }
        )
    return events, counts


def _object_boxes(url, wanted):
    """Stream a (large) geometry file and keep only the boxes of the wanted object IDs."""
    boxes = defaultdict(dict)
    with urllib.request.urlopen(url, timeout=600) as response:
        for raw in response:
            line = raw.decode(errors="replace")
            match = GEOM.search(line)
            if not match:
                continue
            object_id = int(match.group(1) or match.group(4))
            if object_id not in wanted:
                continue
            frame = int(match.group(2) or match.group(3))
            box = BOX.search(line)
            if box:
                boxes[object_id][frame] = [int(v) for v in box.groups()]
    return boxes


def fetch_clip(clip, output):
    target = Path(output) / "clips" / f"{clip['name']}.json"
    if target.exists():
        return json.loads(target.read_text())
    for subset in SUBSETS:
        base = f"{REPO}{subset}/{clip['date']}/{clip['hour']}/{clip['name']}"
        activities = _get(base + ".activities.yml")
        if activities is None:
            continue
        types = _get(base + ".types.yml") or b""
        events, counts = _activity_events(activities.decode(), types.decode())
        boxes = _object_boxes(base + ".geom.yml", {e["object_id"] for e in events}) if events else {}
        kept = []
        for event in events:
            track = boxes.get(event["object_id"], {})
            if not track:
                continue
            frames = sorted(track)
            event["box_first"], event["box_last"] = track[frames[0]], track[frames[-1]]
            kept.append(event)
        record = {**clip, "subset": subset, "events": kept, "activity_counts": dict(counts)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record) + "\n")
        return record
    return None


def fetch_annotations(output, sites=None, limit=None, workers=8):
    clips = list_clips()
    if sites:
        clips = [c for c in clips if c["site"] in set(sites)]
    if limit:
        clips = clips[:limit]
    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(lambda c: fetch_clip(c, output), clips))
    found = [r for r in records if r]
    summary = {
        "clips_listed": len(clips),
        "clips_with_annotations": len(found),
        "put_events": sum(e["kind"] == "put" for r in found for e in r["events"]),
        "pick_events": sum(e["kind"] == "pick" for r in found for e in r["events"]),
    }
    (Path(output) / "fetch_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _normalize(box):
    w, h = FRAME_SIZE
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center_in(box, region):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _same_place(a, b):
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return _iou(a, b) >= 0.3 or (dx * dx + dy * dy) ** 0.5 <= 0.5 * size


def _region(box, scale=2.0, minimum=0.06):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half_w = max((box[2] - box[0]) * scale, minimum) / 2
    half_h = max((box[3] - box[1]) * scale, minimum) / 2
    return [round(max(0.0, cx - half_w), 4), round(max(0.0, cy - half_h), 4),
            round(min(1.0, cx + half_w), 4), round(min(1.0, cy + half_h), 4)]  # fmt: skip


def build_streams(records, max_gap=2.0):
    """Chain time-contiguous clips of the same camera into streams with clip offsets."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["date"], record["site"], record["camera"])].append(record)
    streams = {}
    for (date, site, camera), clips in groups.items():
        clips.sort(key=lambda c: c["start"])
        chain = [clips[0]]
        for clip in clips[1:] + [None]:
            if clip is not None and abs(clip["start"] - chain[-1]["end"]) <= max_gap:
                chain.append(clip)
                continue
            start = chain[0]["start"]
            name = f"{date}.{site}.{camera}.{start // 3600:02d}{start % 3600 // 60:02d}"
            streams[name] = {
                "camera": f"{site}.{camera}",
                "frame_size": list(FRAME_SIZE),
                "duration": chain[-1]["end"] - start,
                "clips": [
                    {
                        "name": c["name"],
                        "video": f"{c['date']}/{c['hour']}/{c['name']}.r13.avi",
                        "offset": c["start"] - start,
                        "events": c["events"],
                    }
                    for c in chain
                ],
            }
            chain = [clip] if clip is not None else []
    return streams


def link_events(stream):
    """Stream-time put/pick events and resting intervals linked by place."""
    events = []
    for clip in stream["clips"]:
        for e in clip["events"]:
            t0, t1 = (clip["offset"] + f / FPS for f in e["frames"])
            box = _normalize(e["box_last"] if e["kind"] == "put" else e["box_first"])
            events.append({**e, "t0": t0, "t1": t1, "box": box, "clip": clip["name"]})
    events.sort(key=lambda e: e["t0"])
    used, rests = set(), []
    for i, put in enumerate(events):
        if put["kind"] != "put":
            continue
        pick = next(
            (
                j
                for j, e in enumerate(events)
                if e["kind"] == "pick" and j not in used and e["t0"] > put["t1"] and _same_place(put["box"], e["box"])
            ),
            None,
        )
        if pick is not None:
            used.add(pick)
        rests.append(
            {
                "box": put["box"],
                "put": i,
                "pick": pick,
                "start": put["t1"],
                "end": events[pick]["t0"] if pick is not None else stream["duration"],
                "removed": events[pick]["t1"] if pick is not None else None,
                "censored": pick is None,
            }
        )
    return events, rests


def _quiet(region, a, b, events, rests):
    """No put/pick in the region during [a, b] and no object resting there at any point of it."""
    if any(_center_in(e["box"], region) and e["t1"] >= a and e["t0"] <= b for e in events):
        return False
    return not any(_center_in(r["box"], region) and r["end"] >= a and r["start"] <= b for r in rests)


def _window(span, query_time):
    return next((w for w in WINDOWS if w >= span + 10 and w <= query_time), None)


def _ask(kind, stream_id, tq, window, region, options, answer, gap, provenance, censored=False):
    minutes = f"{window / 60:g} minutes"
    question = {
        "placed_and_removed": f"In the last {minutes}, did someone put down an object inside the red box "
        "and later pick it up again?",
        "duration": "Someone put down an object inside the red box and it is still there. "
        "How long has it been there?",
        "count": f"How many times did someone put down an object inside the red box in the last {minutes}?",
        "still_there": "Earlier, someone put down an object inside the red box. Is it still there now?",
    }[kind]
    key = f"{stream_id}|{kind}|{tq:.2f}|{region}"
    return {
        "id": hashlib.sha1(key.encode()).hexdigest()[:16],
        "stream": stream_id,
        "type": kind,
        "query_time": round(tq, 2),
        "window": window,
        "region": region,
        "question": question,
        "options": options,
        "answer": answer,
        "gap": None if gap is None else round(gap, 2),
        "censored": censored,
        "provenance": provenance,
    }


def questions_for(stream_id, stream, rng):
    events, rests = link_events(stream)
    end = stream["duration"] - 1
    out = []
    for rest in rests:
        put = events[rest["put"]]
        region = _region(put["box"])
        prov = {"put": [put["clip"], put["frames"]]}
        others = [r for r in rests if r is not rest and _center_in(r["box"], region)]

        def busy(tq):
            # Another object resting in the same box would make the question ambiguous.
            return any(r["start"] <= tq <= r["end"] for r in others)

        def completed(a, b):
            # Another put-down and pick-up in the box within [a, b] would turn a "No" into a "Yes".
            return any(
                r["removed"] is not None and events[r["put"]]["t0"] >= a and r["removed"] <= b for r in others
            )

        if rest["pick"] is not None:
            pick = events[rest["pick"]]
            prov["pick"] = [pick["clip"], pick["frames"]]
            for gap in rng.sample(GAPS, 2):
                tq = rest["removed"] + gap
                window = _window(tq - put["t0"], tq)
                if tq <= end and window and not busy(tq):
                    out.append(_ask("placed_and_removed", stream_id, tq, window, region, ["Yes", "No"], 0, gap, prov))
                    out.append(_ask("still_there", stream_id, tq, window, region, ["Yes", "No"], 1, gap, prov))
            # Hard negative: the same region shortly before anything happened there.
            tq = put["t0"] - 5
            window = next((w for w in WINDOWS if w <= tq and _quiet(region, tq - w, tq, events, rests)), None)
            if window:
                out.append(_ask("placed_and_removed", stream_id, tq, window, region, ["Yes", "No"], 1, None, prov))
                out.append(_ask("count", stream_id, tq, window, region, COUNT_OPTIONS, 0, None, prov))
        for duration in DURATIONS:
            tq = rest["start"] + duration
            if tq > min(rest["end"] - 5, end):
                break
            if duration < 60:
                answer = 0
            elif duration < 300:
                answer = 1
            elif duration < 900:
                answer = 2
            else:
                answer = 3
            window = _window(duration, tq) or (WINDOWS[-1] if tq >= WINDOWS[-1] else None)
            if window is None or busy(tq):
                continue
            censored = rest["censored"]
            out.append(_ask("duration", stream_id, tq, window, region, DURATION_OPTIONS, answer, duration, prov, censored))
            # Placed but still present: not "picked up again".
            if duration >= 40 and not completed(tq - window, tq):
                out.append(_ask("placed_and_removed", stream_id, tq, window, region, ["Yes", "No"], 1, duration, prov, censored))
                out.append(_ask("still_there", stream_id, tq, window, region, ["Yes", "No"], 0, duration, prov, censored))
        gap = rng.choice(GAPS)
        tq = put["t1"] + gap
        window = _window(gap, tq)
        if tq <= end and window:
            n = sum(1 for e in events if e["kind"] == "put" and _center_in(e["box"], region) and tq - window <= e["t0"] <= tq)
            out.append(_ask("count", stream_id, tq, window, region, COUNT_OPTIONS, min(n, 3), gap, prov))
    return out


def balance(questions, rng, per_answer_cap=None):
    """Equalize answers within each question type by downsampling the majority answers.

    Balancing per gap bucket would discard every hard negative (they have no gap); gap analysis
    therefore reports balanced accuracy per bucket instead.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for q in questions:
        groups[q["type"]][q["answer"]].append(q)
    kept = []
    for answers in groups.values():
        n = min(len(v) for v in answers.values()) if len(answers) > 1 else 0
        if per_answer_cap:
            n = min(n, per_answer_cap)
        for items in answers.values():
            kept.extend(rng.sample(items, n))
    return sorted(kept, key=lambda q: (q["stream"], q["query_time"]))


def gap_bucket(gap):
    if gap is None:
        return "none"
    for limit, name in ((30, "<30s"), (120, "30s-2m"), (300, "2-5m"), (600, "5-10m")):
        if gap < limit:
            return name
    return ">10m"


def build(annotations, output, seed=7, splits=(0.7, 0.1, 0.2), balance_answers=True):
    rng = random.Random(seed)
    records = [json.loads(p.read_text()) for p in sorted((Path(annotations) / "clips").glob("*.json"))]
    streams = build_streams(records)
    questions = []
    for stream_id in sorted(streams):
        questions.extend(questions_for(stream_id, streams[stream_id], rng))
    unique = {q["id"]: q for q in questions}
    questions = list(unique.values())
    if balance_answers:
        questions = balance(questions, rng)
    cameras = sorted({s["camera"] for s in streams.values()}, key=lambda c: hashlib.sha1(f"{seed}:{c}".encode()).hexdigest())
    n_train, n_val = round(len(cameras) * splits[0]), round(len(cameras) * splits[1])
    split_of = {c: "train" if i < n_train else "val" if i < n_train + n_val else "test" for i, c in enumerate(cameras)}
    for q in questions:
        q["split"] = split_of[streams[q["stream"]]["camera"]]
    used = {q["stream"] for q in questions}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    spec = {k: {**v, "clips": [{x: c[x] for x in ("name", "video", "offset")} for c in v["clips"]]} for k, v in streams.items() if k in used}
    (output / "streams.json").write_text(json.dumps(spec, indent=1) + "\n")
    write_audit(questions, spec, output / "audit.csv", rng)
    linked = {}
    for name in used:
        events, rests = link_events(streams[name])
        linked[name] = {"events": events, "rests": rests}
    (output / "events.json").write_text(json.dumps(linked) + "\n")
    report = {
        "streams": len(spec),
        "cameras": dict(Counter(split_of.values())),
        "questions": len(questions),
        "by_split": dict(Counter(q["split"] for q in questions)),
        "by_type": dict(Counter(q["type"] for q in questions)),
        "by_type_answer": {f"{k[0]}:{k[1]}": v for k, v in Counter((q["type"], q["answer"]) for q in questions).items()},
        "by_gap": dict(Counter(gap_bucket(q["gap"]) for q in questions)),
        "censored_fraction": sum(q["censored"] for q in questions) / max(1, len(questions)),
        "stream_minutes": round(sum(s["duration"] for s in spec.values()) / 60, 1),
        "seed": seed,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def write_audit(questions, streams, path, rng, per_type=40):
    """Sample test questions for human verification of the heuristic labels."""
    rows = []
    for kind in sorted({q["type"] for q in questions}):
        pool = [q for q in questions if q["type"] == kind and q["split"] == "test"]
        for q in rng.sample(pool, min(per_type, len(pool))):
            clips = streams[q["stream"]]["clips"]
            at = max((c for c in clips if c["offset"] <= q["query_time"]), key=lambda c: c["offset"])
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "question": q["question"],
                    "answer": q["options"][q["answer"]],
                    "query_clip": at["name"],
                    "query_seconds_in_clip": round(q["query_time"] - at["offset"], 1),
                    "window_seconds": q["window"],
                    "region_xyxy": " ".join(map(str, q["region"])),
                    "evidence": json.dumps(q["provenance"]),
                    "label_correct(y/n)": "",
                    "notes": "",
                }
            )
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)


def video_urls(streams_path, output, split_questions=None):
    """Resolve S3 keys for the clips used by the benchmark; rewrite streams.json video paths."""
    streams = json.loads(Path(streams_path).read_text())
    wanted = None
    if split_questions:
        wanted = {json.loads(x)["stream"] for x in Path(split_questions).read_text().splitlines() if x.strip()}
    urls, missing = [], []
    top = (_get(f"{S3}/?list-type=2&delimiter=/") or b"").decode()
    drops = ["drops-123-r13/"] + [d for d in re.findall(r"<Prefix>([^<]+/)</Prefix>", top) if d != "drops-123-r13/"]
    for stream_id, stream in streams.items():
        if wanted is not None and stream_id not in wanted:
            continue
        for clip in stream["clips"]:
            date, hour = clip["name"][:10], clip["name"][11:13]
            keys = []
            for drop in drops:
                listing = _get(f"{S3}/?list-type=2&prefix={drop}{date}/{hour}/{clip['name']}") or b""
                keys = re.findall(r"<Key>([^<]+\.avi)</Key>", listing.decode())
                if keys:
                    break
            if not keys:
                missing.append(clip["name"])
                continue
            key = keys[0]
            clip["video"] = key.split("/", 1)[1]
            urls.append(f"{S3}/{key}\n  out={clip['video']}")
    Path(streams_path).write_text(json.dumps(streams, indent=1) + "\n")
    Path(output).write_text("\n".join(urls) + "\n")
    return {"clips": len(urls), "missing": missing, "aria2c_input": str(output)}
