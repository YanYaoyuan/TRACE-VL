"""Reviewed training-only evidence pairs and separately labelled refusal probes.

Never infer unanswerability from an arbitrary crop. A reviewer must inspect the
actual sampled frames and attest that neither the question nor remaining frames
suffice. OVO/HLD is evaluation only, never a source of training labels.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .data import decode_observation, sha256
from .prepare import digest

UNKNOWN = 'Insufficient visual evidence to answer.'


def add_option(row, position):
    options = list(row['options'])
    if len(options) >= 26 or UNKNOWN in options:
        raise ValueError('cannot insert evidence option')
    options.insert(position, UNKNOWN)
    return {**row, 'options': options, 'answer': row['answer'] + int(row['answer'] >= position),
            'abstention_index': position}


def probe_rows(rows, limit=200, seed=17):
    """Fixed source-answerable MC subset; not human-verified cropped evidence.

    Exclude HLD/any explicit unknown option. Report option-selection rate as a
    diagnostic proxy, never a camera false-alarm rate or definitive false refusal.
    """
    eligible = [r for r in rows if r.get('task') != 'HLD' and
                not any(x in o.lower() for o in r['options']
                        for x in ('unable to', 'cannot determine', 'not enough', 'insufficient', 'unknown', 'not sure'))]
    eligible.sort(key=lambda r: digest([seed, 'abstention-probe-v1', r['id']]))
    counts, result = Counter(), []
    for row in eligible[:limit]:
        n = len(row['options']) + 1
        position = counts[n] % n  # balanced position within each option-count stratum
        counts[n] += 1
        result.append({**add_option(row, position), 'source_question_id': row['id'],
                       'id': 'abstention-probe-' + row['id'], 'split': 'val',
                       'question': row['question'] + '\nUse only the supplied video evidence; choose the insufficient-evidence option if necessary.'})
    return result


def reviewed_training_pairs(path, training, *, steps, pixels):
    """Strict intake; empty/unreviewed/held-out pairs fail before GPU use.

    Schema is documented in docs/core_vlm/preservation_pilot.md. Byte hashes and
    decoded RGB hashes bind review to exactly the training observation.
    """
    source = {r['id']: r for r in training}
    pairs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not pairs:
        raise ValueError('review file has no pairs')
    used, result = set(), []
    for pair in pairs:
        sid = pair['source_id']
        if sid not in source or sid in used:
            raise ValueError('review must reference a unique frozen training question')
        used.add(sid)
        row = source[sid]
        if row.get('split') != 'train' or row.get('task') == 'HLD':
            raise ValueError('reviewed evidence supervision is training-only')
        if not all(isinstance(pair.get(k), str) and pair[k].strip() for k in ('reviewer', 'reason')):
            raise ValueError('human review identity and evidence rationale required')
        if pair.get('question_and_sampled_frames_reviewed') is not True:
            raise ValueError('review of question and sampled frames required')
        if sha256(row['video']) != pair['video_sha256']:
            raise ValueError('reviewed video changed')
        intervals = pair['evidence_intervals']
        if not intervals or any(len(x) != 2 or not all(math.isfinite(v) for v in x)
                                or not 0 <= x[0] <= x[1] for x in intervals):
            raise ValueError('finite required evidence intervals needed')
        views = pair['views']
        if len(views) != 2 or {v.get('answerable') for v in views} != {True, False}:
            raise ValueError('each reviewed pair needs answerable and insufficient views')
        position = int(digest(['review-option-v1', sid])[:8], 16) % (len(row['options']) + 1)
        rgb = set()
        for view in views:
            if type(view['answerable']) is not bool:
                raise ValueError('answerable must be boolean')
            candidate = {**add_option(row, position), 'observation': view['observation'],
                         'query_time': view['query_time'],
                         'question': row['question'] + '\nUse only the supplied video evidence; choose the insufficient-evidence option if necessary.'}
            _, times, resolved, audit = decode_observation(candidate, steps=steps, pixels=pixels)
            if audit['decoded_rgb_sha256'] != view['decoded_rgb_sha256']:
                raise ValueError('reviewed sampled frames changed')
            rgb.add(audit['decoded_rgb_sha256'])
            seen = [any(lo <= float(t) <= hi for t in times.flat) for lo, hi in intervals]
            if (view['answerable'] and not all(seen)) or (not view['answerable'] and any(seen)):
                raise ValueError('sampled evidence does not match reviewed label')
            result.append({**resolved, 'id': 'reviewed-' + sid + '-' + str(view['answerable']).lower(),
                           'answer': resolved['answer'] if view['answerable'] else position,
                           'mixture_family': 'reviewed', 'review_source_id': sid,
                           'review': {'reviewer': pair['reviewer'], 'reason': pair['reason'],
                                      'answerable': view['answerable'], 'audit': audit}})
        if len(rgb) != 2:
            raise ValueError('paired views must differ')
    return result
