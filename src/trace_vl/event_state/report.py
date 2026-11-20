from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np

from .common import read_json, read_rows, write_json, write_rows


def outcome(row, threshold):
    alert = next((t for t, p in zip(row['ticks'], row['scores']) if p >= threshold), None)
    early = alert is not None and alert < row['response_s']
    delay = None if alert is None else alert-row['response_s']
    return {'early': early, 'on_time': delay is not None and 0 <= delay <= 3,
            'miss': alert is None, 'late': delay is not None and delay > 3, 'delay': delay}


def calibrate(rows, target):
    if not rows:
        raise ValueError('Empty calibration set')
    # Fixed grid and tie-break; NEVER tune with dev predictions.
    thresholds = np.linspace(0, 1.000001, 1001)
    valid = []
    for th in thresholds:
        outcomes = [outcome(r, float(th)) for r in rows]
        early = np.mean([o['early'] for o in outcomes])
        score = np.mean([o['on_time'] for o in outcomes])
        if early <= target:
            valid.append((score, float(th)))
    return max(valid)[1]


def summary(rows, threshold):
    outcomes = [outcome(r, threshold) for r in rows]
    delay = [o['delay'] for o in outcomes if o['on_time']]
    return {'n': len(rows), 'videos': len({r['source_video_id'] for r in rows}), 'threshold': threshold,
            'on_time_3s_percent': 100*float(np.mean([o['on_time'] for o in outcomes])),
            'early_percent': 100*float(np.mean([o['early'] for o in outcomes])),
            'miss_percent': 100*float(np.mean([o['miss'] for o in outcomes])),
            'late_percent': 100*float(np.mean([o['late'] for o in outcomes])),
            'p95_delay_matched_s': float(np.percentile(delay, 95)) if delay else None,
            'sidecar_seconds_per_tick': sum(r['seconds'] for r in rows)/sum(len(r['ticks']) for r in rows)}


def paired(a, b, ta, tb, seed=7, samples=4000):
    aa, bb = {r['id']: r for r in a}, {r['id']: r for r in b}
    if len(aa) != len(a) or len(bb) != len(b) or set(aa) != set(bb):
        raise ValueError('Coverage mismatch or duplicate prediction IDs')
    groups = defaultdict(list)
    for key, x in aa.items():
        y = bb[key]
        if any(x[k] != y[k] for k in ('source_video_id', 'ticks', 'response_s', 'split')):
            raise ValueError('Paired metadata mismatch')
        groups[x['source_video_id']].append(int(outcome(x, ta)['on_time'])-int(outcome(y, tb)['on_time']))
    vals = list(groups.values())
    sums, counts = np.array([sum(v) for v in vals]), np.array([len(v) for v in vals])
    rng = np.random.default_rng(seed)
    ci = []
    for _ in range(samples):
        selected = rng.integers(len(vals), size=len(vals))
        ci.append(100*sums[selected].sum()/counts[selected].sum())
    return {'delta_pp': 100*float(sums.sum()/counts.sum()), 'ci95_pp': np.percentile(ci, [2.5, 97.5]).tolist(),
            'bootstrap_unit': 'source_video', 'videos': len(vals), 'samples': samples}


def curves(out, arms):
    colors = ['#e69f00', '#009e73', '#56b4e9', '#cc79a7', '#0072b2']
    series = {a: read_rows(Path(out)/'train'/a/'metrics.jsonl') for a in arms}
    mx = max(r['step'] for s in series.values() for r in s)
    my = max(1., max(r['loss'] for s in series.values() for r in s))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480" viewBox="0 0 960 480">',
             '<rect width="960" height="480" fill="white"/>',
             '<text x="60" y="28" font-size="18">Response timing: training loss (raw, no smoothing)</text>',
             '<path d="M60 50 V410 H900" fill="none" stroke="#333"/>',
             '<text x="400" y="460">Optimizer update</text>']
    for i, (arm, s) in enumerate(series.items()):
        points = ' '.join(f"{60+840*r['step']/mx:.1f},{410-350*r['loss']/my:.1f}" for r in s)
        parts += [f'<polyline points="{points}" fill="none" stroke="{colors[i%len(colors)]}"/>',
                  f'<text x="{70+i*170}" y="435" fill="{colors[i%len(colors)]}">{escape(arm)}</text>']
    for y in (0, my/2, my):
        parts.append(f'<text x="5" y="{410-350*y/my:.1f}">{y:.2f}</text>')
    parts += [f'<text x="880" y="430">{mx}</text>', '</svg>']
    (Path(out)/'training-curves.svg').write_text('\n'.join(parts))


def make_report(out, cfg):
    out = Path(out)
    thresholds, dev, scores, params = {}, {}, {}, {}
    for arm in cfg['arms']:
        rows = read_rows(out/'eval'/arm/'predictions.jsonl')
        cal = [r for r in rows if r['split'] == 'calibration']
        dev[arm] = [r for r in rows if r['split'] == 'dev']
        thresholds[arm] = calibrate(cal, cfg['early_rate_target'])
        scores[arm] = {'calibration': summary(cal, thresholds[arm]),
                       'dev': summary(dev[arm], thresholds[arm])}
        scores[arm]['by_source'] = {source: summary([r for r in dev[arm] if r.get('source', 'unknown') == source],
                                                        thresholds[arm])
                                    for source in sorted({r.get('source', 'unknown') for r in dev[arm]})}
        failures = [{**r, 'outcome': outcome(r, thresholds[arm]), 'threshold': thresholds[arm]}
                    for r in dev[arm] if not outcome(r, thresholds[arm])['on_time']]
        write_rows(out/'eval'/arm/'failure-cases.jsonl', failures[:50])
        params[arm] = read_json(out/'train'/arm/'complete.json')['parameters']
    comparisons = {f'event_state_minus_{a}': paired(dev['event_state'], dev[a],
                    thresholds['event_state'], thresholds[a], cfg['seed'])
                   for a in cfg['arms'] if a != 'event_state'}
    probes = {}
    for mode in ('zero_video', 'shuffled_query'):
        rows = read_rows(out/'eval'/'event_state'/f'{mode}.jsonl')
        probes[mode] = summary(rows, thresholds['event_state'])
    matched = abs(params['temporal']-params['event_state'])/params['event_state'] <= .1
    result = {'complete': True, 'protocol': cfg['protocol'], 'scores': scores, 'paired': comparisons,
              'parameters': params, 'temporal_parameter_matched_within_10_percent': matched,
              'diagnostic_ablations': probes, 'calibration_target': 'early response on <=5% trajectories',
              'not_surveillance_faph': True, 'not_confirmatory_benchmark': True,
              'next_stage': 'review response-timing and shortcut controls before expanding; no automatic scale-up',
              'limits': ['one seed; response timestamps may contain instruction/timing shortcuts',
                         'no earliest-knowable, physical event-clear, exhaustive normal-hour labels',
                         'not unseen-rule-composition validation; no official benchmark score generated',
                         'sidecar timings exclude vision/text feature extraction; see feature profiles',
                         'paired intervals are unadjusted exploratory comparisons; dev set is not final test']}
    write_json(out/'report.json', result)
    curves(out, cfg['arms'])
    lines = ['# 条件响应时机实验报告', '', '本轮测量标注响应时机，不等同于物业误报率或事件起止能力。', '',
             '| 模块 | 参数量 | 开发集 3 秒内响应率 | 提前响应率 | 未响应率 |', '|---|---:|---:|---:|---:|']
    for a in cfg['arms']:
        s = scores[a]['dev']
        lines.append(f"| {a} | {params[a]:,} | {s['on_time_3s_percent']:.2f}% | "
                     f"{s['early_percent']:.2f}% | {s['miss_percent']:.2f}% |")
    lines += ['', '阈值只由 calibration 选择，开发集不重新调阈值。所有比较见 report.json。', '',
              '![训练损失](training-curves.svg)', '', '## 解释限制', '']+[f'- {s}' for s in result['limits']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('VERIFIED: data, cached features, five final-checkpoint arms, calibration, dev and reports complete.', flush=True)
