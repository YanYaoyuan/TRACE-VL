"""Complete 250-source PTS inventory and stage outcome-independent past-state targets."""
import fcntl
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trace_vl.core_vlm.data import presentation_frames, probe_video, sha256
from trace_vl.core_vlm.longstream import schedule
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.streaming_vision.data import _streaming_rows, write_rows

ROOT = Path('/data/yyy/tracevl-data/streaming-delayed-states-20261125-001')
INV = Path('/data/yyy/tracevl-tools/delayed-state-development-inventory-20261125.json')
OLD = Path('/nas-data/vlm/yyy/tracevl-runs/streaming-vision-a566104-001/streamingbench.jsonl')
CSV = Path('/nas-data/vlm/data/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv')
EXPECTED_INVENTORY_GZIP = 'ac087c6939224b96fc758b2cf3d69af5e60c233ec980306126983b3ce1c95ebb'
STATE_TASKS = {'Action Perception', 'Object Perception'}


def signature(p):
    s = p.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def stage_file(source, destination):
    before = signature(source)
    source_hash = sha256(source)
    if destination.exists():
        assert sha256(destination) == source_hash
    else:
        tmp = destination.with_suffix('.partial')
        try:
            with source.open('rb') as src, tmp.open('wb') as dst:
                for block in iter(lambda: src.read(8 << 20), b''):
                    dst.write(block)
            assert sha256(tmp) == source_hash
            assert signature(source) == before
            tmp.replace(destination)
        finally:
            tmp.unlink(missing_ok=True)
    assert signature(source) == before
    return source_hash


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert sha256(INV.with_suffix('.json.gz')) == EXPECTED_INVENTORY_GZIP
    assert gzip.decompress(INV.with_suffix('.json.gz').read_bytes()) == INV.read_bytes()
    assert sha256(OLD) == '33457ec0ea3319fa97d2a9b289b2feadde5c19ed0574afb5ce87dbd94497a757'
    assert sha256(CSV) == '0121417a71e1dcf367222ae97e9512e3e99725d72e60d50760a050816ccd58da'
    inv = json.loads(INV.read_text())
    old = {r['source_group']: r for r in map(json.loads, OLD.read_text().splitlines())}
    assert len(old) == 250 and len(inv['metadata']) == 227 and len(inv['metadata_unresolved']) == 23
    # Freeze the semantic/availability rule BEFORE resolving the missing metadata.
    rule = {'tasks': sorted(STATE_TASKS), 'temporal_clue_type': 'Concurrent',
            'minimum_target_seconds': 60, 'delays_seconds': [0, 15, 60, 180, 300],
            'one_target_per_source': 'earliest eligible cutoff, then lexicographic question ID',
            'eligibility': 'all250 existing development sources, fullPTS supports cutoff+300',
            'counting': 'inventory only; cumulative counting is not a snapshot-local task',
            'selection_uses_model_outcomes': False, 'new_reserved_sources': 0,
            'scope': 'New controlled anchored past-state task; not official StreamingBench or unseen sources. No question/gold/media modification except explicit original-cutoff anchoring at evaluation. Existing results unchanged.'}
    rule['id'] = digest(rule)
    rulepath = ROOT / 'selection-rule.json'
    if rulepath.exists():
        assert json.loads(rulepath.read_text()) == rule
    else:
        write_json(rulepath, rule)
    (ROOT / 'metadata-repairs').mkdir(exist_ok=True)
    (ROOT / 'media').mkdir(exist_ok=True)
    (ROOT / 'metadata').mkdir(exist_ok=True)
    (ROOT / 'sources').mkdir(exist_ok=True)
    repaired = {}

    def repair(item):
        group = item['source_group']
        source = Path(old[group]['video']).resolve(strict=True)
        receipt = ROOT / 'metadata-repairs' / (group.replace(':', '-') + '.json')
        before = signature(source)
        if receipt.exists():
            result = json.loads(receipt.read_text())
            assert result['source_signature'] == before
            assert sha256(result['metadata']) == result['metadata_sha256']
            return group, result
        meta = probe_video(source, 900)
        assert signature(source) == before
        pts, keep = presentation_frames(meta['frames'], drop_bad_pts=True)
        path = ROOT / 'metadata-repairs' / (group.replace(':', '-') + '.metadata.json')
        write_json(path, meta)
        result = {'source_group': group, 'video': str(source), 'source_signature': before,
                  'metadata': str(path), 'metadata_sha256': sha256(path),
                  'last_valid_time': float((pts - pts[0])[keep][-1]),
                  'provenance': 'fresh_full_ffprobe_on_original_media', 'source_media_sha256_rechecked_now': False}
        write_json(receipt, result)
        print('STATE_METADATA_REPAIRED=' + json.dumps({'group': group, 'last': result['last_valid_time']}), flush=True)
        return group, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        repaired.update(pool.map(repair, inv['metadata_unresolved']))
    metadata = {**inv['metadata'], **repaired}
    assert set(metadata) == set(old)
    parsed = {r['id']: r for r in _streaming_rows(CSV)}
    annotations = {q['id']: q for q in inv['questions']}
    eligible = [q for q in inv['questions'] if q['task'] in STATE_TASKS
                and q['temporal_clue_type'] == 'Concurrent' and q['target_time'] >= 60
                and q['target_time'] + 300 <= metadata[q['source_group']]['last_valid_time'] + 1e-5]
    groups = sorted({q['source_group'] for q in eligible})
    selected = [min((q for q in eligible if q['source_group'] == g), key=lambda q: (q['target_time'], q['id'])) for g in groups]
    rows = [parsed[q['id']] for q in selected]
    spec = {'selection_rule_id': rule['id'], 'question_ids': sorted(q['id'] for q in rows),
            'source_groups': groups, 'original_targets': len(rows), 'original_sources': len(groups),
            'inventory_sha256': sha256(INV), 'completed_inventory_sources': 250,
            'resolved_missing_full_metadata_sources': len(repaired),
            'annotation_sha256': sha256(CSV), 'old_development_manifest_sha256': sha256(OLD),
            'model_calls': 0, 'model_outcomes_used': False, 'new_reserved_source_groups_consumed': 0,
            'final_evaluation_protocol_frozen': False, 'scope': rule['scope']}
    spec['id'] = digest(spec)
    write_json(ROOT / 'candidate-spec.json', spec)
    train = json.loads(Path('/data/yyy/tracevl-runs/core-pool-2efc7cc-001/train/plain/run.json').read_text())
    training_hashes = set(train['videos'].values())

    def stage(group):
        source = Path(old[group]['video']).resolve(strict=True)
        destination = ROOT / 'media' / (group.replace(':', '-') + source.suffix)
        checksum = stage_file(source, destination)
        assert checksum not in training_hashes
        info = metadata[group]
        if group in repaired:
            assert signature(source) == info['source_signature']
            assert sha256(info['metadata']) == info['metadata_sha256']
            meta = json.loads(Path(info['metadata']).read_text())
        else:
            path = Path(info['metadata_path'])
            saved = json.loads(path.read_text())
            meta = saved['metadata'] if 'metadata' in saved else saved
            assert checksum == info['cached_video_sha256']
        pts, keep = presentation_frames(meta['frames'], drop_bad_pts=True)
        last = float((pts - pts[0])[keep][-1])
        row = next(q for q in rows if q['source_group'] == group)
        assert row['query_time'] + 300 <= last + 1e-5
        indices, times, dropped = schedule(meta, row['query_time'] + 300)
        path = ROOT / 'metadata' / (checksum + '.metadata.json')
        write_json(path, meta)
        result = {'completed': True, 'source_group': group, 'original_video': str(source),
                  'video': str(destination), 'video_sha256': checksum, 'bytes': destination.stat().st_size,
                  'metadata': str(path), 'metadata_sha256': sha256(path),
                  'metadata_provenance': info['provenance'], 'last_valid_source_time': last,
                  'target_ids': [row['id']], 'target_times': [row['query_time']],
                  'planned_maximum_stream_steps': len(times), 'sampled_frame_count': len(indices),
                  'dropped_bad_pts_frames': dropped, 'full_sequential_decode_pending': True,
                  'training_exact_byte_overlap': False}
        write_json(ROOT / 'sources' / (group.replace(':', '-') + '.json'), result)
        print('STATE_SOURCE_READY=' + json.dumps({'group': group, 'bytes': result['bytes'], 'steps': len(times)}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = list(pool.map(stage, groups))
    by_group = {s['source_group']: s for s in sources}
    for row in rows:
        original = annotations[row['id']]
        row['video'] = by_group[row['source_group']]['video']
        row.pop('sample', None)
        row.update(role='controlled-existing-development-past-state-diagnostic', cohort='natural_delayed_states',
                   temporal_clue_type='Concurrent', frames_required=original['frames_required'])
    write_rows(ROOT / 'questions.jsonl', sorted(rows, key=lambda r: (r['source_group'], r['query_time'], r['id'])))
    ready = {'completed': True, 'prepared_unix': time.time(), 'candidate_spec_id': spec['id'],
             'selection_rule_id': rule['id'], 'questions': len(rows), 'original_sources': len(groups),
             'sources': sources, 'bytes': sum(s['bytes'] for s in sources),
             'questions_sha256': sha256(ROOT / 'questions.jsonl'), 'script_sha256': sha256(__file__),
             'model_calls': 0, 'new_reserved_source_groups_consumed': 0, 'training_exact_byte_overlap': False,
             'scope': rule['scope'] + ' Preparation only; sequential selected-frame pixel decode pending.'}
    ready['id'] = digest(ready)
    write_json(ROOT / 'ready.json', ready)
    print('STATE_DATA_READY=' + json.dumps({'root': str(ROOT), 'id': ready['id'], 'ready_sha256': sha256(ROOT / 'ready.json'),
          'questions': len(rows), 'sources': len(groups), 'bytes': ready['bytes']}), flush=True)


if __name__ == '__main__':
    main()
