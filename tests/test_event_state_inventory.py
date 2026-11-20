import io
import tarfile
from pathlib import Path

import pytest

from trace_vl.event_state.common import read_json
from trace_vl.event_state.media_inventory import choose_available, inventory_archives


def archive(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as t:
        for name in names:
            entry = tarfile.TarInfo(name)
            entry.size = 3
            t.addfile(entry, io.BytesIO(b'raw'))


def test_missing_sources_do_not_consume_selection_budget():
    rows = {f'missing{i}.mp4': [] for i in range(1000)}
    rows.update({name: [] for name in ('a.mp4', 'b.mp4', 'c.mp4', 'duplicate.mp4')})
    members = {'b.mp4': [{'archive': 'b', 'member': 'b.mp4'}],
               'c.mp4': [{'archive': 'c', 'member': 'c.mp4'}],
               'duplicate.mp4': [{'archive': 'x'}, {'archive': 'y'}]}
    cfg = {'seed': 7, 'max_videos': 3}
    selected, report = choose_available(rows, {'a.mp4': {'/original/a.mp4'}}, members, cfg)
    assert set(selected) == {'a.mp4', 'b.mp4', 'c.mp4'}
    assert len(report['unavailable_names']) == 1000
    assert list(report['ambiguous_archive_names']) == ['duplicate.mp4']
    again, _ = choose_available(dict(reversed(list(rows.items()))), {'a.mp4': {'/original/a.mp4'}}, members, cfg)
    assert again == selected


def test_inventory_cache_reuses_unchanged_archives_and_invalidates_changes(tmp_path, monkeypatch):
    import trace_vl.v3.ovo_transfer as transfer
    root, cache = tmp_path/'downloads', tmp_path/'cache'
    archive(root/'one.tar.gz', ['src/a.mp4', 'meta.txt'])
    _, names, report = inventory_archives([root], cache, workers=2)
    assert list(names) == ['a.mp4'] and report['cache_hits'] == 0
    original = transfer._members
    def forbidden(*args):
        raise AssertionError('unchanged archive was reopened')
    monkeypatch.setattr(transfer, '_members', forbidden)
    _, again, report = inventory_archives([root], cache)
    assert again == names and report['cache_hits'] == 1
    monkeypatch.setattr(transfer, '_members', original)
    archive(root/'one.tar.gz', ['src/b.mp4', 'src/c.mp4'])
    _, changed, report = inventory_archives([root], cache)
    assert set(changed) == {'b.mp4', 'c.mp4'} and report['cache_hits'] == 0
    for p in cache.glob('*.json'):
        p.write_text('{broken')
    _, repaired, report = inventory_archives([root], cache)
    assert repaired == changed and report['cache_hits'] == 0


def test_inventory_multipart_and_duplicate_members_are_not_collapsed(tmp_path):
    root = tmp_path/'downloads'
    source = tmp_path/'split.tar.gz'
    archive(source, ['src/a.mp4', 'src/a.mp4'])
    data = source.read_bytes()
    root.mkdir()
    (root/'split.tar.gz.partaa').write_bytes(data[:len(data)//2])
    (root/'split.tar.gz.partab').write_bytes(data[len(data)//2:])
    _, names, report = inventory_archives([root, root], tmp_path/'cache')
    assert len(report['archives']) == 1 and not report['errors'] and not report['skipped']
    assert len(names['a.mp4']) == 2


def test_failed_inventory_never_publishes_partial_candidates(tmp_path, monkeypatch):
    import trace_vl.v3.ovo_transfer as transfer
    root = tmp_path/'downloads'
    archive(root/'bad.tar.gz', ['src/a.mp4'])
    def broken(_):
        yield 'src/a.mp4', None
        raise OSError('truncated archive')
    monkeypatch.setattr(transfer, '_members', broken)
    _, names, report = inventory_archives([root], tmp_path/'cache')
    assert not names and report['errors']
    assert not list((tmp_path/'cache').glob('*.json'))


def test_filtered_catalog_does_not_reopen_unrelated_archives(tmp_path, monkeypatch):
    import trace_vl.v3.ovo_transfer as transfer
    root = tmp_path/'downloads'
    archive(root/'a.tar.gz', ['src/a.mp4'])
    archive(root/'z.tar.gz', ['src/z.mp4'])
    catalog, _, _ = inventory_archives([root], tmp_path/'cache')
    original = transfer._members
    def check(a):
        assert Path(a['parts'][0]).name == 'a.tar.gz'
        yield from original(a)
    monkeypatch.setattr(transfer, '_members', check)
    r = transfer.ensure_media(['a.mp4'], tmp_path/'originals', [root], tmp_path/'out',
                              probe=lambda p: 1 if Path(p).is_file() else None,
                              archive_catalog=[catalog[0]])
    assert not r['still_missing']
    assert (tmp_path/'out/media/a.mp4').read_bytes() == b'raw'


def test_inventory_changes_during_scan_are_rejected(tmp_path, monkeypatch):
    import trace_vl.v3.ovo_transfer as transfer
    root = tmp_path/'downloads'
    p = root/'a.tar.gz'
    archive(p, ['a.mp4'])
    def changed(_):
        yield 'a.mp4', None
        p.write_bytes(p.read_bytes()+b'change')
    monkeypatch.setattr(transfer, '_members', changed)
    _, names, report = inventory_archives([root], tmp_path/'cache')
    assert not names and 'changed' in report['errors'][0]['error']


def test_prepare_draws_real_videos_before_cap_and_keeps_split_gate(tmp_path, monkeypatch):
    from test_event_state import annotation, make_video

    from trace_vl.event_state.common import write_json
    from trace_vl.event_state.data import prepare, split_for

    monkeypatch.setenv('TRACE_DATA_WORKERS', '1')
    cfg = read_json(Path(__file__).parents[1]/'configs/event_state/pilot.json')
    cfg.update(reserve_gb=0, max_videos=3, require_benchmark_inventory=False, benchmark_roots=[],
               min_videos={'train': 1, 'calibration': 1, 'dev': 1},
               min_conditions={'train': 1, 'calibration': 1, 'dev': 1})
    names = {}
    for i in range(100):
        names.setdefault(split_for(str(i), cfg['seed']), f'v_{i}.avi')
    root = tmp_path/'raw'
    root.mkdir()
    rows = [annotation(f'absent-{i}.mp4') for i in range(1000)]
    with tarfile.open(root/'videos.tar.gz', 'w:gz') as t:
        for split, name in names.items():
            p = tmp_path/'staging'/name
            make_video(p)
            with open(p, 'ab') as f:
                f.write(split.encode())
            t.add(p, arcname='src/'+name)
            rows.append(annotation(name))
    write_json(tmp_path/'ann.json', rows)
    kept = prepare(tmp_path/'ann.json', [root], tmp_path/'out', cfg, 'test')
    assert {r['split'] for r in kept} == {'train', 'calibration', 'dev'}
    selection = read_json(tmp_path/'out/selection.json')
    assert selection['available_videos'] == selection['selected_videos'] == 3
    assert len(selection['unavailable_names']) == 1000
    assert read_json(tmp_path/'out/data-audit.json')['ready']
    cfg['min_videos']['dev'] = 2
    with pytest.raises(ValueError, match='Data gate failed'):
        prepare(tmp_path/'ann.json', [root], tmp_path/'insufficient', cfg, 'test')
