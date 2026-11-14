"""Inventory downloaded archive names before sampling JoyAI's multi-source annotations.

No media are extracted here. Gzip TARs need one sequential pass; ZIPs only need
their directory. Successful inventories survive across runs, keyed by resolved
part paths, sizes and mtimes (not a claim of content-hash verification).
"""
from __future__ import annotations

import fcntl
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import digest, read_json, write_json

VERSION = 'archive-member-inventory-v1'
VIDEO_EXT = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}


def signature(archive):
    return [{'path': str(Path(p).resolve()), 'bytes': Path(p).stat().st_size,
             'mtime_ns': Path(p).stat().st_mtime_ns} for p in archive['parts']]


def inventory_archives(roots, cache, workers=8, progress=None):
    from trace_vl.v3.ovo_transfer import _members, find_archives

    if not 1 <= workers <= 32:
        raise ValueError('TRACE_ARCHIVE_WORKERS must be between 1 and 32')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archives, skipped = find_archives(roots)
    lock, stop = threading.Lock(), threading.Event()
    active, finished, hits = {}, 0, 0
    if progress:
        progress.update('archive-member-inventory', archives_total=len(archives), archive_workers=workers)

    def notify(i, **values):
        nonlocal finished, hits
        with lock:
            if values.pop('done', False):
                active.pop(i, None)
                finished += 1
                hits += values.pop('cached', False)
            else:
                active.setdefault(i, {'archive_index': i+1, 'archive': archives[i]['name']}).update(values)
            if progress:
                progress.update(archives_total=len(archives),
                                archives_finished=finished, archives_cached=hits,
                                archive_workers=workers, archives_active=[dict(v) for _, v in sorted(active.items())])

    def scan(pair):
        i, archive = pair
        try:
            before = signature(archive)
            identity = {'version': VERSION, 'parts': before}
            path = cache/(digest(identity)+'.json')
            notify(i, members_scanned=0)
            if path.is_file():
                try:
                    saved = read_json(path)
                    if (saved['identity'] == identity and isinstance(saved['members'], list)
                            and all(isinstance(m, str) for m in saved['members'])
                            and saved['members_sha256'] == digest(saved['members'])):
                        notify(i, done=True, cached=True)
                        return archive, saved, None
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # Incomplete/corrupt cache is rebuilt; originals stay read-only.
            names = []
            for count, (name, _) in enumerate(_members({**archive, '_stop': stop}), 1):
                if stop.is_set():
                    raise InterruptedError('inventory cancelled')
                if Path(name).suffix.lower() in VIDEO_EXT:
                    # Preserve duplicate member names: they are ambiguous, not a set.
                    names.append(name)
                if count == 1 or count % 250 == 0:
                    notify(i, members_scanned=count, current_member=name, video_members=len(names))
            if signature(archive) != before:
                raise ValueError('archive changed during inventory; stop downloads before retrying')
            saved = {'identity': identity, 'members': names, 'members_sha256': digest(names)}
            write_json(path, saved)
            notify(i, done=True, cached=False)
            return archive, saved, None
        except Exception as error:
            notify(i, done=True, cached=False)
            return archive, None, str(error)

    # Serialise independent launchers writing this shared cache; internal reads
    # remain concurrent. A killed job leaves each completed archive reusable.
    with open(cache/'.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            results = list(pool.map(scan, enumerate(archives)))
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
    by_name, records, errors = defaultdict(list), [], []
    for archive, saved, error in results:
        if error:
            errors.append({'archive': archive['name'], 'error': error})
            continue
        records.append({'name': archive['name'], **saved['identity'],
                        'members_sha256': saved['members_sha256'], 'video_members': len(saved['members'])})
        for member in saved['members']:
            by_name[Path(member).name].append({'archive': archive['name'], 'member': member})
    report = {'version': VERSION, 'archives': records, 'errors': errors,
              'skipped': [{'name': a['name'], 'reason': a['reason']} for a in skipped],
              'cache': str(cache), 'cache_hits': hits, 'unique_video_names': len(by_name),
              'identity_scope': 'part paths/sizes/mtimes plus member-list hash; selected media fully decoded and SHA256 hashed later'}
    return archives, by_name, report


def choose_available(by_video, originals, archive_members, cfg):
    """Selection uses availability and fixed hashes, never scores or quota refill.

    Archive-only basename collisions are conservatively excluded. Existing
    originals still undergo the existing content-ambiguity and decode checks.
    """
    available, missing, ambiguous = [], [], {}
    for name in sorted(by_video):
        if originals.get(name) or len(archive_members.get(name, [])) == 1:
            available.append(name)
        elif archive_members.get(name):
            ambiguous[name] = archive_members[name]
        else:
            missing.append(name)
    selected = sorted(available, key=lambda n: digest([cfg['seed'], n]))[:cfg['max_videos']]
    return selected, {'eligible_videos': len(by_video), 'available_videos': len(available),
                      'selected_videos': len(selected), 'selected_names': selected,
                      'unavailable_names': missing, 'ambiguous_archive_names': ambiguous,
                      'rule': 'fixed hash order over available media; no score-based or post-decode refill'}


def default_cache(out):
    return Path(os.environ.get('TRACE_MEDIA_INVENTORY_CACHE', str(Path(out).parent/'archive-member-cache-v1')))
