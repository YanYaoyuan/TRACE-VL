"""Bounded concurrent archive reads, consumed in the original deterministic order."""
from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def scan_archives(archives, wanted, staging, workers, progress=None):
    """Read at most `workers` archives ahead; callers mutate `wanted` after each yield.

    Gzip decompression and filesystem reads release the GIL. Only independent
    archives run concurrently; each multipart tar remains one ordered stream.
    A faster later archive can never override an earlier path match.
    """
    from .ovo_transfer import _members, _path_match

    if workers < 1:
        raise ValueError('archive_workers must be positive')
    stop, lock = threading.Event(), threading.Lock()
    active, finished = {}, 0

    def update(index, **values):
        nonlocal finished
        with lock:
            if values.pop('finished', False):
                active.pop(index, None)
                finished += 1
            else:
                active.setdefault(index, {'archive_index': index+1, 'archive': archives[index]['name']}).update(values)
            if progress:
                progress(archive_workers=workers, archives_total=len(archives), archives_finished=finished,
                         archives_active=[dict(v) for _, v in sorted(active.items())])

    def scan(index, targets):
        archive = archives[index]
        directory = Path(staging)/str(index)
        directory.mkdir(parents=True, exist_ok=True)
        names = defaultdict(list)
        for v in sorted(targets):
            names[Path(v).name].append(v)
        by_path, by_name = defaultdict(list), defaultdict(list)
        update(index, members_scanned=0, matched_members=0, matched_bytes=0)
        matches = written = 0
        try:
            for count, (name, opener) in enumerate(_members({**archive, '_stop': stop}), 1):
                if stop.is_set():
                    raise InterruptedError('archive scan cancelled')
                update(index, members_scanned=count, current_member=name)
                for v in names.get(Path(name).name, ()):
                    kind = _path_match(name, v)
                    if kind is None:
                        continue
                    matches += 1
                    target = directory/str(matches)
                    with opener() as source, open(target, 'wb') as dest:
                        while True:
                            if stop.is_set():
                                raise InterruptedError('archive scan cancelled')
                            block = source.read(1 << 20)
                            if not block:
                                break
                            dest.write(block)
                            written += len(block)
                    (by_path if kind == 'path' else by_name)[v].append((f"{archive['name']}::{name}", target))
                    update(index, matched_members=matches, matched_bytes=written)
            return archive, by_path, by_name, None
        except Exception as error:
            # Never commit a candidate from an archive that failed mid-read.
            return archive, {}, {}, f'unreadable: {error}'
        finally:
            update(index, finished=True)

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='trace-archive')
    pending, submitted = {}, 0
    try:
        for index in range(len(archives)):
            if not wanted:
                break
            while submitted < min(len(archives), index+workers):
                pending[submitted] = pool.submit(scan, submitted, frozenset(wanted))
                submitted += 1
            yield pending.pop(index).result()
    finally:
        stop.set()
        for future in pending.values():
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
