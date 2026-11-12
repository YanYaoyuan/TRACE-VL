"""Bounded CPU preparation workers and persistent, visible progress."""
from __future__ import annotations

import multiprocessing as mp
import os
import resource
import threading
import time
from pathlib import Path

from .common import digest, read_json, sha, write_json


def worker_count():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    mem = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            mem = int(line.split()[1])*1024
    default = max(1, min(16, cpus, int(mem/2**30/2) or 1))
    count = int(os.environ.get('TRACE_DATA_WORKERS', default))
    if not 1 <= count <= cpus:
        raise ValueError(f'TRACE_DATA_WORKERS must be between 1 and {cpus}')
    return count


class Progress:
    def __init__(self, root, workers):
        self.path = Path(root)/'preparation-progress.json'
        self.state = {'phase': 'starting', 'workers': workers, 'status': 'running'}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.start = time.monotonic()

    def emit(self):
        with self.lock:
            state = {**self.state, 'elapsed_seconds': round(time.monotonic()-self.start, 1),
                     'parent_cpu_seconds': round(time.process_time(), 2),
                     'parent_peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1)}
            write_json(self.path, state)
        print('DATA PROGRESS', state, flush=True)

    def update(self, phase=None, **values):
        with self.lock:
            if phase is not None and phase != self.state['phase']:
                self.state = {'phase': phase, 'workers': self.state['workers'], 'status': 'running'}
            self.state.update(values)
        if phase is not None:
            self.emit()

    def _beat(self):
        while not self.stop.wait(10):
            self.emit()

    def __enter__(self):
        self.thread = threading.Thread(target=self._beat, daemon=True)
        self.thread.start()
        self.emit()
        return self

    def __exit__(self, kind, value, tb):
        self.stop.set()
        self.thread.join()
        self.update(status='failed' if kind else 'complete', error=str(value) if kind else None)
        self.emit()


def _limits():
    import cv2
    cv2.setNumThreads(1)


def _verify_job(job):
    from .data import verify_video
    name, path, receipt, previous = job
    p, receipt = Path(path), Path(receipt)
    try:
        old = read_json(receipt) if receipt.is_file() else previous
        before = p.stat()
        resumed = False
        if (old.get('size'), old.get('mtime_ns')) == (before.st_size, before.st_mtime_ns):
            resumed = old.get('sha256') == sha(p)
        result = old if resumed else verify_video(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('media changed during verification')
        write_json(receipt, result)
        return name, result, None, resumed
    except Exception as e:
        return name, None, str(e), False


def verify_many(names, root, previous, workers, progress):
    root = Path(root)
    jobs = [(n, str(root/'media'/n), str(root/'verification'/f'{digest(n)}.json'), previous.get(n, {}))
            for n in names if (root/'media'/n).is_file()]
    verified, errors, resumed = {}, {}, 0
    progress.update('full-video-verification', completed=0, total=len(jobs), cached=0)

    def consume(result):
        nonlocal resumed
        name, value, error, cached = result
        if error:
            errors[name] = error
        else:
            verified[name] = value
        resumed += cached
        progress.update(completed=len(verified)+len(errors), total=len(jobs), cached=resumed,
                        errors=len(errors), verified_frames=sum(v['frames'] for v in verified.values()), last_video=name)

    if workers == 1:
        _limits()
        for job in jobs:
            consume(_verify_job(job))
    else:
        # Spawn avoids inheriting CUDA/library state. Pool context terminates children
        # on cancellation. Each worker handles one video at a time, never all frames in RAM.
        with mp.get_context('spawn').Pool(min(workers, max(1, len(jobs))), initializer=_limits) as pool:
            for result in pool.imap_unordered(_verify_job, jobs, chunksize=1):
                consume(result)
    progress.emit()
    return verified, errors


def reuse_extracted(root, identity, names):
    """Opt-in reuse of accepted media only, never old manifests/features/results.

    The old job must be stopped first. Inputs must match except for code revision;
    all reused files go through the new run's full decode and SHA verification.
    Reuse does not add filenames to the selection pool or change their order.
    """
    import shutil

    previous = os.environ.get('TRACE_REUSE_PREPARATION')
    if not previous:
        return
    previous, root = Path(previous).resolve(), Path(root).resolve()
    if previous == root:
        raise ValueError('Reuse preparation must be a separate previous directory')
    old = read_json(previous/'identity.json')
    if {k: v for k, v in old.items() if k != 'code'} != {k: v for k, v in identity.items() if k != 'code'}:
        raise ValueError('Previous preparation has different inputs/settings; refusing media reuse')
    reused = []
    for name in names:
        source, dest = previous/'media-extracted'/name, root/'media-extracted'/name
        if not source.is_file() or source.is_symlink() or dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name+'.reuse.tmp')
        if temp.exists():
            temp.unlink()
        try:
            os.link(source, temp)
        except OSError:
            shutil.copy2(source, temp)
        temp.replace(dest)
        reused.append(name)
    write_json(root/'reused-preparation.json', {'source': str(previous), 'source_code': old['code'],
               'files': reused, 'count': len(reused), 'note': 'Media only; full verification still required.'})
    print(f'DATA REUSE: {len(reused)} extracted files; selection unchanged; full verification still required.', flush=True)
