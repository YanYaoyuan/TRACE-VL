"""Bounded CPU preparation; preserve source order and all numerical GPU work."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Full, Queue
from threading import Event, Thread


def ordered_parallel(function, items, *, workers=4, depth=8):
    """Prepare future sources on CPU without changing their consumption order."""
    if workers < 1 or depth < workers:
        raise ValueError("positive workers and depth >= workers required")
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(depth):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass


@contextmanager
def prefetch(factory, *, depth=2):
    """Run an owned CPU iterator ahead of the GPU, with bounded resident batches.

    The producer owns and closes its decoder, including on consumer failure.
    Exceptions cross the queue rather than leaving the consumer hanging.
    """
    if depth < 1:
        raise ValueError("positive prefetch depth required")
    queue, stopped = Queue(depth), Event()

    def put(value):
        while not stopped.is_set():
            try:
                queue.put(value, timeout=0.05)
                return
            except Full:
                pass

    def produce():
        iterator = None
        try:
            iterator = iter(factory())
            for value in iterator:
                if stopped.is_set():
                    break
                put(("value", value))
        except BaseException as error:
            put(("error", error))
        finally:
            try:
                if hasattr(iterator, "close"):
                    iterator.close()
            finally:
                put(("end", None))

    def consume():
        while True:
            kind, value = queue.get()
            if kind == "error":
                raise value
            if kind == "end":
                return
            yield value

    thread = Thread(target=produce, name="trace-cpu-prefetch", daemon=True)
    thread.start()
    try:
        yield consume()
    finally:
        stopped.set()
        thread.join()
