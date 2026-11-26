"""Phase-wide deadlines and cleanup restricted to explicitly launched process groups."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


class BudgetExceeded(TimeoutError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def budget(seconds):
    """Linux main-thread phase watchdog, including startup, reporting and bundling."""
    if seconds <= 0 or signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise ValueError("positive nonnested budget required")
    def expired(signum, frame):
        if signum == signal.SIGALRM:
            raise BudgetExceeded(f"phase elapsed-time ceiling reached: {seconds}s")
        raise InterruptedError(f"phase interrupted by signal {signum}")
    old = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for s in old:
            signal.signal(s, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield time.monotonic() + seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in old.items():
            signal.signal(s, handler)


def run_jobs(jobs, env, log_root, deadline, receipt, *, cleanup_grace=10):
    """All groups are born here with start_new_session; never discover/kill user jobs."""
    Path(log_root).mkdir(parents=True, exist_ok=True)
    active, handles, records = [], [], {}
    started = time.monotonic()
    state = {"status": "running", "workers": records, "started_unix": time.time()}
    try:
        for name, command, devices in jobs:
            if time.monotonic() >= deadline:
                raise BudgetExceeded("phase deadline before worker launch")
            log = (Path(log_root) / f"{name}.log").open("ab", buffering=0)
            handles.append(log)
            child = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices))},
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            active.append((name, child))
            records[name] = {"pid": child.pid, "gpu_ids": devices, "started_unix": time.time()}
            atomic_json(receipt, state)
            print(f"START {name} PID={child.pid} GPUs={devices}", flush=True)
        while active:
            if time.monotonic() >= deadline:
                raise BudgetExceeded("phase elapsed-time ceiling reached")
            remaining = []
            for name, child in active:
                code = child.poll()
                if code is None:
                    remaining.append((name, child))
                else:
                    records[name].update(exit_code=code, completed_unix=time.time())
                    records[name]["wall_seconds"] = records[name]["completed_unix"] - records[name]["started_unix"]
                    if code:
                        raise RuntimeError(f"{name} exited {code}; inspect {log_root}/{name}.log")
                    print(f"DONE {name}", flush=True)
            active = remaining
            atomic_json(receipt, state)
            if active:
                time.sleep(.2)
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="held_or_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if active:
            # On failure, shield the entire TERM/wait/KILL sequence. Disarming
            # unconditionally would remove the phase alarm after a SUCCESSFUL
            # GPU check, leaving subsequent workers/reporting unbounded.
            signals = {signal.SIGALRM, signal.SIGTERM, signal.SIGINT}
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
            handlers = {s: signal.getsignal(s) for s in signals}
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for s in signals:
                    signal.signal(s, signal.SIG_IGN)
                for _, child in active:
                    if child.poll() is None:
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                grace = time.monotonic() + cleanup_grace
                for _, child in active:
                    try:
                        child.wait(timeout=max(.01, grace - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=5)
            finally:
                # Discard pending cleanup-time signals while ignored; our
                # original exception still marks the phase incomplete.
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
                for s, handler in handlers.items():
                    signal.signal(s, handler)
        state.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        atomic_json(receipt, state)
        for log in handles:
            log.close()
