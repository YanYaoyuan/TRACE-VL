"""Replay existing core experiment logs/reports to SwanLab, optionally while running."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from trace_vl.v3.swanlab_tracking import SwanTracker, scalars

SKIP = {"source", "wandb", "swanlog", ".swanlab-upload", "questions", "__pycache__", "tmp", "media"}
TEXT = {".json", ".jsonl", ".log", ".md", ".txt"}


def files_to_upload(root):
    root = Path(root)
    result = []
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(("checkpoint-", ".")))
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix in TEXT and not name.startswith('.') and not path.is_symlink():
                # Data manifests contain training QA; only experiment metadata is collected.
                if name.endswith('.jsonl') and name != 'metrics.jsonl' and name != 'predictions.jsonl':
                    continue
                result.append(path)
    return sorted(result)


def log_series(path):
    """Parse native metrics or rank-zero JSON lines embedded in an operator log."""
    if path.name != 'metrics.jsonl' and path.suffix != '.log':
        return []
    result = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            # A live final line can be incomplete; middle corruption in a native
            # metrics file must not be mistaken for a complete replay.
            if path.name == 'metrics.jsonl' and i != len(lines) - 1 and line.strip():
                raise ValueError(f'corrupt metrics line: {path}:{i + 1}') from None
            continue
        if isinstance(row, dict) and type(row.get('update', row.get('step'))) is int:
            result.append(row)
    return result


def upload(root, operator_logs=(), watch=0):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('experiment directory does not exist: ' + str(root))
    extras = [Path(p).resolve() for p in operator_logs]
    if any(not p.is_file() for p in extras):
        raise ValueError('operator log does not exist')
    target = root / '.swanlab-upload'
    project = os.environ.get('TRACE_SWANLAB_PROJECT', 'trace-vl-core')
    run_id = 'trace-' + hashlib.sha256(json.dumps([str(root), project,
             os.environ.get('TRACE_SWANLAB_WORKSPACE', '')]).encode()).hexdigest()[:32]
    tracker = SwanTracker.start(True, root.name + '-records', root.name, 'existing-records',
        config={'source': str(root), 'replayed': True, 'watch_seconds': watch,
                'scope': 'Existing logs and scores only. No model execution or scientific reclassification.'},
        directory=target / 'swanlog', run_id=run_id)
    if not tracker.active:
        raise RuntimeError(f'SwanLab startup failed ({tracker.error or "no active run"}); '
                           'see preceding warning; original logs retained')
    seen, mtime = {}, {}
    completed = False
    try:
        while True:
            files = files_to_upload(root)
            for path in files:
                stamp = (path.stat().st_size, path.stat().st_mtime_ns)
                if mtime.get(path) == stamp:
                    continue
                mtime[path] = stamp
                label = str(path.relative_to(root)).replace('.', '_')
                rows = log_series(path)
                if rows and rows[-1].get('update', rows[-1].get('step')) < seen.get(path, -1):
                    raise ValueError('metrics were truncated/replaced: ' + str(path))
                for row in rows:
                    step = row.get('update', row.get('step'))
                    if step > seen.get(path, -1):
                        tracker.log(scalars(row, 'curves/' + label + '/'), step=step)
                        seen[path] = step
            print(f'SWANLAB REPLAY: {len(files)} files; {len(seen)} metric series; mode={tracker.mode}', flush=True)
            # Continuous mode follows an existing process without restarting it.
            if not watch:
                break
            time.sleep(watch)
        completed = True
    except KeyboardInterrupt:
        print('Stopping live collection; saving the current log/report snapshot', flush=True)
        completed = True
    finally:
        try:
            files = files_to_upload(root)
            for path in files:
                if path.suffix == '.json' and path.name in {'report.json', 'summary.json', 'batch-report.json'}:
                    payload = json.loads(path.read_text())
                    tracker.log(scalars(payload, 'reports/' + str(path.relative_to(root)).replace('.', '_') + '/'))
            names = {str(p): 'operator/' + str(i) + '-' + p.name for i, p in enumerate(extras)}
            uploaded = tracker.upload(root.name + '-records', 'experiment-records', [*files, *extras],
                                      base=root, names=names)
            if tracker.error or uploaded is None:
                raise RuntimeError('SwanLab recording error; source files retained, rerun the upload command')
            mode = tracker.mode
        finally:
            tracker.finish()
        if tracker.error:
            raise RuntimeError('SwanLab finish failed; local records retained')
        if completed:
            target.mkdir(parents=True, exist_ok=True)
            receipt = {'mode': mode, 'experiment_id': run_id, 'files': uploaded,
                       'note': 'SDK logging/finish completed; online delivery uses the SDK transport. '
                               'An offline receipt is not a cloud-upload confirmation.'}
            temp = target / 'receipt.tmp'
            temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
            temp.replace(target / 'receipt.json')
            print(f'SWANLAB RECORDS COMPLETE: mode={mode}, receipt={target / "receipt.json"}', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--operator-log', action='append', default=[])
    parser.add_argument('--watch', type=float, default=0, help='Poll interval in seconds; Ctrl-C finishes the snapshot')
    args = parser.parse_args()
    if args.watch < 0:
        parser.error('--watch must be nonnegative')
    upload(args.runs, args.operator_log, args.watch)


if __name__ == '__main__':
    main()
