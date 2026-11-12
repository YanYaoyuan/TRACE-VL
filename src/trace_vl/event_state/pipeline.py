"""One entry point: audit/extract -> frozen features -> queued arms -> calibrated report."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .common import bind, digest, read_json, read_rows, sha, write_json, write_rows


def queue(jobs, devices, root, label):
    """Bounded independent jobs, one per visible GPU; no cross-rank collectives."""
    pending = list(jobs)
    active = {}
    completed, last_heartbeat = [], 0.
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending:
                    continue
                name, args = pending.pop(0)
                log = open(root/f'{name}.log', 'a', buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), PYTHONUNBUFFERED='1')
                proc = subprocess.Popen([sys.executable, '-m', 'trace_vl.event_state.pipeline', *args],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
                active[device] = (proc, log, name)
                print(f'{label} START {name} GPU={device} pid={proc.pid}', flush=True)
            for device, (proc, log, name) in list(active.items()):
                status = proc.poll()
                if status is not None:
                    log.close()
                    del active[device]
                    if status:
                        raise RuntimeError(f'{label} {name} failed ({status}); see {root/name}.log')
                    completed.append(name)
                    print(f'{label} DONE {name}', flush=True)
            if time.monotonic()-last_heartbeat >= 20 or not active:
                status = {'phase': label, 'queued': [n for n, _ in pending], 'completed_this_queue': completed,
                          'active': {d: {'name': n, 'pid': p.pid} for d, (p, _, n) in active.items()}}
                write_json(root.parent/'progress.json', status)
                print('PROGRESS', status, flush=True)
                last_heartbeat = time.monotonic()
            if active:
                time.sleep(1)
    finally:
        # Only child jobs launched by this queue; do not touch other GPU users.
        for proc, log, _ in active.values():
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()


def worker(args, cfg):
    from .features import encode_shard, valid_tensor
    from .train import predict, train_arm
    out = Path(args.out)
    rows = read_rows(out/'data'/'manifest.jsonl')
    cache = out/'features'
    if args.worker == 'features':
        encode_shard(rows, args.model, cache, cfg, args.shard, args.shards)
        return
    arm = args.arm
    train_arm(rows, cache, out/'train'/arm, cfg, arm, 'cuda:0', args.code)
    checkpoint = out/'train'/arm/'checkpoint.pt'
    if not valid_tensor(checkpoint):
        raise ValueError('Training did not produce a valid checkpoint')
    dest = out/'eval'/arm
    identity = {'checkpoint_sha256': sha(checkpoint), 'manifest': digest(rows), 'code': args.code,
                'cfg': cfg, 'evaluator': 'response-readiness-first-crossing-v1'}
    bind(dest/'identity.json', identity)
    done = dest/'complete.json'
    expected = ['predictions.jsonl'] + (['zero_video.jsonl', 'shuffled_query.jsonl'] if arm == 'event_state' else [])
    if done.is_file():
        result = read_json(done)
        if result.get('identity') != digest(identity) or any(
                not (dest/n).is_file() or result['files'].get(n) != sha(dest/n) for n in expected):
            raise ValueError('Invalid evaluation receipt; inspect before restarting')
        return
    targets = [r for r in rows if r['split'] in {'calibration', 'dev'}]
    predictions = predict(targets, cache, checkpoint, cfg, arm, 'cuda:0')
    if {r['id'] for r in predictions} != {r['id'] for r in targets}:
        raise ValueError('Prediction coverage incomplete')
    write_rows(dest/'predictions.jsonl', predictions)
    if arm == 'event_state':
        for mode in ('zero_video', 'shuffled_query'):
            write_rows(dest/f'{mode}.jsonl', predict([r for r in targets if r['split'] == 'dev'],
                       cache, checkpoint, cfg, arm, 'cuda:0', mode))
    write_json(done, {'identity': digest(identity), 'files': {n: sha(dest/n) for n in expected}})


def all_stages(args, cfg):
    from .data import prepare
    from .features import bind_features, model_complete, model_identity, valid_tensor
    from .report import make_report
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'opencv-python-headless')}
    bind(out/'run-identity.json', {'cfg': cfg, 'code': args.code, 'versions': versions,
                                'annotation': sha(args.annotation), 'media_roots': args.media_root,
                                'model_root': str(Path(args.model).resolve())})
    write_json(out/'protocol.json', cfg)
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    data = data_root/(args.code[:12]+'-'+digest(cfg)[:12])
    if (out/'data').is_symlink():
        if (out/'data').resolve() != data:
            raise ValueError('Output points to another data preparation')
    elif (out/'data').exists():
        raise ValueError('Unexpected data directory; use a new output path')
    else:
        (out/'data').symlink_to(data, target_is_directory=True)
    print('[1/5] CPU annotation/exclusion inventory, selected extraction and full video decode', flush=True)
    rows = prepare(args.annotation, args.media_root, data, cfg, args.code)
    print('[2/5] Verify or obtain original frozen model; fingerprint every weight file', flush=True)
    model = Path(args.model)
    if not model_complete(model):
        if args.no_download:
            raise ValueError('Model missing and --no-download set')
        if shutil.disk_usage(model.parent if model.parent.exists() else out).free < (cfg['reserve_gb']+10)*1e9:
            raise ValueError('Not enough disk for model + reserve')
        from huggingface_hub import snapshot_download
        snapshot_download(cfg['model_id'], revision=cfg['model_revision'], local_dir=str(model),
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'], max_workers=4)
    hashes = model_identity(model)
    expected = cfg.get('model_weight_sha256', {})
    if any(hashes.get(name) != value for name, value in expected.items()):
        raise ValueError('Model weights do not match the pinned official SHA256; no experiments started')
    if read_json(model/'config.json')['text_config']['hidden_size'] != cfg['model_hidden_size']:
        raise ValueError('Model hidden size differs from protocol')
    bind(out/'model-identity.json', {'requested_id': cfg['model_id'], 'requested_revision': cfg['model_revision'],
                                  'local_content': hashes, 'note': 'local files identified by hashes; '
                                  'pre-existing model path is not assumed to be an official revision'})
    bind_features(out/'features', rows, cfg, hashes, args.code)
    unique_ticks = {(r['source_video_id'], t) for r in rows for t in r['ticks']}
    estimated = len(unique_ticks)*cfg['grid']**2*cfg['model_hidden_size']*2
    existing = sum(p.stat().st_size for p in (out/'features').glob('*.pt'))
    if shutil.disk_usage(out).free < max(0, estimated-existing)+cfg['reserve_gb']*1e9:
        raise ValueError(f'Feature storage insufficient: estimated {estimated/1e9:.1f} GB plus reserve')
    devices = args.devices.split(',')
    # The parent only queries CUDA free memory and never constructs a model.
    import torch
    for d in devices:
        if not d.isdigit():
            raise ValueError('Devices must be physical integer GPU IDs')
        free, _ = torch.cuda.mem_get_info(int(d))
        if free/2**30 < args.min_free_gib:
            raise ValueError(f'GPU {d}: {free/2**30:.1f} GiB free < {args.min_free_gib}')
    common = ['--out', str(out), '--config', str(out/'protocol.json'), '--code', args.code, '--model', str(model)]
    print('[3/5] Frozen visual/query features, independent GPU shards', flush=True)
    queue([(f'features-{i}', [*common, '--worker', 'features', '--shard', str(i), '--shards', str(len(devices))])
           for i in range(len(devices))], devices, out/'logs', 'FEATURE')
    expected = {f"video-{digest(r['source_video_id'])[:24]}.pt" for r in rows}
    expected.update(f"query-{r['id']}.pt" for r in rows)
    if not all(valid_tensor(out/'features'/name) for name in expected):
        raise ValueError('Incomplete/corrupt feature cache')
    print('[4/5] Shared-data experiments: clock, frame, GRU, temporal Transformer, event state', flush=True)
    queue([(a, [*common, '--worker', 'experiment', '--arm', a]) for a in cfg['arms']],
          devices[:args.parallel_arms], out/'logs', 'EXPERIMENT')
    print('[5/5] Calibrated timing metrics, paired intervals, shortcut probes, curves and archive', flush=True)
    make_report(out, cfg)
    bundle = out/'reports.tar.gz'
    with tarfile.open(bundle.with_suffix('.tmp'), 'w:gz') as tar:
        for p in sorted(out.rglob('*')):
            if (p.is_file() and p.suffix in {'.json', '.jsonl', '.md', '.svg', '.log', '.txt'}
                    and 'source' not in p.relative_to(out).parts):
                tar.add(p, arcname=str(p.relative_to(out)), recursive=False)
        for n in ('manifest.jsonl', 'data-audit.json', 'identity.json'):
            tar.add(data/n, arcname='data/'+n)
    bundle.with_suffix('.tmp').replace(bundle)
    print(f'REPORT BUNDLE: {bundle}', flush=True)


def main():
    def terminate(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--code', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--annotation')
    p.add_argument('--media-root', action='append', default=[])
    p.add_argument('--data-root', default='/nas-data/vlm/yyy/tracevl-data/event-state-v1')
    p.add_argument('--devices', default='0,1,2,3')
    p.add_argument('--parallel-arms', type=int, default=4)
    p.add_argument('--min-free-gib', type=float, default=24)
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--worker', choices=['features', 'experiment'])
    p.add_argument('--arm')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    cfg = read_json(args.config)
    required_arms = {'clock', 'frame', 'gru', 'temporal', 'event_state'}
    if set(cfg['arms']) != required_arms or len(cfg['arms']) != len(required_arms):
        p.error('This protocol requires each of the five controls exactly once')
    if cfg['dim'] % cfg['heads'] or cfg['dim'] % 2:
        p.error('dim must be even and divisible by heads')
    if any(cfg[k] < 1 for k in ('dim', 'heads', 'layers', 'window', 'state_tokens', 'batch', 'epochs',
                               'max_updates', 'max_videos', 'max_conditions_per_video')):
        p.error('Model/data/training sizes must be positive')
    if args.parallel_arms < 1:
        p.error('parallel-arms must be positive')
    if args.worker:
        worker(args, cfg)
    else:
        if len(set(args.devices.split(','))) != len(args.devices.split(',')):
            p.error('duplicate GPU IDs')
        if not args.annotation or not args.media_root:
            p.error('--annotation and --media-root are required')
        all_stages(args, cfg)


if __name__ == '__main__':
    main()
