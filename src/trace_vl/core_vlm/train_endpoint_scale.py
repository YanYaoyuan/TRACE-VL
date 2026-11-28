"""Synchronous source-parallel continuation of the native endpoint Reservoir reader."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import checkpoint_identity
from .data import sha256
from .endpoint_reader import EndpointReaderEpisode
from .train import checked_checkpoint, latest_checkpoint, restore_checkpoint, save_checkpoint, tensor_digest
from .train_endpoint_reader import source_observations, training_sources


def source_batches(sources, world):
    """Fixed source-only order; nearby durations reduce synchronous idle time."""
    if world < 1 or len(sources) % world or len(sources) < 2 * world:
        raise ValueError('whole source batches without padding and at least two updates required')
    # Retain shortest/longest cost coverage in the first two global updates.
    first = sources[:2 * world]
    tail = []
    for offset in range(2 * world, len(sources), 8 * world):
        tail.extend(sorted(sources[offset:offset + 8 * world],
                           key=lambda rows: (rows[0]['query_time'], rows[0]['source_id'])))
    ordered = first + tail
    return [ordered[i:i + world] for i in range(0, len(ordered), world)]


def average_gradients(parameters):
    """Mean source gradients across ranks BEFORE one global gradient clip."""
    world = dist.get_world_size()
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError('every trainable LM LoRA parameter must participate')
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def retain_checkpoints(out, current):
    keep = {current, current - 1, 1, 16, 64, 128, 256}
    for folder in out.glob('checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9]'):
        if int(folder.name.split('-')[1]) not in keep:
            shutil.rmtree(folder)


def run(args):
    import transformers

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != 4:
        raise ValueError('this scale stage requires exactly four source-parallel GPUs')
    torch.cuda.set_device(local)
    torch.manual_seed(17 + rank)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.training.lock').open('a')
    if rank == 0:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = Path(args.manifest)
    if sha256(manifest) != args.manifest_sha256:
        raise ValueError('expanded training manifest changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    sources = training_sources(rows)
    if len(sources) != args.expected_sources:
        raise ValueError('declared training source count differs')
    if args.expected_sources == 1024:
        if any(r['split'] != 'candidate_train_scale' for r in rows):
            raise ValueError('expanded run accepts the new training partition only')
        prepared = json.loads((manifest.parent / 'report.json').read_text())
        if not prepared['completed'] or prepared['manifest_sha256'] != args.manifest_sha256:
            raise ValueError('expanded data qualification is incomplete')
    batches = source_batches(sources, world)
    if args.stage_updates:
        if not 1 <= args.stage_updates <= len(batches):
            raise ValueError('stage update count must fit the unchanged full-cohort schedule')
        batches = batches[:args.stage_updates]
    init = Path(args.init_checkpoint)
    init_report = json.loads((init.parent / 'report.json').read_text())
    init_run = json.loads((init.parent / 'run.json').read_text())
    initial = checked_checkpoint(init, init_run['id'])
    if (not init_report['completed'] or initial['update'] != 64
            or initial['adapter_sha256'] != 'd60a97dee0a669ae3887da530a581510aef09b24901e11fe2b4d71c48bd0fef4'):
        raise ValueError('the frozen64-source Reservoir candidate is required')
    base = init_run['initialization']
    # Shared filesystem qualification is done once before any distributed collectives.
    if rank == 0:
        if checkpoint_identity(base['model']) != base['model_identity']:
            raise ValueError('local base model identity changed')
        for records in sources:
            record = records[0]
            if (not Path(record['video']).resolve().is_relative_to(manifest.parent.resolve() / 'videos')
                    or sha256(record['video']) != record['video_sha256']):
                raise ValueError('training video path or bytes changed')
            if 'metadata' in record and sha256(record['metadata']) != record['metadata_sha256']:
                raise ValueError('qualified PTS metadata changed')
    identity = {'protocol': 'native-endpoint-scale1024-v1', 'manifest_sha256': args.manifest_sha256,
                'sources': len(sources), 'QA': len(rows), 'world_size': world,
                'stage_updates': len(batches), 'stage_sources': len(batches) * world,
                'stage_QA': sum(len(records) for batch in batches for records in batch),
                'batch_source_ids': [[r[0]['source_id'] for r in batch] for batch in batches],
                'QA_ids': [[r['id'] for r in records] for records in sources],
                'initial_checkpoint': initial, 'initial_base_model': base['model_identity'],
                'optimizer': {'lr': 1e-5, 'weight_decay': 0., 'clip': 1.,
                              'carry_candidate_Adam_moments': True, 'initial_Adam_steps': 64},
                'retention': {'policy': 'reservoir', 'recent': 8, 'capacity': 32, 'seed': 17},
                'prefix_gradients': False, 'source_objective': 'mean source of mean question answer-token CE',
                'prefix_adapter': {'mode': 'frozen64' if args.frozen_ingest else 'shared_trainable',
                                   'frozen_checkpoint': initial if args.frozen_ingest else None,
                                   'separate_backbone_allocation': bool(args.frozen_ingest)},
                'code': {name: sha256(Path(__file__).with_name(name)) for name in
                         ('train_endpoint_scale.py', 'train_endpoint_reader.py', 'endpoint_reader.py',
                          'train.py', 'native.py', 'data.py')}, 'torch': str(torch.__version__),
                'transformers': transformers.__version__}
    identity['id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ensure_run_manifest(out / 'run.json', identity)
    previous = latest_checkpoint(out, identity['id']) if rank == 0 else None
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        pointer = [str(previous) if previous else None]
        dist.broadcast_object_list(pointer, src=0)
        bb = Backbone(BackboneConfig(name=base['model'], dtype='bfloat16', local_files_only=True),
                      StreamConfig(pixels_per_frame=256 * 256), device=f'cuda:{local}')
        bb.add_lora(8, 16, dropout=0., adapter=str(init / 'adapter'))
        bb.model.eval()
        params = {name: p for name, p in bb.model.named_parameters() if p.requires_grad}
        if not params or any('lora_' not in name or 'language_model' not in name for name in params):
            raise ValueError('only LM LoRA parameters may train')
        if tensor_digest(params) != initial['adapter_sha256']:
            raise ValueError('loaded candidate adapter differs')
        prefix_bb = None
        if args.frozen_ingest:
            prefix_bb = Backbone(BackboneConfig(name=base['model'], dtype='bfloat16', local_files_only=True),
                                 StreamConfig(pixels_per_frame=256 * 256), device=f'cuda:{local}')
            prefix_bb.add_lora(8, 16, dropout=0., adapter=str(init / 'adapter'))
            prefix_bb.model.eval().requires_grad_(False)
            frozen_lora = {name: p for name, p in prefix_bb.model.named_parameters() if 'lora_' in name}
            if tensor_digest(frozen_lora) != initial['adapter_sha256']:
                raise ValueError('frozen ingestion adapter differs from the original64-source candidate')
            prefix_versions = [(id(p), p._version) for p in prefix_bb.model.parameters()]
        optimizer = torch.optim.AdamW(list(params.values()), lr=1e-5, weight_decay=0.)
        if pointer[0] is None:
            state = torch.load(init / 'training.pt', map_location='cpu', weights_only=True)
            if list(state['parameters']) != list(params):
                raise ValueError('candidate optimizer parameter set differs')
            if any(group['lr'] != 1e-5 or group['weight_decay'] != 0 for group in state['optimizer']['param_groups']):
                raise ValueError('candidate optimizer hyperparameters differ')
            optimizer.load_state_dict(state['optimizer'])
            start, metrics = 0, []
        else:
            start, metrics = restore_checkpoint(pointer[0], identity['id'], params, optimizer, rank, world)
        if len(metrics) != start:
            raise ValueError('checkpoint update history differs')
        expected = [source[0]['source_id'] for batch in batches[:start] for source in batch]
        if [source['source_id'] for m in metrics for source in m['sources']] != expected:
            raise ValueError('checkpoint source schedule differs')
        limit = args.stop_after or len(batches)
        if not start <= limit <= len(batches):
            raise ValueError('stop-after must preserve the frozen complete schedule')
        base_versions = [(id(p), p._version) for p in bb.model.parameters() if not p.requires_grad]
        torch.cuda.reset_peak_memory_stats()
        for update in range(start, limit):
            record_group = batches[update][rank]
            tick = time.monotonic()
            atomic_text(out / f'rank-{rank}.json', json.dumps({'stage': 'source', 'update': update + 1,
                        'source_id': record_group[0]['source_id'], 'rank': rank}) + '\n')
            head, blocks, questions, audit = source_observations(bb, record_group, parity=update == start)
            episode = EndpointReaderEpisode(bb, head, 'reservoir', recent=8, capacity=32, seed=17,
                                            prefix_bb=prefix_bb)
            for block in blocks:
                episode.append(block)
            episode.seal(record_group[0]['query_time'])
            if episode.stream.audit()['kv_steps'] != 40:
                raise ValueError('scale run changed retention budget')
            stream = episode.stream.audit()
            result = episode.train_step(questions, optimizer, synchronize_gradients=average_gradients)
            torch.cuda.synchronize()
            if (base_versions != [(id(p), p._version) for p in bb.model.parameters() if not p.requires_grad]
                    or any(p.grad is not None for p in bb.model.parameters() if not p.requires_grad)):
                raise ValueError('frozen base changed or received gradients')
            if prefix_bb is not None:
                if (prefix_versions != [(id(p), p._version) for p in prefix_bb.model.parameters()]
                        or any(p.grad is not None or p.requires_grad for p in prefix_bb.model.parameters())):
                    raise ValueError('fixed ingestion model changed or received gradients')
            audit.update(**result, stream=stream, rank=rank,
                         source_seconds=time.monotonic() - tick,
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            audit['frozen_ingest'] = prefix_bb is not None
            gathered = [None] * world
            dist.all_gather_object(gathered, audit)
            metrics.append({'update': update + 1, 'sources': gathered,
                            'global_mean_source_CE': sum(m['mean_question_answer_ce'] for m in gathered) / world})
            if any(int(state['step']) != 64 + update + 1 for state in optimizer.state.values()):
                raise ValueError('candidate Adam update count differs')
            del episode, blocks, questions
            save_checkpoint(out, update + 1, identity['id'], bb, params, optimizer, rank, world, metrics)
            if rank == 0:
                checked_checkpoint(out / f'checkpoint-{update+1:06d}', identity['id'])
                retain_checkpoints(out, update + 1)
                atomic_text(out / 'progress.json', json.dumps({'stage': 'checkpointed',
                            'completed_sources': (update + 1) * world, 'target_sources': len(batches) * world,
                            'completed_updates': update + 1, 'last_update': metrics[-1]}, indent=2) + '\n')
                print(json.dumps({'update': update + 1, 'sources': (update + 1) * world,
                                  'slowest_source_seconds': max(m['source_seconds'] for m in gathered)}), flush=True)
            dist.barrier()
        if rank == 0:
            result = {'completed': limit == len(batches), 'completed_sources': limit * world,
                      'target_sources': len(batches) * world, 'eligible_sources': len(sources),
                      'frozen_ingest': bool(args.frozen_ingest), 'stage_optimizer_updates': limit,
                      'initial_optimizer_updates': 64, 'run': identity['id'],
                      'last_checkpoint': f'checkpoint-{limit:06d}', 'prefix_gradients': False,
                      'held_out_evaluated': False}
            atomic_text(out / 'report.json', json.dumps(result, indent=2) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--expected-sources', type=int, default=1024)
    p.add_argument('--init-checkpoint', default='/data/yyy/tracevl-runs/endpoint-reader-pilot-798eee2-001/reservoir/checkpoint-000064')
    p.add_argument('--out', required=True)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--stage-updates', type=int, default=0,
                   help='fixed first N updates of the full source schedule;0 means all')
    p.add_argument('--frozen-ingest', action='store_true',
                   help='separate frozen64-source ingestion backbone; train question/answer LoRA only')
    run(p.parse_args())


if __name__ == '__main__':
    main()
