from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import bind, digest, read_json, sha, write_json, write_rows
from .features import save_tensor, valid_tensor
from .model import ResponseHead


def sample(row, cache):
    cache = Path(cache)
    v = torch.load(cache/f"video-{digest(row['source_video_id'])[:24]}.pt", weights_only=True)
    q = torch.load(cache/f"query-{row['id']}.pt", weights_only=True)
    index = {t: i for i, t in enumerate(v['ticks'])}
    x = v['visual'][[index[t] for t in row['ticks']]].float()
    elapsed = torch.tensor(row['ticks']).float()-row['query_s']
    labels = (torch.tensor(row['ticks']) >= row['response_s']).float()
    return x, q['query'].float(), elapsed, labels


def batch_rows(rows, cache, device):
    values = [sample(r, cache) for r in rows]
    b, t = len(values), max(v[0].shape[0] for v in values)
    k, c = values[0][0].shape[1:]
    x = torch.zeros(b, t, k, c)
    q = torch.stack([v[1] for v in values])
    clock, labels, mask = torch.zeros(b, t), torch.zeros(b, t), torch.zeros(b, t)
    for i, (xx, _, ee, yy) in enumerate(values):
        n = len(xx)
        x[i, :n], clock[i, :n], labels[i, :n], mask[i, :n] = xx, ee, yy, 1
    return [v.to(device) for v in (x, q, clock, labels, mask)]


def timing_loss(logits, labels, mask):
    loss = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction='none')
    # Equal weight per trajectory, then equal wait/ready contribution where present.
    pos, neg = labels*mask, (1-labels)*mask
    return ((loss*pos).sum(1)/pos.sum(1).clamp_min(1) +
            (loss*neg).sum(1)/neg.sum(1).clamp_min(1)).mean()/2


def train_arm(rows, cache, out, cfg, arm, device, code):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    selected = sorted([r for r in rows if r['split'] == 'train'], key=lambda r: r['id'])
    identity = {'cfg': cfg, 'arm': arm, 'code': code, 'rows': digest(rows),
                'features': read_json(Path(cache)/'identity.json')}
    bind(out/'identity.json', identity)
    if (out/'complete.json').is_file():
        receipt = read_json(out/'complete.json')
        if receipt.get('identity') != digest(identity) or not valid_tensor(out/'checkpoint.pt'):
            raise ValueError('Invalid completed training checkpoint')
        if receipt.get('checkpoint_sha256') != sha(out/'checkpoint.pt'):
            raise ValueError('Completed checkpoint hash mismatch')
        saved = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=True)
        expected_steps = min(cfg['max_updates'], cfg['epochs']*math.ceil(len(selected)/cfg['batch']))
        if (saved.get('identity') != digest(identity) or saved.get('step') != expected_steps
                or receipt.get('updates') != expected_steps or len(saved.get('history', [])) != expected_steps
                or not saved.get('model') or not saved.get('optimizer')):
            raise ValueError('Completed checkpoint content is incomplete or inconsistent')
        print(f'SKIP verified {arm}', flush=True)
        return
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(cfg['seed'])
    width = sample(selected[0], cache)[0].shape[-1]
    model = ResponseHead(width, cfg, arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # Fixed source order per epoch. Re-created on resume; no dependence on process hash randomization.
    schedule = []
    for epoch in range(cfg['epochs']):
        order = list(range(len(selected)))
        random.Random(cfg['seed']+epoch).shuffle(order)
        schedule += [order[i:i+cfg['batch']] for i in range(0, len(order), cfg['batch'])]
    schedule = schedule[:cfg['max_updates']]
    if not schedule:
        raise ValueError('Empty training schedule')
    start_step, history = 0, []
    if valid_tensor(out/'checkpoint.pt'):
        ck = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=True)
        if ck['identity'] != digest(identity):
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_step, history = ck['step'], ck['history']
        torch.set_rng_state(ck['torch_rng'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
    params = sum(p.numel() for p in model.parameters())
    start = time.monotonic()
    cuda = str(device).startswith('cuda')
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(start_step, len(schedule)):
        batch = [selected[i] for i in schedule[step]]
        x, q, clock, labels, mask = batch_rows(batch, cache, device)
        warmup = max(1, math.ceil(.05*len(schedule)))
        ratio = (step+1)/warmup if step < warmup else .1+.9*(
            1+math.cos(math.pi*(step-warmup)/max(1, len(schedule)-warmup)))/2
        for group in optimizer.param_groups:
            group['lr'] = cfg['lr']*ratio
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cuda):
            logits = model(x, q, clock)
            loss = timing_loss(logits, labels, mask)
        if not torch.isfinite(loss):
            raise ValueError(f'Non-finite loss at {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if cuda:
            torch.cuda.synchronize(device)
        item = {'step': step+1, 'loss': float(loss.detach()), 'grad_norm': float(grad),
                'elapsed_seconds_this_attempt': time.monotonic()-start,
                'examples': len(batch), 'lr': optimizer.param_groups[0]['lr'],
                'peak_gib': torch.cuda.max_memory_allocated(device)/2**30 if cuda else 0.}
        history.append(item)
        if (step+1) % 10 == 0 or step == 0 or step+1 == len(schedule):
            print(arm, item, flush=True)
        if (step+1) % 25 == 0 or step+1 == len(schedule):
            save_tensor(out/'checkpoint.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step+1, 'history': history, 'identity': digest(identity), 'width': width,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if cuda else []})
            write_rows(out/'metrics.jsonl', history)
    write_json(out/'complete.json', {'identity': digest(identity), 'updates': len(schedule), 'parameters': params,
                                    'checkpoint_sha256': sha(out/'checkpoint.pt'), 'final_checkpoint_only': True})


@torch.inference_mode()
def predict(rows, cache, checkpoint, cfg, arm, device, ablation='none'):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ResponseHead(ck['width'], cfg, arm).to(device).eval()
    results = []
    model.load_state_dict(ck['model'])
    cuda = str(device).startswith('cuda')
    candidates = sorted(rows, key=lambda r: r['id'])
    for i, r in enumerate(candidates):
        x, q, clock, _, _ = batch_rows([r], cache, device)
        if ablation == 'zero_video':
            x = torch.zeros_like(x)
        elif ablation == 'shuffled_query':
            other = candidates[(i+1) % len(candidates)]
            q = sample(other, cache)[1][None].to(device)
        elif ablation != 'none':
            raise ValueError(ablation)
        if cuda:
            torch.cuda.synchronize(device)
        start = time.monotonic()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cuda):
            scores = model(x, q, clock)[0].float().sigmoid().cpu().tolist()
        if cuda:
            torch.cuda.synchronize(device)
        results.append({'id': r['id'], 'source_video_id': r['source_video_id'], 'split': r['split'],
                        'source': r.get('source', 'unknown'), 'condition': r.get('condition', ''),
                        'query_s': r.get('query_s'),
                        'ticks': r['ticks'], 'response_s': r['response_s'], 'scores': scores,
                        'seconds': time.monotonic()-start, 'ablation': ablation})
    return results
