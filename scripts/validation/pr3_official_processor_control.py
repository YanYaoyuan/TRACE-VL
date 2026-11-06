"""Compare public HF full forward and native streaming on identical processor-default inputs.

Only the decoder backend changes to HF's public OpenCV reader because PyAV/torchcodec
are absent. Sampling, resizing, token replacement and chat formatting remain HF code.
No benchmark examples enter training. BF16 score differences are diagnostic, not an
FP32 equality gate. One unconstrained greedy next-token and constrained A-D are logged.
"""
import argparse
import gc
import json
import time
import traceback
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers.video_utils import read_video_opencv

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.session import StreamSession

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--records', required=True)
p.add_argument('--video-root', required=True)
p.add_argument('--output', required=True)
p.add_argument('--limit-videos', type=int, default=30)
p.add_argument('--device', default='cuda:0')
p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
p.add_argument('--align-outer-markers', action='store_true')
args = p.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
cfg, _ = load_v3_config(args.config)
cfg.backbone.dtype = args.dtype
cfg.memory.policy = 'full'
bb = Backbone(cfg.backbone, cfg.stream, args.device)
processor = bb.processor
vp = processor.video_processor
vp_defaults = {key: asdict(getattr(vp, key)) if is_dataclass(getattr(vp, key)) else getattr(vp, key)
               for key in ['fps', 'max_frames', 'min_frames', 'size', 'cap_pixels_per_frame']}
(out / 'settings.json').write_text(json.dumps({'config': cfg.to_dict(), 'processor_defaults': vp_defaults,
    'sampling': 'HF sample_frames defaults', 'decoder': 'HF read_video_opencv', 'dtype': args.dtype,
    'reference': 'public processor + full model forward, logits_to_keep=1',
    'align_outer_markers': args.align_outer_markers,
    'matched': 'same processor pixel_values, per-patch grid, timestamps, question and template; no eviction'}, indent=2)+'\n')
class AlignedSession(StreamSession):
    @torch.no_grad()
    def _prefill_prompt(self):
        ids = self.bb.ids(self.bb.head_text) + [self.bb.vision_start]
        self.bb.lm(self.bb.embed_ids(ids), self.pos.text(len(ids)), self.cache)
        self.memory.append(len(ids), 'prompt', -1., ('prompt',), protected=True)

    @torch.no_grad()
    def _fork(self):
        cache, pos = super()._fork()
        # Public processor wraps the entire video's per-patch markers in another start/end pair.
        self.bb.lm(self.bb.embed_ids([self.bb.vision_end]), pos.text(1, self.bb.device), cache)
        return cache, pos
groups = defaultdict(list)
for line in Path(args.records).read_text().splitlines():
    r = json.loads(line)
    groups[r['video']].append(r)
result_file = out / 'predictions.jsonl'
completed = {json.loads(x)['id'] for x in result_file.read_text().splitlines()} if result_file.exists() else set()
for video, records in list(groups.items())[:args.limit_videos]:
    if all(r['id'] in completed for r in records):
        continue
    original_preprocess, original_resize = bb.preprocess, bb.resize_step
    try:
        started = time.perf_counter()
        frames, meta = read_video_opencv(str(Path(args.video_root) / video), sample_indices_fn=vp.sample_frames)
        timestamps = processor._calculate_timestamps(meta.frames_indices, meta.fps, vp.temporal_patch_size)
        session = None
        original_preprocess, original_resize = bb.preprocess, bb.resize_step
        for record in records:
            lines = [f'{chr(65+i)}. {option}' for i, option in enumerate(record['options'])]
            question = '\n' + record['question'] + '\n' + '\n'.join(lines) + '\nAnswer with the option letter only.'
            messages = [{'role': 'user', 'content': [{'type': 'video', 'video': frames},
                                                      {'type': 'text', 'text': question}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                return_tensors='pt', add_generation_prompt=True, enable_thinking=False,
                video_metadata=[meta], do_sample_frames=False)
            grid = inputs['video_grid_thw'][0]
            nt, gh, gw = map(int, grid)
            assert nt == len(timestamps), (nt, len(timestamps))
            pixels = inputs['pixel_values_videos']
            if session is None:
                session = (AlignedSession if args.align_outer_markers else StreamSession)(bb, cfg, mode='native')
                per_step = gh * gw
                for k, t in enumerate(timestamps):
                    patch_pixels = pixels[k * per_step:(k + 1) * per_step]
                    patch_grid = grid.clone()[None]
                    patch_grid[0, 0] = 1
                    bb.resize_step = lambda x: x
                    bb.preprocess = lambda _frames, pv=patch_pixels, g=patch_grid: (pv, g)
                    session.step(np.zeros((2, 32, 32, 3), dtype=np.uint8), float(t))
                bb.preprocess, bb.resize_step = original_preprocess, original_resize
            if record['id'] in completed:
                continue
            with torch.no_grad():
                matched_choice, matched_scores = session.ask(record['question'], record['options'])
                # Clear the multimodal wrapper's own rope-delta state between independent examples.
                bb.model.model.rope_deltas = None
                data = {k: v.to(bb.device) for k, v in inputs.items() if torch.is_tensor(v)}
                data['pixel_values_videos'] = data['pixel_values_videos'].to(bb.dtype)
                logits = bb.model(**data, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                letters = [bb.ids(chr(65+i))[0] for i in range(len(record['options']))]
                official_scores = logits.log_softmax(-1)[letters].tolist()
                official_choice = int(np.argmax(official_scores))
                greedy_id = int(logits.argmax())
            row = {'id': record['id'], 'video': video, 'task': record['task'], 'answer': record['answer'],
                'official_prediction': official_choice, 'matched_prediction': matched_choice,
                'official_scores': official_scores, 'matched_scores': matched_scores,
                'max_abs_logp_difference': max(abs(a-b) for a,b in zip(official_scores, matched_scores)),
                'greedy_next_token': bb.tokenizer.decode([greedy_id]), 'greedy_is_option': greedy_id in letters,
                'sampled_frames': len(frames), 'grid_thw': [nt, gh, gw],
                'input_tokens': inputs['input_ids'].shape[1], 'kv_tokens': bb.kv_length(session.cache),
                'peak_gb': torch.cuda.max_memory_allocated(bb.device)/1e9,
                'elapsed_video': round(time.perf_counter()-started, 3)}
            with result_file.open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in {'official_scores','matched_scores'}}), flush=True)
            del data, logits, inputs
        del session, frames, pixels
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        bb.preprocess, bb.resize_step = original_preprocess, original_resize
        failure = {'video': video, 'error': traceback.format_exc()}
        with (out/'errors.jsonl').open('a') as handle:
            handle.write(json.dumps(failure)+'\n')
        print(json.dumps(failure), flush=True)
        raise
rows = [json.loads(x) for x in result_file.read_text().splitlines()]
summary = {'n': len(rows), 'videos': len({r['video'] for r in rows}), 'non_thinking': True,
    'official_accuracy': sum(r['official_prediction']==r['answer'] for r in rows)/len(rows),
    'matched_accuracy': sum(r['matched_prediction']==r['answer'] for r in rows)/len(rows),
    'prediction_agreement': sum(r['official_prediction']==r['matched_prediction'] for r in rows)/len(rows),
    'greedy_option_rate': sum(r['greedy_is_option'] for r in rows)/len(rows),
    'max_abs_logp_difference': max(r['max_abs_logp_difference'] for r in rows),
    'mean_abs_logp_difference': float(np.mean([r['max_abs_logp_difference'] for r in rows])),
    'partial_benchmark': True}
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('SUMMARY', json.dumps(summary), flush=True)
