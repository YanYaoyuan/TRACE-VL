"""Distributed native-video QA LoRA training with identity-checked recovery.

Microbatches contain independent questions with per-question mean answer CE,
including EOS. Every rank consumes the same number of questions. Native video
positions and the global shuffled question stream are preserved when regrouping.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest
from trace_vl.v3.tracking import Tracker, enabled_from_env

from .control import checkpoint_identity
from .data import decode_observation, read_split, sha256
from .native import NativeLayout, NativeVideoQALoss, check_native_forward, encode_native, prepare_native
from .preservation import original_answer_logits
from .qa import VisualStep
from .temporal import attach_temporal


def initial_checkpoint(args):
    path = getattr(args, "init_checkpoint", None)
    if not path:
        return None
    path = Path(path)
    previous = json.loads((path.parent / "run.json").read_text())
    record = checked_checkpoint(path, previous["id"])
    if previous["model"] != checkpoint_identity(args.model):
        raise ValueError("initial checkpoint base weights differ")
    for key in ("mode", "pool_grid", "rank", "steps", "pixels", "timestamp_policy"):
        if previous["config"][key] != getattr(args, key):
            raise ValueError(f"initial checkpoint configuration differs: {key}")
    if previous["config"].get("temporal", "none") != "none":
        raise ValueError("continuation pilot must start from the frozen plain-grid baseline")
    return record


def sample_indices(size, update, world, accumulation, rank, seed):
    """A single shuffled global stream, partitioned without distributed padding.

    At an epoch boundary the next deterministic shuffle starts immediately;
    fixed-update experiments may repeat examples across epochs, never silently
    pad a short rank. World/accumulation are part of the saved run identity.
    """
    if min(size, world, accumulation) < 1 or update < 0 or not 0 <= rank < world:
        raise ValueError("invalid sample schedule")
    result, orders = [], {}
    for micro in range(accumulation):
        offset = update * world * accumulation + micro * world + rank
        epoch, index = divmod(offset, size)
        if epoch not in orders:
            orders[epoch] = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        result.append(orders[epoch][index])
    return result


def tensor_digest(parameters):
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def collective_list(value, world):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def checked_checkpoint(folder, run_id):
    """Completion requires every recorded file, not a log line or adapter name."""
    folder = Path(folder)
    record = json.loads((folder / "complete.json").read_text())
    if record["run"] != run_id or not record["files"]:
        raise ValueError("checkpoint identity differs")
    required = {"training.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
    if not required.issubset(record["files"]):
        raise ValueError("checkpoint lacks adapter or training state")
    for name, expected in record["files"].items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or sha256(folder / p) != expected:
            raise ValueError(f"checkpoint file differs: {name}")
    from safetensors.torch import load_file
    adapter = load_file(folder / "adapter/adapter_model.safetensors")
    state = torch.load(folder / "training.pt", map_location="cpu", weights_only=True)
    if (state.get("run") != run_id or state.get("update") != record["update"]
            or not state.get("parameters") or not state.get("rng")
            or "optimizer" not in state or "metrics" not in state or not adapter):
        raise ValueError("invalid checkpoint training state")
    if tensor_digest(state["parameters"]) != record["adapter_sha256"]:
        raise ValueError("checkpoint parameter identity differs")
    return record


def latest_checkpoint(out, run_id):
    # Ignore temporary directories, never accept a corrupt committed checkpoint.
    folders = sorted(Path(out).glob("checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not folders:
        return None
    for folder in folders:
        record = checked_checkpoint(folder, run_id)
        if int(folder.name.split("-")[1]) != record["update"]:
            raise ValueError("checkpoint step does not match directory")
    return folders[-1]


def save_checkpoint(out, update, run_id, bb, params, optimizer, rank, world, metrics):
    rng = collective_list({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, world)
    hashes = collective_list(tensor_digest(params), world)
    if len(set(hashes)) != 1:
        raise RuntimeError("adapter parameters diverged across ranks")
    if rank == 0:
        out = Path(out)
        target = out / f"checkpoint-{update:06d}"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=out))
        try:
            bb.peft.save_pretrained(temporary / "adapter", safe_serialization=True)
            state = {"run": run_id, "update": update, "parameters": {n: p.detach().cpu() for n, p in params.items()},
                     "optimizer": optimizer.state_dict(), "rng": rng, "metrics": metrics}
            torch.save(state, temporary / "training.pt")
            files = {str(p.relative_to(temporary)): sha256(p) for p in temporary.rglob("*") if p.is_file()}
            atomic_text(temporary / "complete.json", json.dumps({"run": run_id, "update": update,
                        "adapter_sha256": hashes[0], "files": files}, indent=2) + "\n")
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    if world > 1:
        dist.barrier()


def restore_checkpoint(folder, run_id, params, optimizer, rank, world):
    record = checked_checkpoint(folder, run_id)
    state = torch.load(Path(folder) / "training.pt", map_location="cpu", weights_only=True)
    if state["run"] != run_id or state["update"] != record["update"] or len(state["rng"]) != world:
        raise ValueError("training checkpoint identity differs")
    if params.keys() != state["parameters"].keys():
        raise ValueError("trainable parameter set differs")
    with torch.no_grad():
        for name, parameter in params.items():
            if parameter.shape != state["parameters"][name].shape:
                raise ValueError(f"adapter shape differs: {name}")
            parameter.copy_(state["parameters"][name])
    if tensor_digest(params) != record["adapter_sha256"]:
        raise ValueError("loaded adapter digest differs")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"][rank]["cpu"])
    torch.cuda.set_rng_state(state["rng"][rank]["cuda"])
    return state["update"], state["metrics"]


class Samples:
    """Bounded CPU feature cache. Vision is frozen; no LM state is cached."""
    def __init__(self, bb, args):
        self.bb, self.args = bb, args
        self.cache, self.bytes = OrderedDict(), 0

    def decode(self, row):
        return decode_observation(row, steps=self.args.steps, pixels=self.args.pixels)

    def batch(self, rows):
        missing = {row["id"]: row for row in rows if row["id"] not in self.cache}
        workers = min(len(missing), getattr(self.args, "decode_workers", 1))
        if workers < 2:
            return [self.get(row) for row in rows]
        # Only CPU decoding runs concurrently. Vision encoding and cache
        # mutation keep the original sample order on the rank's CUDA stream.
        with ThreadPoolExecutor(workers) as pool:
            decoded = dict(zip(missing, pool.map(self.decode, missing.values()), strict=True))
        return [self.get(row, decoded.get(row["id"])) for row in rows]

    def get(self, row, decoded=None):
        key = row["id"]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        images, times, record, audit = self.decode(row) if decoded is None else decoded
        layout = prepare_native(self.bb.processor, images, times, record, audit,
                                allow_native_timestamp_approximation=self.args.timestamp_policy == "native-index-fps")
        self.bb.visual.eval()
        steps = encode_native(self.bb, layout)
        record = {**record, "native_layout": layout}
        if not self.cache and self.args.native_check:
            # Native parity is an engineering guard; gold answers never enter it.
            self.bb.language_model.eval()
            check_native_forward(self.bb, layout, steps, record)
        steps = [VisualStep(s.tokens.detach().cpu(), s.grid, s.end, s.frame_times) for s in steps]
        # Pixels are no longer needed for the frozen-feature training path.
        layout = NativeLayout({k: v for k, v in layout.inputs.items() if k != "pixel_values_videos"},
                              layout.frame_times, layout.question, layout.audit)
        sample = (steps, {**record, "native_layout": layout})
        nbytes = sum(s.tokens.numel() * s.tokens.element_size() for s in steps)
        nbytes += sum(x.numel() * x.element_size() for x in layout.inputs.values())
        limit = self.args.cache_mb * 1024**2
        while self.cache and self.bytes + nbytes > limit:
            _, (_, removed) = self.cache.popitem(last=False)
            self.bytes -= removed
        if nbytes <= limit:
            self.cache[key] = (sample, nbytes)
            self.bytes += nbytes
        return sample


def run(args):
    import transformers
    world, rank, local = (int(os.environ.get(k, default))
                          for k, default in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    microbatch = getattr(args, "microbatch_size", 1)
    if not 1 <= getattr(args, "decode_workers", 1) <= 16:
        raise ValueError("decode_workers must be in [1, 16]")
    if min(args.updates, args.accumulation, args.rank, args.save_every, args.steps, microbatch) < 1 or args.updates >= 1_000_000:
        raise ValueError("positive training counts required")
    if args.examples < 0 or args.cache_mb < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid examples/cache/lr")
    if args.stop_after is not None and not 0 < args.stop_after <= args.updates:
        raise ValueError("stop-after must be within the training schedule")
    preserve = getattr(args, "preservation_teacher", False)
    weight = getattr(args, "preservation_weight", 0.)
    temperature = getattr(args, "preservation_temperature", 2.)
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid preservation objective")
    if weight and not preserve:
        raise ValueError("positive preservation weight requires the original teacher")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=torch.device(f"cuda:{local}"))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = sorted(read_split(Path(args.manifest), "train"), key=lambda r: r["id"])
    initialization = initial_checkpoint(args)
    if preserve:
        if initialization is not None or args.mode != 'grid':
            raise ValueError('preservation pilot starts from the original grid backbone')
        families = {r.get('mixture_family') for r in rows}
        if not {'general', 'temporal'} <= families or not families <= {'general', 'temporal', 'reviewed'}:
            raise ValueError('preservation needs declared general/temporal training families')
    if args.examples:
        if len(rows) < args.examples:
            raise ValueError("not enough training questions")
        rows = rows[:args.examples]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (out / ".train.lock").open("a")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        configuration = {k: v for k, v in vars(args).items() if k not in {"out", "stop_after"}}
        paths = sorted({r["video"] for r in rows})
        with ThreadPoolExecutor(8) as pool:
            videos = dict(zip(paths, pool.map(sha256, paths), strict=True))
        identity = {"protocol": "core-native-qa-train-v1", "config": configuration, "world_size": world,
                    "objective": ("mean per-question answer CE + configured weight * T^2 KL(original || student) "
                                  "on general-family answer positions incl EOS" if preserve else
                                  "mean per-question answer-token CE including EOS"),
                    "manifest_sha256": sha256(args.manifest), "ids": [r["id"] for r in rows],
                    "videos": videos,
                    "model": checkpoint_identity(args.model), "torch": str(torch.__version__),
                    "transformers": transformers.__version__,
                    "deterministic_algorithms": True, "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    "initial_checkpoint": initialization,
                    "code": {name: sha256(Path(__file__).parent / name)
                             for name in ("train.py", "native.py", "qa.py", "data.py", "control.py", "temporal.py", "preservation.py",
                                          "../v3/backbone.py", "../v3/positions.py", "../v3/run_io.py")}}
        identity["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ensure_run_manifest(out / "run.json", identity)
    if world > 1:
        dist.barrier()
    run_id = json.loads((out / "run.json").read_text())["id"]
    previous = latest_checkpoint(out, run_id)
    if previous and checked_checkpoint(previous, run_id)["update"] == args.updates:
        if rank == 0:
            print(f"VERIFIED already complete: {previous}", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    bb = Backbone(BackboneConfig(name=args.model, dtype="bfloat16", local_files_only=True),
                  StreamConfig(pixels_per_frame=args.pixels), device=f"cuda:{local}")
    adapter = str(Path(args.init_checkpoint) / "adapter") if getattr(args, "init_checkpoint", None) else None
    bb.add_lora(args.rank, 2 * args.rank, dropout=0., adapter=adapter)
    bb.enable_checkpointing()
    if bb.tokenizer.eos_token_id is None:
        raise ValueError("answer EOS required")
    net = NativeVideoQALoss(bb, mode=args.mode, pool_grid=tuple(args.pool_grid),
                           answer_end_token_id=bb.tokenizer.eos_token_id)
    attach_temporal(net, getattr(args, "temporal", "none"))
    params = {n: p for n, p in net.named_parameters() if p.requires_grad}
    if not params or any("lora_" not in n and not n.startswith("temporal.") for n in params):
        raise ValueError("only LM LoRA and the declared temporal adapter may train")
    if rank == 0:
        # Recorded before restoring a resumed optimizer/checkpoint: both arms
        # must originate from identical base weights and identical LoRA tensors.
        initial = {"run": run_id, "lora_sha256": tensor_digest({
            n: p for n, p in params.items() if "lora_" in n}),
            "temporal": getattr(args, "temporal", "none"),
            "trainable_parameters": sum(p.numel() for p in params.values())}
        ensure_run_manifest(out / "initialization.json", {"id": run_id, **initial})
    # Temporal construction consumes random numbers; keep subsequent stochastic
    # LM operations on the same stream in the two fresh-base arms.
    torch.manual_seed(args.seed + 100000)
    torch.cuda.manual_seed_all(args.seed + 100000)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    start, metrics = (0, []) if previous is None else restore_checkpoint(previous, run_id, params, optimizer, rank, world)
    wrapped = DDP(net, device_ids=[local], broadcast_buffers=False) if world > 1 else net
    samples = Samples(bb, args)
    tracker = Tracker()
    if rank == 0:
        # Drop any log suffix beyond the restored checkpoint before appending.
        atomic_text(out / "metrics.jsonl", "".join(json.dumps(x) + "\n" for x in metrics))
        group = os.environ.get("TRACE_WANDB_GROUP") or out.parent.parent.name
        tracker = Tracker.start(enabled_from_env(), name=f"{group}-{out.name}", group=group,
            job_type="core-train", config={**configuration, "run_id": run_id, "world_size": world,
                "effective_batch_size": world * args.accumulation * microbatch}, directory=out / "wandb")
    begin = time.monotonic()
    end = args.stop_after or args.updates
    for update in range(start, end):
        update_begin = time.monotonic()
        net.train()
        optimizer.zero_grad(set_to_none=True)
        lr = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * update / max(1, args.updates - 1))))
        for group in optimizer.param_groups:
            group["lr"] = lr
        selected = sample_indices(len(rows), update, world, args.accumulation * microbatch, rank, args.seed)
        local_loss, local_tokens, selected_ids = 0., 0, []
        local_ce, local_kl, local_anchors = 0., 0., 0
        sample_seconds = 0.
        for micro in range(args.accumulation):
            begin_samples = time.monotonic()
            batch = samples.batch([rows[index] for index in selected[micro * microbatch:(micro + 1) * microbatch]])
            sample_seconds += time.monotonic() - begin_samples
            bb.language_model.train()
            if microbatch == 1:
                steps, record = batch[0]
                anchors = int(preserve and record.get('mixture_family') == 'general')
                teacher = original_answer_logits(net, steps, record) if anchors else None
                applied_weight = weight if anchors else 0.
            else:
                steps, record = [x[0] for x in batch], [x[1] for x in batch]
                anchor_indices = [i for i, r in enumerate(record) if preserve and r.get('mixture_family') == 'general']
                teacher = [None] * microbatch
                if anchor_indices:
                    values = original_answer_logits(net, [steps[i] for i in anchor_indices],
                                                    [record[i] for i in anchor_indices])
                    for i, value in zip(anchor_indices, values, strict=True):
                        teacher[i] = value
                anchors, applied_weight = len(anchor_indices), weight
            context = wrapped.no_sync() if world > 1 and micro < args.accumulation - 1 else nullcontext()
            with context:
                loss, meta = wrapped(steps, record, teacher_logits=teacher,
                    preservation_weight=applied_weight, preservation_temperature=temperature)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite answer loss")
                (loss / args.accumulation).backward()
            local_loss += float(loss.detach()) * microbatch
            local_ce += meta['answer_ce'] * microbatch
            local_kl += meta['preservation_kl'] * microbatch
            local_anchors += anchors
            del teacher
            local_tokens += meta["answer_tokens"]
            selected_ids.extend(x[1]["id"] for x in batch)
        norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True))
        temporal_norm = sum(float(p.grad.detach().float().square().sum()) for n, p in params.items()
                            if n.startswith("temporal.") and p.grad is not None) ** .5
        if hasattr(net, "temporal") and (not math.isfinite(temporal_norm) or temporal_norm <= 0):
            raise RuntimeError("temporal adapter receives no finite QA gradient")
        if norm <= 0:
            raise RuntimeError("LM adapter receives no answer gradient")
        optimizer.step()
        totals = torch.tensor([local_loss, local_tokens, local_ce, local_kl, local_anchors], dtype=torch.float64, device=f"cuda:{local}")
        if world > 1:
            dist.all_reduce(totals)
        memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                              device=f"cuda:{local}", dtype=torch.float64)
        if world > 1:
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        rank_ids = collective_list(selected_ids, world)
        batch_size = world * args.accumulation * microbatch
        entry = {"update": update + 1, "loss": float(totals[0]) / batch_size,
                 "answer_tokens": int(totals[1]), "question_ids_by_rank": rank_ids, "lr": lr,
                 "answer_ce": float(totals[2]) / batch_size,
                 "preservation_kl_per_question": float(totals[3]) / batch_size,
                 "preservation_kl_per_anchor": float(totals[3]) / max(1., float(totals[4])),
                 "teacher_questions": int(totals[4]),
                 "gradient_norm": norm, "wall_seconds_this_invocation": time.monotonic() - begin,
                 "microbatch_size": microbatch, "effective_batch_size": batch_size,
                 "seconds_per_update": time.monotonic() - update_begin,
                 "sample_preparation_seconds_rank0": sample_seconds,
                 "peak_allocated_gib_max": float(memory[0]) / 2**30,
                 "peak_reserved_gib_max": float(memory[1]) / 2**30,
                 "temporal_gradient_norm_after_clip": temporal_norm,
                 "peak_allocated_gib_rank0": torch.cuda.max_memory_allocated() / 2**30}
        metrics.append(entry)
        if rank == 0:
            entry["samples_per_second"] = batch_size / entry["seconds_per_update"]
            entry["answer_tokens_per_second"] = entry["answer_tokens"] / entry["seconds_per_update"]
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            tracker.log(entry, step=update + 1)
        if (update + 1) % args.save_every == 0 or update + 1 == end:
            save_checkpoint(out, update + 1, run_id, bb, params, optimizer, rank, world, metrics)
    if rank == 0:
        final = latest_checkpoint(out, run_id)
        record = checked_checkpoint(final, run_id)
        atomic_text(out / "status.json", json.dumps({"completed": record["update"] == args.updates,
                    "run": run_id, "update": record["update"], "target_updates": args.updates,
                    "checkpoint": final.name, "world_size": world,
                    "trainable_parameters": sum(p.numel() for p in params.values()),
                    "adapter_sha256": record["adapter_sha256"]}, indent=2) + "\n")
        print(f"{'VERIFIED training complete' if end == args.updates else 'CHECKPOINTED partial run'}: {out}", flush=True)
    if world > 1:
        dist.destroy_process_group()
    if rank == 0:
        tracker.summary({"completed_updates": end, "completed": end == args.updates,
                         "checkpoint": str(final), "run_id": run_id})
        tracker.upload(out.name + "-training", "core-training", [out / "run.json", out / "metrics.jsonl",
                       out / "status.json"], base=out)
        tracker.finish()
    if lock:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--mode", choices=("full", "grid"), default="grid")
    p.add_argument("--temporal", choices=("none", "causal", "causal-pool"), default="none")
    p.add_argument("--preservation-teacher", action="store_true")
    p.add_argument("--preservation-weight", type=float, default=0.)
    p.add_argument("--preservation-temperature", type=float, default=2.)
    p.add_argument("--init-checkpoint", help="new optimizer/schedule, initialized from a verified plain baseline")
    p.add_argument("--pool-grid", nargs=2, type=int, default=[4, 5])
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--pixels", type=int, default=256 * 256)
    p.add_argument("--examples", type=int, default=0, help="zero uses all saved training rows")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--microbatch-size", type=int, default=1, help="questions processed together per GPU")
    p.add_argument("--decode-workers", type=int, default=1, help="parallel CPU decoders per GPU process")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--cache-mb", type=int, default=512)
    p.add_argument("--timestamp-policy", choices=("strict", "native-index-fps"), default="strict")
    p.add_argument("--native-check", action="store_true")
    p.add_argument("--stop-after", type=int, help="checkpoint early; preserves the original total-update schedule")
    run(p.parse_args())


if __name__ == "__main__":
    main()
