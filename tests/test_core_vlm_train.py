import hashlib
import json

import pytest
import torch

from trace_vl.core_vlm.train import checked_checkpoint, latest_checkpoint, sample_indices, tensor_digest


def test_global_sampler_has_no_padding_and_restarts_shuffle_deterministically():
    # Ten samples across five two-GPU updates form one exact shuffled epoch.
    seen = [sample_indices(10, u, 2, 1, r, 17)[0] for u in range(5) for r in range(2)]
    assert sorted(seen) == list(range(10))
    assert seen == torch.randperm(10, generator=torch.Generator().manual_seed(17)).tolist()
    next_seen = [sample_indices(10, u, 2, 1, r, 17)[0] for u in range(5, 10) for r in range(2)]
    assert next_seen == torch.randperm(10, generator=torch.Generator().manual_seed(18)).tolist()
    # Same global batch under single-GPU accumulation and two-GPU accumulation.
    assert sample_indices(10, 2, 1, 4, 0, 17) == [
        sample_indices(10, 2, 2, 2, r, 17)[m] for m in range(2) for r in range(2)]


def test_resume_sampler_needs_no_mutable_dataloader_cursor():
    whole = [sample_indices(7, u, 2, 3, 1, 8) for u in range(11)]
    resumed = [sample_indices(7, u, 2, 3, 1, 8) for u in range(4, 11)]
    assert whole[4:] == resumed
    with pytest.raises(ValueError):
        sample_indices(7, 0, 2, 3, 2, 8)


def checkpoint(tmp_path, update=2):
    from safetensors.torch import save_file
    path = tmp_path / f"checkpoint-{update:06d}"
    (path / "adapter").mkdir(parents=True)
    params = {"lora": torch.ones(2)}
    torch.save({"run": "fixed", "update": update, "parameters": params,
                "optimizer": {}, "rng": [{"cpu": torch.get_rng_state()}], "metrics": []}, path / "training.pt")
    save_file(params, path / "adapter/adapter_model.safetensors")
    (path / "adapter/adapter_config.json").write_text('{}')
    files = {}
    for name in ("training.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
        files[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "complete.json").write_text(json.dumps({"run": "fixed", "update": update, "files": files,
                                                    "adapter_sha256": tensor_digest(params)}))
    return path


def test_resume_ignores_temporary_work_but_refuses_corrupt_committed_checkpoint(tmp_path):
    (tmp_path / ".checkpoint-partial").mkdir()
    assert latest_checkpoint(tmp_path, "fixed") is None
    path = checkpoint(tmp_path)
    assert latest_checkpoint(tmp_path, "fixed") == path
    with pytest.raises(ValueError, match="identity"):
        checked_checkpoint(path, "other")
    (path / "adapter/adapter_model.safetensors").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="file differs"):
        latest_checkpoint(tmp_path, "fixed")


def test_empty_checkpoint_cannot_be_complete(tmp_path):
    path = checkpoint(tmp_path)
    record = json.loads((path / "complete.json").read_text())
    del record["files"]["adapter/adapter_model.safetensors"]
    (path / "complete.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="lacks adapter"):
        checked_checkpoint(path, "fixed")


def test_tensor_digest_detects_dtype_shape_and_values():
    value = torch.arange(6, dtype=torch.float32)
    digest = tensor_digest({"a": value})
    assert digest == tensor_digest({"a": value.clone()})
    assert digest != tensor_digest({"a": value.reshape(2, 3)})
    assert digest != tensor_digest({"a": value.to(torch.bfloat16)})
    assert digest != tensor_digest({"a": value + 1})
