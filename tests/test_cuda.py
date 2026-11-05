"""Small real-GPU checks. Skipped only when the process cannot see CUDA devices."""

import pytest
import torch

from trace_vl.config import ModelConfig
from trace_vl.data.dataset import VideoDataset, collate_samples
from trace_vl.models.model import TraceVL
from trace_vl.runtime import move_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not visible")
@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_cuda_forward_backward_and_generation(fixture_manifest, precision):
    device = torch.device("cuda:0")
    model = TraceVL(ModelConfig()).to(device)
    dataset = VideoDataset(fixture_manifest, "train")
    batch = move_batch(collate_samples([dataset[0], dataset[1]]), device)
    with torch.autocast("cuda", dtype=precision, enabled=precision != torch.float32):
        output = model(batch)
    output["loss"].backward()
    assert torch.isfinite(model.blocks[0].q.weight.grad).all()
    with torch.no_grad(), torch.autocast("cuda", dtype=precision, enabled=precision != torch.float32):
        model.eval()
        answers, info = model.answer(output["state"].detach(), batch["questions"], max_new_tokens=2)
    assert len(answers) == 2
    assert (info["known_at"][info["valid"]] <= 3.5).all()
