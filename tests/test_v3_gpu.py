"""CUDA-only regressions that CPU tests cannot catch (skipped without a GPU; no backbone weights needed)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_alert_loss_is_autocast_safe_with_choice_targets():
    """PR #3 review: F.binary_cross_entropy on the confidence crashed under bf16 autocast."""
    from trace_vl.v3.alert import DecisionMemory, EncodedQuestions, Question, alert_loss
    from trace_vl.v3.config import V3Config
    from trace_vl.v3.model import TraceV3Modules
    from trace_vl.v3.modules import grid_coords

    cfg = V3Config.from_dict({"cst": {"dim": 32, "layers": 1, "heads": 4}})
    modules = TraceV3Modules(cfg, model_dim=16).cuda().train()
    questions = [Question("abnormal", "noul", "Is anything abnormal?"),
                 Question("incident", "choice", "Which?", ["none", "fall"])]  # fmt: skip
    enc = EncodedQuestions(questions, torch.randn(2, 16, device="cuda"), [None, torch.randn(2, 16, device="cuda")])
    taps = torch.randn(len(cfg.alert.taps), 6, 16, device="cuda")
    memory = DecisionMemory(8)
    memory.add(torch.randn(3, 16, device="cuda"), 0.0)
    states = torch.randn(*modules.alert.state_shape, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        results = modules.alert(enc, taps, grid_coords(2, 3, "cuda"), memory, 1.0, torch.zeros(4), states)
        loss, stats = alert_loss(results, {"abnormal": 1.0, "incident": 1})
    loss.backward()
    assert torch.isfinite(loss) and "alert_confidence" in stats
