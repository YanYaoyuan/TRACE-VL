"""Original-backbone answer-distribution anchor, with no second model copy."""
from __future__ import annotations

import torch

from .native import NativeVideoQALoss


@torch.no_grad()
def original_answer_logits(student, steps, record):
    """Same pixels/prompt/grid/gold prefix, original LM, no temporal adapter.

    Only frozen base parameters are allowed outside LoRA. Restore every training
    flag and the active adapters before student forward AND checkpointed backward.
    No recurrent cache is created. This anchors answer-token distributions only.
    """
    bb = student.bb
    if not hasattr(bb, 'peft'):
        raise ValueError('preservation requires a PEFT backbone')
    if any(p.requires_grad and 'lora_' not in name for name, p in bb.model.named_parameters()):
        raise ValueError('base weights must be frozen for the shared original teacher')
    flags = [(module, module.training) for module in bb.model.modules()]
    try:
        bb.model.eval()
        with bb.peft.disable_adapter():
            teacher = NativeVideoQALoss(bb, mode=student.mode, pool_grid=student.pool_grid,
                                       answer_end_token_id=student.answer_end_token_id)
            logits, _, _ = teacher.answer_logits(steps, record)
            return logits.detach()
    finally:
        for module, training in flags:
            module.training = training
