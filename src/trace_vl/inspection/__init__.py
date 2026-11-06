"""Fixed-camera property inspection with change-triggered transition memory.

Pipeline: a causal dual-background change detector turns the stream into region-level transitions
(before/after crops with onset, confirmation and end times). A bounded store keeps them. At question
or alert time, an evidence packer spends a fixed visual-token budget on the most relevant
transitions and a pretrained video VLM answers. Baselines (uniform frames, recent frames, a text
caption log) receive the same budget.
"""
