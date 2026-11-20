"""TRACE-VL v3: transition-tokenized streaming on a hybrid VLM (Qwen3.5-2B).

Modules (docs/v3/design.md): causal spatio-temporal adapter (CST), transition encoder (latent
P-frames), transition KV memory in the full-attention layers. All new modules are gated so that at
initialization the model reproduces the backbone exactly.
"""
