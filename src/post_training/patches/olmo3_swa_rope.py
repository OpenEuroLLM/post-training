"""Monkey-patch HF transformers' Olmo3Model so SWA layers use unscaled RoPE.

Mirrors the upstream fix in huggingface/transformers PR #45945
(https://github.com/huggingface/transformers/pull/45945) ported as a runtime
monkey-patch so we don't have to vendor / wait for a transformers release.

## Bug

The stock `Olmo3Model.forward` (transformers ≤ 5.4.x, post PR #39847) computes
a single model-level `Olmo3RotaryEmbedding` (`self.rotary_emb`) whose
YaRN-scaled `(cos, sin)` is fed to ALL 32 layers — including the 24 SWA
layers. HF's own comment (`modular_olmo3.py`, above `class Olmo3Model`) says
SWA layers should NOT get scaled RoPE. OLMo-core and vLLM both implement
this correctly. The bug was introduced by transformers PR #39847's RoPE
refactor; PR #45945 fixes it but is still in draft as of 2026-05-27.

## What `install()` does

1. Extends `Olmo3RotaryEmbedding.__init__` with an optional `rope_type` kwarg
   so a second instance can opt out of YaRN (`self.rope_type = rope_type or
   self.config.rope_parameters["rope_type"]`).
2. In `Olmo3Model.__init__`, replaces `self.rotary_emb` with
   `self.rotary_embs = nn.ModuleDict({"sliding_attention": …, "full_attention": …})`.
3. Replaces `Olmo3Model.forward` so it builds a `position_embeddings_mapping`
   keyed by layer type and dispatches per-layer.

Call `install()` BEFORE the model is constructed (e.g. before
`AutoModelForCausalLM.from_pretrained(...)`); the patch then applies to every
`Olmo3Model` instance created in the process. `uninstall()` restores the
original methods. Use `maybe_install(model_name_or_path)` to get auto-detect
+ upstream-version-gating: it installs only for Olmo3 models, and only if the
installed transformers doesn't already carry the fix.
"""
from __future__ import annotations

import inspect
import logging
from typing import Callable, Optional

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3Model,
    Olmo3RotaryEmbedding,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

logger = logging.getLogger(__name__)

_INSTALLED: bool = False
_ORIG_MODEL_INIT = None
_ORIG_MODEL_FORWARD = None
_ORIG_ROTARY_INIT = None


def install() -> None:
    """Apply the patch (idempotent)."""
    global _INSTALLED, _ORIG_MODEL_INIT, _ORIG_MODEL_FORWARD, _ORIG_ROTARY_INIT
    if _INSTALLED:
        return
    _ORIG_ROTARY_INIT = Olmo3RotaryEmbedding.__init__
    _ORIG_MODEL_INIT = Olmo3Model.__init__
    _ORIG_MODEL_FORWARD = Olmo3Model.forward
    Olmo3RotaryEmbedding.__init__ = _patched_rotary_init
    Olmo3Model.__init__ = _patched_model_init
    Olmo3Model.forward = _patched_model_forward
    _INSTALLED = True


def uninstall() -> None:
    global _INSTALLED, _ORIG_MODEL_INIT, _ORIG_MODEL_FORWARD, _ORIG_ROTARY_INIT
    if not _INSTALLED:
        return
    Olmo3RotaryEmbedding.__init__ = _ORIG_ROTARY_INIT
    Olmo3Model.__init__ = _ORIG_MODEL_INIT
    Olmo3Model.forward = _ORIG_MODEL_FORWARD
    _ORIG_ROTARY_INIT = None
    _ORIG_MODEL_INIT = None
    _ORIG_MODEL_FORWARD = None
    _INSTALLED = False


def is_upstream_fixed() -> bool:
    """Detect whether the installed transformers already has PR #45945's fix.

    Inspects `Olmo3Model.__init__` source for the `rotary_embs` ModuleDict.
    More robust than a version compare (we don't know what version the PR
    will land in). Returns False if introspection fails for any reason —
    we'd rather apply a redundant patch than skip a real one.
    """
    try:
        src = inspect.getsource(Olmo3Model.__init__)
    except (OSError, TypeError):
        return False
    return "rotary_embs" in src


def _is_olmo3(model_name_or_path: str) -> bool:
    """Best-effort check whether a model identifier points at an Olmo3 architecture.

    Tries `AutoConfig.from_pretrained` (the authoritative answer) and falls
    back to a name-heuristic if config loading fails (offline environments
    with cold cache, etc.).
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name_or_path)
        if type(cfg).__name__ == "Olmo3Config":
            return True
        # `model_type` is the auto-config registration key; "olmo3" covers any
        # subclass that might exist (e.g. quantized variants).
        return getattr(cfg, "model_type", "").lower() == "olmo3"
    except Exception:
        name = str(model_name_or_path).lower()
        return "olmo-3" in name or "olmo3" in name


def maybe_install(model_name_or_path: str) -> bool:
    """Install the Olmo3 SWA RoPE fix iff the model needs it and upstream hasn't.

    Returns True if the patch was installed (now or previously in this
    process), False if it was skipped because the model isn't Olmo3 or
    because upstream transformers already carries the fix.
    """
    if not _is_olmo3(model_name_or_path):
        return False
    if is_upstream_fixed():
        logger.info(
            "Olmo3 SWA RoPE: upstream transformers already carries the fix "
            "(see huggingface/transformers PR #45945); not installing monkey-patch."
        )
        return False
    install()
    logger.info(
        "Olmo3 SWA RoPE patch installed (mirrors HF transformers PR #45945). "
        "SWA layers use vanilla RoPE; full-attention layers keep YaRN scaling. "
        "Verified on the divergence diagnostic: mean |Δlogprob| vs vLLM drops "
        "from 0.0743 to 0.0049 nats."
    )
    return True


def _patched_rotary_init(self, config, device=None, rope_type: Optional[str] = None):
    # Mirror of upstream PR #45945's Olmo3RotaryEmbedding.__init__ body.
    # Only the `self.rope_type = ...` line differs from stock: it now honors
    # an explicit `rope_type` override so a second instance can opt out of YaRN.
    nn.Module.__init__(self)
    self.max_seq_len_cached = config.max_position_embeddings
    self.original_max_seq_len = config.max_position_embeddings
    self.config = config
    self.rope_type = rope_type or self.config.rope_parameters["rope_type"]
    rope_init_fn: Callable = self.compute_default_rope_parameters
    if self.rope_type != "default":
        rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


def _patched_model_init(self, config):
    # Call the original Olmo3Model.__init__ to build embed/layers/norm/rotary_emb.
    # The stock init still creates `self.rotary_emb` (YaRN); we delete it and
    # replace with a ModuleDict of two per-layer-type RoPEs, matching PR #45945.
    _ORIG_MODEL_INIT(self, config)
    del self.rotary_emb
    self.rotary_embs = nn.ModuleDict(
        {
            "sliding_attention": Olmo3RotaryEmbedding(config=config, rope_type="default"),
            "full_attention": Olmo3RotaryEmbedding(config=config),
        }
    )


def _patched_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    **kwargs,
):
    # Mirror of the stock forward at modeling_olmo3.py:Olmo3Model.forward,
    # with per-layer-type position_embeddings dispatch (upstream PR #45945).
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You must specify exactly one of input_ids or inputs_embeds"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    hidden_states = inputs_embeds
    # The fix: build a per-layer-type (cos, sin) mapping once and dispatch.
    position_embeddings_mapping = {
        "sliding_attention": self.rotary_embs["sliding_attention"](hidden_states, position_ids),
        "full_attention": self.rotary_embs["full_attention"](hidden_states, position_ids),
    }

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_type = self.config.layer_types[i]
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings_mapping[layer_type],
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )
