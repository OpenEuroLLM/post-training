"""Callback that logs Model FLOP Utilization (MFU) at every logging step."""

from __future__ import annotations

import logging
import os
import re
import time

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# Peak FLOP/s (BF16/FP16) for common GPU architectures.
# Keys are substrings matched against torch.cuda.get_device_properties().name.
# Note: "H200" matches "GH200" via substring, so Grace Hopper chips resolve
# to the correct H-series peak without a dedicated entry.
_GPU_PEAK_TFLOPS: dict[str, float] = {
    "H200": 989.0,  # https://www.nvidia.com/en-us/data-center/h200/
    "H100": 989.0,  # https://www.nvidia.com/en-us/data-center/h100/
    "A100": 312.0,  # https://www.nvidia.com/en-us/data-center/a100/
}

# Matches parameter names belonging to a routed MoE expert, e.g.
# "model.layers.0.mlp.experts.5.gate_proj.weight".  Shared experts (e.g.
# ".shared_experts.") are deliberately excluded because they activate on
# every token and should count as dense.
_ROUTED_EXPERT_PATTERN = re.compile(r"\.experts\.\d+\.")


def _detect_peak_flops() -> float | None:
    """Return estimated peak FLOP/s for the current GPU, or *None* if unknown."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).name
    for key, tflops in _GPU_PEAK_TFLOPS.items():
        if key in name:
            logger.info("MFUCallback: detected GPU '%s' → peak %.0f TFLOP/s", name, tflops)
            return tflops * 1e12
    logger.warning(
        "MFUCallback: unknown GPU '%s'; set peak_flops_per_device manually to enable MFU logging.",
        name,
    )
    return None


def _detect_moe(model) -> tuple[int, int] | None:
    """Return ``(num_experts, num_experts_per_tok)`` if *model* is MoE, else ``None``.

    Handles the common field-name variants across Mixtral, Qwen-MoE, OLMoE,
    and DeepSeek-MoE.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    num_experts = (
        getattr(config, "num_experts", None)
        or getattr(config, "num_local_experts", None)
        or getattr(config, "n_routed_experts", None)
    )
    num_experts_per_tok = getattr(config, "num_experts_per_tok", None) or getattr(
        config, "num_experts_per_token", None
    )

    if num_experts and num_experts_per_tok and int(num_experts_per_tok) < int(num_experts):
        return int(num_experts), int(num_experts_per_tok)
    return None


def _count_params(model) -> tuple[int, int]:
    """Return ``(total_params, routed_expert_params)``.

    Uses ``p.ds_numel`` for DeepSpeed ZeRO-3 sharded parameters so the full
    (unpartitioned) size is reported.
    """
    total = 0
    routed_expert = 0
    for name, p in model.named_parameters():
        n = p.ds_numel if hasattr(p, "ds_numel") else p.numel()
        total += n
        if _ROUTED_EXPERT_PATTERN.search(name):
            routed_expert += n
    return total, routed_expert


class MFUCallback(TrainerCallback):
    """Logs ``throughput/mfu`` at every logging step.

    MFU (Model FLOP Utilization) measures what fraction of the GPU's
    theoretical peak throughput is being achieved::

        MFU = (tokens_per_sec_per_gpu × flops_per_token_coeff × active_params) / peak_flops_per_gpu

    The coefficient depends on the training method:

    - **SFT** (default): ``6N`` per token — 2N forward + 4N backward.
    - **DPO**: ``8N`` per token — policy forward (2N) + policy backward (4N)
      + reference forward (2N, no backward).

    Gradient-checkpointing rematerialization is intentionally excluded from
    the numerator; this is the standard MFU convention (HFU would include it).

    For dense models, ``active_params == total_params``.  For MoE models,
    only a subset of expert weights participate in each token's forward
    pass, so::

        active_params = dense_params
                      + routed_expert_params × (num_experts_per_tok / num_experts)

    Requires ``include_num_input_tokens_seen=True`` on the trainer so that
    ``num_input_tokens_seen`` is present in the logs at each logging step.

    Parameters
    ----------
    peak_flops_per_device:
        Theoretical peak FLOP/s for a single GPU.  When *None* (default) the
        value is auto-detected from the GPU device name via a built-in lookup
        table.  Pass an explicit value for unsupported or custom hardware.
    flops_per_token_coeff:
        FLOPs-per-token multiplier in units of the active parameter count.
        Default ``6.0`` (SFT).  Pass ``8.0`` for DPO to account for the
        reference-model forward pass.
    """

    def __init__(
        self,
        peak_flops_per_device: float | None = None,
        flops_per_token_coeff: float = 6.0,
    ) -> None:
        self._configured_peak_flops = peak_flops_per_device
        self._flops_per_token_coeff = flops_per_token_coeff
        self._peak_flops: float | None = None
        self._active_params: int | None = None
        self._last_time: float = 0.0
        self._last_tokens: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is not None:
            total_params, routed_expert_params = _count_params(model)
            moe = _detect_moe(model)

            if moe is not None and routed_expert_params > 0:
                num_experts, num_experts_per_tok = moe
                dense_params = total_params - routed_expert_params
                active_expert_params = int(
                    routed_expert_params * num_experts_per_tok / num_experts
                )
                self._active_params = dense_params + active_expert_params
                logger.info(
                    "MFUCallback: MoE detected — total=%.2fB, active=%.2fB "
                    "(%d/%d experts per token, dense=%.2fB, routed=%.2fB)",
                    total_params / 1e9,
                    self._active_params / 1e9,
                    num_experts_per_tok,
                    num_experts,
                    dense_params / 1e9,
                    routed_expert_params / 1e9,
                )
            else:
                self._active_params = total_params
                logger.info(
                    "MFUCallback: dense model — num_params=%d (%.2fB)",
                    total_params,
                    total_params / 1e9,
                )

        self._peak_flops = self._configured_peak_flops or _detect_peak_flops()
        self._last_time = time.perf_counter()
        self._last_tokens = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None or self._active_params is None or self._peak_flops is None:
            return

        tokens = logs.get("num_input_tokens_seen")
        if tokens is None:
            return

        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed <= 0:
            return

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        tps_per_gpu = (tokens - self._last_tokens) / elapsed / world_size

        tflops_per_gpu = tps_per_gpu * self._flops_per_token_coeff * self._active_params / 1e12
        mfu = tflops_per_gpu / (self._peak_flops / 1e12)
        logs["throughput/tflops_per_gpu"] = round(tflops_per_gpu, 2)
        logs["throughput/mfu"] = round(mfu, 4)

        self._last_time = now
        self._last_tokens = tokens
