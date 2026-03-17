"""Callback that logs Model FLOP Utilization (MFU) at every logging step."""

from __future__ import annotations

import logging
import os
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
_GPU_PEAK_TFLOPS: dict[str, float] = {
    "H200": 989.0,  # https://www.nvidia.com/en-us/data-center/h200/
    "H100": 989.0,  # https://www.nvidia.com/en-us/data-center/h100/
    "A100": 312.0,  # https://www.nvidia.com/en-us/data-center/a100/
}


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


class MFUCallback(TrainerCallback):
    """Logs ``throughput/mfu`` at every logging step.

    MFU (Model FLOP Utilization) measures what fraction of the GPU's
    theoretical peak throughput is being achieved::

        MFU = (tokens_per_sec_per_gpu × 6 × num_params) / peak_flops_per_gpu

    The ``6 × num_params`` term is the standard FLOPs-per-token estimate for
    transformer models (2N forward + 4N backward).

    Requires ``include_num_input_tokens_seen=True`` on the trainer so that
    ``num_input_tokens_seen`` is present in the logs at each logging step.

    Parameters
    ----------
    peak_flops_per_device:
        Theoretical peak FLOP/s for a single GPU.  When *None* (default) the
        value is auto-detected from the GPU device name via a built-in lookup
        table.  Pass an explicit value for unsupported or custom hardware.
    """

    def __init__(self, peak_flops_per_device: float | None = None) -> None:
        self._configured_peak_flops = peak_flops_per_device
        self._peak_flops: float | None = None
        self._num_params: int | None = None
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
            # With DeepSpeed ZeRO-3 parameters are sharded: p.numel() returns
            # only the local shard size.  p.ds_numel holds the full count.
            self._num_params = sum(
                p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters()
            )
            logger.info(
                "MFUCallback: num_params=%d (%.2fB)", self._num_params, self._num_params / 1e9
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
        if logs is None or self._num_params is None or self._peak_flops is None:
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

        # 6N FLOPs per token: 2N (forward) + 4N (backward)
        mfu = tps_per_gpu * 6 * self._num_params / self._peak_flops
        logs["throughput/mfu"] = round(mfu, 4)

        self._last_time = now
        self._last_tokens = tokens
