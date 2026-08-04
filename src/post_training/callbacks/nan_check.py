"""Callback that scans model parameters for NaN/Inf after each optimizer step.

Diagnostic aid for runs that diverge: catches the first step at which a
parameter goes non-finite, and names the offending parameter per rank.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


def _rank() -> int:
    """Current distributed rank, or 0 when running without ``torch.distributed``."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _local_shard(param: torch.nn.Parameter) -> torch.Tensor:
    """Return the tensor actually holding this rank's data.

    Under DeepSpeed ZeRO-3 the ``.data`` of a partitioned parameter is an
    empty placeholder and the real shard lives in ``ds_tensor``. Checking
    ``.data`` there would silently scan nothing.
    """
    if param.numel() == 0 and getattr(param, "ds_tensor", None) is not None:
        return param.ds_tensor
    return param.data


class NaNCheckCallback(TrainerCallback):
    """Report parameters that contain NaN/Inf at the end of a training step.

    Parameters
    ----------
    every_n_steps:
        Scan interval in global steps. ``1`` (the default) checks every step.
    stop_on_detect:
        When ``True``, request a training stop as soon as a non-finite
        parameter is found instead of letting the run continue.
    """

    def __init__(self, every_n_steps: int = 1, stop_on_detect: bool = False) -> None:
        self.every_n_steps = max(1, every_n_steps)
        self.stop_on_detect = stop_on_detect

    # ------------------------------------------------------------------

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is None:
            return
        if state.global_step % self.every_n_steps != 0:
            return

        rank = _rank()
        found = False

        for n, p in model.named_parameters():
            shard = _local_shard(p)
            if shard.numel() == 0 or not shard.is_floating_point():
                continue
            if torch.isnan(shard).any() or torch.isinf(shard).any():
                found = True
                # print, not logger: setup_logging() installs a NullHandler on
                # non-zero ranks, so logger output from them is discarded.
                print(
                    f"[rank {rank}] {n} is NaN/Inf after step {state.global_step}",
                    flush=True,
                )

        if found and self.stop_on_detect:
            control.should_training_stop = True
