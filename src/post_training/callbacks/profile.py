"""Callback that logs GPU utilization, memory, achieved TFLOPS, and MFU."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

# Bf16 peak TFLOPS (with structured sparsity) by GPU name substring.
# Values match NVIDIA's headline figures, which is the convention used in most
# MFU reporting (e.g. NanoGPT, LLaMA).  Keys are matched case-insensitively
# against torch.cuda.get_device_name(); the first match wins.
_GPU_BF16_PEAK_TFLOPS: dict[str, float] = {
    "h200": 1979.0,  # H200 SXM
    "h100": 989.0,  # H100 SXM5 / PCIe
    "a100": 312.0,  # A100 SXM4 / PCIe
    "l40s": 733.0,  # L40S
    "l40": 362.1,  # L40  (matched after l40s)
    "a40": 149.7,  # A40
    "a10g": 31.2,  # A10G
    "v100": 125.0,  # V100 fp16 (no native bf16; fp16 used as proxy)
}


def _get_peak_tflops(device: int = 0) -> float | None:
    """Return the bf16 peak TFLOPS for *device*, or ``None`` if unknown."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device).lower()
    for key, tflops in _GPU_BF16_PEAK_TFLOPS.items():
        if key in name:
            logger.info("profile: matched GPU '%s' → %.1f peak TFLOPS (bf16).", name, tflops)
            return tflops
    logger.warning(
        "profile: GPU '%s' not in peak-TFLOPS table — MFU will not be computed. "
        "Set profile.peak_tflops to provide the value manually.",
        name,
    )
    return None


def _count_params(model) -> int:
    """Count total parameters, handling DeepSpeed ZeRO-3 sharding."""
    total = 0
    for p in model.parameters():
        # p.ds_numel holds the unsharded size under ZeRO-3; fall back to numel().
        total += getattr(p, "ds_numel", p.numel())
    return total


class ProfileCallback(TrainerCallback):
    """Logs GPU utilization, memory, achieved TFLOPS, and MFU each logging step.

    This callback must be registered **after** ``ThroughputCallback`` so that
    ``throughput/tokens_per_gpu_per_sec`` is already present in ``logs`` when
    ``on_log`` fires.  :func:`~post_training.methods.common.build_callbacks`
    guarantees this ordering.

    All metrics are published under the ``profile/`` prefix:

    * ``profile/gpu_utilization_pct``        — average compute utilisation across local GPUs (%)
    * ``profile/gpu_memory_used_gb``         — average memory used across local GPUs (GB)
    * ``profile/gpu_memory_utilization_pct`` — average memory utilisation across local GPUs (%)
    * ``profile/tflops_per_gpu``             — achieved BF16 TFLOPS per GPU
    * ``profile/mfu``                        — model FLOP utilisation (0–1), if peak TFLOPS is known

    GPU stats are queried via :mod:`torch.cuda` (no extra dependencies).  MFU
    uses the standard ``6N`` FLOP-per-token approximation (forward + backward,
    excluding the ``O(L·H·S)`` attention term that is typically small).

    When *trace_dir* is provided, a PyTorch Profiler session is started on
    LOCAL_RANK 0 with ``profile_memory=True`` and ``with_stack=True``.  The
    profiler runs for one cycle (``wait → warmup → active`` steps) and writes
    a TensorBoard-compatible trace to *trace_dir*.  Use TensorBoard's
    *PyTorch Profiler* plugin or Chrome's ``chrome://tracing`` to view the
    results (``pip install torch_tb_profiler``).

    Parameters
    ----------
    peak_tflops:
        Manual override for the GPU's bf16 peak TFLOPS used in the MFU
        denominator.  When ``None`` (default) the value is looked up from
        :data:`_GPU_BF16_PEAK_TFLOPS` by GPU device name.
    trace_dir:
        Directory where PyTorch Profiler traces are written.  ``None``
        disables memory profiling.
    profiler_wait, profiler_warmup, profiler_active:
        Profiler schedule parameters (number of steps for each phase).
    """

    def __init__(
        self,
        peak_tflops: float | None = None,
        trace_dir: Path | None = None,
        profiler_wait: int = 1,
        profiler_warmup: int = 1,
        profiler_active: int = 3,
    ) -> None:
        self._peak_tflops_override = peak_tflops
        self._peak_tflops: float | None = None
        self._num_params: int | None = None

        self._trace_dir = trace_dir
        self._profiler_wait = profiler_wait
        self._profiler_warmup = profiler_warmup
        self._profiler_active = profiler_active
        self._profiler: torch.profiler.profile | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gpu_stats(self) -> dict[str, float]:
        """Query utilization and memory for all local GPUs; return averages."""
        n = torch.cuda.device_count()
        if n == 0:
            return {}

        util_pcts: list[float] = []
        mem_used_gbs: list[float] = []
        mem_pcts: list[float] = []

        for device in range(n):
            try:
                util_pcts.append(float(torch.cuda.utilization(device)))
            except Exception:  # noqa: BLE001
                pass
            try:
                free, total = torch.cuda.mem_get_info(device)
                used = total - free
                mem_used_gbs.append(used / 1e9)
                mem_pcts.append(100.0 * used / total)
            except Exception:  # noqa: BLE001
                pass

        out: dict[str, float] = {}
        if util_pcts:
            out["profile/gpu_utilization_pct"] = round(sum(util_pcts) / len(util_pcts), 1)
        if mem_used_gbs:
            out["profile/gpu_memory_used_gb"] = round(sum(mem_used_gbs) / len(mem_used_gbs), 2)
            out["profile/gpu_memory_utilization_pct"] = round(sum(mem_pcts) / len(mem_pcts), 1)
        return out

    def _start_profiler(self) -> None:
        """Initialise and enter the PyTorch Profiler context (LOCAL_RANK 0 only)."""
        from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

        self._trace_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        self._profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self._profiler_wait,
                warmup=self._profiler_warmup,
                active=self._profiler_active,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(str(self._trace_dir)),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )
        self._profiler.__enter__()
        logger.info(
            "profile: PyTorch Profiler started (wait=%d, warmup=%d, active=%d). "
            "Traces will be written to %s",
            self._profiler_wait,
            self._profiler_warmup,
            self._profiler_active,
            self._trace_dir,
        )

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is not None:
            self._num_params = _count_params(model)
            logger.info("profile: model has %d parameters.", self._num_params)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._peak_tflops = (
            self._peak_tflops_override
            if self._peak_tflops_override is not None
            else _get_peak_tflops(local_rank)
        )

        if self._trace_dir is not None and local_rank == 0:
            self._start_profiler()

    def _stop_profiler(self) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            logger.info("profile: PyTorch Profiler stopped. Traces in %s", self._trace_dir)
            self._profiler = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._profiler is None:
            return
        self._profiler.step()
        # Close the context as soon as the cycle is done so the kineto hooks
        # (including the CUDA allocator hook from profile_memory=True) are
        # deregistered and don't add overhead to the remaining training steps.
        cycle_steps = self._profiler_wait + self._profiler_warmup + self._profiler_active
        if self._profiler.step_num >= cycle_steps:
            self._stop_profiler()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Guard against the case where training ends before the cycle completes.
        self._stop_profiler()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return

        # GPU utilization and memory ---------------------------------------
        logs.update(self._gpu_stats())

        # Achieved TFLOPS and MFU ------------------------------------------
        # throughput/tokens_per_gpu_per_sec is set by ThroughputCallback
        # (which runs before this callback in the list).
        tps_per_gpu = logs.get("throughput/tokens_per_gpu_per_sec")
        if tps_per_gpu is not None and self._num_params is not None:
            flops_per_token = 6 * self._num_params
            tflops_per_gpu = tps_per_gpu * flops_per_token / 1e12
            logs["profile/tflops_per_gpu"] = round(tflops_per_gpu, 2)

            if self._peak_tflops is not None:
                logs["profile/mfu"] = round(tflops_per_gpu / self._peak_tflops, 4)
