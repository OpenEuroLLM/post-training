"""Callback that logs token throughput at every logging step."""

from __future__ import annotations

import os
import time

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class ThroughputCallback(TrainerCallback):
    """Logs ``throughput/tokens_per_sec`` and ``throughput/tokens_per_gpu_per_sec``.

    Requires ``include_num_input_tokens_seen`` to be set on the trainer so that
    ``num_input_tokens_seen`` is present in the logs at each logging step.
    """

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._last_time: float = time.perf_counter()
        self._last_tokens: int = 0

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
        tokens = logs.get("num_input_tokens_seen")
        if tokens is None:
            return

        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed <= 0:
            return

        delta_tokens = tokens - self._last_tokens
        tps = delta_tokens / elapsed
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        logs["throughput/tokens_per_sec"] = round(tps)
        logs["throughput/tokens_per_gpu_per_sec"] = round(tps / world_size)

        self._last_time = now
        self._last_tokens = tokens
