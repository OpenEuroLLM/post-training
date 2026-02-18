"""Callback that saves minimal inference-only checkpoints.

Unlike the full training checkpoints managed by TRL (which include
optimizer / scheduler states), these contain only the model weights and
tokenizer — just enough to load the model for inference or evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class InferenceCheckpointCallback(TrainerCallback):
    """Save model + tokenizer every *save_steps* global steps.

    Parameters
    ----------
    save_steps:
        Interval (in global steps) between inference checkpoint saves.
    output_dir:
        Directory under which ``step-<N>/`` sub-directories are created.
    """

    def __init__(self, save_steps: int, output_dir: str | Path) -> None:
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs,
    ) -> None:
        # Treat non-positive save interval as \"disabled\" (extra safety in case of
        # misconfiguration; such values should normally prevent the callback from
        # being constructed at all).
        if self.save_steps <= 0:
            return

        if state.global_step == 0:
            return
        if state.global_step % self.save_steps != 0:
            return

        # Only save on the main process.
        if not state.is_world_process_zero:
            return

        save_path = self.output_dir / f"step-{state.global_step}"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Saving inference checkpoint at step %d → %s",
            state.global_step,
            save_path,
        )

        if model is not None:
            model.save_pretrained(save_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
