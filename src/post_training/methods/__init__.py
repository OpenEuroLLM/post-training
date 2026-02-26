"""Training method factory.

Each method (SFT, DPO, …) lives in its own module and exposes a
``build_*_trainer`` function.  The :func:`build_trainer` factory
dispatches to the correct builder based on ``config.method``.

Imports are lazy so that HuggingFace / TRL are not loaded until after
offline-mode environment variables have been set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Trainer

    from post_training.config import PostTrainingConfig


def build_trainer(config: PostTrainingConfig, run_dir: Path) -> Trainer:
    """Build the trainer for the configured method.

    Parameters
    ----------
    config:
        Fully resolved configuration (``config.method`` selects the
        training method).
    run_dir:
        Run output directory (already created).

    Returns
    -------
    Trainer
        A TRL trainer (e.g. ``SFTTrainer``, ``DPOTrainer``) ready to
        call ``.train()``.
    """
    if config.method == "sft":
        from post_training.methods.sft import build_sft_trainer

        return build_sft_trainer(config, run_dir)

    if config.method == "dpo":
        from post_training.methods.dpo import build_dpo_trainer

        return build_dpo_trainer(config, run_dir)

    raise ValueError(f"Unknown method '{config.method}'. Choose from: sft, dpo")
