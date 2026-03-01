"""Shared utilities for all training method builders.

Functions in this module are used by every method (SFT, DPO, …) and
contain **no** method-specific logic.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer

from post_training.callbacks.inference_checkpoint import InferenceCheckpointCallback
from post_training.callbacks.throughput import ThroughputCallback
from post_training.chat_templates.registry import get_chat_template

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

logger = logging.getLogger(__name__)

_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Map a string like ``"bfloat16"`` to a :class:`torch.dtype`."""
    if name not in _TORCH_DTYPE_MAP:
        raise ValueError(f"Unknown torch_dtype '{name}'. Choose from {list(_TORCH_DTYPE_MAP)}")
    return _TORCH_DTYPE_MAP[name]


def build_tokenizer(config: PostTrainingConfig) -> AutoTokenizer:
    """Load the tokenizer and apply the configured chat template."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_str = get_chat_template(config.data.chat_template)
    tokenizer.chat_template = template_str
    logger.info("Chat template set to '%s'.", config.data.chat_template)
    return tokenizer


def build_model_init_kwargs(config: PostTrainingConfig) -> dict[str, Any]:
    """Return model kwargs forwarded to TRL's model loader."""
    dtype = resolve_torch_dtype(config.model.dtype)
    logger.info(
        "Model '%s' will be loaded by TRL (dtype=%s)",
        config.model.name_or_path,
        dtype,
    )
    return {
        "attn_implementation": config.model.attn_implementation,
        "dtype": dtype,
    }


def build_common_training_kwargs(
    config: PostTrainingConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """Return ``TrainingArguments`` kwargs shared across all methods."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = config.resolve_gradient_accumulation_steps(world_size)
    logger.info("world_size=%d, gradient_accumulation_steps=%d", world_size, grad_accum)

    t = config.training
    ds_config = config.load_deepspeed_config() if config.deepspeed.config_path else None

    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(run_dir / "logs"))

    # Determine training duration kwargs. When num_train_epochs is set, max_steps
    # must be -1 (disabled) so the Trainer uses epoch-based stopping. Otherwise,
    # max_steps is always set (possibly derived from num_training_samples/tokens).
    if t.num_train_epochs is not None:
        duration_kwargs: dict[str, Any] = {
            "num_train_epochs": t.num_train_epochs,
            "max_steps": -1,
        }
        logger.info("Training duration: %.2f epochs", t.num_train_epochs)
    else:
        duration_kwargs = {"max_steps": t.max_steps}
        logger.info("Training duration: %d steps", t.max_steps)

    return dict(
        output_dir=str(run_dir / "checkpoints"),
        **duration_kwargs,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        lr_scheduler_kwargs=dataclasses.asdict(t.lr_scheduler_kwargs),
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs=dataclasses.asdict(t.gradient_checkpointing_kwargs),
        use_liger_kernel=t.use_liger_kernel,
        bf16=t.bf16,
        seed=t.seed,
        # Checkpointing
        save_strategy="steps",
        save_steps=config.checkpointing.save_steps,
        save_total_limit=config.checkpointing.save_total_limit,
        # Logging
        report_to=config.logging.report_to,
        logging_steps=config.logging.logging_steps,
        run_name=run_dir.name,
        include_num_input_tokens_seen=config.logging.include_num_input_tokens_seen,
        # DeepSpeed
        deepspeed=ds_config,
    )


def build_callbacks(config: PostTrainingConfig, run_dir: Path) -> list:
    """Build the callback list (shared across methods)."""
    callbacks: list = []

    # ThroughputCallback must come first so ProfileCallback can read
    # throughput/tokens_per_gpu_per_sec from logs when computing MFU.
    callbacks.append(ThroughputCallback())

    steps = config.checkpointing.inference_checkpoint_steps
    # Treat ``None`` or non-positive values as \"disabled\" for inference checkpoints.
    if steps is not None and steps > 0:
        inference_ckpt_dir = run_dir / config.checkpointing.inference_checkpoint_path
        inference_ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            InferenceCheckpointCallback(
                save_steps=steps,
                output_dir=inference_ckpt_dir,
            )
        )

    if config.profile.enabled:
        from post_training.callbacks.profile import ProfileCallback

        p = config.profile
        trace_dir = run_dir / "profiler_traces" if p.memory else None
        callbacks.append(
            ProfileCallback(
                peak_tflops=p.peak_tflops,
                trace_dir=trace_dir,
                profiler_wait=p.profiler_wait,
                profiler_warmup=p.profiler_warmup,
                profiler_active=p.profiler_active,
            )
        )
        logger.info(
            "profile: ProfileCallback enabled%s.",
            f", memory trace → {trace_dir}" if trace_dir else "",
        )

    return callbacks
