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
from transformers.integrations import WandbCallback

from post_training.callbacks.inference_checkpoint import InferenceCheckpointCallback
from post_training.callbacks.mfu import MFUCallback
from post_training.callbacks.throughput import ThroughputCallback
from post_training.chat_templates.registry import get_chat_template, terminator_from_render

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
    tokenizer_name_or_path, revision = config.model.resolve_tokenizer()
    logger.info("Loading tokenizer '%s' (revision=%s)", tokenizer_name_or_path, revision or "main")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_str = get_chat_template(config.data.chat_template)
    tokenizer.chat_template = template_str
    logger.info("Chat template set to '%s'.", config.data.chat_template)

    # The template decides what ends a turn, and SFT trains the model to emit
    # exactly that. When it is not the tokenizer's `eos_token`, generation has
    # nothing to stop on: the model emits the token it was trained to emit and
    # nobody is listening, so it runs to `max_new_tokens` on every prompt.
    # `qwen3` ends on `<|im_end|>`; the `olmo3-*` templates already end on
    # `eos_token`, for which this is a no-op.
    # Set after the pad fallback above, so `pad_token` keeps the model's own eos
    # rather than inheriting the turn terminator.
    # Rendered through the tokenizer's own `apply_chat_template`, so what we
    # inspect is what training will actually produce — and through the public API
    # rather than transformers' private jinja helpers.
    terminator = None
    try:
        probe = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        rendered = tokenizer.apply_chat_template(probe, tokenize=False)
        terminator = terminator_from_render(rendered, tokenizer.get_added_vocab())
    except Exception:  # noqa: BLE001 - an unrenderable template must not break the run
        logger.warning(
            "Could not render chat template '%s' to find its turn terminator; "
            "leaving eos_token as %s.",
            config.data.chat_template,
            tokenizer.eos_token,
        )
    if terminator is not None and terminator != tokenizer.eos_token:
        logger.info(
            "Chat template '%s' terminates turns with %s, not the tokenizer's "
            "eos_token %s. Setting eos_token to %s so generation stops on what "
            "the model is trained to emit.",
            config.data.chat_template,
            terminator,
            tokenizer.eos_token,
            terminator,
        )
        tokenizer.eos_token = terminator
    return tokenizer


def align_generation_eos(trainer: Any) -> None:
    """Make the trained checkpoint stop on the token the template taught it.

    Runs after the trainer exists, because the model's ``generation_config`` is
    what ``generate()`` actually reads — **not** ``tokenizer.eos_token``. The two
    are independent, and the model's copy comes from the checkpoint: Prelude
    ships no ``generation_config.json`` at all and falls back to ``config.json``'s
    ``eos_token_id``, while OLMo's is an empty ``{}``. Either way the value is the
    model's pretraining terminator, which the SFT data never contains.

    The terminator is read off the tokenizer, which :func:`build_tokenizer` has
    already aligned to the template, and the pretraining eos is read off the
    model, which is where it lives. The pretraining eos is KEPT as a secondary
    stop id: the model has a prior to emit it that SFT decays without erasing, so
    a stray one should stop cleanly rather than render as text. Qwen ships a
    two-element list for the same reason.

    A no-op when the two already agree — which is the case for every ``olmo3-*``
    template, since those terminate on ``eos_token``.
    """
    model = getattr(trainer, "model", None)
    tokenizer = getattr(trainer, "processing_class", None)
    if model is None or tokenizer is None:
        return
    gc = getattr(model, "generation_config", None)
    terminator_id = getattr(tokenizer, "eos_token_id", None)
    if gc is None or terminator_id is None:
        return

    existing = gc.eos_token_id
    existing = existing if isinstance(existing, list) else [existing]
    existing = [i for i in existing if i is not None]
    if existing == [terminator_id]:
        return  # already correct; touch nothing

    gc.eos_token_id = [terminator_id] + [i for i in existing if i != terminator_id]
    logger.info(
        "generation_config.eos_token_id set to %s (%s from the chat template, "
        "then the model's own).",
        gc.eos_token_id,
        tokenizer.eos_token,
    )


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
        "revision": config.model.revision,
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
    # Normalize a falsy config (e.g. `{}`) to None so it isn't forwarded to
    # TrainingArguments as if it were an enabled DeepSpeed config.
    ds_config = config.deepspeed or None

    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(run_dir / "logs"))

    # Determine training duration kwargs. When num_train_epochs is set, max_steps
    # must be -1 (disabled) so the Trainer uses epoch-based stopping. Otherwise,
    # resolve max_steps without mutating the configured sample/token budget.
    if t.num_train_epochs is not None:
        duration_kwargs: dict[str, Any] = {
            "num_train_epochs": t.num_train_epochs,
            "max_steps": -1,
        }
        logger.info("Training duration: %.2f epochs", t.num_train_epochs)
    else:
        max_steps = config.resolve_max_steps()
        if max_steps is None:
            raise ValueError("Step-based training requires a resolvable max_steps value.")
        duration_kwargs = {"max_steps": max_steps}
        logger.info("Training duration: %d steps", max_steps)

    return dict(
        output_dir=str(run_dir / "checkpoints"),
        **duration_kwargs,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_train_batch_size,
        adam_beta1=t.adam_beta1,
        adam_beta2=t.adam_beta2,
        weight_decay=t.weight_decay,
        adam_epsilon=t.adam_epsilon,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=t.warmup_steps,
        lr_scheduler_type=t.lr_scheduler_type,
        lr_scheduler_kwargs=(
            None
            if t.lr_scheduler_kwargs is None
            else {
                k: v for k, v in dataclasses.asdict(t.lr_scheduler_kwargs).items() if v is not None
            }
        ),
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            None
            if t.gradient_checkpointing_kwargs is None
            else dataclasses.asdict(t.gradient_checkpointing_kwargs)
        ),
        use_liger_kernel=t.use_liger_kernel,
        liger_kernel_config=(
            None
            if t.liger_kernel_config is None
            else {
                k: v for k, v in dataclasses.asdict(t.liger_kernel_config).items() if v is not None
            }
        ),
        bf16=t.bf16,
        seed=t.seed,
        # Checkpointing
        save_strategy=config.checkpointing.save_strategy,
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


def sanitize_generation_config(trainer: Any) -> None:
    """Fix inconsistent ``generation_config`` so checkpoint saves don't fail.

    Some upstream models (notably Olmo-3 Think variants) ship a
    ``generation_config.json`` that sets sampling-only parameters
    (``temperature``, ``top_p``) while leaving ``do_sample=False``.  This is
    benign at training time — we never call ``model.generate`` — but
    ``transformers >= 5.x`` runs strict validation inside
    ``GenerationConfig.save_pretrained`` and refuses to write the file::

        ValueError: GenerationConfig is invalid:
          - `temperature` is set to 0.6 -- this flag is only used in
            sample-based generation modes. You should set `do_sample=True`
            or unset `temperature`.

    Every checkpoint save ultimately calls ``model.save_pretrained`` which
    writes the generation config, so an unfixed model crashes the very first
    save.  We patch ``do_sample`` to ``True`` once, on the in-memory model
    object, immediately after the trainer has been constructed.  The fix is
    local to this run — the upstream model files on the Hub are unchanged.
    """
    model = getattr(trainer, "model", None)
    if model is None:
        return
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    _FLOAT_SAMPLING_PARAMS = (
        "temperature",
        "top_p",
        "min_p",
        "top_h",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
    )
    has_sampling_param = any(
        getattr(gc, p, None) is not None for p in _FLOAT_SAMPLING_PARAMS
    ) or getattr(gc, "top_k", None) not in (None, 0)
    if has_sampling_param and not getattr(gc, "do_sample", False):
        logger.info(
            "Sanitizing generation_config: setting do_sample=True so that "
            "checkpoint saves can write generation_config.json without "
            "tripping transformers' strict validation."
        )
        gc.do_sample = True


def build_callbacks(config: PostTrainingConfig, run_dir: Path) -> list:
    """Build the callback list (shared across methods)."""
    callbacks: list = []

    callbacks.append(ThroughputCallback())
    callbacks.append(MFUCallback())

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

    return callbacks


def prioritize_metric_callbacks(trainer: Any) -> None:
    """Move ``ThroughputCallback``/``MFUCallback`` to run just before ``WandbCallback``.

    ``Trainer.__init__`` always places reporting-integration callbacks
    (``WandbCallback``, ``TensorBoardCallback``, ...) ahead of user-supplied
    ones, regardless of the order passed to ``callbacks=``. ``WandbCallback``
    reads the shared ``logs`` dict and ships it off inside its own
    ``on_log``, so keys added by callbacks that run *after* it (like the
    throughput/MFU keys) never reach it. Splicing our callbacks in right
    before it lets their ``logs[...] = ...`` mutations land in time, in the
    same on_log call.
    """
    metric_callback_types = (ThroughputCallback, MFUCallback)
    callbacks = trainer.callback_handler.callbacks
    metric_callbacks = [cb for cb in callbacks if isinstance(cb, metric_callback_types)]
    if not metric_callbacks:
        return
    rest = [cb for cb in callbacks if not isinstance(cb, metric_callback_types)]
    wandb_index = next((i for i, cb in enumerate(rest) if isinstance(cb, WandbCallback)), None)
    if wandb_index is None:
        return
    rest[wandb_index:wandb_index] = metric_callbacks
    trainer.callback_handler.callbacks = rest
