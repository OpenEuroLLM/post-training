"""Interactive pre-submission guardrails.

Prints a structured summary of all critical configuration values and
requires a single explicit confirmation before the job is submitted.
For very large GPU counts the user must type the exact count to confirm,
similar to how repository-deletion prompts work.

Pass ``--confirm`` to ``submit.py`` to skip all guardrails.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

# Total GPU count at or above this threshold triggers a type-to-confirm prompt.
_LARGE_GPU_THRESHOLD = 64

# ANSI colours (disabled when not writing to a terminal).
_USE_COLOUR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def _bold(text: str) -> str:
    return _c(text, "1")


def _red(text: str) -> str:
    return _c(text, "31")


def _yellow(text: str) -> str:
    return _c(text, "33")


def _cyan(text: str) -> str:
    return _c(text, "36")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_COL_WIDTH = 22  # label column width


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n  {_bold(_cyan(title))}")
    print(f"  {bar}")


def _row(label: str, value: str, warn: bool = False) -> None:
    formatted = _yellow(value) if warn else value
    print(f"    {label:<{_COL_WIDTH}}: {formatted}")


# ---------------------------------------------------------------------------
# DeepSpeed introspection
# ---------------------------------------------------------------------------


def _deepspeed_summary(config: PostTrainingConfig) -> str:
    """Return a short description of the DeepSpeed config, e.g. 'zero2.yaml (ZeRO stage 2)'."""
    ds_path = config.deepspeed.config_path
    if not ds_path:
        return "disabled"
    path = Path(ds_path)
    try:
        resolved = path if path.is_absolute() else Path.cwd() / path
        with open(resolved) as fh:
            ds_cfg = yaml.safe_load(fh)
        stage = ds_cfg.get("zero_optimization", {}).get("stage", "?")
        return f"{path.name}  (ZeRO stage {stage})"
    except Exception:
        return str(ds_path)


# ---------------------------------------------------------------------------
# Training-duration helpers
# ---------------------------------------------------------------------------


def _duration_summary(config: PostTrainingConfig) -> str:
    t = config.training
    if t.num_training_tokens is not None:
        return f"{t.num_training_tokens:,} tokens → {t.max_steps:,} steps"
    if t.num_training_samples is not None:
        return f"{t.num_training_samples:,} samples → {t.max_steps:,} steps"
    if t.max_steps is not None:
        return f"{t.max_steps:,} steps"
    if t.num_train_epochs is not None:
        return f"{t.num_train_epochs} epoch(s)"
    return "unknown"


def _batch_summary(config: PostTrainingConfig, total_gpus: int) -> tuple[str, str]:
    """Return (batch_line, grad_acc_line)."""
    t = config.training
    try:
        gas = t.effective_batch_size // (t.per_device_train_batch_size * total_gpus)
        remainder = t.effective_batch_size % (t.per_device_train_batch_size * total_gpus)
        gas_str = str(gas) if remainder == 0 else f"{gas} (⚠ not exact — check batch sizes)"
    except ZeroDivisionError:
        gas_str = "?"
    batch_line = (
        f"{t.per_device_train_batch_size} per device"
        f"  ×  {total_gpus} GPUs"
        f"  ×  {gas_str} grad-acc"
        f"  =  {t.effective_batch_size} effective"
    )
    return batch_line, gas_str


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_guardrails(config: PostTrainingConfig, run_dir: Path) -> None:
    """Print a full config summary and ask for confirmation.

    Parameters
    ----------
    config:
        Fully resolved post-training configuration.
    run_dir:
        The run directory that will be used for this job.
    """
    total_gpus = config.slurm.num_nodes * config.slurm.gpus_per_node

    print()
    print(_bold("╔══════════════════════════════════════════════════════════════╗"))
    print(_bold("║         SUBMISSION REVIEW — verify before proceeding         ║"))
    print(_bold("╚══════════════════════════════════════════════════════════════╝"))

    # ------------------------------------------------------------------
    # SLURM
    # ------------------------------------------------------------------
    _section("SLURM")
    _row("Job name", config.slurm.job_name)
    _row("Partition", config.slurm.partition)
    _row("Nodes", str(config.slurm.num_nodes))
    gpu_summary = (
        f"{config.slurm.gpus_per_node} per node"
        f"  ×  {config.slurm.num_nodes} nodes"
        f"  =  {total_gpus} total"
    )
    _row("GPUs", gpu_summary, warn=total_gpus >= _LARGE_GPU_THRESHOLD)
    _row("CPUs per task", str(config.slurm.cpus_per_task))
    _row("Wall time", config.slurm.wall_time)
    _row("Signal time", f"{config.slurm.signal_time_seconds}s before timeout")
    _row("Max failures", str(config.slurm.max_failures))

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    _section("Environment")
    using_container = bool(config.container.image)
    if using_container:
        _row("Container image", config.container.image)
        if config.container.bind_mounts:
            _row("Bind mounts", config.container.bind_mounts[0])
            for mount in config.container.bind_mounts[1:]:
                _row("", mount)
        _row("Env file", config.container.env_file or "none")
    else:
        _row("Container", "none  (bare-metal)")
    _row("Offline mode", str(config.offline))

    # ------------------------------------------------------------------
    # Model & data
    # ------------------------------------------------------------------
    _section("Model & Data")
    _row("Model", config.model.name_or_path)
    _row("Attention impl", config.model.attn_implementation)
    _row("Dtype", config.model.dtype)
    _row("Chat template", config.data.chat_template)
    for i, entry in enumerate(config.data.datasets):
        label = "Dataset" if i == 0 else ""
        parts = [entry.path]
        if entry.subset:
            parts.append(f"subset={entry.subset}")
        parts.append(f"split={entry.split}")
        parts.append(f"weight={entry.weight}")
        if entry.transform:
            parts.append(f"transform={entry.transform}")
        _row(label, f"[{entry.name}]  " + "  ".join(parts))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    _section(f"Training  ({config.method.upper()})")
    _row("Backend", config.backend)
    _row("Duration", _duration_summary(config))
    _row("Learning rate", f"{config.training.learning_rate:.2e}")
    lr_sched = config.training.lr_scheduler_type
    min_lr = config.training.lr_scheduler_kwargs.min_lr_rate
    _row("LR scheduler", f"{lr_sched}  (min_lr_rate={min_lr})")
    _row("Warmup ratio", str(config.training.warmup_ratio))
    batch_line, _ = _batch_summary(config, total_gpus)
    _row("Batch sizes", batch_line)
    _row("Grad checkpoint", str(config.training.gradient_checkpointing))
    _row("Liger kernel", str(config.training.use_liger_kernel))
    _row("Seed", str(config.training.seed))

    if config.method == "sft":
        sft = config.sft
        _row("Max seq length", str(sft.max_seq_length))
        _row("Packing", str(sft.packing))
    elif config.method == "dpo":
        dpo = config.dpo
        _row("Beta", str(dpo.beta))
        _row("Loss type", dpo.loss_type)
        _row("Max seq length", str(dpo.max_seq_length))
        if dpo.ref_model_name_or_path:
            _row("Ref model", dpo.ref_model_name_or_path)

    # ------------------------------------------------------------------
    # Training (Distributed)
    # ------------------------------------------------------------------
    _section("Training (Distributed)")
    _row("DeepSpeed", _deepspeed_summary(config))
    _row("Mixed precision", config.accelerate.mixed_precision)
    _row("Dynamo backend", config.accelerate.dynamo_backend)
    _row("RDZV backend", config.accelerate.rdzv_backend)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    _section("Checkpointing")
    ckpt = config.checkpointing
    _row("Save every", f"{ckpt.save_steps} steps  (keep last {ckpt.save_total_limit})")
    if ckpt.inference_checkpoint_steps:
        _row(
            "Inference ckpts",
            f"every {ckpt.inference_checkpoint_steps} steps → {ckpt.inference_checkpoint_path}/",
        )
    else:
        _row("Inference ckpts", "disabled")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    _section("Output")
    _row("Run directory", str(run_dir))
    _row("Logging", ", ".join(config.logging.report_to))
    if "wandb" in config.logging.report_to:
        _row("WandB project", config.logging.wandb_project)
    if config.debug.enabled:
        _row("Debug mode", _red("*** ENABLED — output dir may be overwritten ***"), warn=True)
    else:
        _row("Debug mode", "disabled")

    # ------------------------------------------------------------------
    # Final confirmation
    # ------------------------------------------------------------------
    print()
    print(_bold("╔══════════════════════════════════════════════════════════════╗"))

    if total_gpus >= _LARGE_GPU_THRESHOLD:
        print(
            _bold(f"║  Large job: {total_gpus} GPUs requested.{' ' * (46 - len(str(total_gpus)))}║")
        )
        print(_bold("╚══════════════════════════════════════════════════════════════╝"))
        print()
        answer = input(
            f"  Type '{total_gpus}' to confirm submission, or press Enter to abort: "
        ).strip()
        if answer != str(total_gpus):
            print("\nAborted.")
            sys.exit(1)
    else:
        print(_bold("╚══════════════════════════════════════════════════════════════╝"))
        print()
        answer = input("  Submit this job? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("\nAborted.")
            sys.exit(1)

    print("\nConfirmed. Submitting job...\n")
