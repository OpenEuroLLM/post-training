"""Configuration schema and loader for post-training.

All configuration is defined via nested dataclasses with OmegaConf for
YAML loading, merging, and CLI overrides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import MISSING, DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model identity and loading options."""

    name_or_path: str = "allenai/Olmo-3-1025-7B"
    attn_implementation: str = "flash_attention_3"
    dtype: str = "bfloat16"


@dataclass
class LRSchedulerKwargs:
    """Extra keyword arguments forwarded to the LR scheduler."""

    min_lr_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Core training hyper-parameters shared across all methods."""

    max_steps: Optional[int] = None
    num_training_samples: Optional[int] = None
    num_training_tokens: Optional[int] = None

    learning_rate: float = 2.0e-5
    effective_batch_size: int = 512
    per_device_train_batch_size: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: LRSchedulerKwargs = field(default_factory=LRSchedulerKwargs)

    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42
    use_liger_kernel: bool = True


# ---------------------------------------------------------------------------
# Method-specific configs
# ---------------------------------------------------------------------------


@dataclass
class SFTMethodConfig:
    """Parameters unique to supervised fine-tuning."""

    max_seq_length: int = 4096
    packing: bool = True


@dataclass
class DPOMethodConfig:
    """Parameters unique to direct preference optimisation."""

    beta: float = 5.0
    loss_type: str = "sigmoid"
    ref_model_name_or_path: Optional[str] = None
    max_seq_length: int = 2048


@dataclass
class CheckpointingConfig:
    """Checkpoint saving strategy."""

    save_steps: int = 500
    save_total_limit: int = 2
    # When set to ``None`` (or a non-positive value via CLI overrides), inference
    # checkpoints are disabled and no `inference_checkpoints/` directory is created.
    inference_checkpoint_steps: Optional[int] = 200
    inference_checkpoint_path: str = "inference_checkpoints"


@dataclass
class DatasetEntry:
    """A single dataset in the data mix."""

    name: str = MISSING
    path: str = MISSING
    data_dir: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    weight: float = 1.0
    transform: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset mixing and chat template selection."""

    chat_template: str = "default"
    num_proc: Optional[int] = None  # None = auto (capped at 32)
    datasets: list[DatasetEntry] = field(default_factory=list)


@dataclass
class DeepSpeedConfig:
    """Pointer to the DeepSpeed YAML config file.  Set to ``null`` to disable."""

    config_path: Optional[str] = "configs/deepspeed/zero2.yaml"


@dataclass
class AccelerateConfig:
    """Flags forwarded to ``accelerate launch`` for explicit multi-node control."""

    mixed_precision: str = "bf16"
    use_deepspeed: bool = True
    deepspeed_multinode_launcher: str = "standard"
    same_network: bool = True
    rdzv_backend: str = "static"
    dynamo_backend: str = "inductor"


@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""

    report_to: list[str] = field(default_factory=lambda: ["wandb", "tensorboard"])
    wandb_project: str = "post-training"
    logging_steps: int = 1
    include_num_input_tokens_seen: str = "non_padding"


@dataclass
class SlurmConfig:
    """SLURM job scheduler parameters."""

    partition: str = "gpu"
    num_nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_gpu: int = 32
    wall_time: str = "02:00:00"
    job_name: str = "post-training"
    signal_time_seconds: int = 300
    max_failures: int = 3


@dataclass
class DebugConfig:
    """Debug / quick-iteration mode."""

    enabled: bool = False
    override_existing: bool = False


@dataclass
class PathsConfig:
    """Output directory layout."""

    output_base: str = "outputs"
    debug_base: str = "outputs/debug"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


_SUPPORTED_METHODS = ("sft", "dpo")


@dataclass
class PostTrainingConfig:
    """Root configuration that composes every sub-config."""

    method: str = "sft"
    run_name: Optional[str] = None
    offline: bool = False

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Method-specific (only the active method's block is used at runtime).
    sft: SFTMethodConfig = field(default_factory=SFTMethodConfig)
    dpo: DPOMethodConfig = field(default_factory=DPOMethodConfig)

    # Infrastructure.
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    accelerate: AccelerateConfig = field(default_factory=AccelerateConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        yaml_path: str | Path,
        cli_overrides: list[str] | None = None,
    ) -> "PostTrainingConfig":
        """Load config from *yaml_path*, merge CLI dot-list overrides, validate.

        Parameters
        ----------
        yaml_path:
            Path to the YAML configuration file.
        cli_overrides:
            Optional list of ``"key=value"`` strings (OmegaConf dot-list
            notation) that take precedence over the YAML values.

        Returns
        -------
        PostTrainingConfig
            Fully resolved and validated configuration object.
        """
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(yaml_path)
        merged: DictConfig = OmegaConf.merge(schema, file_cfg)

        if cli_overrides:
            cli_cfg = OmegaConf.from_dotlist(cli_overrides)
            merged = OmegaConf.merge(merged, cli_cfg)

        config: PostTrainingConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
        config._validate()
        return config

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save(self, yaml_path: str | Path) -> None:
        """Save the configuration to a YAML file."""
        OmegaConf.save(self, yaml_path)

    # ------------------------------------------------------------------
    # Validation & derived values
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Run cross-field validation and compute derived values."""
        # Validate training method.
        if self.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Supported methods: {', '.join(_SUPPORTED_METHODS)}"
            )

        t = self.training

        # Validate effective batch size.
        if t.effective_batch_size <= 0:
            raise ValueError("training.effective_batch_size must be positive.")
        if t.per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive.")

        # Training length: exactly one of max_steps / num_training_samples /
        # num_training_tokens must be specified.
        length_fields = {
            "training.max_steps": t.max_steps,
            "training.num_training_samples": t.num_training_samples,
            "training.num_training_tokens": t.num_training_tokens,
        }
        specified = [name for name, value in length_fields.items() if value is not None]
        if len(specified) == 0:
            raise ValueError(
                "Exactly one of training.max_steps, training.num_training_samples, "
                "or training.num_training_tokens must be specified."
            )
        if len(specified) > 1:
            raise ValueError(
                "Training length is over-specified. Choose exactly one of "
                f"training.max_steps, training.num_training_samples, or "
                f"training.num_training_tokens. You set: {', '.join(specified)}."
            )

        # Token-based length is only supported for SFT with packing enabled.
        if t.num_training_tokens is not None:
            if self.method != "sft" or not self.sft.packing:
                raise ValueError(
                    "training.num_training_tokens is only valid for method='sft' "
                    "when sft.packing=true."
                )
            if t.num_training_tokens <= 0:
                raise ValueError(
                    "training.num_training_tokens must be a positive integer."
                )
            if self.sft.max_seq_length <= 0:
                raise ValueError("sft.max_seq_length must be positive.")

            tokens_per_step = t.effective_batch_size * self.sft.max_seq_length
            t.max_steps = math.ceil(t.num_training_tokens / tokens_per_step)

        # Derive max_steps from num_training_samples when applicable.
        if t.num_training_samples is not None:
            if t.num_training_samples <= 0:
                raise ValueError(
                    "training.num_training_samples must be a positive integer."
                )
            t.max_steps = math.ceil(t.num_training_samples / t.effective_batch_size)

    def resolve_gradient_accumulation_steps(self, world_size: int) -> int:
        """Compute gradient accumulation steps from the effective batch size.

        Parameters
        ----------
        world_size:
            Total number of GPUs across all nodes.

        Returns
        -------
        int
            ``effective_batch_size / (per_device_train_batch_size * world_size)``

        Raises
        ------
        ValueError
            If the result is not an integer (i.e. the batch sizes are
            incompatible).
        """
        t = self.training
        gas = t.effective_batch_size / (t.per_device_train_batch_size * world_size)
        if gas != int(gas):
            raise ValueError(
                f"effective_batch_size ({t.effective_batch_size}) is not evenly "
                f"divisible by per_device_train_batch_size ({t.per_device_train_batch_size}) "
                f"* world_size ({world_size}). Got gradient_accumulation_steps={gas}."
            )
        return int(gas)

    def load_deepspeed_config(self) -> dict[str, Any]:
        """Load the DeepSpeed YAML config and return it as a plain dict."""
        ds_path = Path(self.deepspeed.config_path)
        if not ds_path.is_absolute():
            # Resolve relative to the project root (cwd).
            ds_path = Path.cwd() / ds_path
        with open(ds_path) as f:
            return yaml.safe_load(f)
