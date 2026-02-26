"""Configuration schema and loader for post-training.

All configuration is defined via nested dataclasses with OmegaConf for
YAML loading, merging, and CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
class GradientCheckpointingKwargs:
    """Extra keyword arguments forwarded to ``gradient_checkpointing_enable()``."""

    use_reentrant: bool = False
    determinism_check: str = "default"
    debug: bool = False
    early_stop: bool = True


@dataclass
class TrainingConfig:
    """Core training hyper-parameters shared across all methods."""

    max_steps: int | None = None
    num_train_epochs: float | None = None
    num_training_samples: int | None = None
    num_training_tokens: int | None = None

    learning_rate: float = 2.0e-5
    effective_batch_size: int = 512
    per_device_train_batch_size: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: LRSchedulerKwargs = field(default_factory=LRSchedulerKwargs)

    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: GradientCheckpointingKwargs = field(
        default_factory=GradientCheckpointingKwargs
    )
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
    dataset_num_proc: int | None = None


@dataclass
class DPOMethodConfig:
    """Parameters unique to direct preference optimisation."""

    beta: float = 5.0
    loss_type: str = "sigmoid"
    ref_model_name_or_path: str | None = None
    max_seq_length: int = 2048
    dataset_num_proc: int | None = None


@dataclass
class CheckpointingConfig:
    """Checkpoint saving strategy."""

    save_steps: int = 500
    save_total_limit: int = 2
    # When set to ``None`` (or a non-positive value via CLI overrides), inference
    # checkpoints are disabled and no `inference_checkpoints/` directory is created.
    inference_checkpoint_steps: int | None = 200
    inference_checkpoint_path: str = "inference_checkpoints"


@dataclass
class DatasetEntry:
    """A single dataset in the data mix."""

    name: str = MISSING
    path: str = MISSING
    data_dir: str | None = None
    subset: str | None = None
    split: str = "train"
    weight: float = 1.0
    transform: str | None = None


@dataclass
class DataConfig:
    """Dataset mixing and chat template selection."""

    chat_template: str = "default"
    num_proc: int | None = None  # None = auto (capped at 32)
    datasets: list[DatasetEntry] = field(default_factory=list)


@dataclass
class DeepSpeedConfig:
    """Pointer to the DeepSpeed YAML config file.  Set to ``null`` to disable."""

    config_path: str | None = "configs/deepspeed/zero2.yaml"


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
class ContainerConfig:
    """Singularity / Apptainer container settings."""

    image: str | None = None
    bind_mounts: list[str] = field(default_factory=list)
    env_file: str | None = None


@dataclass
class SlurmConfig:
    """SLURM job scheduler parameters."""

    partition: str = "gpu"
    num_nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 32
    wall_time: str = "02:00:00"
    job_name: str = "post-training"
    signal_time_seconds: int = 300
    max_failures: int = 3
    modules: list[str] = field(default_factory=list)
    module_purge: bool = False


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


@dataclass
class PostTrainingConfig:
    """Root configuration that composes every sub-config."""

    method: str = "sft"
    run_name: str | None = None
    offline: bool = False
    backend: str = "trl"
    llamafactory: dict | None = None
    container: ContainerConfig = field(default_factory=ContainerConfig)

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
    ) -> PostTrainingConfig:
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
        from post_training.backend import get_backend

        get_backend(self.backend).validate(self)

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
