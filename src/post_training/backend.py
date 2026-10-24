"""Training backend dispatch."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

    from post_training.config import PostTrainingConfig
    from post_training.utils.prefetch import PrefetchedPaths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class Backend(ABC):
    @abstractmethod
    def validate(self, config: PostTrainingConfig) -> None:
        """Backend-specific config validation."""

    @abstractmethod
    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        """Return an auto-generated run name."""

    @abstractmethod
    def run_dir_subdirs(self) -> list[str]:
        """Extra subdirectories to create inside the run directory."""

    @abstractmethod
    def render_slurm_script(
        self,
        config: PostTrainingConfig,
        run_dir: Path,
        frozen_config_path: str,
        *,
        tokenize_only: bool = False,
    ) -> Path:
        """Render the SLURM batch script and return its path."""

    @abstractmethod
    def post_freeze(
        self,
        config: PostTrainingConfig,
        run_dir: Path,
        prefetched: PrefetchedPaths | None = None,
    ) -> None:
        """Copy any extra artifacts into the run directory after config freeze.

        *prefetched* holds the local snapshot directories from the login-node
        prefetch, or None when the prefetch did not run.
        """

    def check_explicit_keys(self, explicit: dict) -> None:
        """Check the keys that the YAML file and the CLI overrides set.

        *explicit* holds only those keys, before the schema defaults fill in.
        The default accepts every key.
        """


# ---------------------------------------------------------------------------
# TRL
# ---------------------------------------------------------------------------

_SUPPORTED_METHODS = ("sft", "dpo")

# Config sections that only the TRL backend reads. A LlamaFactory run takes all
# training settings from its llamafactory block, so it rejects these.
TRL_SECTIONS = (
    "model",
    "training",
    "checkpointing",
    "data",
    "sft",
    "dpo",
    "deepspeed",
    "accelerate",
    "logging",
)


def _shorten_model_name(name_or_path: str) -> str:
    short = name_or_path.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9\-]", "", short.lower())


def _shorten_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "_", name.lower())


def _dataset_mix_hash(datasets: list) -> str:
    canonical = sorted(
        [{"name": d.name, "path": d.path, "weight": d.weight} for d in datasets],
        key=lambda x: x["name"],
    )
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()[:8]


class TRLBackend(Backend):
    def validate(self, config: PostTrainingConfig) -> None:
        if config.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{config.method}'. "
                f"Supported methods: {', '.join(_SUPPORTED_METHODS)}"
            )

        if isinstance(config.deepspeed, dict) and "config_path" in config.deepspeed:
            raise ValueError(
                "deepspeed.config_path is no longer supported. Inline the DeepSpeed "
                "JSON/YAML directly under the `deepspeed:` key (see configs/trl/sft.yaml "
                "for an example), or set `deepspeed: null` to disable DeepSpeed."
            )

        # Container validation (only when container.image is set)
        if config.container is not None and config.container.image:
            if not config.container.bind_mounts:
                raise ValueError(
                    "container.bind_mounts must be non-empty when container.image is set."
                )
            if not config.container.env_file:
                raise ValueError("container.env_file must be set when container.image is set.")

        total_weight = 0.0
        for entry in config.data.datasets:
            if entry.weight < 0:
                raise ValueError(
                    f"data.datasets[].weight must be non-negative, got {entry.weight} "
                    f"for dataset '{entry.name}'."
                )
            total_weight += entry.weight
        if config.data.datasets and total_weight <= 0:
            raise ValueError("Sum of data.datasets[].weight must be > 0.")

        t = config.training

        if t.effective_batch_size <= 0:
            raise ValueError("training.effective_batch_size must be positive.")
        if t.per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive.")

        # Training length: exactly one of max_steps / num_train_epochs /
        # num_training_samples / num_training_tokens must be specified.
        length_fields = {
            "training.max_steps": t.max_steps,
            "training.num_train_epochs": t.num_train_epochs,
            "training.num_training_samples": t.num_training_samples,
            "training.num_training_tokens": t.num_training_tokens,
        }
        specified = [name for name, value in length_fields.items() if value is not None]
        if len(specified) == 0:
            raise ValueError(
                "Exactly one of training.max_steps, training.num_train_epochs, "
                "training.num_training_samples, or training.num_training_tokens "
                "must be specified."
            )
        if len(specified) > 1:
            raise ValueError(
                "Training length is over-specified. Choose exactly one of "
                "training.max_steps, training.num_train_epochs, "
                "training.num_training_samples, or training.num_training_tokens. "
                f"You set: {', '.join(specified)}."
            )

        if t.max_steps is not None and t.max_steps <= 0:
            raise ValueError("training.max_steps must be a positive integer.")

        if t.num_train_epochs is not None and t.num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be a positive number.")

        if t.num_training_tokens is not None:
            if config.method != "sft" or not config.sft.packing:
                raise ValueError(
                    "training.num_training_tokens is only valid for method='sft' "
                    "when sft.packing=true."
                )
            if t.num_training_tokens <= 0:
                raise ValueError("training.num_training_tokens must be a positive integer.")
            if config.sft.max_seq_length <= 0:
                raise ValueError("sft.max_seq_length must be positive.")

        if t.num_training_samples is not None:
            if t.num_training_samples <= 0:
                raise ValueError("training.num_training_samples must be a positive integer.")

    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        model_short = _shorten_model_name(config.model.name_or_path)
        datasets = config.data.datasets
        if len(datasets) == 1:
            ds_part = _shorten_dataset_name(datasets[0].name)
        else:
            ds_part = f"mix_{_dataset_mix_hash(datasets)}"
        return f"{config.method}-{model_short}-{ds_part}-{timestamp}"

    def run_dir_subdirs(self) -> list[str]:
        return ["checkpoints", "inference_checkpoints"]

    def render_slurm_script(self, config, run_dir, frozen_config_path, *, tokenize_only=False):
        if config.container is not None and config.container.image:
            from post_training.slurm.launcher import render_trl_container_slurm_script

            return render_trl_container_slurm_script(
                config, run_dir, frozen_config_path, tokenize_only=tokenize_only
            )

        from post_training.slurm.launcher import render_trl_slurm_script

        return render_trl_slurm_script(
            config, run_dir, frozen_config_path, tokenize_only=tokenize_only
        )

    def post_freeze(self, config, run_dir, prefetched=None):
        """Freeze src/post_training and scripts/ into run_dir for containerized runs.

        *prefetched* is not used: submit.py writes the TRL snapshot paths into
        config.yaml itself.

        Done here (at submission time) rather than inside the SLURM job script,
        so a job that sits queued isn't affected by later repo changes. Each
        directory is copied into a temp path and atomically swapped into place,
        so an interrupted previous attempt can never leave a partial freeze that
        gets mistaken for a completed one.
        """
        if config.container is None or not config.container.image:
            return

        repo_dir = Path.cwd()
        self._freeze_dir(repo_dir / "src" / "post_training", run_dir / "src" / "post_training")
        self._freeze_dir(repo_dir / "scripts", run_dir / "scripts")

    @staticmethod
    def _freeze_dir(src: Path, dst: Path) -> None:
        if dst.exists():
            logger.warning("%s already exists — leaving frozen source in place.", dst)
            return

        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp)
        os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# LlamaFactory
# ---------------------------------------------------------------------------

# The LlamaFactory release that the container runs. The key names and stage choices
# below come from its hparams.
LLAMAFACTORY_VERSION = "0.9.5"

# llamafactory keys that hold local input paths. The job runs in the run directory, so a
# relative value that exists is made absolute at submission. Output keys (export_dir,
# logging_dir, offload_folder, swanlab_logdir, profiler_output_dir) stay relative and land
# in the run directory.
_LLAMAFACTORY_INPUT_PATH_KEYS = (
    "model_name_or_path",
    "adapter_folder",
    "cache_dir",
    "new_special_tokens_config",
    "export_quantization_dataset",
    "kt_weight_path",
    "kt_expert_checkpoint_path",
    "hyper_parallel_args",
    "media_dir",
    "tokenized_path",
    "ref_model",
    "ref_model_adapters",
    "reward_model",
    "reward_model_adapters",
    "resume_from_checkpoint",
    "fsdp_config",
)
# Keys that hold a comma-separated list of input paths.
_LLAMAFACTORY_INPUT_PATH_LIST_KEYS = ("adapter_name_or_path",)

_LLAMAFACTORY_DEFAULT_DATASET_DIR = "data"
_LLAMAFACTORY_DEFAULT_STAGE = "sft"
_LLAMAFACTORY_STAGES = ("pt", "sft", "rm", "ppo", "dpo", "kto")
# LlamaFactory does not resume these stages with full or freeze tuning: a requeued job
# starts again from step 0.
_LLAMAFACTORY_NON_RESUMABLE_STAGES = ("rm", "ppo")
_LLAMAFACTORY_REGISTRY = "dataset_info.json"


def llamafactory_dataset_names(
    lf: dict, keys: tuple[str, ...] = ("dataset", "eval_dataset")
) -> list[str]:
    """The dataset names under *keys*, split like LlamaFactory's split_arg()."""
    names: list[str] = []
    for key in keys:
        value = lf.get(key)
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list):
            names.extend(str(name).strip() for name in value if str(name).strip())
    return names


def _is_local_registry(dataset_dir: str) -> bool:
    """False for the dataset_dir values that LlamaFactory resolves on the Hub."""
    return dataset_dir != "ONLINE" and not dataset_dir.startswith("REMOTE:")


def _absolute_if_exists(value: str) -> str:
    """Return a relative path that exists as an absolute path; return anything else unchanged."""
    path = Path(value)
    if path.is_absolute() or not path.exists():
        return value
    return str(path.resolve())


def _freeze_file(dst: Path, render: Callable[[], str]) -> None:
    """Write render() to *dst* once; an existing *dst* is kept, so a resumed run keeps its inputs.

    The text goes to a temp file first and is swapped into place, so an interrupted
    submission never leaves a partial copy that a later one would keep.
    """
    if dst.exists():
        logger.warning("%s already exists — keeping the frozen copy.", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(render())
    os.replace(tmp, dst)


def _registry_pointing_at(source_dir: Path) -> str:
    """Return source_dir's dataset_info.json with relative local entries made absolute."""
    entries = json.loads((source_dir / _LLAMAFACTORY_REGISTRY).read_text())
    for entry in entries.values():
        for key in ("file_name", "script_url"):
            if isinstance(entry.get(key), str) and not Path(entry[key]).is_absolute():
                entry[key] = str(source_dir / entry[key])
    return json.dumps(entries, indent=2, ensure_ascii=False) + "\n"


class LlamaFactoryBackend(Backend):
    def validate(self, config: PostTrainingConfig) -> None:
        if not config.llamafactory:
            raise ValueError("llamafactory must be set when backend='llamafactory'.")
        if config.container is None or not config.container.image:
            raise ValueError("container.image must be set when backend='llamafactory'.")
        if not config.llamafactory.get("model_name_or_path"):
            raise ValueError("llamafactory.model_name_or_path must be set.")

        lf = config.llamafactory
        stage = lf.get("stage", _LLAMAFACTORY_DEFAULT_STAGE)
        if stage not in _LLAMAFACTORY_STAGES:
            raise ValueError(
                f"llamafactory.stage '{stage}' is not a LlamaFactory {LLAMAFACTORY_VERSION} "
                f"stage. Choose from: {', '.join(_LLAMAFACTORY_STAGES)}."
            )
        if config.method != stage:
            raise ValueError(
                f"method '{config.method}' does not match llamafactory.stage '{stage}'. "
                f"Set method: {stage}."
            )

        # post_freeze() owns these two settings, so a requeued or resubmitted job resumes.
        if "output_dir" in lf:
            raise ValueError(
                "llamafactory.output_dir must not be set: the submission writes checkpoints to "
                "<run_dir>/checkpoints so that a requeued job resumes. Remove it."
            )
        if lf.get("overwrite_output_dir"):
            raise ValueError(
                "llamafactory.overwrite_output_dir: true stops a requeued job from resuming "
                "from its checkpoints. Remove it."
            )

        # post_freeze() copies these files, so a missing one must fail before submission.
        deepspeed = lf.get("deepspeed")
        if isinstance(deepspeed, str) and not Path(deepspeed).is_file():
            raise ValueError(
                f"llamafactory.deepspeed file '{deepspeed}' not found. Relative paths start at "
                "the directory where submit.py runs."
            )
        dataset_dir = lf.get("dataset_dir") or _LLAMAFACTORY_DEFAULT_DATASET_DIR
        if (lf.get("dataset") or lf.get("eval_dataset")) and _is_local_registry(dataset_dir):
            registry = Path(dataset_dir) / _LLAMAFACTORY_REGISTRY
            if not registry.is_file():
                raise ValueError(
                    f"{registry} not found: named datasets need a {_LLAMAFACTORY_REGISTRY} in "
                    "llamafactory.dataset_dir. Relative paths start at the directory where "
                    "submit.py runs."
                )

    def check_explicit_keys(self, explicit: dict) -> None:
        """Reject TRL sections, including null values and CLI overrides."""
        trl_keys: list[str] = []
        for section in TRL_SECTIONS:
            if section not in explicit:
                continue
            value = explicit[section]
            if isinstance(value, dict) and value:
                trl_keys.extend(f"{section}.{key}" for key in value)
            else:
                trl_keys.append(section)
        if trl_keys:
            raise ValueError(
                "backend: llamafactory takes all training settings from the llamafactory "
                f"block. Remove these TRL keys: {', '.join(trl_keys)}."
            )

    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        """llamafactory-<stage>-<model>-<datasets joined with +>-<timestamp>.

        Built from the llamafactory block only; the TRL key `method` is not used.
        """
        lf = config.llamafactory
        parts = [
            "llamafactory",
            str(lf.get("stage") or _LLAMAFACTORY_DEFAULT_STAGE),
            _shorten_model_name(str(lf["model_name_or_path"]).rstrip("/")),
        ]
        names = llamafactory_dataset_names(lf, keys=("dataset",))
        if names:
            parts.append("+".join(_shorten_dataset_name(name) for name in names))
        parts.append(timestamp)
        return "-".join(parts)

    def run_dir_subdirs(self) -> list[str]:
        return ["checkpoints"]

    def render_slurm_script(self, config, run_dir, frozen_config_path, *, tokenize_only=False):
        # LlamaFactory has no --tokenize-only equivalent; submit.py rejects it before this point.
        del tokenize_only
        from post_training.slurm.launcher import render_llamafactory_slurm_script

        return render_llamafactory_slurm_script(config, run_dir)

    def post_freeze(self, config, run_dir, prefetched=None):
        """Freeze the job's inputs into run_dir and write llamafactory_config.yaml.

        The DeepSpeed JSON and dataset_info.json are copied into run_dir/llamafactory/,
        so a queued or resumed job does not read later edits. Only
        llamafactory_config.yaml carries the rewritten values; config.yaml keeps the
        user's block, so a new submit from it passes validation again.
        """
        run_dir = run_dir.resolve()
        frozen_dir = run_dir / "llamafactory"
        lf_config = copy.deepcopy(config.llamafactory)

        # Every rank then loads a plain local directory, not a Hub ID through the cache locks.
        if prefetched is not None:
            lf_config["model_name_or_path"] = prefetched.model
            if prefetched.ref_model is not None:
                lf_config["ref_model"] = prefetched.ref_model

        for key in _LLAMAFACTORY_INPUT_PATH_KEYS:
            if isinstance(lf_config.get(key), str):
                lf_config[key] = _absolute_if_exists(lf_config[key])
        for key in _LLAMAFACTORY_INPUT_PATH_LIST_KEYS:
            if isinstance(lf_config.get(key), str):
                parts = lf_config[key].split(",")
                lf_config[key] = ",".join(_absolute_if_exists(part.strip()) for part in parts)

        deepspeed = lf_config.get("deepspeed")
        if isinstance(deepspeed, str):
            frozen_deepspeed = frozen_dir / Path(deepspeed).name
            _freeze_file(frozen_deepspeed, lambda: Path(deepspeed).read_text())
            lf_config["deepspeed"] = str(frozen_deepspeed)

        dataset_dir = lf_config.get("dataset_dir") or _LLAMAFACTORY_DEFAULT_DATASET_DIR
        if _is_local_registry(dataset_dir):
            source_dir = Path(dataset_dir).resolve()
            frozen_registry = frozen_dir / _LLAMAFACTORY_REGISTRY
            if (source_dir / _LLAMAFACTORY_REGISTRY).is_file() or frozen_registry.exists():
                _freeze_file(frozen_registry, lambda: _registry_pointing_at(source_dir))
                lf_config["dataset_dir"] = str(frozen_dir)
                # media_dir defaults to dataset_dir; keep relative media paths on the source.
                lf_config.setdefault("media_dir", str(source_dir))
            elif source_dir.exists():
                lf_config["dataset_dir"] = str(source_dir)

        # LlamaFactory resumes from the last checkpoint-N in output_dir only when
        # overwrite_output_dir is false.
        lf_config["output_dir"] = str(run_dir / "checkpoints")
        lf_config["overwrite_output_dir"] = False

        lf_config_path = run_dir / "llamafactory_config.yaml"
        with open(lf_config_path, "w") as f:
            yaml.dump(lf_config, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Backend] = {
    "trl": TRLBackend(),
    "llamafactory": LlamaFactoryBackend(),
}


def get_backend(name: str) -> Backend:
    """Return the backend implementation for *name*."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Supported: {', '.join(_BACKENDS)}")
    return _BACKENDS[name]
