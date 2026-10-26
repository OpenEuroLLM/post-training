"""Generate a SLURM batch script from the config and submit it.

The Jinja template ``job.sh.jinja`` (shipped alongside this module) is
rendered with values from the ``slurm`` and ``accelerate`` config sections
and written into the run directory before ``sbatch`` is called.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent
_TEMPLATE_NAME = "job.sh.jinja"
_LLAMAFACTORY_TEMPLATE_NAME = "job_llamafactory.sh.jinja"
_TRL_CONTAINER_TEMPLATE_NAME = "job_trl_container.sh.jinja"


def _freeze_accelerate_config(config: PostTrainingConfig, run_dir: Path) -> str | None:
    """Freeze the optional Accelerate launch profile next to ``job.sh``.

    A queued job must not read a mutable file from the working checkout. The
    copied path is also inside ``run_dir``, which the container launcher already
    exposes to the training process.
    """
    source_value = config.accelerate.config_file
    if source_value is None:
        return None

    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"accelerate.config_file '{source_value}' not found.")

    destination = run_dir / "slurm" / "accelerate.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return str(destination.resolve())


def render_trl_slurm_script(
    config: PostTrainingConfig,
    run_dir: Path,
    config_path: str,
    *,
    tokenize_only: bool = False,
) -> Path:
    """Render the TRL SLURM batch script and write it to *run_dir/slurm/job.sh*.

    Parameters
    ----------
    config:
        Fully resolved configuration.
    run_dir:
        Run output directory (must already exist).
    config_path:
        Path to the YAML config file that ``scripts/train.py`` will
        receive at launch time.

    Returns
    -------
    Path
        The path to the generated ``job.sh`` file.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATE_NAME)
    accelerate_config_file = _freeze_accelerate_config(config, run_dir)

    rendered = template.render(
        # SLURM parameters
        job_name=config.slurm.job_name,
        partition=config.slurm.partition,
        account=config.slurm.account,
        qos=config.slurm.qos,
        num_nodes=config.slurm.num_nodes,
        gpus_per_node=config.slurm.gpus_per_node,
        cpus_per_task=config.slurm.cpus_per_task,
        cpus_per_gpu=config.slurm.cpus_per_gpu,
        mem=config.slurm.mem,
        wall_time=config.slurm.wall_time,
        signal_time_seconds=config.slurm.signal_time_seconds,
        max_failures=config.slurm.max_failures,
        modules=config.slurm.modules,
        module_purge=config.slurm.module_purge,
        run_dir=str(run_dir),
        config_path=config_path,
        tokenize_only=tokenize_only,
        # Accelerate flags
        accelerate_config_file=accelerate_config_file,
        mixed_precision=config.accelerate.mixed_precision,
        dynamo_backend=config.accelerate.dynamo_backend,
        use_deepspeed=config.accelerate.use_deepspeed and bool(config.deepspeed),
        deepspeed_multinode_launcher=config.accelerate.deepspeed_multinode_launcher,
        same_network=config.accelerate.same_network,
        rdzv_backend=config.accelerate.rdzv_backend,
    )

    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script_path = slurm_dir / "job.sh"
    script_path.write_text(rendered)
    script_path.chmod(0o755)

    logger.info("SLURM script written to %s", script_path)
    return script_path


def render_trl_container_slurm_script(
    config: PostTrainingConfig,
    run_dir: Path,
    config_path: str,
    *,
    tokenize_only: bool = False,
) -> Path:
    """Render the containerized TRL SLURM batch script into *run_dir/slurm/job.sh*.

    Uses ``singularity exec`` to run ``accelerate launch`` inside a container,
    following the same patterns as the LlamaFactory containerized template.
    """
    if config.container is None or not config.container.image:
        raise ValueError("container.image must be set to render a containerized TRL script.")

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(_TRL_CONTAINER_TEMPLATE_NAME)
    accelerate_config_file = _freeze_accelerate_config(config, run_dir)

    rendered = template.render(
        # SLURM parameters
        job_name=config.slurm.job_name,
        partition=config.slurm.partition,
        account=config.slurm.account,
        qos=config.slurm.qos,
        num_nodes=config.slurm.num_nodes,
        gpus_per_node=config.slurm.gpus_per_node,
        cpus_per_task=config.slurm.cpus_per_task,
        cpus_per_gpu=config.slurm.cpus_per_gpu,
        mem=config.slurm.mem,
        wall_time=config.slurm.wall_time,
        signal_time_seconds=config.slurm.signal_time_seconds,
        max_failures=config.slurm.max_failures,
        modules=config.slurm.modules,
        module_purge=config.slurm.module_purge,
        run_dir=str(run_dir.resolve()),
        config_path=config_path,
        tokenize_only=tokenize_only,
        # Accelerate flags
        accelerate_config_file=accelerate_config_file,
        mixed_precision=config.accelerate.mixed_precision,
        dynamo_backend=config.accelerate.dynamo_backend,
        use_deepspeed=config.accelerate.use_deepspeed and bool(config.deepspeed),
        deepspeed_multinode_launcher=config.accelerate.deepspeed_multinode_launcher,
        same_network=config.accelerate.same_network,
        rdzv_backend=config.accelerate.rdzv_backend,
        # Container
        container_image=config.container.image,
        bind_mounts=config.container.bind_mounts,
        env_file=config.container.env_file,
        container_path=config.container.path,
    )

    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script_path = slurm_dir / "job.sh"
    script_path.write_text(rendered)
    script_path.chmod(0o755)

    logger.info("Containerized TRL SLURM script written to %s", script_path)
    return script_path


def render_llamafactory_slurm_script(
    config: PostTrainingConfig,
    run_dir: Path,
) -> Path:
    """Render the LlamaFactory SLURM batch script into *run_dir/slurm/job.sh*."""
    if config.container is None or not config.container.image:
        raise ValueError("container.image must be set to render a LlamaFactory script.")

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(_LLAMAFACTORY_TEMPLATE_NAME)

    rendered = template.render(
        # SLURM parameters
        job_name=config.slurm.job_name,
        partition=config.slurm.partition,
        account=config.slurm.account,
        qos=config.slurm.qos,
        num_nodes=config.slurm.num_nodes,
        gpus_per_node=config.slurm.gpus_per_node,
        cpus_per_task=config.slurm.cpus_per_task,
        cpus_per_gpu=config.slurm.cpus_per_gpu,
        mem=config.slurm.mem,
        wall_time=config.slurm.wall_time,
        signal_time_seconds=config.slurm.signal_time_seconds,
        max_failures=config.slurm.max_failures,
        run_dir=str(run_dir.resolve()),
        # Container
        container_image=config.container.image,
        bind_mounts=config.container.bind_mounts,
        env_file=config.container.env_file,
        container_path=config.container.path,
        # LlamaFactory
        llamafactory_config=str(run_dir / "llamafactory_config.yaml"),
        repo_dir=str(Path.cwd()),
    )

    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script_path = slurm_dir / "job.sh"
    script_path.write_text(rendered)
    script_path.chmod(0o755)

    logger.info("LlamaFactory SLURM script written to %s", script_path)
    return script_path


def submit_job(script_path: Path) -> str:
    """Submit the SLURM script via ``sbatch`` and return the job ID.

    Raises
    ------
    RuntimeError
        If ``sbatch`` exits with a non-zero return code.
    """
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed (rc={result.returncode}): {result.stderr}")

    # sbatch output: "Submitted batch job <ID>"
    job_id = result.stdout.strip().split()[-1]
    logger.info("Submitted SLURM job %s", job_id)
    return job_id


def generate_and_submit(
    config: PostTrainingConfig,
    run_dir: Path,
    config_path: str,
    *,
    tokenize_only: bool = False,
) -> str:
    """Render the SLURM script and submit it in one call.

    Returns
    -------
    str
        The SLURM job ID.
    """
    from post_training.backend import get_backend

    script_path = get_backend(config.backend).render_slurm_script(
        config, run_dir, config_path, tokenize_only=tokenize_only
    )
    return submit_job(script_path)
