"""Generate a SLURM batch script from the config and submit it.

The Jinja template ``job.sh.jinja`` (shipped alongside this module) is
rendered with values from the ``slurm`` and ``accelerate`` config sections
and written into the run directory before ``sbatch`` is called.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent
_TEMPLATE_NAME = "job.sh.jinja"


def render_slurm_script(
    config: PostTrainingConfig,
    run_dir: Path,
    config_path: str,
) -> Path:
    """Render the SLURM batch script and write it to *run_dir/slurm/job.sh*.

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

    rendered = template.render(
        # SLURM parameters
        job_name=config.slurm.job_name,
        partition=config.slurm.partition,
        num_nodes=config.slurm.num_nodes,
        gpus_per_node=config.slurm.gpus_per_node,
        cpus_per_task=config.slurm.cpus_per_task,
        wall_time=config.slurm.wall_time,
        signal_time_seconds=config.slurm.signal_time_seconds,
        max_failures=config.slurm.max_failures,
        modules=config.slurm.modules,
        module_purge=config.slurm.module_purge,
        run_dir=str(run_dir),
        config_path=config_path,
        # Accelerate flags
        mixed_precision=config.accelerate.mixed_precision,
        dynamo_backend=config.accelerate.dynamo_backend,
        use_deepspeed=config.accelerate.use_deepspeed,
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
) -> str:
    """Render the SLURM script and submit it in one call.

    Returns
    -------
    str
        The SLURM job ID.
    """
    script_path = render_slurm_script(config, run_dir, config_path)
    return submit_job(script_path)
