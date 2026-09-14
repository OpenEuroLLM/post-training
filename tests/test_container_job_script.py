"""Run the rendered container job scripts under bash with stub SLURM commands.

The text checks in test_slurm_render.py cannot show what bash does with a
script under ``set -euo pipefail``. These tests execute the script instead.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from post_training.config import ContainerConfig, PostTrainingConfig
from post_training.slurm.launcher import (
    REQUIRED_HF_CACHE_VARS,
    render_llamafactory_slurm_script,
    render_trl_container_slurm_script,
)

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not available")

# Stub bodies, keyed by command name. Each stub logs to $STUB_LOG_DIR/<name>.log.
_STUBS = {
    "srun": 'echo "$@" >> "$STUB_LOG_DIR/srun.log"\nexit "${FAKE_SRUN_EXIT:-0}"\n',
    "scontrol": 'case "$1" in\n    show) echo "$SLURM_JOB_NODELIST" ;;\nesac\n',
    "sbatch": 'echo "$@" >> "$STUB_LOG_DIR/sbatch.log"\n',
    "getent": 'echo "127.0.0.1 $2"\n',
}

_RENDERERS = {
    "trl_container": lambda config, run_dir: render_trl_container_slurm_script(
        config, run_dir, "configs/trl/sft.yaml"
    ),
    "llamafactory": render_llamafactory_slurm_script,
}


@pytest.fixture(params=sorted(_RENDERERS))
def job_script(request, tmp_path):
    """Render one container job script that sources an empty cluster env file."""
    env_file = tmp_path / "cluster.env"
    env_file.write_text("")
    config = PostTrainingConfig()
    config.container = ContainerConfig(
        image="/shared/containers/image.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file=str(env_file),
    )
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)
    return _RENDERERS[request.param](config, run_dir)


@pytest.fixture
def log_dir(tmp_path):
    path = tmp_path / "stub_logs"
    path.mkdir()
    return path


@pytest.fixture
def job_env(tmp_path, log_dir):
    """A fresh job environment: stubs first on PATH, SLURM job and HF cache variables."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        stub = bin_dir / name
        stub.write_text("#!/bin/bash\n" + body)
        stub.chmod(0o755)
    return {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "STUB_LOG_DIR": str(log_dir),
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_NODELIST": "node001",
        "SLURM_NNODES": "1",
        "HF_HOME": "/hf",
        "HF_HUB_CACHE": "/hf/hub",
        "HUGGINGFACE_HUB_CACHE": "/hf/hub",
    }


def _run_job(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(script)], env=env, capture_output=True, text=True, timeout=30)


# ---------------------------------------------------------------------------
# HF cache variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", REQUIRED_HF_CACHE_VARS)
def test_unset_hf_variable_stops_before_srun(job_script, job_env, log_dir, missing):
    """A missing HF cache variable stops the job with a clear message before srun starts."""
    del job_env[missing]

    result = _run_job(job_script, job_env)

    assert result.returncode == 1
    assert f"ERROR: {missing} not set" in result.stderr
    assert "unbound variable" not in result.stderr
    assert not (log_dir / "srun.log").exists()
    assert not (log_dir / "sbatch.log").exists()


def test_hf_datasets_cache_is_optional(job_script, job_env, log_dir):
    """An unset HF_DATASETS_CACHE falls back to $HF_HOME/datasets instead of stopping the job."""
    assert "HF_DATASETS_CACHE" not in job_env

    result = _run_job(job_script, job_env)

    assert result.returncode == 0, result.stderr
    assert "HF_DATASETS_CACHE=/hf/datasets" in (log_dir / "srun.log").read_text()
