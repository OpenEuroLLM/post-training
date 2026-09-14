"""Run the rendered container job scripts under bash with stub SLURM commands.

The text checks in test_slurm_render.py cannot show what bash does with a
script under ``set -euo pipefail``. These tests execute the script instead.
"""

import os
import shutil
import signal
import subprocess
import time
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

# Stub bodies, keyed by command name. The stubs record their calls in $STUB_LOG_DIR.
_STUBS = {
    "srun": (
        'echo "$@" >> "$STUB_LOG_DIR/srun.log"\n'
        'if [ -n "${FAKE_SRUN_SLEEP:-}" ]; then\n'
        '    touch "$STUB_LOG_DIR/srun.started"\n'
        '    sleep "$FAKE_SRUN_SLEEP"\n'
        "fi\n"
        'exit "${FAKE_SRUN_EXIT:-0}"\n'
    ),
    "scontrol": (
        'case "$1" in\n'
        '    show) echo "$SLURM_JOB_NODELIST" ;;\n'
        "    requeue)\n"
        '        echo "$@" >> "$STUB_LOG_DIR/requeue.log"\n'
        '        exit "${FAKE_REQUEUE_EXIT:-0}"\n'
        "        ;;\n"
        "esac\n"
    ),
    "sbatch": 'echo "$@" >> "$STUB_LOG_DIR/sbatch.log"\n',
    "getent": 'echo "127.0.0.1 $2"\n',
}

_RENDERERS = {
    "trl_container": lambda config, run_dir: render_trl_container_slurm_script(
        config, run_dir, "configs/trl/sft.yaml"
    ),
    "llamafactory": render_llamafactory_slurm_script,
}

_MAX_FAILURES = PostTrainingConfig().slurm.max_failures


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


def _run_job_with_sigusr1(
    script: Path, env: dict[str, str], log_dir: Path, tmp_path: Path
) -> tuple[int, str]:
    """Run the job, send SIGUSR1 to the batch shell while srun runs, return (exit code, stderr).

    Output goes to files, not pipes: the sleeping srun stub outlives the batch
    shell and would hold a pipe open. The stub is killed with its process group.
    """
    env = {**env, "FAKE_SRUN_SLEEP": "5"}
    stderr_path = tmp_path / "job.err"
    with (tmp_path / "job.out").open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            [BASH, str(script)], env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 10
            while not (log_dir / "srun.started").exists():
                assert time.monotonic() < deadline, "the srun stub did not start"
                time.sleep(0.05)
            proc.send_signal(signal.SIGUSR1)
            returncode = proc.wait(timeout=30)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return returncode, stderr_path.read_text()


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


# ---------------------------------------------------------------------------
# Failure handling and requeue
# ---------------------------------------------------------------------------


def test_failed_srun_is_counted_and_resubmitted(job_script, job_env, log_dir):
    """A failed srun increments failure_count, resubmits the script and keeps the exit code."""
    job_env["FAKE_SRUN_EXIT"] = "3"

    result = _run_job(job_script, job_env)

    assert result.returncode == 3, result.stderr
    assert (job_script.parent / "failure_count").read_text().strip() == "1"
    assert (log_dir / "sbatch.log").read_text().strip() == str(job_script)


def test_failed_srun_at_max_failures_stops(job_script, job_env, log_dir):
    """The failure that reaches max_failures stops the job instead of resubmitting."""
    (job_script.parent / "failure_count").write_text(f"{_MAX_FAILURES - 1}\n")
    job_env["FAKE_SRUN_EXIT"] = "3"

    result = _run_job(job_script, job_env)

    assert result.returncode == 1, result.stderr
    assert (job_script.parent / "failure_count").read_text().strip() == str(_MAX_FAILURES)
    assert not (log_dir / "sbatch.log").exists()


def test_sigusr1_requeues_without_resubmit(job_script, job_env, log_dir, tmp_path):
    """SIGUSR1 requeues the job, and the failure handler does not also resubmit it."""
    returncode, stderr = _run_job_with_sigusr1(job_script, job_env, log_dir, tmp_path)

    assert returncode == 0, stderr
    assert (log_dir / "requeue.log").read_text().strip() == "requeue 12345"
    assert not (log_dir / "sbatch.log").exists()
    assert (job_script.parent / "failure_count").read_text().strip() == "0"


def test_sigusr1_with_failed_requeue_exits(job_script, job_env, log_dir, tmp_path):
    """If scontrol requeue fails, the job exits at once with an error and does not resubmit."""
    job_env["FAKE_REQUEUE_EXIT"] = "1"

    returncode, stderr = _run_job_with_sigusr1(job_script, job_env, log_dir, tmp_path)

    assert returncode != 0
    assert "scontrol requeue failed" in stderr
    assert not (log_dir / "sbatch.log").exists()
    assert (job_script.parent / "failure_count").read_text().strip() == "0"
