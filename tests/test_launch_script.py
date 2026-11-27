"""Tests for the launch script that an external orchestrator runs in an srun step."""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

from post_training.config import ContainerConfig, FSDPConfig, PostTrainingConfig
from post_training.slurm.launcher import (
    render_trl_container_slurm_script,
    render_trl_launch_script,
)

# scripts/ is not a Python package, so load the script by path.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "render_launch_script.py"
_spec = importlib.util.spec_from_file_location("render_launch_script", _SCRIPT_PATH)
render_launch_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_launch_script)

_ENV_FILE = """\
export HF_HOME=/scratch/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
"""


@pytest.fixture
def config(tmp_path):
    env_file = tmp_path / "cluster.env"
    env_file.write_text(_ENV_FILE)
    cfg = PostTrainingConfig()
    cfg.container = ContainerConfig(
        image="/shared/containers/trl.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file=str(env_file),
    )
    cfg.slurm.num_nodes = 2
    cfg.slurm.gpus_per_node = 4
    return cfg


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / "outputs" / "my-run"
    path.mkdir(parents=True)
    return path


def _accelerate_block(content: str) -> list[str]:
    """Return the rendered lines from ``accelerate launch`` to the train.py invocation."""
    lines = content.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("accelerate launch"))
    end = next(i for i, line in enumerate(lines) if "scripts/train.py" in line)
    return lines[start : end + 1]


# ---------------------------------------------------------------------------
# Rendered content
# ---------------------------------------------------------------------------


def test_launch_script_has_no_job_control(config, run_dir):
    """The caller owns the allocation: no batch header, requeue or resubmission."""
    script = render_trl_launch_script(config, run_dir, "config.yaml")
    lines = [line.strip() for line in script.read_text().splitlines()]

    assert script == run_dir.resolve() / "slurm" / "launch.sh"
    assert os.access(script, os.X_OK)
    assert not any(line.startswith("#SBATCH ") for line in lines)
    assert not any("scontrol requeue" in line for line in lines if not line.startswith("#"))
    # No line starts a scheduler command (the error text of the step check names srun).
    assert not any(re.match(r"(exec\s+)?(sbatch|scancel|srun)\b", line) for line in lines)


def test_launch_script_reads_only_step_variables(config, run_dir):
    """Allocation-wide variables describe the host of a pool, not this step."""
    content = render_trl_launch_script(config, run_dir, "config.yaml").read_text()
    code = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("#"))

    assert "SLURM_JOB_NODELIST" not in code
    assert "SLURM_NNODES" not in code
    assert "SLURM_STEP_NODELIST" in code
    assert "SLURM_STEP_NUM_NODES" in code


def test_launch_script_freezes_env_file(config, run_dir):
    """The env file is copied, so the step needs no path relative to the repository."""
    content = render_trl_launch_script(config, run_dir, "config.yaml").read_text()

    frozen_env = run_dir.resolve() / "slurm" / "launch.env"
    assert frozen_env.read_text() == _ENV_FILE
    assert f'source "{frozen_env}"' in content


def test_launch_script_requires_container(config, run_dir):
    config.container = None
    with pytest.raises(ValueError, match="container.image"):
        render_trl_launch_script(config, run_dir, "config.yaml")


@pytest.mark.parametrize("fsdp_version", [None, 1, 2])
@pytest.mark.parametrize("tokenize_only", [False, True])
def test_launch_script_matches_job_script_accelerate_command(
    config, run_dir, fsdp_version, tokenize_only
):
    """The launch script and the sbatch job script start the same accelerate command."""
    if fsdp_version is not None:
        config.fsdp = FSDPConfig(version=fsdp_version)
        config.deepspeed = None

    job = render_trl_container_slurm_script(
        config, run_dir, "config.yaml", tokenize_only=tokenize_only
    ).read_text()
    launch = render_trl_launch_script(
        config, run_dir, "config.yaml", tokenize_only=tokenize_only
    ).read_text()

    assert _accelerate_block(launch) == _accelerate_block(job)


# ---------------------------------------------------------------------------
# Behavior in a (stubbed) srun step
# ---------------------------------------------------------------------------


def _write_stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def _run_launch_script(tmp_path, config, run_dir, step_env):
    """Run the rendered launch script with stub cluster commands; return the process."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_stub(stubs / "scontrol", 'echo "$3" | tr "," "\\n"\n')  # show hostnames <list>
    _write_stub(stubs / "getent", 'echo "10.0.0.5 $2"\n')  # hosts <name>
    # Drop the singularity options and the image, then run the container command.
    _write_stub(
        stubs / "singularity",
        f'while [ "$1" != "{config.container.image}" ]; do shift; done\nshift\nexec "$@"\n',
    )
    _write_stub(stubs / "accelerate", 'echo "ACCELERATE $*"\necho "UMASK $(umask)"\n')
    config.container.path = f"{stubs}:/usr/bin:/bin"

    script = render_trl_launch_script(config, run_dir, str(run_dir / "config.yaml"))
    env = {
        "PATH": f"{stubs}:/usr/bin:/bin",
        "HOME": str(tmp_path),  # keep the login shell away from the user's profile
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        **step_env,
    }
    return subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=60
    )


_STEP_ENV = {
    "SLURM_JOB_ID": "1000",
    "SLURM_STEP_ID": "7",
    "SLURM_STEP_NUM_NODES": "2",
    "SLURM_STEP_NODELIST": "node07,node08",
    "SLURM_NODEID": "1",
    # The full allocation is larger than the step.
    "SLURM_JOB_NODELIST": "node01,node02,node07,node08",
    "SLURM_NNODES": "4",
}


def test_launch_script_uses_the_step_layout(tmp_path, config, run_dir):
    result = _run_launch_script(tmp_path, config, run_dir, _STEP_ENV)

    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines() if line.startswith("ACCELERATE"))
    assert "--num_machines 2 " in command
    assert "--num_processes 8 " in command
    assert "--machine_rank 1 " in command
    assert "--main_process_ip 10.0.0.5 " in command
    assert f"--main_process_port {29500 + (1000 + 7) % 2000} " in command
    assert f"scripts/train.py --config {run_dir / 'config.yaml'}" in command
    assert "UMASK 0002" in result.stdout


def test_launch_script_rejects_a_different_node_count(tmp_path, config, run_dir):
    result = _run_launch_script(
        tmp_path, config, run_dir, {**_STEP_ENV, "SLURM_STEP_NUM_NODES": "3"}
    )

    assert result.returncode != 0
    assert "the step has 3 node(s), but the config expects 2" in result.stderr
    assert "ACCELERATE" not in result.stdout


def test_launch_script_fails_outside_an_srun_step(tmp_path, config, run_dir):
    step_env = {k: v for k, v in _STEP_ENV.items() if not k.startswith("SLURM_STEP_")}
    result = _run_launch_script(tmp_path, config, run_dir, step_env)

    assert result.returncode != 0
    assert "must run in an srun step" in result.stderr
    assert "ACCELERATE" not in result.stdout


# ---------------------------------------------------------------------------
# scripts/render_launch_script.py — a run directory that already exists
# ---------------------------------------------------------------------------


@pytest.fixture
def prepared_run_dir(tmp_path, config):
    """A run directory as submit.py leaves it: frozen config and frozen code."""
    config.run_name = "my-run"
    config.paths.output_base = str(tmp_path / "outputs")
    config.prefetch_assets = False
    config.training.max_steps = 1
    path = tmp_path / "outputs" / "my-run"
    (path / "src" / "post_training").mkdir(parents=True)
    (path / "scripts").mkdir()
    (path / "scripts" / "train.py").write_text("# frozen\n")
    (path / "checkpoints" / "checkpoint-250").mkdir(parents=True)
    config.save(path / "config.yaml")
    return path


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def test_render_for_run_dir_only_adds_launch_files(prepared_run_dir):
    before = _snapshot(prepared_run_dir)

    script = render_launch_script.render_for_run_dir(prepared_run_dir)

    after = _snapshot(prepared_run_dir)
    assert script == prepared_run_dir / "slurm" / "launch.sh"
    assert set(after) - set(before) == {"slurm", "slurm/launch.sh", "slurm/launch.env"}
    assert all(after[name] == content for name, content in before.items())
    assert f"--config {prepared_run_dir / 'config.yaml'}" in script.read_text()


def test_render_for_run_dir_tokenize_only_uses_one_gpu(prepared_run_dir):
    content = render_launch_script.render_for_run_dir(
        prepared_run_dir, tokenize_only=True
    ).read_text()

    assert "EXPECTED_NODES=1" in content
    assert "export GPUS_PER_NODE=1" in content
    assert "--tokenize-only" in content


def test_render_for_run_dir_rejects_missing_frozen_code(prepared_run_dir):
    (prepared_run_dir / "scripts" / "train.py").unlink()
    with pytest.raises(FileNotFoundError, match="no frozen code"):
        render_launch_script.render_for_run_dir(prepared_run_dir)


def test_render_for_run_dir_rejects_a_moved_run_dir(prepared_run_dir):
    moved = prepared_run_dir.with_name("moved-run")
    prepared_run_dir.rename(moved)
    with pytest.raises(ValueError, match="frozen config points to"):
        render_launch_script.render_for_run_dir(moved)
