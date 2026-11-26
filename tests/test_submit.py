"""Tests for the submission-phase checks in scripts/submit.py."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

# scripts/ is not a Python package, so load the script by path.
_SUBMIT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "submit.py"
_spec = importlib.util.spec_from_file_location("submit", _SUBMIT_PATH)
submit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(submit)


class _ReachedRunDirSetup(Exception):
    """Raised in place of run-directory setup to show the checks passed."""


def _run_submit(tmp_path, monkeypatch, *, effective_batch_size, extra_args=()):
    """Run submit.main() for a 1 node x 3 GPU config, stopping before any side effect."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "prefetch_assets": False,
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": effective_batch_size,
                    "per_device_train_batch_size": 1,
                },
                "slurm": {"num_nodes": 1, "gpus_per_node": 3},
            }
        )
    )

    def _stop(*args, **kwargs):
        raise _ReachedRunDirSetup

    monkeypatch.setattr(submit, "setup_run_directory", _stop)
    monkeypatch.setattr(sys, "argv", ["submit.py", "--config", str(config_path), *extra_args])
    submit.main()


def test_inexact_gradient_accumulation_rejected_at_submission(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="not evenly divisible"):
        _run_submit(tmp_path, monkeypatch, effective_batch_size=8)


def test_exact_gradient_accumulation_passes(tmp_path, monkeypatch):
    with pytest.raises(_ReachedRunDirSetup):
        _run_submit(tmp_path, monkeypatch, effective_batch_size=6)


def test_tokenize_only_checks_against_a_single_gpu(tmp_path, monkeypatch):
    """--tokenize-only submits a 1-GPU job, so 8 / (1 * 1) is exact."""
    with pytest.raises(_ReachedRunDirSetup):
        _run_submit(tmp_path, monkeypatch, effective_batch_size=8, extra_args=["--tokenize-only"])


# ---------------------------------------------------------------------------
# --prepare-only — the submission phase without sbatch
# ---------------------------------------------------------------------------


def _write_prepare_config(tmp_path, *, container=True):
    env_file = tmp_path / "cluster.env"
    env_file.write_text("export HF_HOME=/scratch/hf\n")
    config = {
        "method": "sft",
        "backend": "trl",
        "run_name": "my-run",
        "prefetch_assets": False,
        "training": {"max_steps": 1, "effective_batch_size": 4, "per_device_train_batch_size": 1},
        "slurm": {"num_nodes": 1, "gpus_per_node": 4},
        "paths": {"output_base": str(tmp_path / "outputs")},
    }
    if container:
        config["container"] = {
            "image": "/shared/containers/trl.sif",
            "bind_mounts": ["/scratch"],
            "env_file": str(env_file),
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_prepare_only_writes_the_launch_script_and_does_not_submit(tmp_path, monkeypatch):
    config_path = _write_prepare_config(tmp_path)

    def _no_submission(*args, **kwargs):
        raise AssertionError("--prepare-only must not submit a job")

    monkeypatch.setattr(submit, "generate_and_submit", _no_submission)
    monkeypatch.setattr(submit, "_apply_hf_env_from_file", lambda env_file: None)
    monkeypatch.chdir(_SUBMIT_PATH.parent.parent)  # post_freeze copies the code from the cwd
    monkeypatch.setattr(
        sys, "argv", ["submit.py", "--config", str(config_path), "--confirm", "--prepare-only"]
    )
    submit.main()

    run_dir = tmp_path / "outputs" / "my-run"
    assert (run_dir / "slurm" / "launch.sh").is_file()
    assert not (run_dir / "slurm" / "job.sh").exists()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "src" / "post_training" / "config.py").is_file()
    assert (run_dir / "scripts" / "train.py").is_file()


def test_prepare_only_requires_a_trl_container_config(tmp_path, monkeypatch):
    config_path = _write_prepare_config(tmp_path, container=False)
    monkeypatch.setattr(
        sys, "argv", ["submit.py", "--config", str(config_path), "--confirm", "--prepare-only"]
    )
    with pytest.raises(ValueError, match="--prepare-only is only supported"):
        submit.main()
    assert not (tmp_path / "outputs").exists()
