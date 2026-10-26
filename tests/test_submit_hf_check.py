"""Tests for the submission-time HF cache variable check in scripts/submit.py.

A container job without one of these variables stops as soon as it starts,
possibly after days in the queue, so submit.py must refuse it up front.
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

import post_training.utils.prefetch as prefetch_module
from post_training.config import ContainerConfig, PostTrainingConfig
from post_training.slurm.launcher import REQUIRED_HF_CACHE_VARS

# scripts/ is not a Python package, so load submit.py from its path.
_SUBMIT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "submit.py"
_spec = importlib.util.spec_from_file_location("submit", _SUBMIT_PATH)
_submit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_submit)


@pytest.fixture(autouse=True)
def clean_hf_env():
    """Hide the developer's HF cache variables and undo what _apply_hf_env_from_file sets."""
    with mock.patch.dict(os.environ):
        for var in {*_submit._HF_CACHE_VARS, *REQUIRED_HF_CACHE_VARS}:
            os.environ.pop(var, None)
        yield


def _write_env_file(tmp_path: Path, skip: tuple[str, ...] = (), extra: str = "") -> Path:
    """Write an env file that exports every required variable except those in *skip*."""
    lines = [
        f"export {var}=/cluster/hf/{var.lower()}\n"
        for var in REQUIRED_HF_CACHE_VARS
        if var not in skip
    ]
    env_file = tmp_path / "cluster.env"
    env_file.write_text("".join(lines) + extra)
    return env_file


def _container_config(env_file: Path) -> PostTrainingConfig:
    config = PostTrainingConfig()
    config.container = ContainerConfig(
        image="/shared/containers/image.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file=str(env_file),
    )
    return config


def test_all_variables_exported_passes(tmp_path):
    env_file = _write_env_file(tmp_path)
    _submit._apply_hf_env_from_file(str(env_file))

    _submit._check_required_hf_vars(_container_config(env_file))


@pytest.mark.parametrize("missing", REQUIRED_HF_CACHE_VARS)
def test_missing_variable_is_rejected(tmp_path, missing):
    env_file = _write_env_file(tmp_path, skip=(missing,))
    _submit._apply_hf_env_from_file(str(env_file))

    with pytest.raises(ValueError, match=missing) as excinfo:
        _submit._check_required_hf_vars(_container_config(env_file))
    assert str(env_file) in str(excinfo.value)


def test_variable_from_login_environment_passes(tmp_path, monkeypatch):
    """sbatch passes the submitting shell's environment to the job, so a login-shell value counts."""
    env_file = _write_env_file(tmp_path, skip=("HF_HUB_CACHE",))
    monkeypatch.setenv("HF_HUB_CACHE", "/login/hf/hub")
    _submit._apply_hf_env_from_file(str(env_file))

    _submit._check_required_hf_vars(_container_config(env_file))


def test_empty_variable_is_rejected(tmp_path):
    """An empty value fails the job's [ -z ] test, so it is rejected like an unset one."""
    env_file = _write_env_file(tmp_path, skip=("HF_HUB_CACHE",), extra="export HF_HUB_CACHE=\n")
    _submit._apply_hf_env_from_file(str(env_file))

    with pytest.raises(ValueError, match="HF_HUB_CACHE"):
        _submit._check_required_hf_vars(_container_config(env_file))


def test_bare_metal_job_is_not_checked():
    """The bare-metal job script does not forward the HF cache variables."""
    config = PostTrainingConfig()
    config.container = None

    _submit._check_required_hf_vars(config)


def test_submit_stops_before_prefetch_and_run_directory(tmp_path, monkeypatch):
    """main() rejects the job before it downloads assets or creates the run directory."""
    env_file = _write_env_file(tmp_path, skip=("HF_HUB_CACHE",))
    output_base = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "training": {"max_steps": 1},
                "container": {
                    "image": "/shared/containers/image.sif",
                    "bind_mounts": ["/scratch:/scratch"],
                    "env_file": str(env_file),
                },
                "paths": {"output_base": str(output_base)},
            }
        )
    )

    def fail_prefetch(config):
        raise AssertionError("prefetch_assets must not run")

    monkeypatch.setattr(prefetch_module, "prefetch_assets", fail_prefetch)
    monkeypatch.setattr(sys, "argv", ["submit.py", "--config", str(config_path), "--confirm"])

    with pytest.raises(ValueError, match="HF_HUB_CACHE"):
        _submit.main()
    assert not output_base.exists()
