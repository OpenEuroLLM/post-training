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
