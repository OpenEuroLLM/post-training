"""End-to-end tests of scripts/submit.py for backend: llamafactory, without sbatch or the Hub."""

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

import post_training.utils.prefetch as prefetch_module
from post_training.slurm.launcher import REQUIRED_HF_CACHE_VARS

# scripts/ is not a Python package, so load submit.py from its path.
_SUBMIT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "submit.py"
_spec = importlib.util.spec_from_file_location("submit", _SUBMIT_PATH)
_submit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_submit)

_SNAPSHOT = "/cache/org/lf-model@main"


@pytest.fixture(autouse=True)
def clean_hf_env():
    """Hide the developer's HF cache variables and undo what _apply_hf_env_from_file sets."""
    with mock.patch.dict(os.environ):
        for var in {*_submit._HF_CACHE_VARS, *REQUIRED_HF_CACHE_VARS}:
            os.environ.pop(var, None)
        yield


@pytest.fixture
def submitted(tmp_path, monkeypatch):
    """Run submit.py main() on a LlamaFactory config; return the run directory and job scripts."""
    env_file = tmp_path / "cluster.env"
    env_file.write_text("".join(f"export {var}=/cluster/hf\n" for var in REQUIRED_HF_CACHE_VARS))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "llamafactory",
                "run_name": "lf-run",
                "container": {
                    "image": "/shared/containers/llamafactory.sif",
                    "bind_mounts": ["/scratch:/scratch"],
                    "env_file": str(env_file),
                },
                "paths": {"output_base": str(tmp_path / "outputs")},
                "llamafactory": {
                    "model_name_or_path": "org/lf-model",
                    "stage": "sft",
                    "do_train": True,
                },
            }
        )
    )
    submitted_scripts = []

    def fake_prefetch(config):
        return prefetch_module.PrefetchedPaths(model=_SNAPSHOT, tokenizer=_SNAPSHOT)

    def fake_generate_and_submit(config, run_dir, config_path, *, tokenize_only=False):
        submitted_scripts.append(run_dir)
        return "12345"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prefetch_module, "prefetch_assets", fake_prefetch)
    monkeypatch.setattr(_submit, "generate_and_submit", fake_generate_and_submit)
    monkeypatch.setattr(sys, "argv", ["submit.py", "--config", str(config_path), "--confirm"])

    _submit.main()

    return tmp_path / "outputs" / "lf-run", submitted_scripts


def test_llamafactory_job_loads_the_prefetched_snapshot(submitted):
    run_dir, submitted_scripts = submitted

    lf_config = yaml.safe_load((run_dir / "llamafactory_config.yaml").read_text())
    assert lf_config["model_name_or_path"] == _SNAPSHOT
    assert len(submitted_scripts) == 1


def test_llamafactory_config_yaml_keeps_the_hub_id_without_trl_sections(submitted):
    """config.yaml keeps the user's llamafactory block and holds no TRL section."""
    run_dir, _ = submitted

    frozen = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert frozen["llamafactory"]["model_name_or_path"] == "org/lf-model"
    trl_sections = {"model", "training", "checkpointing", "data", "sft", "dpo", "deepspeed"}
    assert not set(frozen) & (trl_sections | {"accelerate", "logging"})


def test_llamafactory_resubmit_from_frozen_config(submitted, monkeypatch):
    """A new submit from the frozen config.yaml passes the TRL-section check."""
    run_dir, submitted_scripts = submitted
    frozen = str(run_dir / "config.yaml")
    monkeypatch.setattr(sys, "argv", ["submit.py", "--config", frozen, "--confirm"])

    _submit.main()

    assert len(submitted_scripts) == 2
