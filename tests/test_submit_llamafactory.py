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


def _tokenize_only_submit(tmp_path, monkeypatch, llamafactory_block):
    """Run submit.py --tokenize-only; return the run directory and the submit calls."""
    env_file = tmp_path / "cluster.env"
    env_file.write_text("".join(f"export {var}=/cluster/hf\n" for var in REQUIRED_HF_CACHE_VARS))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "llamafactory",
                "run_name": "lf-tok",
                "container": {
                    "image": "/shared/containers/llamafactory.sif",
                    "bind_mounts": ["/scratch:/scratch"],
                    "env_file": str(env_file),
                },
                "paths": {"output_base": str(tmp_path / "outputs")},
                "slurm": {"num_nodes": 4, "gpus_per_node": 8},
                "llamafactory": {
                    "model_name_or_path": "org/lf-model",
                    "stage": "sft",
                    "do_train": True,
                    **llamafactory_block,
                },
            }
        )
    )
    calls = []

    def fake_generate_and_submit(config, run_dir, config_path, *, tokenize_only=False):
        calls.append((config.slurm.num_nodes, config.slurm.gpus_per_node, tokenize_only))
        return "12345"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        prefetch_module,
        "prefetch_assets",
        lambda config: prefetch_module.PrefetchedPaths(model=_SNAPSHOT, tokenizer=_SNAPSHOT),
    )
    monkeypatch.setattr(_submit, "generate_and_submit", fake_generate_and_submit)
    monkeypatch.setattr(
        sys, "argv", ["submit.py", "--config", str(config_path), "--confirm", "--tokenize-only"]
    )

    _submit.main()

    return tmp_path / "outputs" / "lf-tok", calls


def test_llamafactory_tokenize_only_makes_tokenized_path_absolute(tmp_path, monkeypatch):
    """A relative tokenized_path starts at the submit directory, on 1 node with 1 GPU."""
    run_dir, calls = _tokenize_only_submit(
        tmp_path, monkeypatch, {"tokenized_path": "data/tokenized/sft"}
    )

    lf_config = yaml.safe_load((run_dir / "llamafactory_config.yaml").read_text())
    assert lf_config["tokenized_path"] == str(tmp_path.resolve() / "data" / "tokenized" / "sft")
    assert calls == [(1, 1, True)]


def test_llamafactory_tokenize_only_needs_tokenized_path(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="needs llamafactory.tokenized_path"):
        _tokenize_only_submit(tmp_path, monkeypatch, {})


def test_llamafactory_tokenize_only_refuses_existing_data(tmp_path, monkeypatch):
    tokenized = tmp_path / "data" / "tokenized"
    tokenized.mkdir(parents=True)
    (tokenized / "dataset_dict.json").write_text("{}")

    with pytest.raises(ValueError, match="already holds data"):
        _tokenize_only_submit(tmp_path, monkeypatch, {"tokenized_path": "data/tokenized"})
