"""Tests for scripts/prepare_run.py — the prepare command for an external orchestrator."""

import importlib.util
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest
import yaml

# scripts/ is not a Python package, so load the script by path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "prepare_run", _REPO_ROOT / "scripts" / "prepare_run.py"
)
prepare_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_run)


def _write_config(tmp_path, *, run_name="my-run", slurm=None, container=True, model=None):
    env_file = tmp_path / "cluster.env"
    env_file.write_text("export HF_HOME=/scratch/hf\n")
    config = {
        "method": "sft",
        "backend": "trl",
        "prefetch_assets": False,
        "training": {"max_steps": 1, "effective_batch_size": 8, "per_device_train_batch_size": 1},
        "paths": {"output_base": str(tmp_path / "outputs")},
    }
    if run_name:
        config["run_name"] = run_name
    if model is not None:
        # A local model with no tokenizer path, and the prefetch step of a real run.
        config["model"] = {"name_or_path": str(model)}
        config["prefetch_assets"] = True
    if slurm is not None:
        config["slurm"] = slurm
    if container:
        config["container"] = {
            "image": "/shared/containers/trl.sif",
            "bind_mounts": ["/scratch"],
            "env_file": str(env_file),
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.fixture
def run(monkeypatch, capsys):
    """Call prepare_run.main() with *args* and return (exit code, result, stderr)."""

    def _no_submission(*args, **kwargs):
        raise AssertionError("prepare_run.py must not submit a job")

    monkeypatch.setattr(prepare_run.submit, "generate_and_submit", _no_submission)
    monkeypatch.setattr(prepare_run.submit, "_apply_hf_env_from_file", lambda env_file: None)
    monkeypatch.chdir(_REPO_ROOT)  # the code freeze copies the code from the cwd

    def _run(*args):
        monkeypatch.setattr(sys, "argv", ["prepare_run.py", *map(str, args)])
        code = 0
        umask = os.umask(0o022)  # main() changes the umask of this process
        try:
            prepare_run.main()
        except SystemExit as exit_:
            code = exit_.code
        finally:
            os.umask(umask)
        captured = capsys.readouterr()
        results = [
            json.loads(line[len(prepare_run.RESULT_PREFIX) :])
            for line in captured.out.splitlines()
            if line.startswith(prepare_run.RESULT_PREFIX)
        ]
        return code, (results[-1] if results else None), captured.err

    return _run


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


_NODES = ("--num-nodes", 2, "--gpus-per-node", 4)


# ---------------------------------------------------------------------------
# A new run
# ---------------------------------------------------------------------------


def test_new_run_is_prepared_and_reported(tmp_path, run):
    config_path = _write_config(tmp_path)

    code, result, _ = run("--config", config_path, *_NODES)

    run_dir = tmp_path / "outputs" / "my-run"
    assert code == 0
    assert result == {
        "run_dir": str(run_dir),
        "launch_script": str(run_dir / "slurm" / "launch.sh"),
        "config": str(run_dir / "config.yaml"),
        "new_run": True,
        "checked_only": False,
        "num_nodes": 2,
        "gpus_per_node": 4,
    }
    assert (run_dir / "src" / "post_training" / "config.py").is_file()
    assert (run_dir / "scripts" / "train.py").is_file()
    assert not (run_dir / "slurm" / "job.sh").exists()
    # The orchestrator's values are in the frozen config and in the launch script.
    frozen = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert frozen["slurm"]["num_nodes"] == 2
    assert "EXPECTED_NODES=2" in (run_dir / "slurm" / "launch.sh").read_text()


def test_new_run_directory_is_writable_for_the_group(tmp_path, run):
    """The job can run as a different Unix user; the caller's umask (022 here) must not matter."""
    config_path = _write_config(tmp_path)

    code, result, _ = run("--config", config_path, *_NODES)

    assert code == 0
    run_dir = Path(result["run_dir"])
    for path in (
        run_dir.parent,
        run_dir,
        run_dir / "checkpoints",
        run_dir / "logs",
        run_dir / "slurm",
    ):
        assert path.stat().st_mode & stat.S_IWGRP, f"{path} is not writable for the group"
    assert (run_dir / "config.yaml").stat().st_mode & stat.S_IWGRP


def test_run_name_can_be_an_override(tmp_path, run):
    config_path = _write_config(tmp_path, run_name=None)

    code, result, _ = run("--config", config_path, "run_name=other-run")

    assert code == 0
    assert result["run_dir"] == str(tmp_path / "outputs" / "other-run")


def test_missing_run_name_is_an_error(tmp_path, run):
    config_path = _write_config(tmp_path, run_name=None)

    code, result, err = run("--config", config_path)

    assert code == 1
    assert result is None
    assert "run_name is mandatory" in err
    assert not (tmp_path / "outputs").exists()


def test_config_without_container_is_an_error(tmp_path, run):
    config_path = _write_config(tmp_path, container=False)

    code, _, err = run("--config", config_path)

    assert code == 1
    assert "container.image" in err
    assert not (tmp_path / "outputs").exists()


def test_check_writes_nothing(tmp_path, run):
    config_path = _write_config(tmp_path)

    code, result, _ = run("--config", config_path, "--check")

    assert code == 0
    assert result["new_run"] is True
    assert result["checked_only"] is True
    assert not (tmp_path / "outputs").exists()


# ---------------------------------------------------------------------------
# The values that the orchestrator owns
# ---------------------------------------------------------------------------


def test_equal_node_count_in_the_file_passes(tmp_path, run):
    config_path = _write_config(tmp_path, slurm={"num_nodes": 2, "gpus_per_node": 4})

    code, result, _ = run("--config", config_path, *_NODES)

    assert code == 0
    assert result["num_nodes"] == 2


@pytest.mark.parametrize(
    ("slurm", "key"),
    [
        ({"num_nodes": 1, "gpus_per_node": 4}, "slurm.num_nodes is 1"),
        ({"num_nodes": 2, "gpus_per_node": 2}, "slurm.gpus_per_node is 2"),
    ],
)
def test_different_value_in_the_file_is_an_error(tmp_path, run, slurm, key):
    config_path = _write_config(tmp_path, slurm=slurm)

    code, result, err = run("--config", config_path, *_NODES)

    assert code == 1
    assert result is None
    assert key in err
    assert not (tmp_path / "outputs").exists()


def test_override_of_an_owned_key_is_an_error(tmp_path, run):
    config_path = _write_config(tmp_path)

    code, _, err = run("--config", config_path, *_NODES, "slurm.num_nodes=2")

    assert code == 1
    assert "Remove the override of slurm.num_nodes" in err
    assert not (tmp_path / "outputs").exists()


def test_inexact_gradient_accumulation_is_an_error_before_any_write(tmp_path, run):
    config_path = _write_config(tmp_path)

    code, _, err = run("--config", config_path, "--num-nodes", 3, "--gpus-per-node", 4)

    assert code == 1
    assert "not evenly divisible" in err
    assert not (tmp_path / "outputs").exists()


# ---------------------------------------------------------------------------
# A continued run
# ---------------------------------------------------------------------------


@pytest.fixture
def prepared(tmp_path, run):
    """A prepared run directory that already has a checkpoint."""
    config_path = _write_config(tmp_path)
    code, result, _ = run("--config", config_path, *_NODES)
    assert code == 0
    run_dir = Path(result["run_dir"])
    (run_dir / "checkpoints" / "checkpoint-250").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint-250" / "trainer_state.json").write_text("{}")
    return config_path, run_dir


def test_second_call_changes_nothing(prepared, run):
    config_path, run_dir = prepared
    before = _snapshot(run_dir)
    frozen_mtime = (run_dir / "config.yaml").stat().st_mtime_ns

    code, result, _ = run("--config", config_path, *_NODES)

    assert code == 0
    assert result["new_run"] is False
    assert _snapshot(run_dir) == before
    assert (run_dir / "config.yaml").stat().st_mtime_ns == frozen_mtime


def test_second_call_writes_a_removed_launch_script_again(prepared, run):
    config_path, run_dir = prepared
    before = _snapshot(run_dir)
    (run_dir / "slurm" / "launch.sh").unlink()

    code, _, _ = run("--config", config_path, *_NODES)

    assert code == 0
    assert _snapshot(run_dir) == before


def test_frozen_config_can_be_the_config(prepared, run):
    _, run_dir = prepared
    before = _snapshot(run_dir)

    code, result, _ = run("--config", run_dir / "config.yaml", *_NODES)

    assert code == 0
    assert result["new_run"] is False
    assert _snapshot(run_dir) == before


def test_changed_config_for_a_continued_run_is_an_error(prepared, run):
    config_path, run_dir = prepared
    before = _snapshot(run_dir)

    code, result, err = run("--config", config_path, *_NODES, "training.learning_rate=0.5")

    assert code == 1
    assert result is None
    assert "training.learning_rate" in err
    assert "now 0.5" in err
    assert _snapshot(run_dir) == before


def test_different_node_count_for_a_continued_run_is_an_error(prepared, run):
    config_path, run_dir = prepared
    before = _snapshot(run_dir)

    code, _, err = run("--config", config_path, "--num-nodes", 4, "--gpus-per-node", 2)

    assert code == 1
    assert "slurm.num_nodes: frozen 2, now 4" in err
    assert _snapshot(run_dir) == before


def test_check_of_a_continued_run_writes_nothing(prepared, run):
    config_path, run_dir = prepared
    (run_dir / "slurm" / "launch.sh").unlink()
    before = _snapshot(run_dir)

    code, result, _ = run("--config", config_path, *_NODES, "--check")

    assert code == 0
    assert result["new_run"] is False
    assert _snapshot(run_dir) == before


def test_delete_option_of_debug_runs_is_refused(tmp_path, run):
    config_path = _write_config(tmp_path)

    code, _, err = run(
        "--config", config_path, "debug.enabled=true", "debug.override_existing=true"
    )

    assert code == 1
    assert "debug.override_existing" in err


# ---------------------------------------------------------------------------
# A Hub repo id, which submit.py freezes as its local snapshot directory
# ---------------------------------------------------------------------------


def _freeze_model_as_snapshot(run_dir: Path, repo_id: str) -> None:
    frozen = yaml.safe_load((run_dir / "config.yaml").read_text())
    cache_name = "models--" + repo_id.replace("/", "--")
    frozen["model"]["name_or_path"] = f"/cache/hub/{cache_name}/snapshots/abc123"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(frozen))


def test_same_repo_id_passes_for_a_continued_run(prepared, run):
    config_path, run_dir = prepared
    _freeze_model_as_snapshot(run_dir, "my-org/model-a")

    code, result, _ = run("--config", config_path, *_NODES, "model.name_or_path=my-org/model-a")

    assert code == 0
    assert result["new_run"] is False


def test_different_repo_id_is_an_error_for_a_continued_run(prepared, run):
    config_path, run_dir = prepared
    _freeze_model_as_snapshot(run_dir, "my-org/model-a")
    before = _snapshot(run_dir)

    code, _, err = run("--config", config_path, *_NODES, "model.name_or_path=my-org/model-b")

    assert code == 1
    assert "model.name_or_path" in err
    assert "now 'my-org/model-b'" in err
    assert _snapshot(run_dir) == before


def test_repo_id_with_the_same_prefix_is_an_error(prepared, run):
    """models--my-org--model-a must not pass for the repo my-org/model."""
    config_path, run_dir = prepared
    _freeze_model_as_snapshot(run_dir, "my-org/model-a")

    code, _, err = run("--config", config_path, *_NODES, "model.name_or_path=my-org/model")

    assert code == 1
    assert "model.name_or_path" in err


# ---------------------------------------------------------------------------
# A config that takes the tokenizer from the model (no tokenizer path)
# ---------------------------------------------------------------------------


def test_second_call_passes_when_the_tokenizer_comes_from_a_local_model(tmp_path, run):
    """submit.py freezes the tokenizer as the model path; the config still has no tokenizer."""
    model = tmp_path / "model"
    model.mkdir()
    config_path = _write_config(tmp_path, model=model)

    assert run("--config", config_path, *_NODES)[0] == 0
    run_dir = tmp_path / "outputs" / "my-run"
    frozen = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert frozen["model"]["tokenizer_name_or_path"] == str(model)
    before = _snapshot(run_dir)

    code, result, err = run("--config", config_path, *_NODES)

    assert code == 0, err
    assert result["new_run"] is False
    assert _snapshot(run_dir) == before


def _freeze_model_and_tokenizer_as_snapshot(run_dir: Path, repo_id: str) -> None:
    frozen = yaml.safe_load((run_dir / "config.yaml").read_text())
    cache_name = "models--" + repo_id.replace("/", "--")
    snapshot = f"/cache/hub/{cache_name}/snapshots/abc123"
    frozen["model"]["name_or_path"] = snapshot
    frozen["model"]["tokenizer_name_or_path"] = snapshot
    (run_dir / "config.yaml").write_text(yaml.safe_dump(frozen))


def test_second_call_passes_when_the_tokenizer_comes_from_a_hub_model(prepared, run):
    config_path, run_dir = prepared
    _freeze_model_and_tokenizer_as_snapshot(run_dir, "my-org/model-a")

    code, result, err = run("--config", config_path, *_NODES, "model.name_or_path=my-org/model-a")

    assert code == 0, err
    assert result["new_run"] is False


def test_tokenizer_of_a_different_model_is_an_error(prepared, run):
    config_path, run_dir = prepared
    _freeze_model_and_tokenizer_as_snapshot(run_dir, "my-org/model-a")

    code, _, err = run("--config", config_path, *_NODES, "model.name_or_path=my-org/model-b")

    assert code == 1
    assert "model.name_or_path" in err
    assert "model.tokenizer_name_or_path" in err
    assert "now None" in err  # the message shows the value of the config, not the derived one


# ---------------------------------------------------------------------------
# A preparation that stopped after config.yaml and before the code freeze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["scripts", "src"])
def test_interrupted_preparation_is_done_again(tmp_path, run, missing):
    config_path = _write_config(tmp_path)
    assert run("--config", config_path, *_NODES)[0] == 0
    run_dir = tmp_path / "outputs" / "my-run"
    complete = _snapshot(run_dir)
    # The state that an interrupted submission phase leaves: no code, no launch script.
    shutil.rmtree(run_dir / missing)
    shutil.rmtree(run_dir / "slurm")

    code, result, err = run("--config", config_path, *_NODES)

    assert code == 0, err
    assert result["new_run"] is True
    assert _snapshot(run_dir) == complete


def test_missing_code_of_a_run_with_checkpoints_is_an_error(prepared, run):
    config_path, run_dir = prepared  # has checkpoints/checkpoint-250
    shutil.rmtree(run_dir / "scripts")
    before = _snapshot(run_dir)

    code, result, err = run("--config", config_path, *_NODES)

    assert code == 1
    assert result is None
    assert "has checkpoints but no frozen code" in err
    assert _snapshot(run_dir) == before


# ---------------------------------------------------------------------------
# A debug run: train.py sets logging.report_to to "none" in the frozen config
# ---------------------------------------------------------------------------


def _set_frozen_report_to_none(run_dir: Path) -> None:
    frozen = yaml.safe_load((run_dir / "config.yaml").read_text())
    frozen["logging"]["report_to"] = ["none"]
    (run_dir / "config.yaml").write_text(yaml.safe_dump(frozen))


def test_debug_run_can_be_prepared_again_after_it_ran(tmp_path, run):
    config_path = _write_config(tmp_path)
    debug = ("debug.enabled=true", f"paths.debug_base={tmp_path / 'debug'}")
    code, result, _ = run("--config", config_path, *_NODES, *debug)
    assert code == 0
    run_dir = Path(result["run_dir"])
    assert run_dir == tmp_path / "debug" / "my-run"
    _set_frozen_report_to_none(run_dir)

    code, result, err = run("--config", config_path, *_NODES, *debug)

    assert code == 0, err
    assert result["new_run"] is False


def test_changed_reporting_of_a_normal_run_is_still_an_error(prepared, run):
    config_path, run_dir = prepared
    _set_frozen_report_to_none(run_dir)

    code, _, err = run("--config", config_path, *_NODES)

    assert code == 1
    assert "logging.report_to" in err


# ---------------------------------------------------------------------------
# --check finds what the real preparation would stop on
# ---------------------------------------------------------------------------


def test_check_finds_a_missing_env_file_of_a_new_run(tmp_path, run):
    config_path = _write_config(tmp_path)
    (tmp_path / "cluster.env").unlink()

    code, result, err = run("--config", config_path, *_NODES, "--check")

    assert code == 1
    assert result is None
    assert "container.env_file" in err and "not found" in err
    assert not (tmp_path / "outputs").exists()


def test_check_finds_a_missing_env_file_of_a_continued_run(prepared, run, tmp_path):
    config_path, run_dir = prepared
    (tmp_path / "cluster.env").unlink()
    before = _snapshot(run_dir)

    code, _, err = run("--config", config_path, *_NODES, "--check")

    assert code == 1
    assert "container.env_file" in err
    assert _snapshot(run_dir) == before


def test_check_finds_missing_code_of_a_run_with_checkpoints(prepared, run):
    config_path, run_dir = prepared
    (run_dir / "scripts" / "train.py").unlink()

    code, _, err = run("--config", config_path, *_NODES, "--check")

    assert code == 1
    assert "has checkpoints but no frozen code" in err
