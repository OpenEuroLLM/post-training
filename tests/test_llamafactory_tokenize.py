"""Tests for the --tokenize-only script of the LlamaFactory backend, without llamafactory."""

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

# The script is not a post_training module: the container runs it from the run directory.
_SCRIPT = Path(__file__).resolve().parents[1] / "src/post_training/slurm/llamafactory_tokenize.py"
_spec = importlib.util.spec_from_file_location("llamafactory_tokenize", _SCRIPT)
_tokenize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tokenize)


def _write_config(tmp_path, tokenized_path) -> str:
    config = tmp_path / "llamafactory_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": "org/model",
                "stage": "sft",
                "tokenized_path": str(tokenized_path),
                "deepspeed": "/run/llamafactory/zero3.json",
            }
        )
    )
    return str(config)


@pytest.fixture
def tokenize_calls(monkeypatch):
    """Replace the LlamaFactory step with one that saves a marker into tokenized_path."""
    calls = []

    def fake_tokenize(args):
        calls.append(args)
        os.makedirs(args["tokenized_path"])
        Path(args["tokenized_path"], "dataset_dict.json").write_text("{}")

    monkeypatch.setattr(_tokenize, "tokenize", fake_tokenize)
    return calls


def test_partial_path_is_named_after_the_slurm_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4711")

    assert _tokenize.partial_path("/data/tokenized/sft/") == "/data/tokenized/sft.partial-4711"


def test_partial_path_uses_the_pid_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    assert _tokenize.partial_path("/data/sft") == f"/data/sft.partial-{os.getpid()}"


def test_main_saves_into_its_partial_directory_and_publishes_it(
    tmp_path, monkeypatch, tokenize_calls, capsys
):
    monkeypatch.setenv("SLURM_JOB_ID", "4711")
    tokenized = tmp_path / "tokenized"

    _tokenize.main(_write_config(tmp_path, tokenized))

    assert [call["tokenized_path"] for call in tokenize_calls] == [f"{tokenized}.partial-4711"]
    assert "deepspeed" not in tokenize_calls[0]
    assert (tokenized / "dataset_dict.json").exists()
    assert not Path(f"{tokenized}.partial-4711").exists()
    assert f"Tokenized dataset saved at {tokenized}" in capsys.readouterr().out


def test_main_replaces_an_empty_tokenized_path_directory(tmp_path, monkeypatch, tokenize_calls):
    monkeypatch.setenv("SLURM_JOB_ID", "4711")
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()

    _tokenize.main(_write_config(tmp_path, tokenized))

    assert (tokenized / "dataset_dict.json").exists()


def test_main_removes_the_interrupted_save_of_a_previous_attempt(
    tmp_path, monkeypatch, tokenize_calls
):
    """A requeued job has the same job ID, so its partial directory must not be loaded."""
    monkeypatch.setenv("SLURM_JOB_ID", "4711")
    tokenized = tmp_path / "tokenized"
    stale = Path(f"{tokenized}.partial-4711")
    stale.mkdir()
    (stale / "state.json").write_text("interrupted")

    _tokenize.main(_write_config(tmp_path, tokenized))

    assert sorted(p.name for p in tokenized.iterdir()) == ["dataset_dict.json"]


def test_main_does_nothing_when_tokenized_path_holds_data(
    tmp_path, monkeypatch, tokenize_calls, capsys
):
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    (tokenized / "dataset_dict.json").write_text("{}")

    _tokenize.main(_write_config(tmp_path, tokenized))

    assert tokenize_calls == []
    assert "already holds data" in capsys.readouterr().out


def test_main_discards_its_data_when_another_job_published_first(tmp_path, monkeypatch, capsys):
    """Two jobs on one tokenized_path: the second publish fails and leaves the first intact."""
    monkeypatch.setenv("SLURM_JOB_ID", "4711")
    tokenized = tmp_path / "tokenized"

    def tokenize_while_another_job_publishes(args):
        os.makedirs(args["tokenized_path"])
        Path(args["tokenized_path"], "dataset_dict.json").write_text("second job")
        tokenized.mkdir()
        (tokenized / "dataset_dict.json").write_text("first job")

    monkeypatch.setattr(_tokenize, "tokenize", tokenize_while_another_job_publishes)

    _tokenize.main(_write_config(tmp_path, tokenized))

    assert (tokenized / "dataset_dict.json").read_text() == "first job"
    assert not Path(f"{tokenized}.partial-4711").exists()
    assert "Another job saved tokenized data" in capsys.readouterr().out


def test_publish_keeps_the_first_published_data(tmp_path):
    tokenized = tmp_path / "tokenized"
    first, second = tmp_path / "tokenized.partial-1", tmp_path / "tokenized.partial-2"
    for partial in (first, second):
        partial.mkdir()
        (partial / "dataset_dict.json").write_text(partial.name)

    assert _tokenize.publish(str(first), str(tokenized)) is True
    assert _tokenize.publish(str(second), str(tokenized)) is False
    assert (tokenized / "dataset_dict.json").read_text() == "tokenized.partial-1"
    assert not first.exists()
    assert not second.exists()


def test_publish_raises_other_errors(tmp_path):
    """A target that is a file is a setup error, not a lost race."""
    tokenized = tmp_path / "tokenized"
    tokenized.write_text("not a directory")
    partial = tmp_path / "tokenized.partial-1"
    partial.mkdir()

    with pytest.raises(OSError):
        _tokenize.publish(str(partial), str(tokenized))
    assert partial.exists()
