"""Tests for the submission-time behavior of TRLBackend and LlamaFactoryBackend."""

import logging
from pathlib import Path

import pytest
import yaml

from post_training.backend import LlamaFactoryBackend, TRLBackend
from post_training.config import ContainerConfig, PostTrainingConfig


def _make_fake_repo(repo_dir, marker: str = "live") -> None:
    (repo_dir / "src" / "post_training").mkdir(parents=True)
    (repo_dir / "src" / "post_training" / "marker.txt").write_text(marker)
    (repo_dir / "scripts").mkdir(parents=True)
    (repo_dir / "scripts" / "marker.txt").write_text(marker)


def _config_with_container() -> PostTrainingConfig:
    cfg = PostTrainingConfig()
    cfg.container = ContainerConfig(
        image="/shared/containers/trl.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file="/shared/env/cluster.env",
    )
    return cfg


def test_post_freeze_noop_without_container(tmp_path, monkeypatch):
    """No container configured — nothing is copied, run_dir stays untouched."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir)
    monkeypatch.chdir(repo_dir)

    cfg = PostTrainingConfig()
    cfg.container = None
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    TRLBackend().post_freeze(cfg, run_dir)

    assert not (run_dir / "src").exists()
    assert not (run_dir / "scripts").exists()


def test_post_freeze_copies_when_fresh(tmp_path, monkeypatch):
    """Neither directory exists yet — both get copied from the live repo."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (run_dir / "src" / "post_training" / "marker.txt").read_text() == "live"
    assert (run_dir / "scripts" / "marker.txt").read_text() == "live"


def test_post_freeze_skips_when_fully_frozen(tmp_path, monkeypatch, caplog):
    """Both already frozen — existing copies are preserved, not overwritten."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live-later")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    (run_dir / "src" / "post_training").mkdir(parents=True)
    (run_dir / "src" / "post_training" / "marker.txt").write_text("frozen")
    (run_dir / "scripts").mkdir(parents=True)
    (run_dir / "scripts" / "marker.txt").write_text("frozen")

    with caplog.at_level(logging.WARNING):
        TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (run_dir / "src" / "post_training" / "marker.txt").read_text() == "frozen"
    assert (run_dir / "scripts" / "marker.txt").read_text() == "frozen"
    assert "already exists" in caplog.text


def test_post_freeze_completes_partial_state(tmp_path, monkeypatch, caplog):
    """Only one directory exists (interrupted prior attempt) — the missing one
    is copied to complete the freeze; the existing one is left untouched."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    (run_dir / "scripts").mkdir(parents=True)
    (run_dir / "scripts" / "marker.txt").write_text("frozen")
    # src/post_training deliberately left missing.

    with caplog.at_level(logging.WARNING):
        TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (run_dir / "src" / "post_training" / "marker.txt").read_text() == "live"
    assert (run_dir / "scripts" / "marker.txt").read_text() == "frozen"
    assert "already exists" in caplog.text


def test_post_freeze_recovers_from_interrupted_copy(tmp_path, monkeypatch):
    """A killed process mid-copytree leaves only a .tmp dir, not a corrupted
    frozen dir — the next post_freeze must redo the copy in full rather than
    silently adopting the half-written content.

    Regression test for the bug reported on PR #46: since post_freeze's
    existence check was the only signal used to decide "already frozen,
    leave it", a directory that existed but was only partially copied got
    permanently treated as a completed freeze, and the job silently ran
    with incomplete/stale code (`cd`'d into run_dir and imported from there).
    """
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    frozen_src = run_dir / "src" / "post_training"
    tmp_src = frozen_src.with_name(frozen_src.name + ".tmp")
    tmp_src.mkdir(parents=True)
    (tmp_src / "half-written.py").write_text("partial")
    # frozen_src itself deliberately does not exist yet — simulates the
    # process dying after shutil.copytree started writing but before it finished.

    TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (frozen_src / "marker.txt").read_text() == "live"
    assert not (frozen_src / "half-written.py").exists()
    assert not tmp_src.exists()


# ---------------------------------------------------------------------------
# LlamaFactoryBackend — output directory and resume
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _llamafactory_config(**block) -> PostTrainingConfig:
    cfg = PostTrainingConfig()
    cfg.backend = "llamafactory"
    cfg.container = ContainerConfig(
        image="/shared/containers/llamafactory.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file="/shared/env/cluster.env",
    )
    cfg.llamafactory = {"stage": "sft", "do_train": True, **block}
    return cfg


def _frozen_llamafactory_config(run_dir) -> dict:
    return yaml.safe_load((run_dir / "llamafactory_config.yaml").read_text())


def test_llamafactory_validate_rejects_output_dir():
    """The submission owns output_dir, so a user value is refused before any queue wait."""
    with pytest.raises(ValueError, match="llamafactory.output_dir"):
        LlamaFactoryBackend().validate(_llamafactory_config(output_dir="output/my-run"))


def test_llamafactory_validate_rejects_overwrite_output_dir():
    """overwrite_output_dir: true turns off LlamaFactory's resume after a requeue."""
    with pytest.raises(ValueError, match="llamafactory.overwrite_output_dir"):
        LlamaFactoryBackend().validate(_llamafactory_config(overwrite_output_dir=True))


def test_llamafactory_validate_allows_overwrite_output_dir_false():
    LlamaFactoryBackend().validate(_llamafactory_config(overwrite_output_dir=False))


def test_llamafactory_example_config_is_valid(monkeypatch):
    """The shipped example config passes validation from the repository root."""
    monkeypatch.chdir(_REPO_ROOT)

    PostTrainingConfig.load(_REPO_ROOT / "configs" / "llamafactory" / "long-context.yaml")


def test_llamafactory_run_dir_has_checkpoints_subdir():
    assert LlamaFactoryBackend().run_dir_subdirs() == ["checkpoints"]


def test_llamafactory_post_freeze_writes_resumable_output_dir(tmp_path):
    """Checkpoints go to <run_dir>/checkpoints with resume enabled; config.yaml keeps the user block."""
    cfg = _llamafactory_config(learning_rate=1e-5)
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    LlamaFactoryBackend().post_freeze(cfg, run_dir)

    frozen = _frozen_llamafactory_config(run_dir)
    assert frozen["output_dir"] == str(run_dir.resolve() / "checkpoints")
    assert frozen["overwrite_output_dir"] is False
    assert frozen["learning_rate"] == 1e-5
    assert "output_dir" not in cfg.llamafactory
    assert "overwrite_output_dir" not in cfg.llamafactory
