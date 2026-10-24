"""Tests for the submission-time behavior of TRLBackend and LlamaFactoryBackend."""

import json
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
    cfg.llamafactory = {
        "model_name_or_path": "org/model",
        "stage": "sft",
        "do_train": True,
        **block,
    }
    return cfg


def _frozen_llamafactory_config(run_dir) -> dict:
    return yaml.safe_load((run_dir / "llamafactory_config.yaml").read_text())


def test_llamafactory_validate_requires_model_name_or_path():
    cfg = _llamafactory_config()
    del cfg.llamafactory["model_name_or_path"]

    with pytest.raises(ValueError, match="llamafactory.model_name_or_path"):
        LlamaFactoryBackend().validate(cfg)


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


def test_llamafactory_post_freeze_writes_resumable_output_dir(tmp_path, monkeypatch):
    """Checkpoints go to <run_dir>/checkpoints with resume enabled; config.yaml keeps the user block."""
    monkeypatch.chdir(tmp_path)
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


# ---------------------------------------------------------------------------
# LlamaFactoryBackend — frozen DeepSpeed and dataset configs, input paths
# ---------------------------------------------------------------------------

_DATASET_INFO = {
    "local_file": {"file_name": "sft.jsonl", "formatting": "sharegpt"},
    "absolute_file": {"file_name": "/datasets/abs.jsonl"},
    "local_script": {"script_url": "my_script"},
    "hub": {"hf_hub_url": "org/dataset"},
}


def _make_llamafactory_repo(repo_dir: Path) -> None:
    """A submission directory with a DeepSpeed JSON and a LlamaFactory dataset registry."""
    (repo_dir / "configs" / "deepspeed").mkdir(parents=True)
    (repo_dir / "configs" / "deepspeed" / "zero3.json").write_text(
        '{"zero_optimization": {"stage": 3}}'
    )
    (repo_dir / "data" / "llamafactory").mkdir(parents=True)
    (repo_dir / "data" / "llamafactory" / "dataset_info.json").write_text(json.dumps(_DATASET_INFO))


def test_llamafactory_validate_rejects_missing_deepspeed_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="llamafactory.deepspeed"):
        LlamaFactoryBackend().validate(_llamafactory_config(deepspeed="configs/ds/missing.json"))


def test_llamafactory_validate_accepts_inline_deepspeed_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    LlamaFactoryBackend().validate(
        _llamafactory_config(deepspeed={"zero_optimization": {"stage": 2}})
    )


@pytest.mark.parametrize(
    "block",
    [
        {"dataset": "long_sft", "dataset_dir": "data/llamafactory"},
        {"dataset": "long_sft"},  # LlamaFactory's default dataset_dir is "data"
        {"eval_dataset": "long_sft_eval"},
    ],
)
def test_llamafactory_validate_rejects_missing_dataset_info(tmp_path, monkeypatch, block):
    """A named dataset needs <dataset_dir>/dataset_info.json, or the job stops at start."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="dataset_info.json"):
        LlamaFactoryBackend().validate(_llamafactory_config(**block))


@pytest.mark.parametrize("dataset_dir", ["ONLINE", "REMOTE:org/registry"])
def test_llamafactory_validate_skips_registry_check_for_hub_registries(
    tmp_path, monkeypatch, dataset_dir
):
    monkeypatch.chdir(tmp_path)

    LlamaFactoryBackend().validate(
        _llamafactory_config(dataset="org/dataset", dataset_dir=dataset_dir)
    )


def test_llamafactory_post_freeze_copies_deepspeed_json(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    _make_llamafactory_repo(repo_dir)
    monkeypatch.chdir(repo_dir)
    cfg = _llamafactory_config(deepspeed="configs/deepspeed/zero3.json")
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    LlamaFactoryBackend().post_freeze(cfg, run_dir)

    copy_path = run_dir.resolve() / "llamafactory" / "zero3.json"
    assert _frozen_llamafactory_config(run_dir)["deepspeed"] == str(copy_path)
    assert copy_path.read_text() == (repo_dir / "configs" / "deepspeed" / "zero3.json").read_text()
    assert cfg.llamafactory["deepspeed"] == "configs/deepspeed/zero3.json"


def test_llamafactory_post_freeze_keeps_inline_deepspeed_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inline = {"zero_optimization": {"stage": 2}}
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    LlamaFactoryBackend().post_freeze(_llamafactory_config(deepspeed=inline), run_dir)

    assert _frozen_llamafactory_config(run_dir)["deepspeed"] == inline


def test_llamafactory_post_freeze_freezes_dataset_registry(tmp_path, monkeypatch):
    """Relative local entries point back at the source; the job reads the frozen registry."""
    repo_dir = tmp_path / "repo"
    _make_llamafactory_repo(repo_dir)
    monkeypatch.chdir(repo_dir)
    cfg = _llamafactory_config(dataset="local_file", dataset_dir="data/llamafactory")
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    LlamaFactoryBackend().post_freeze(cfg, run_dir)

    source_dir = (repo_dir / "data" / "llamafactory").resolve()
    frozen_dir = run_dir.resolve() / "llamafactory"
    frozen = _frozen_llamafactory_config(run_dir)
    assert frozen["dataset_dir"] == str(frozen_dir)
    assert frozen["media_dir"] == str(source_dir)
    registry = json.loads((frozen_dir / "dataset_info.json").read_text())
    assert registry["local_file"] == {
        "file_name": str(source_dir / "sft.jsonl"),
        "formatting": "sharegpt",
    }
    assert registry["absolute_file"] == _DATASET_INFO["absolute_file"]
    assert registry["local_script"] == {"script_url": str(source_dir / "my_script")}
    assert registry["hub"] == _DATASET_INFO["hub"]
    assert cfg.llamafactory["dataset_dir"] == "data/llamafactory"


def test_llamafactory_post_freeze_online_registry_is_not_copied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    LlamaFactoryBackend().post_freeze(
        _llamafactory_config(dataset="org/dataset", dataset_dir="ONLINE"), run_dir
    )

    frozen = _frozen_llamafactory_config(run_dir)
    assert frozen["dataset_dir"] == "ONLINE"
    assert "media_dir" not in frozen
    assert not (run_dir / "llamafactory" / "dataset_info.json").exists()


def test_llamafactory_post_freeze_makes_existing_input_paths_absolute(tmp_path, monkeypatch):
    """Existing relative input paths survive the job's cd; hub IDs, new paths and outputs do not change."""
    repo_dir = tmp_path / "repo"
    for local in ("models/local-model", "adapters/a", "adapters/b", "media", "exports"):
        (repo_dir / local).mkdir(parents=True)
    monkeypatch.chdir(repo_dir)
    cfg = _llamafactory_config(
        model_name_or_path="models/local-model",
        adapter_name_or_path="adapters/a,adapters/b",
        media_dir="media",
        ref_model="org/ref-model",
        tokenized_path="data/tokenized",
        export_dir="exports",
        dataset_dir="ONLINE",
    )
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    LlamaFactoryBackend().post_freeze(cfg, run_dir)

    root = repo_dir.resolve()
    frozen = _frozen_llamafactory_config(run_dir)
    assert frozen["model_name_or_path"] == str(root / "models" / "local-model")
    assert frozen["adapter_name_or_path"] == f"{root / 'adapters' / 'a'},{root / 'adapters' / 'b'}"
    assert frozen["media_dir"] == str(root / "media")
    assert frozen["ref_model"] == "org/ref-model"
    assert frozen["tokenized_path"] == "data/tokenized"
    assert frozen["export_dir"] == "exports"


def test_llamafactory_post_freeze_makes_v095_input_paths_absolute(tmp_path, monkeypatch):
    """The KTransformers and HyperParallel input files that LlamaFactory 0.9.5 added."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "kt").mkdir(parents=True)
    for local in ("kt/weights.kt", "kt/experts.safetensors", "hyper_parallel.json"):
        (repo_dir / local).write_text("{}")
    monkeypatch.chdir(repo_dir)
    cfg = _llamafactory_config(
        kt_weight_path="kt/weights.kt",
        kt_expert_checkpoint_path="kt/experts.safetensors",
        hyper_parallel_args="hyper_parallel.json",
        profiler_output_dir="profiler",
        dataset_dir="ONLINE",
    )
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    LlamaFactoryBackend().post_freeze(cfg, run_dir)

    root = repo_dir.resolve()
    frozen = _frozen_llamafactory_config(run_dir)
    assert frozen["kt_weight_path"] == str(root / "kt" / "weights.kt")
    assert frozen["kt_expert_checkpoint_path"] == str(root / "kt" / "experts.safetensors")
    assert frozen["hyper_parallel_args"] == str(root / "hyper_parallel.json")
    assert frozen["profiler_output_dir"] == "profiler"


def test_llamafactory_post_freeze_keeps_existing_copies(tmp_path, monkeypatch, caplog):
    """A new submit into the same run directory keeps the copies its checkpoints were made with."""
    repo_dir = tmp_path / "repo"
    _make_llamafactory_repo(repo_dir)
    monkeypatch.chdir(repo_dir)
    cfg = _llamafactory_config(
        deepspeed="configs/deepspeed/zero3.json",
        dataset="local_file",
        dataset_dir="data/llamafactory",
    )
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)
    LlamaFactoryBackend().post_freeze(cfg, run_dir)
    frozen_dir = run_dir / "llamafactory"
    first_deepspeed = (frozen_dir / "zero3.json").read_text()
    first_registry = (frozen_dir / "dataset_info.json").read_text()

    (repo_dir / "configs" / "deepspeed" / "zero3.json").write_text(
        '{"zero_optimization": {"stage": 2}}'
    )
    (repo_dir / "data" / "llamafactory" / "dataset_info.json").write_text("{}")
    with caplog.at_level(logging.WARNING):
        LlamaFactoryBackend().post_freeze(cfg, run_dir)

    assert (frozen_dir / "zero3.json").read_text() == first_deepspeed
    assert (frozen_dir / "dataset_info.json").read_text() == first_registry
    assert "already exists" in caplog.text


def test_llamafactory_post_freeze_recovers_from_interrupted_copy(tmp_path, monkeypatch):
    """A leftover .tmp from a killed submission is replaced by a complete copy."""
    repo_dir = tmp_path / "repo"
    _make_llamafactory_repo(repo_dir)
    monkeypatch.chdir(repo_dir)
    run_dir = tmp_path / "outputs" / "my-run"
    (run_dir / "llamafactory").mkdir(parents=True)
    (run_dir / "llamafactory" / "zero3.json.tmp").write_text('{"zero_opt')

    LlamaFactoryBackend().post_freeze(
        _llamafactory_config(deepspeed="configs/deepspeed/zero3.json"), run_dir
    )

    copy_path = run_dir / "llamafactory" / "zero3.json"
    assert copy_path.read_text() == (repo_dir / "configs" / "deepspeed" / "zero3.json").read_text()
    assert not (run_dir / "llamafactory" / "zero3.json.tmp").exists()


# ---------------------------------------------------------------------------
# Prefetched snapshots
# ---------------------------------------------------------------------------


def test_trl_post_freeze_accepts_prefetched_paths(tmp_path):
    """TRL substitutes snapshots into config.yaml in submit.py, so post_freeze ignores them."""
    from post_training.utils.prefetch import PrefetchedPaths

    cfg = PostTrainingConfig()
    cfg.container = None
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    TRLBackend().post_freeze(cfg, run_dir, PrefetchedPaths(model="/m", tokenizer="/m"))

    assert not (run_dir / "src").exists()


def test_llamafactory_post_freeze_writes_prefetched_snapshots(tmp_path, monkeypatch):
    """The job loads the prefetched snapshot directories; config.yaml keeps the Hub IDs."""
    from post_training.utils.prefetch import PrefetchedPaths

    monkeypatch.chdir(tmp_path)
    cfg = _llamafactory_config(model_name_or_path="org/model", ref_model="org/ref")
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)
    prefetched = PrefetchedPaths(
        model="/cache/org/model@main",
        tokenizer="/cache/org/model@main",
        ref_model="/cache/org/ref@main",
    )

    LlamaFactoryBackend().post_freeze(cfg, run_dir, prefetched)

    frozen = _frozen_llamafactory_config(run_dir)
    assert frozen["model_name_or_path"] == "/cache/org/model@main"
    assert frozen["ref_model"] == "/cache/org/ref@main"
    assert cfg.llamafactory["model_name_or_path"] == "org/model"
    assert cfg.llamafactory["ref_model"] == "org/ref"


# ---------------------------------------------------------------------------
# LlamaFactoryBackend — run name from the llamafactory block
# ---------------------------------------------------------------------------

_TIMESTAMP = "20260917-101500"


def _llamafactory_run_name(cfg: PostTrainingConfig) -> str:
    return LlamaFactoryBackend().generate_run_name(cfg, _TIMESTAMP)


def test_llamafactory_run_name_single_dataset():
    """eval_dataset does not enter the name."""
    cfg = _llamafactory_config(
        model_name_or_path="ontocord/1.7b-MixtureVitae-300BT",
        dataset="long_sft",
        eval_dataset="long_sft_eval",
    )

    assert (
        _llamafactory_run_name(cfg)
        == f"llamafactory-sft-17b-mixturevitae-300bt-long_sft-{_TIMESTAMP}"
    )


@pytest.mark.parametrize(
    "datasets", ["dolci_instruct_sft, Tulu3-SFT", ["dolci_instruct_sft", "Tulu3-SFT"]]
)
def test_llamafactory_run_name_joins_datasets_with_plus(datasets):
    cfg = _llamafactory_config(dataset=datasets)

    assert (
        _llamafactory_run_name(cfg)
        == f"llamafactory-sft-model-dolci_instruct_sft+tulu3-sft-{_TIMESTAMP}"
    )


def test_llamafactory_run_name_without_datasets():
    assert _llamafactory_run_name(_llamafactory_config()) == f"llamafactory-sft-model-{_TIMESTAMP}"


def test_llamafactory_run_name_uses_the_stage():
    cfg = _llamafactory_config(stage="dpo", dataset="prefs")

    assert _llamafactory_run_name(cfg) == f"llamafactory-dpo-model-prefs-{_TIMESTAMP}"


def test_llamafactory_run_name_ignores_the_top_level_method():
    """Without llamafactory.stage the name uses LlamaFactory's default stage, not the TRL key."""
    cfg = _llamafactory_config()
    del cfg.llamafactory["stage"]
    cfg.method = "dpo"

    assert _llamafactory_run_name(cfg) == f"llamafactory-sft-model-{_TIMESTAMP}"


def test_llamafactory_run_name_local_model_with_trailing_slash():
    cfg = _llamafactory_config(model_name_or_path="/scratch/models/Prelude-9B/")

    assert _llamafactory_run_name(cfg) == f"llamafactory-sft-prelude-9b-{_TIMESTAMP}"


def test_llamafactory_run_name_hub_dataset_id():
    cfg = _llamafactory_config(dataset_dir="ONLINE", dataset="allenai/Dolci-Instruct-SFT")

    assert (
        _llamafactory_run_name(cfg)
        == f"llamafactory-sft-model-allenai_dolci-instruct-sft-{_TIMESTAMP}"
    )
