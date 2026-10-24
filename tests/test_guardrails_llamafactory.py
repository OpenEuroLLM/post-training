"""Tests for the pre-submission review of LlamaFactory runs."""

import builtins

import pytest

from post_training.backend import LlamaFactoryBackend
from post_training.config import ContainerConfig, PostTrainingConfig
from post_training.utils.guardrails import run_guardrails


def _llamafactory_config(**block) -> PostTrainingConfig:
    cfg = PostTrainingConfig()
    cfg.backend = "llamafactory"
    cfg.method = block.get("stage", "sft")
    cfg.container = ContainerConfig(
        image="/shared/containers/llamafactory.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file="/shared/env/cluster.env",
    )
    cfg.slurm.num_nodes = 2
    cfg.slurm.gpus_per_node = 4
    cfg.llamafactory = {"model_name_or_path": "org/lf-model", "dataset": "a,b", **block}
    return cfg


def _review(cfg, tmp_path, monkeypatch, capsys) -> str:
    monkeypatch.setattr(builtins, "input", lambda _: "y")
    run_guardrails(cfg, tmp_path / "outputs" / "my-run")
    return capsys.readouterr().out


def test_shows_the_llamafactory_block_not_trl_defaults(tmp_path, monkeypatch, capsys):
    cfg = _llamafactory_config(
        stage="sft",
        finetuning_type="full",
        template="qwen3",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
    )

    out = _review(cfg, tmp_path, monkeypatch, capsys)

    assert "org/lf-model" in out
    assert "llamafactory 0.9.5" in out
    assert "a, b" in out
    assert "2 per device  ×  8 GPUs  ×  3 grad-acc  =  48 effective" in out
    # TRL defaults that have no effect on a LlamaFactory run.
    assert PostTrainingConfig().model.name_or_path not in out
    assert "Liger kernel" not in out
    assert "Inference ckpts" not in out


def test_marks_unset_keys_with_their_defaults(tmp_path, monkeypatch, capsys):
    out = _review(_llamafactory_config(), tmp_path, monkeypatch, capsys)

    assert "lora  (default)" in out
    assert "3.0  (default)" in out
    assert "8 per device  ×  8 GPUs  ×  1 grad-acc  =  64 effective" in out


def test_max_steps_replaces_epochs(tmp_path, monkeypatch, capsys):
    out = _review(_llamafactory_config(max_steps=1000), tmp_path, monkeypatch, capsys)

    assert "1,000 steps" in out
    assert "Epochs" not in out


@pytest.mark.parametrize("stage", ["rm", "ppo"])
@pytest.mark.parametrize("finetuning_type", ["full", "freeze"])
def test_warns_when_the_stage_cannot_resume(tmp_path, monkeypatch, capsys, stage, finetuning_type):
    cfg = _llamafactory_config(stage=stage, finetuning_type=finetuning_type)

    out = _review(cfg, tmp_path, monkeypatch, capsys)

    assert f"cannot resume stage {stage} with {finetuning_type} tuning" in out


def test_warns_when_no_checkpoints_are_saved(tmp_path, monkeypatch, capsys):
    out = _review(_llamafactory_config(save_strategy="no"), tmp_path, monkeypatch, capsys)

    assert "no checkpoints" in out


def test_resumes_from_run_dir_checkpoints(tmp_path, monkeypatch, capsys):
    cfg = _llamafactory_config(stage="dpo", finetuning_type="full")

    out = _review(cfg, tmp_path, monkeypatch, capsys)

    assert f"from the last checkpoint in {tmp_path / 'outputs' / 'my-run' / 'checkpoints'}" in out
    assert "Preference loss" in out


def _write_run_inputs(repo_dir, registry: str) -> None:
    (repo_dir / "data").mkdir(parents=True, exist_ok=True)
    (repo_dir / "data" / "dataset_info.json").write_text(registry)


def test_no_frozen_input_warning_for_a_new_run_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_run_inputs(tmp_path, '{"a": {"file_name": "a.json"}}')

    out = _review(_llamafactory_config(), tmp_path, monkeypatch, capsys)

    assert "Frozen input" not in out


def test_warns_about_a_kept_frozen_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_run_inputs(tmp_path, '{"a": {"file_name": "a.json"}}')
    cfg = _llamafactory_config()
    LlamaFactoryBackend().post_freeze(cfg, tmp_path / "outputs" / "my-run")

    out = _review(cfg, tmp_path, monkeypatch, capsys)

    assert "llamafactory/dataset_info.json already exists in the run directory" in out
    assert "will NOT be replaced" in out


def test_warns_when_a_kept_frozen_registry_differs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_run_inputs(tmp_path, '{"a": {"file_name": "a.json"}}')
    cfg = _llamafactory_config()
    LlamaFactoryBackend().post_freeze(cfg, tmp_path / "outputs" / "my-run")
    _write_run_inputs(tmp_path, '{"a": {"hf_hub_url": "org/a"}}')

    out = _review(cfg, tmp_path, monkeypatch, capsys)

    source = tmp_path.resolve() / "data" / "dataset_info.json"
    assert f"llamafactory/dataset_info.json differs from {source}" in out
    assert "the frozen copy will NOT be replaced" in out
