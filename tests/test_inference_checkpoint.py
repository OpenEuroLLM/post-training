"""Tests for InferenceCheckpointCallback, including the FSDP gather path."""

from types import SimpleNamespace

import pytest

from post_training.callbacks.inference_checkpoint import InferenceCheckpointCallback


class _FakeModel:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, path, state_dict=None):
        self.saved.append((path, state_dict))


class _FakeAccelerator:
    def __init__(self, inner):
        self.inner = inner
        self.gathered = 0

    def get_state_dict(self, model):
        self.gathered += 1
        return {"w": "full"}

    def unwrap_model(self, model):
        return self.inner


def _state(step=10, rank_zero=True):
    return SimpleNamespace(global_step=step, max_steps=100, is_world_process_zero=rank_zero)


def _fsdp_trainer():
    inner = _FakeModel()
    return (
        SimpleNamespace(is_fsdp_enabled=True, model=object(), accelerator=_FakeAccelerator(inner)),
        inner,
    )


def test_saves_callback_model_without_fsdp(tmp_path, monkeypatch):
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    callback = InferenceCheckpointCallback(save_steps=10, output_dir=tmp_path)
    model = _FakeModel()

    callback.on_step_end(None, _state(), None, model=model)

    assert model.saved == [(tmp_path / "step-10", None)]


def test_fsdp_gathers_and_rank_zero_saves_full_state_dict(tmp_path):
    callback = InferenceCheckpointCallback(save_steps=10, output_dir=tmp_path)
    trainer, inner = _fsdp_trainer()
    callback.bind_trainer(trainer)
    shard_model = _FakeModel()

    callback.on_step_end(None, _state(), None, model=shard_model)

    assert trainer.accelerator.gathered == 1
    assert inner.saved == [(tmp_path / "step-10", {"w": "full"})]
    assert shard_model.saved == []


def test_fsdp_non_zero_rank_joins_gather_but_does_not_save(tmp_path):
    callback = InferenceCheckpointCallback(save_steps=10, output_dir=tmp_path)
    trainer, inner = _fsdp_trainer()
    callback.bind_trainer(trainer)

    callback.on_step_end(None, _state(rank_zero=False), None, model=_FakeModel())

    assert trainer.accelerator.gathered == 1
    assert inner.saved == []


def test_fsdp_skips_gather_between_save_steps(tmp_path):
    callback = InferenceCheckpointCallback(save_steps=10, output_dir=tmp_path)
    trainer, _ = _fsdp_trainer()
    callback.bind_trainer(trainer)

    callback.on_step_end(None, _state(step=7), None, model=_FakeModel())

    assert trainer.accelerator.gathered == 0


def test_unbound_callback_under_fsdp_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    callback = InferenceCheckpointCallback(save_steps=10, output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="bind_trainer"):
        callback.on_step_end(None, _state(), None, model=_FakeModel())
