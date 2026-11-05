"""Tests for ResumeCheckCallback (resume sanity check)."""

from types import SimpleNamespace

import pytest
import torch

from post_training.callbacks.resume_check import ResumeCheckCallback


def _trainer():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(2):
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    accelerator = SimpleNamespace(device="cpu", reduce=lambda t, reduction: t)
    return SimpleNamespace(model=model, optimizer=optimizer, accelerator=accelerator)


def _args(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path), process_index=0, world_size=1)


def _state(step):
    return SimpleNamespace(global_step=step, is_world_process_zero=True)


def _saved_callback(tmp_path):
    trainer = _trainer()
    callback = ResumeCheckCallback()
    callback.bind_trainer(trainer)
    callback.on_save(_args(tmp_path), _state(2), None)
    return callback, trainer


def test_on_save_writes_rank_fingerprint(tmp_path):
    _saved_callback(tmp_path)

    assert (tmp_path / "checkpoint-2" / "resume_check" / "rank0.json").exists()


def test_identical_state_passes(tmp_path):
    callback, _ = _saved_callback(tmp_path)

    callback.on_train_begin(_args(tmp_path), _state(2), None)


def test_fresh_run_is_skipped(tmp_path):
    callback = ResumeCheckCallback()
    callback.bind_trainer(_trainer())

    callback.on_train_begin(_args(tmp_path), _state(0), None)


def test_dropped_optimizer_state_fails(tmp_path):
    callback, trainer = _saved_callback(tmp_path)
    trainer.optimizer.state.clear()

    with pytest.raises(RuntimeError, match="resume_check FAILED"):
        callback.on_train_begin(_args(tmp_path), _state(2), None)


def test_changed_weights_fail(tmp_path):
    callback, trainer = _saved_callback(tmp_path)
    with torch.no_grad():
        trainer.model.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="resume_check FAILED"):
        callback.on_train_begin(_args(tmp_path), _state(2), None)


def test_changed_adam_moment_fails(tmp_path):
    callback, trainer = _saved_callback(tmp_path)
    trainer.optimizer.state[trainer.model.bias]["exp_avg"].mul_(2.0)

    with pytest.raises(RuntimeError, match="resume_check FAILED"):
        callback.on_train_begin(_args(tmp_path), _state(2), None)


def test_missing_fingerprint_fails(tmp_path):
    callback = ResumeCheckCallback()
    callback.bind_trainer(_trainer())

    with pytest.raises(RuntimeError, match="resume_check FAILED"):
        callback.on_train_begin(_args(tmp_path), _state(2), None)


def test_callback_built_only_when_enabled(tmp_path):
    from post_training.config import PostTrainingConfig
    from post_training.methods.common import build_callbacks

    cfg = PostTrainingConfig()
    assert not any(isinstance(c, ResumeCheckCallback) for c in build_callbacks(cfg, tmp_path))

    cfg.debug.resume_check = True
    assert any(isinstance(c, ResumeCheckCallback) for c in build_callbacks(cfg, tmp_path))
