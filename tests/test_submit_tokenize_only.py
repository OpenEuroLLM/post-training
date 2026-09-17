"""Tests for the --tokenize-only allocation shrink in scripts/submit.py."""

import importlib.util
from pathlib import Path

import pytest

from post_training.config import PostTrainingConfig

# scripts/ is not a Python package, so load the helper from the file directly.
_SUBMIT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "submit.py"
_spec = importlib.util.spec_from_file_location("submit", _SUBMIT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_shrink_allocation_for_tokenize_only = _mod._shrink_allocation_for_tokenize_only


def _config(method: str, stage: int | None, use_deepspeed: bool = True) -> PostTrainingConfig:
    cfg = PostTrainingConfig()
    cfg.method = method
    cfg.deepspeed = None if stage is None else {"zero_optimization": {"stage": stage}}
    cfg.accelerate.use_deepspeed = use_deepspeed
    cfg.slurm.num_nodes = 4
    cfg.slurm.gpus_per_node = 8
    return cfg


def test_dpo_zero3_keeps_allocation():
    cfg = _config("dpo", stage=3)

    _shrink_allocation_for_tokenize_only(cfg)

    assert (cfg.slurm.num_nodes, cfg.slurm.gpus_per_node) == (1, 8)


@pytest.mark.parametrize(
    "method, stage, use_deepspeed",
    [
        ("dpo", 2, True),
        ("dpo", None, True),
        ("dpo", 3, False),
        ("sft", 3, True),
        ("sft", 2, True),
    ],
)
def test_other_setups_shrink_to_one_gpu(method, stage, use_deepspeed):
    cfg = _config(method, stage=stage, use_deepspeed=use_deepspeed)

    _shrink_allocation_for_tokenize_only(cfg)

    assert (cfg.slurm.num_nodes, cfg.slurm.gpus_per_node) == (1, 1)
