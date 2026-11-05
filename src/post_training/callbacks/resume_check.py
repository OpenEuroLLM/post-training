"""Sanity check that a resume restores the saved weights and optimizer state.

Enable with ``debug.resume_check: true`` and run the same config twice under a
fixed ``run_name``:

1. Train to a checkpoint (e.g. ``training.max_steps=2 checkpointing.save_steps=2``).
   At each save, every rank writes a fingerprint of its local parameter shards
   and optimizer state to ``<checkpoint>/resume_check/rank<N>.json``.
2. Run again with a larger ``training.max_steps``. ``scripts/train.py`` resumes
   from the last checkpoint in a fresh process; before the first update, every
   rank recomputes the fingerprint and compares it with the saved one. The job
   fails if any value differs.

The fingerprints are per rank, so phase 2 must use the same world size and FSDP
or DeepSpeed layout as phase 1.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# Saved and reloaded values should be bit-identical; the tolerance only absorbs
# float64 reduction-order noise.
_REL_TOL = 1e-9


def _local(t: torch.Tensor) -> torch.Tensor:
    """Return this rank's shard (FSDP2 DTensor) or the tensor itself."""
    return t.to_local() if hasattr(t, "to_local") else t


def _stats(t: torch.Tensor) -> dict[str, float]:
    t = _local(t).detach().double()
    return {"sum": t.sum().item(), "sq": t.square().sum().item()}


def _fingerprint(trainer: Any) -> list[dict[str, Any]]:
    """Fingerprint every parameter the optimizer holds, in optimizer order."""
    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    out = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            entry: dict[str, Any] = {"numel": _local(p).numel(), "param": _stats(p), "state": {}}
            for key, value in optimizer.state.get(p, {}).items():
                if torch.is_tensor(value) and _local(value).numel() > 1:
                    entry["state"][key] = _stats(value)
                elif torch.is_tensor(value):
                    entry["state"][key] = {"value": _local(value).double().item()}
                else:
                    entry["state"][key] = {"value": float(value)}
            out.append(entry)
    return out


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=1e-12)


def _compare(saved: list[dict], loaded: list[dict]) -> list[str]:
    """Return one line per difference between the saved and loaded fingerprints."""
    if len(saved) != len(loaded):
        return [f"parameter count differs: saved {len(saved)}, loaded {len(loaded)}"]
    problems = []
    for i, (old, new) in enumerate(zip(saved, loaded)):
        if old["numel"] != new["numel"]:
            problems.append(
                f"param {i}: local numel {old['numel']} -> {new['numel']} (layout changed)"
            )
            continue
        if not all(_close(old["param"][k], new["param"][k]) for k in old["param"]):
            problems.append(f"param {i}: weights differ ({old['param']} -> {new['param']})")
        missing = sorted(set(old["state"]) - set(new["state"]))
        if missing:
            problems.append(f"param {i}: optimizer state missing {missing}")
        for key in sorted(set(old["state"]) & set(new["state"])):
            if not all(
                _close(old["state"][key][k], new["state"][key][k]) for k in old["state"][key]
            ):
                problems.append(
                    f"param {i}: optimizer {key} differs ({old['state'][key]} -> {new['state'][key]})"
                )
    return problems


class ResumeCheckCallback(TrainerCallback):
    """Write fingerprints at each save; verify them after a resume."""

    def __init__(self) -> None:
        # Set by ``bind_trainer``: callbacks get the pre-``prepare`` model, and the
        # optimizer state lives only on the trainer.
        self.trainer: Any = None

    def bind_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def _rank_file(self, args, step: int) -> Path:
        return (
            Path(args.output_dir)
            / f"checkpoint-{step}"
            / "resume_check"
            / f"rank{args.process_index}.json"
        )

    def on_save(self, args, state, control, **kwargs) -> None:
        path = self._rank_file(args, state.global_step)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_fingerprint(self.trainer)))
        if state.is_world_process_zero:
            logger.info("resume_check: wrote fingerprints for step %d", state.global_step)

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        # global_step > 0 means the trainer has just loaded a checkpoint.
        if state.global_step == 0:
            return
        path = self._rank_file(args, state.global_step)
        if not path.exists():
            problems = [f"no fingerprint at {path}; was phase 1 run with debug.resume_check?"]
        else:
            problems = _compare(json.loads(path.read_text()), _fingerprint(self.trainer))

        for line in problems[:20]:
            logger.error("resume_check rank %d: %s", args.process_index, line)
        if len(problems) > 20:
            logger.error(
                "resume_check rank %d: ... %d more", args.process_index, len(problems) - 20
            )

        failed = torch.tensor([len(problems)], device=self.trainer.accelerator.device)
        failed = self.trainer.accelerator.reduce(failed, reduction="sum").item()
        if failed:
            raise RuntimeError(
                f"resume_check FAILED at step {state.global_step}: {failed} difference(s) "
                "between the saved and the resumed state (see the per-rank errors above)."
            )
        if state.is_world_process_zero:
            logger.info(
                "resume_check PASSED at step %d: weights and optimizer state match on all "
                "%d ranks.",
                state.global_step,
                args.world_size,
            )
