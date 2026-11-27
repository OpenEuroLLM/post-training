"""Tests for the selection of the checkpoint that scripts/train.py resumes from."""

import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest

# scripts/ is not a Python package, so load the script by path.
_TRAIN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train", _TRAIN_PATH)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def _write_safetensors(path: Path) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))


def _write_torch_file(path: Path) -> None:
    """torch.save writes a zip archive; a zip with one member is enough for the check."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"data")


def _write_checkpoint(checkpoints_dir: Path, step: int, ranks=(0,), prefix: str = "") -> Path:
    """Write the files of a DeepSpeed Trainer checkpoint that the check reads."""
    checkpoint = checkpoints_dir / f"checkpoint-{step}"
    (checkpoint / f"global_step{step}").mkdir(parents=True)
    _write_safetensors(checkpoint / "model.safetensors")
    for rank in ranks:
        _write_torch_file(
            checkpoint
            / f"global_step{step}"
            / f"{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        )
    _write_torch_file(checkpoint / "rng_state_0.pth")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return checkpoint


def _truncate(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-4])


def test_empty_directory_starts_from_scratch(tmp_path):
    assert train._last_complete_checkpoint(tmp_path) is None


def test_newest_complete_checkpoint_is_selected(tmp_path):
    _write_checkpoint(tmp_path, 250)
    newest = _write_checkpoint(tmp_path, 1000)
    _write_checkpoint(tmp_path, 500)
    (tmp_path / "checkpoint-notes").mkdir()

    assert train._last_complete_checkpoint(tmp_path) == str(newest)


def test_checkpoint_without_trainer_state_is_ignored(tmp_path):
    older = _write_checkpoint(tmp_path, 250)
    (_write_checkpoint(tmp_path, 500) / "trainer_state.json").unlink()

    assert train._last_complete_checkpoint(tmp_path) == str(older)


def test_checkpoint_with_partial_trainer_state_is_ignored(tmp_path):
    older = _write_checkpoint(tmp_path, 250)
    (_write_checkpoint(tmp_path, 500) / "trainer_state.json").write_text('{"global_step": 5')

    assert train._last_complete_checkpoint(tmp_path) == str(older)


@pytest.mark.parametrize(
    "truncated",
    [
        "model.safetensors",
        "global_step500/zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "rng_state_0.pth",
    ],
)
def test_checkpoint_with_truncated_file_is_ignored(tmp_path, truncated):
    older = _write_checkpoint(tmp_path, 250)
    _truncate(_write_checkpoint(tmp_path, 500) / truncated)

    assert train._last_complete_checkpoint(tmp_path) == str(older)


def test_incomplete_checkpoint_stays_on_disk(tmp_path):
    _write_checkpoint(tmp_path, 250)
    incomplete = _write_checkpoint(tmp_path, 500)
    (incomplete / "trainer_state.json").unlink()

    train._last_complete_checkpoint(tmp_path)

    assert (incomplete / "model.safetensors").exists()


def test_no_complete_checkpoint_is_an_error(tmp_path):
    """A silent start from step 0 would discard the progress of the run."""
    (_write_checkpoint(tmp_path, 250) / "trainer_state.json").unlink()

    with pytest.raises(RuntimeError, match="none of them is complete"):
        train._last_complete_checkpoint(tmp_path)


# ---------------------------------------------------------------------------
# DeepSpeed ZeRO: one optimizer file for each rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", "bf16_"])
def test_checkpoint_with_the_optimizer_files_of_all_ranks_is_selected(tmp_path, prefix):
    _write_checkpoint(tmp_path, 250, ranks=range(4), prefix=prefix)
    newest = _write_checkpoint(tmp_path, 500, ranks=range(4), prefix=prefix)

    assert train._last_complete_checkpoint(tmp_path, world_size=4) == str(newest)
    assert train._last_complete_checkpoint(tmp_path) == str(newest)


def test_checkpoint_without_the_optimizer_file_of_the_last_rank_is_ignored(tmp_path):
    """Rank 0 wrote trainer_state.json before rank 3 created its file."""
    older = _write_checkpoint(tmp_path, 250, ranks=range(4))
    _write_checkpoint(tmp_path, 500, ranks=range(3))

    assert train._last_complete_checkpoint(tmp_path, world_size=4) == str(older)


def test_gap_in_the_ranks_is_found_without_the_world_size(tmp_path):
    older = _write_checkpoint(tmp_path, 250, ranks=range(4))
    _write_checkpoint(tmp_path, 500, ranks=(0, 2, 3))

    assert train._last_complete_checkpoint(tmp_path) == str(older)


def test_checkpoint_without_deepspeed_files_does_not_need_the_world_size(tmp_path):
    checkpoint = _write_checkpoint(tmp_path, 250, ranks=())

    assert train._last_complete_checkpoint(tmp_path, world_size=32) == str(checkpoint)
