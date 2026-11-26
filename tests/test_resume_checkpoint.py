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


def _write_checkpoint(checkpoints_dir: Path, step: int) -> Path:
    """Write the files of a DeepSpeed Trainer checkpoint that the check reads."""
    checkpoint = checkpoints_dir / f"checkpoint-{step}"
    (checkpoint / f"global_step{step}").mkdir(parents=True)
    _write_safetensors(checkpoint / "model.safetensors")
    _write_torch_file(
        checkpoint / f"global_step{step}" / "zero_pp_rank_0_mp_rank_00_optim_states.pt"
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
