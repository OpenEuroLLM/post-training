"""Regression tests for the ``scripts/data.py`` CLI commands.

PR #68 made the SFT trainer condition each row on a thinking mode through a
per-row ``chat_template_kwargs`` column, and it renders rows with their
``tools`` declarations.  ``scripts/data.py`` must render rows the same way:
``inspect`` and ``token-stats`` showed prompts and token counts that differed
from training (issue #74), dropped the ``tools`` column, and ``inspect`` did
the whole instruction assignment even for raw-only runs.

No tokenizer is downloaded: ``_CharTokenizer`` renders the real registered
``qwen3`` template through transformers' own generation tracker and treats one
character as one token, so expected counts can be computed through the very
same trainer path (``load_and_mix_datasets`` + ``_filter_sft_rows`` as
``build_sft_trainer`` calls them).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from functools import partial
from pathlib import Path

import pytest
import yaml

from post_training.chat_templates.registry import get_chat_template
from post_training.data.loader import load_and_mix_datasets
from post_training.methods.sft import MESSAGES_FEATURES, _filter_sft_rows

_COMPILED: dict[str, object] = {}


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "data.py"
    spec = importlib.util.spec_from_file_location("data_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_script = _load_script()


class _CharTokenizer:
    """Stand-in for a HF tokenizer, with one character per token."""

    def __init__(self, template_name: str) -> None:
        self.template_name = template_name

    @property
    def chat_template(self) -> str:
        return get_chat_template(self.template_name)

    def _render(self, conversation, tools=None, add_generation_prompt=False, **kwargs):
        utils = pytest.importorskip("transformers.utils.chat_template_utils")
        if self.template_name not in _COMPILED:
            _COMPILED[self.template_name] = utils._compile_jinja_template(self.chat_template)
        return utils._render_with_assistant_indices(
            _COMPILED[self.template_name],
            conversation,
            tools,
            None,
            add_generation_prompt,
            **kwargs,
        )

    def apply_chat_template(
        self,
        conversation,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        desc=None,
        **kwargs,
    ):
        rendered, indices = self._render(conversation, tools, add_generation_prompt, **kwargs)
        if not tokenize:
            return rendered
        input_ids = [ord(character) for character in rendered]
        result = {"input_ids": input_ids}
        if return_dict or return_assistant_tokens_mask:
            masks = [0] * len(input_ids)
            for start, end in indices:
                for position in range(start, end):
                    masks[position] = 1
            result["assistant_masks"] = masks
        return result

    def decode(self, token_ids, **kwargs) -> str:
        return "".join(chr(tid) for tid in token_ids)


@pytest.fixture
def tokenizer():
    return _CharTokenizer("qwen3")


@pytest.fixture(autouse=True)
def _fake_tokenizer(monkeypatch, tokenizer):
    monkeypatch.setattr("post_training.methods.common.build_tokenizer", lambda config: tokenizer)


DIRECT_ROW = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
}
REASONING_ROW = {
    "messages": [
        {"role": "user", "content": "What is 6*7?"},
        {"role": "assistant", "content": "<think>6x7=42.</think>42"},
    ]
}
TOOLS = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Calculator",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
            },
        }
    ]
)


def _write_dataset(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    ds_dir = tmp_path / name
    ds_dir.mkdir()
    (ds_dir / "data.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return ds_dir


def _write_config(tmp_path: Path, ds_dir: Path, transform: str | None = None) -> Path:
    config = {
        "method": "sft",
        "backend": "trl",
        "container": None,
        "deepspeed": None,
        "model": {"name_or_path": "dummy/model"},
        "training": {
            "max_steps": 1,
            "effective_batch_size": 1,
            "per_device_train_batch_size": 1,
        },
        "data": {
            "chat_template": "qwen3",
            "num_proc": 1,
            "seed": 42,
            "datasets": [
                {
                    "name": "test",
                    "path": str(ds_dir),
                    "split": "train",
                    "weight": 1.0,
                    "transform": transform,
                }
            ],
        },
        "sft": {"max_seq_length": 4096},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _inspect_args(config_path: Path, num_samples: int = 2, **flags) -> argparse.Namespace:
    toggles = {
        "show_raw": False,
        "show_transformed": False,
        "show_formatted": False,
        "show_tokens": False,
        "show_stats": False,
    }
    toggles.update(flags)
    return argparse.Namespace(config=str(config_path), num_samples=num_samples, **toggles)


def _token_total(output: str) -> int:
    match = re.search(r"Total number of tokens: (\d+)", output)
    assert match is not None, output
    return int(match.group(1))


def _trainer_token_total(config_path: Path, tokenizer, overrides: list[str]) -> int:
    """Token total the trainer would see, through the same filter call as build_sft_trainer."""
    from post_training.config import PostTrainingConfig

    config = PostTrainingConfig.load(config_path, overrides)
    mc = config.sft
    conditioning = config.thinking_conditioning
    omission = conditioning.default_mode_omission_probability if conditioning.enable else None
    ds = load_and_mix_datasets(
        config.data,
        dataset_filter_fn=partial(
            _filter_sft_rows,
            tokenizer=tokenizer,
            max_length=mc.max_seq_length,
            truncated_span_action=mc.truncated_span_action,
            thinking_omission_probability=omission,
        ),
        columns_to_keep=["messages", "tools"],
        features=MESSAGES_FEATURES,
    )
    total = 0
    for row in ds:
        tools = json.loads(row["tools"]) if isinstance(row.get("tools"), str) else row.get("tools")
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
            **(row.get("chat_template_kwargs") or {}),
        )
        total += len(rendered["input_ids"])
    return total


# ── token-stats ──────────────────────────────────────────────────────────────


def test_token_stats_matches_trainer_with_thinking_conditioning(tmp_path, capsys, tokenizer):
    ds_dir = _write_dataset(tmp_path, "ds", [DIRECT_ROW, REASONING_ROW])
    config_path = _write_config(tmp_path, ds_dir)

    args = argparse.Namespace(command="token-stats", config=str(config_path))
    data_script._run_count_tokens(args, ["thinking_conditioning.enable=true"])
    with_conditioning = _token_total(capsys.readouterr().out)

    data_script._run_count_tokens(args, ["thinking_conditioning.enable=false"])
    without_conditioning = _token_total(capsys.readouterr().out)

    assert with_conditioning == _trainer_token_total(
        config_path, tokenizer, ["thinking_conditioning.enable=true"]
    )
    assert without_conditioning == _trainer_token_total(
        config_path, tokenizer, ["thinking_conditioning.enable=false"]
    )
    # The /no_think system block of the direct-answer row is part of training.
    assert with_conditioning > without_conditioning


def test_token_stats_counts_tool_declarations(tmp_path, capsys, tokenizer):
    rows = [dict(DIRECT_ROW, tools=TOOLS), REASONING_ROW]
    ds_dir = _write_dataset(tmp_path, "ds", rows)
    config_path = _write_config(tmp_path, ds_dir)

    args = argparse.Namespace(command="token-stats", config=str(config_path))
    data_script._run_count_tokens(args, ["thinking_conditioning.enable=true"])
    with_tools = _token_total(capsys.readouterr().out)

    assert with_tools == _trainer_token_total(
        config_path, tokenizer, ["thinking_conditioning.enable=true"]
    )
    # Without the tools column the "# Tools" system preamble is missing.
    no_tools_dir = _write_dataset(tmp_path, "ds_notools", [DIRECT_ROW, REASONING_ROW])
    no_tools_root = tmp_path / "notools"
    no_tools_root.mkdir()
    no_tools_config = _write_config(no_tools_root, no_tools_dir)
    data_script._run_count_tokens(
        argparse.Namespace(command="token-stats", config=str(no_tools_config)),
        ["thinking_conditioning.enable=true"],
    )
    assert with_tools > _token_total(capsys.readouterr().out)


def test_token_stats_with_transform_matches_trainer(tmp_path, capsys, tokenizer):
    rows = [
        {"instruction": "What is 2+2?", "response": "4"},
        {"instruction": "What is 6*7?", "response": "<think>6x7=42.</think>42"},
    ]
    ds_dir = _write_dataset(tmp_path, "ds", rows)
    config_path = _write_config(tmp_path, ds_dir, transform="instruction_response_to_messages")

    args = argparse.Namespace(command="token-stats", config=str(config_path))
    data_script._run_count_tokens(args, ["thinking_conditioning.enable=true"])
    total = _token_total(capsys.readouterr().out)

    assert total == _trainer_token_total(
        config_path, tokenizer, ["thinking_conditioning.enable=true"]
    )


# ── inspect ──────────────────────────────────────────────────────────────────


def test_inspect_renders_thinking_instruction(tmp_path, capsys):
    ds_dir = _write_dataset(tmp_path, "ds", [DIRECT_ROW, REASONING_ROW])
    config_path = _write_config(tmp_path, ds_dir)

    data_script._run_inspect(
        _inspect_args(config_path, show_formatted=True), ["thinking_conditioning.enable=true"]
    )
    out = capsys.readouterr().out
    assert "thinking_instruction='no_think'" in out
    assert "<|im_start|>system\n/no_think<|im_end|>" in out

    data_script._run_inspect(
        _inspect_args(config_path, show_formatted=True), ["thinking_conditioning.enable=false"]
    )
    out = capsys.readouterr().out
    assert "thinking_instruction" not in out
    assert "/no_think" not in out


def test_inspect_renders_tool_declarations(tmp_path, capsys):
    rows = [dict(DIRECT_ROW, tools=TOOLS), REASONING_ROW]
    ds_dir = _write_dataset(tmp_path, "ds", rows)
    config_path = _write_config(tmp_path, ds_dir)

    data_script._run_inspect(_inspect_args(config_path, show_formatted=True), [])
    out = capsys.readouterr().out
    assert "# Tools" in out
    assert "calc" in out


def test_inspect_assigns_instructions_on_transformed_rows(tmp_path, capsys):
    rows = [
        {"instruction": "What is 2+2?", "response": "4"},
        {"instruction": "What is 6*7?", "response": "<think>6x7=42.</think>42"},
    ]
    ds_dir = _write_dataset(tmp_path, "ds", rows)
    config_path = _write_config(tmp_path, ds_dir, transform="instruction_response_to_messages")

    data_script._run_inspect(
        _inspect_args(config_path, show_formatted=True), ["thinking_conditioning.enable=true"]
    )
    out = capsys.readouterr().out
    assert "thinking_instruction='no_think'" in out
    assert "<|im_start|>system\n/no_think<|im_end|>" in out


def test_inspect_raw_only_skips_the_instruction_assignment(tmp_path, capsys, monkeypatch):
    ds_dir = _write_dataset(tmp_path, "ds", [DIRECT_ROW, REASONING_ROW])
    config_path = _write_config(tmp_path, ds_dir)

    calls = []
    from post_training.methods.sft import _with_thinking_instructions as real

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr("post_training.methods.sft._with_thinking_instructions", spy)

    data_script._run_inspect(
        _inspect_args(config_path, show_raw=True), ["thinking_conditioning.enable=true"]
    )
    capsys.readouterr()
    assert calls == []

    data_script._run_inspect(
        _inspect_args(config_path, show_formatted=True), ["thinking_conditioning.enable=true"]
    )
    capsys.readouterr()
    assert len(calls) == 1
