"""Tests for selected dataset transforms."""

import json

import pytest
from datasets import Dataset

from post_training.chat_templates.registry import get_chat_template
from post_training.config import DataConfig, DatasetEntry
from post_training.data import loader
from post_training.data.transforms import (
    _DOLCI_TOOL_PROMPT,
    aegis_safety,
    dolci_tool_use,
    helpsteer2,
    helpsteer3_preference,
)
from post_training.methods.sft import MESSAGES_FEATURES


def test_aegis_safety_keeps_unsafe_non_redacted_response():
    unsafe = {
        "prompt": "How do I do something unsafe?",
        "response": "Unsafe answer",
        "response_label": "unsafe",
    }
    safe = {
        "prompt": "How do I do something benign?",
        "response": "Safe answer",
        "response_label": "safe",
    }

    assert aegis_safety(unsafe)["messages"][1]["content"] == "Unsafe answer"
    assert aegis_safety(safe) == {"messages": []}
    assert aegis_safety({**unsafe, "prompt": "REDACTED"}) == {"messages": []}


def test_helpsteer2_preference_strength_selects_response():
    example = {
        "prompt": "Pick one",
        "response_1": "first",
        "response_2": "second",
    }

    assert helpsteer2({**example, "preference_strength": 1})["messages"][1]["content"] == "second"
    assert helpsteer2({**example, "preference_strength": 0})["messages"][1]["content"] == "first"
    assert helpsteer2({**example, "preference_strength": -1})["messages"][1]["content"] == "first"


def test_helpsteer3_preference_selects_response():
    example = {
        "context": [{"role": "user", "content": "Pick one"}],
        "response1": "first",
        "response2": "second",
    }

    assert (
        helpsteer3_preference({**example, "overall_preference": 1})["messages"][-1]["content"]
        == "second"
    )
    assert (
        helpsteer3_preference({**example, "overall_preference": 0})["messages"][-1]["content"]
        == "first"
    )


_DOLCI_TOOLS = json.dumps(
    [
        {
            "type": "function",
            "function": {"name": "wx.get", "parameters": {"properties": {"q": {"type": "string"}}}},
        }
    ]
)


def _dolci_row(function_calls, content="", system=_DOLCI_TOOL_PROMPT):
    return {
        "messages": [
            {
                "role": "system",
                "content": system,
                "functions": _DOLCI_TOOLS,
                "function_calls": None,
            },
            {"role": "user", "content": "Weather?", "functions": None, "function_calls": "null"},
            {
                "role": "assistant",
                "content": content,
                "functions": None,
                "function_calls": function_calls,
            },
            {
                "role": "environment",
                "content": '{"temp": 5}',
                "functions": None,
                "function_calls": None,
            },
            {"role": "assistant", "content": "It is 5.", "functions": None, "function_calls": None},
        ]
    }


def _arguments(out):
    return [json.loads(c["function"]["arguments"]) for c in out["messages"][1]["tool_calls"]]


def test_dolci_tool_use_converts_to_standard_format():
    out = dolci_tool_use(
        _dolci_row('wx.get(q="Oslo", metric=true)\nwx.get(q="Bergen", days=[1, null])')
    )

    assert out["tools"] == _DOLCI_TOOLS
    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert out["messages"][1]["tool_calls"][0] == {
        "type": "function",
        "function": {"name": "wx.get", "arguments": '{"q": "Oslo", "metric": true}'},
    }
    assert _arguments(out)[1] == {"q": "Bergen", "days": [1, None]}
    assert out["messages"][2] == {"role": "tool", "content": '{"temp": 5}'}


@pytest.mark.parametrize(
    ("function_calls", "expected"),
    [
        ("wx.get(minutes=12*60, rates=[0.5]*2)", [{"minutes": 720, "rates": [0.5, 0.5]}]),
        ("wx.get(chars=[c for c in 'ab'])", [{"chars": ["a", "b"]}]),
        ('Rates API(from="EUR", to="USD")', [{"from": "EUR", "to": "USD"}]),
        ("# look up both\nwx.get(q='a')\n\nwx.get(q='b')", [{"q": "a"}, {"q": "b"}]),
        ("wx.get(q='a')\\nwx.get(q='b')", [{"q": "a"}, {"q": "b"}]),
        ("wx.get(q='a',\n       days=2)", [{"q": "a", "days": 2}]),
        ('wx.get(q="(()")\nwx.get(q="())()")', [{"q": "(()"}, {"q": "())()"}]),
        ("wx.get(q='a')  # (may fail)", [{"q": "a"}]),
        ("wx.get(q='#a')\n# skip\nwx.get(q='b')", [{"q": "#a"}, {"q": "b"}]),
        ('wx.get(q="a)\\nb")', [{"q": "a)\nb"}]),
        ("wx.get(q='a')\nnull", [{"q": "a"}]),
        (
            'Rates API(from="EUR", q="(")\nRates API(from="USD", q=")")',
            [{"from": "EUR", "q": "("}, {"from": "USD", "q": ")"}],
        ),
        (
            'wx.get(html="<div class=\'a\'>", q="from=x")',
            [{"html": "<div class='a'>", "q": "from=x"}],
        ),
        ("wx.get(flags=[true, false]*2)", [{"flags": [True, False, True, False]}]),
        ("wx.get(q='a')\r\nwx.get(q='b')", [{"q": "a"}, {"q": "b"}]),
        ("Get Player's Media(id=1)\nGet Player's Media(id=2)", [{"id": 1}, {"id": 2}]),
    ],
)
def test_dolci_tool_use_repairs_call_lines(function_calls, expected):
    assert _arguments(dolci_tool_use(_dolci_row(function_calls))) == expected


@pytest.mark.parametrize(
    "function_calls",
    [
        "wx.get(q=X)",
        "wx.get(q=[random numbers])",
        "wx.get(q=other(q=1))",
        "min([2, 4, 6])",
        "wx.get(q=(1, 2))",
        "wx.get(q={1: 'a'})",
        "wx.get(q={[]: 1})",
        "wx.get(q=2**10)",
        'say"hi(q=1)',
        "wx.get",
    ],
)
def test_dolci_tool_use_drops_row_with_unconvertible_call(function_calls):
    assert dolci_tool_use(_dolci_row(function_calls)) == {"messages": [], "tools": None}


def test_dolci_tool_use_restores_names_that_lost_an_escaped_letter():
    # Dolci decoded "\reverse" as "\r" + "everse"; "\next" became a plain line break.
    row = _dolci_row("wx.get(q='a')\reverse(s='x')\x0clatten(x=1)\next(n=2)")
    row["messages"][0]["functions"] = json.dumps(
        [
            {"type": "function", "function": {"name": n}}
            for n in ("wx.get", "reverse", "flatten", "next")
        ]
    )
    calls = dolci_tool_use(row)["messages"][1]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["wx.get", "reverse", "flatten", "next"]


def test_dolci_tool_use_null_call_keeps_refusal_and_drops_empty_turn():
    refusal = dolci_tool_use(_dolci_row("null", content="I can't do that."))
    assert refusal["messages"][1] == {"role": "assistant", "content": "I can't do that."}

    empty = dolci_tool_use(_dolci_row("null"))
    assert [m["role"] for m in empty["messages"]] == ["user", "tool", "assistant"]


def test_dolci_tool_use_keeps_task_instructions_in_system_prompt():
    system = (
        _DOLCI_TOOL_PROMPT
        + " You do research.\n\n- Use tools. Format calls like: tool_name(query=...) (below)."
    )
    out = dolci_tool_use(_dolci_row("wx.get(q='a')", system=system))

    assert out["messages"][0] == {
        "role": "system",
        "content": "You do research.\n\n- Use tools. (below).",
    }


def test_dolci_tool_use_passes_rows_without_tools():
    row = {
        "messages": [{"role": "user", "content": "Hi", "functions": None, "function_calls": None}]
    }

    assert dolci_tool_use(row) == {"messages": [{"role": "user", "content": "Hi"}], "tools": None}


def test_dolci_tool_use_survives_features_cast_and_renders_qwen3_tool_call(monkeypatch):
    """The transform output must keep `tool_calls` through the loader's
    MESSAGES_FEATURES cast and render as a Qwen3 / hermes `<tool_call>`."""
    plain = {
        "messages": [{"role": "user", "content": "Hi", "functions": None, "function_calls": None}]
    }
    rows = [_dolci_row('wx.get(q="Oslo")'), plain, _dolci_row("wx.get(q=X)")]
    monkeypatch.setattr(loader, "load_dataset", lambda path, **kwargs: Dataset.from_list(rows))

    ds = loader.load_and_mix_datasets(
        DataConfig(
            num_proc=1,
            datasets=[DatasetEntry(name="dolci", path="dolci", transform="dolci_tool_use")],
        ),
        columns_to_keep=["messages", "tools"],
        features=MESSAGES_FEATURES,
    )
    tool_row = next(r for r in ds if r["tools"])
    plain_row = next(r for r in ds if not r["tools"] and r["messages"])

    assert sum(not r["messages"] for r in ds) == 1  # the X placeholder row

    assert plain_row["messages"] == [{"role": "user", "content": "Hi", "tool_calls": None}]
    chat_template_utils = pytest.importorskip("transformers.utils.chat_template_utils")
    compiled = chat_template_utils._compile_jinja_template(get_chat_template("qwen3"))
    rendered, indices = chat_template_utils._render_with_assistant_indices(
        compiled, tool_row["messages"], json.loads(tool_row["tools"]), None, False
    )
    spans = [rendered[start:end] for start, end in indices]

    assert '<tools>\n{"type": "function", "function": {"name": "wx.get"' in rendered
    assert '<tool_response>\n{"temp": 5}\n</tool_response>' in rendered
    assert (
        spans[0]
        == '<tool_call>\n{"name": "wx.get", "arguments": {"q": "Oslo"}}\n</tool_call><|im_end|>'
    )
