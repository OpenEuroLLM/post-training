"""Render-parity tests: registered templates vs. their pristine upstream.

Splicing `{% generation %}` markers into a vendored chat template is only
safe if it changes *nothing* about what the template renders — the markers
carve up the loss mask, they must not move a single byte of the prompt.  A
template that renders even slightly differently from the one the base model
was trained with is a silent train/inference skew.

`tests/fixtures/qwen3-8b-upstream.jinja` is the unmodified `chat_template`
field from `Qwen/Qwen3-8B`'s `tokenizer_config.json` (4168 bytes, no
trailing newline; the repo publishes no standalone `chat_template.jinja`).
It is a reference copy — never edit it.  To refresh it after an upstream
change, re-extract it verbatim and update `_UPSTREAM_SHA256`:

    python -c "import json,urllib.request as u; \\
        print(json.load(u.urlopen('https://huggingface.co/Qwen/Qwen3-8B/\\
resolve/main/tokenizer_config.json'))['chat_template'], end='')" \\
        > tests/fixtures/qwen3-8b-upstream.jinja
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from post_training.chat_templates.registry import (
    get_chat_template,
    has_generation_markers,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_UPSTREAM = _FIXTURES / "qwen3-8b-upstream.jinja"

# Guards the reference copy against silent edits — including an editor or
# hook appending a trailing newline, which would change what it renders.
_UPSTREAM_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


# ── conversation matrix ────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name."}},
                "required": ["city"],
            },
        },
    }
]

_CONVERSATIONS: dict[str, list[dict]] = {
    "single-turn-reasoning": [
        {"role": "user", "content": "What is 17 * 23?"},
        {"role": "assistant", "content": "<think>\n340 + 51.\n</think>\n\n391."},
    ],
    "multi-turn": [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<think>\nR1\n</think>\n\nA1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<think>\nR2\n</think>\n\nA2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "<think>\nR3\n</think>\n\nA3"},
    ],
    "no-think": [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And Japan?"},
        {"role": "assistant", "content": "Tokyo."},
    ],
    "tool-call": [
        {"role": "user", "content": "Weather in Berlin?"},
        {
            "role": "assistant",
            "content": "<think>\nCall the tool.\n</think>\n\n",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Berlin"}},
                }
            ],
        },
        {"role": "tool", "content": '{"temp_c": 18, "sky": "clear"}'},
        {"role": "assistant", "content": "18C and clear."},
    ],
    "system+tools": [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Weather in Berlin?"},
        {"role": "assistant", "content": "<think>\nLooks clear.\n</think>\n\nSunny."},
    ],
}


def _render(template: str, messages: list[dict], tools: list[dict] | None, agp: bool) -> str:
    """Render *template* through transformers' own chat-template machinery.

    Going through transformers rather than a bare `jinja2.Environment`
    matters: the parity claim is about what `apply_chat_template` actually
    produces, which depends on `trim_blocks`/`lstrip_blocks` and on the
    `{% generation %}` extension being registered.
    """
    # transformers-internal.  The public equivalent is
    # `tokenizer.apply_chat_template(..., tokenize=False)`, which would drag
    # a tokenizer download into what is otherwise a pure-Jinja test.
    utils = pytest.importorskip("transformers.utils.chat_template_utils")

    rendered, _ = utils._render_with_assistant_indices(
        utils._compile_jinja_template(template), messages, tools, None, agp
    )
    return rendered


# ── fixture integrity ──────────────────────────────────────────────────


def test_upstream_fixture_is_unmodified() -> None:
    """The fixture is only meaningful as an *untouched* reference copy."""
    digest = hashlib.sha256(_UPSTREAM.read_bytes()).hexdigest()
    assert digest == _UPSTREAM_SHA256, (
        "tests/fixtures/qwen3-8b-upstream.jinja has been modified. It must stay "
        "byte-identical to Qwen/Qwen3-8B's upstream chat_template; see this "
        "module's docstring for how to refresh it."
    )


def test_upstream_fixture_has_no_generation_markers() -> None:
    """Confirms the fixture really is the pristine upstream template: the
    markers are ours, so their absence here is what makes the render-parity
    comparison below meaningful.
    """
    assert not has_generation_markers(_UPSTREAM.read_text())
    assert has_generation_markers(get_chat_template("qwen3"))


# ── render parity ──────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", sorted(_CONVERSATIONS))
@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
def test_qwen3_renders_identically_to_upstream(
    shape: str, add_generation_prompt: bool, with_tools: bool
) -> None:
    """Our marker-spliced qwen3 template must render byte-identically to the
    pristine upstream template in every combination — assistant content,
    reasoning traces, tool calls, the tools system block, and the trailing
    generation prompt all included.

    If this fails, the marker splice moved a byte of the prompt and the
    template no longer matches what Qwen3 was trained with.
    """
    messages = _CONVERSATIONS[shape]
    tools = _TOOLS if with_tools else None

    ours = _render(get_chat_template("qwen3"), messages, tools, add_generation_prompt)
    upstream = _render(_UPSTREAM.read_text(), messages, tools, add_generation_prompt)

    assert ours == upstream
