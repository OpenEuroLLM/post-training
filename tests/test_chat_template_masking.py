"""Tests for the chat-template `{% generation %}` marker detector.

These guard against the silent-no-op failure mode where TRL's
`assistant_only_loss=True` looks like it's masking but actually isn't,
because the chat template lacks `{% generation %}…{% endgeneration %}`
markers around the assistant content emission.

We test the detector itself (cheap, no deps) plus per-template
regression guards on the registered chat templates.
"""

from __future__ import annotations

from post_training.chat_templates.registry import (
    get_chat_template,
    has_generation_markers,
)


# ── detector unit tests ────────────────────────────────────────────────


def test_detects_plain_markers() -> None:
    assert has_generation_markers(
        "before {% generation %} body {% endgeneration %} after"
    )


def test_detects_whitespace_stripped_markers() -> None:
    assert has_generation_markers(
        "before {%- generation -%} body {%- endgeneration -%} after"
    )


def test_detects_asymmetric_strip_markers() -> None:
    # Mixed strip directions also valid Jinja2.
    assert has_generation_markers(
        "{%- generation %} body {% endgeneration -%}"
    )


def test_rejects_missing_close() -> None:
    assert not has_generation_markers("{% generation %} body, never closes")


def test_rejects_missing_open() -> None:
    assert not has_generation_markers("body without an opener {% endgeneration %}")


def test_rejects_empty_or_none() -> None:
    assert not has_generation_markers("")
    assert not has_generation_markers(None)


# ── registry-level regression guards ───────────────────────────────────


def test_olmo3_instruct_sft_template_has_markers() -> None:
    """The Instruct-SFT template is the one our production SFT config uses.
    If anyone strips `{% generation %}` markers from
    `olmo3-instruct-sft.jinja`, SFT silently regresses to full-sequence loss.
    """
    assert has_generation_markers(get_chat_template("olmo3-instruct-sft"))


def test_olmo3_think_sft_template_has_markers() -> None:
    """The Think-SFT template is the one to use when reproducing
    AllenAI's Olmo-3-7B-Think-SFT recipe via TRL.  Same masking story as
    Instruct-SFT — if `{% generation %}` markers go missing, SFT
    regresses to full-sequence loss.
    """
    assert has_generation_markers(get_chat_template("olmo3-think-sft"))


def test_olmo3_template_lacks_markers() -> None:
    """The legacy `olmo3` template (a re-formatted copy of
    Olmo-3-7B-Think-SFT) does not have `{% generation %}` markers.
    Documented here so the runtime guard's behaviour is explicit:
    starting an SFT run with `chat_template: olmo3` will raise — use
    `olmo3-instruct-sft` or `olmo3-think-sft` instead, depending on
    which checkpoint you are reproducing.
    """
    assert not has_generation_markers(get_chat_template("olmo3"))
