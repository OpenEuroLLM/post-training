"""Chat template registry.

Each chat template is a Jinja file stored under
``src/post_training/chat_templates/templates/``.  The registry maps a
short name (used in the YAML config) to the Jinja file name, and
:func:`get_chat_template` returns the raw Jinja string so it can be
assigned to ``tokenizer.chat_template``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# `{% generation %}` markers are a transformers-specific extension to Jinja2,
# used by ``apply_chat_template(..., return_assistant_tokens_mask=True)`` (and
# therefore by TRL's ``assistant_only_loss=True``).  Accept any of the four
# whitespace-stripping variants (``{%``/``{%-`` and ``%}``/``-%}``).
_GENERATION_OPEN_RE = re.compile(r"\{%-?\s*generation\s*-?%\}")
_GENERATION_CLOSE_RE = re.compile(r"\{%-?\s*endgeneration\s*-?%\}")

# Directory that holds the .jinja template files.
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Mapping: template name -> jinja filename (relative to _TEMPLATES_DIR).
#
# Note on the three ``olmo3*`` entries: they correspond to three different
# views of AllenAI's OLMo-3-7B chat templates.
#   - ``olmo3``: a cosmetically-reformatted copy of
#     ``allenai/Olmo-3-7B-Think-SFT``'s chat template (single→double quotes
#     and whitespace stripping added; rendered output is byte-identical to
#     the upstream Think template).  Appends ``<think>`` to
#     ``add_generation_prompt=True``.  No ``{% generation %}`` markers, so
#     SFT here cannot mask user/system tokens out of the loss — the runtime
#     guard in ``methods/sft.py`` will refuse to start training with this
#     template.  Kept for inference parity / backwards compatibility.
#   - ``olmo3-instruct-sft``: byte-identical (modulo spliced ``{% generation %}``
#     markers) to ``allenai/OLMo-3-7B-Instruct-SFT``'s ``chat_template.jinja``.
#     Use this to reproduce the Instruct-SFT recipe via TRL.
#   - ``olmo3-think-sft``: byte-identical (modulo spliced ``{% generation %}``
#     markers) to ``allenai/Olmo-3-7B-Think-SFT``'s ``chat_template`` field
#     in ``tokenizer_config.json``.  Use this to reproduce the Think-SFT
#     recipe via TRL.  Same assistant-only masking pattern as Instruct-SFT;
#     OLMo-core's reference Think pipeline uses identical offline-baked
#     masks (only learning rate and dataset differ between the two recipes).
#
# Note on ``qwen3``: Unlike the OLMo templates, ``qwen3`` masks *only the
# final exchange*: Qwen3 strips ``<think>`` blocks from assistant turns at
# or before the last user query, so history renders as a bare answer with
# no reasoning trace. Training on that teaches the model to answer directly at
# exactly the position where it opens a ``<think>`` block at inference.  The
# markers therefore wrap the assistant body only when
# ``loop.index0 > ns.last_query_index``.  Multi-step tool exchanges keep every
# assistant turn after the last user query, since those retain their traces.
CHAT_TEMPLATES: dict[str, str] = {
    "chatml": "chatml.jinja",
    "olmo3": "olmo3.jinja",
    "olmo3-instruct-sft": "olmo3-instruct-sft.jinja",
    "olmo3-think-sft": "olmo3-think-sft.jinja",
    "apertus": "apertus.jinja",
    "tulu3": "tulu3.jinja",
    "qwen3": "qwen3.jinja",
}


def register_chat_template(name: str, filename: str) -> None:
    """Register a new chat template.

    Parameters
    ----------
    name:
        Short identifier used in the config YAML.
    filename:
        Jinja file name inside the ``templates/`` directory.
    """
    if name in CHAT_TEMPLATES:
        logger.warning("Overwriting existing chat template '%s'.", name)
    CHAT_TEMPLATES[name] = filename


def has_generation_markers(template: str | None) -> bool:
    """Return ``True`` if *template* wraps content in
    ``{% generation %}…{% endgeneration %}`` markers (any whitespace-strip form).

    Required by transformers' ``return_assistant_tokens_mask`` path, which TRL
    uses to implement ``assistant_only_loss=True``.  Missing markers make the
    mask silently all-zero — SFT then trains on every token in the sequence.
    """
    if not template:
        return False
    return bool(_GENERATION_OPEN_RE.search(template) and _GENERATION_CLOSE_RE.search(template))


def infer_end_token_from_render(rendered: str, added_tokens: dict[str, int]) -> str | None:
    """The added token a rendered conversation ends on, or ``None``.

    Deliberately not named for the eos or for a "stop token": what comes back is
    a TEMPLATE-side observation, and under ``qwen3`` it is ``<|im_end|>``, which
    is not the model's eos and not a stop token until ``align_generation_eos``
    makes it one. Under the ``olmo3-*`` templates it happens to be the eos, but
    only because those templates terminate on it.

    This is how the turn terminator is established: **by looking at what the
    template actually produced**, never from a ``{template: terminator}`` table.
    A table goes stale the moment a template changes and the failure is silent —
    the model learns to emit one token while the config stops on another.

    Pure string logic on purpose, so it is testable without a tokenizer or a
    model. The caller renders (see ``build_tokenizer``) and passes the result in
    together with ``tokenizer.get_added_vocab()``.

    ``None`` is the conservative answer and means *change nothing*: it covers a
    template whose terminator is not an added token, one that could not be
    rendered, and any shape nobody has considered.
    """
    tail = (rendered or "").rstrip("\n")
    matches = [t for t in added_tokens if t and tail.endswith(t)]
    if not matches:
        return None
    return max(matches, key=len)  # longest wins: '<|im_end|>' over 'end|>'


def get_chat_template(name: str) -> str:
    """Return the Jinja source string for the template registered as *name*.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    FileNotFoundError
        If the Jinja file does not exist on disk.
    """
    if name not in CHAT_TEMPLATES:
        available = ", ".join(sorted(CHAT_TEMPLATES.keys()))
        raise KeyError(f"Chat template '{name}' not found. Available: {available}")
    path = _TEMPLATES_DIR / CHAT_TEMPLATES[name]
    if not path.exists():
        raise FileNotFoundError(f"Chat template file not found: {path}")
    return path.read_text()
