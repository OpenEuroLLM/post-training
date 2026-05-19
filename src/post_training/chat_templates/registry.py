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
CHAT_TEMPLATES: dict[str, str] = {
    "chatml": "chatml.jinja",
    "olmo3": "olmo3.jinja",
    "olmo3-instruct-sft": "olmo3-instruct-sft.jinja",
    "olmo3-think-sft": "olmo3-think-sft.jinja",
    "apertus": "apertus.jinja",
    "tulu3": "tulu3.jinja",
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
