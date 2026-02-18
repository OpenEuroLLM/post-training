"""Chat template registry.

Each chat template is a Jinja file stored under
``src/post_training/chat_templates/templates/``.  The registry maps a
short name (used in the YAML config) to the Jinja file name, and
:func:`get_chat_template` returns the raw Jinja string so it can be
assigned to ``tokenizer.chat_template``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory that holds the .jinja template files.
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Mapping: template name -> jinja filename (relative to _TEMPLATES_DIR).
CHAT_TEMPLATES: dict[str, str] = {
    "chatml": "chatml.jinja",
    "olmo3": "olmo3.jinja",
    "apertus": "apertus.jinja",
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
