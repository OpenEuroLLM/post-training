"""Diagnostic shim — re-exports the package-level Olmo3 SWA RoPE patch.

The real implementation lives at `post_training.patches.olmo3_swa_rope`
(installed alongside the training pipeline). This file is kept so the
diagnostic's `import patch_olmo3_swa_rope` still resolves without changes
to `hf_forward.py`. Both call sites end up at the same monkey-patch.
"""
from post_training.patches.olmo3_swa_rope import (  # noqa: F401
    install,
    is_upstream_fixed,
    maybe_install,
    uninstall,
)
