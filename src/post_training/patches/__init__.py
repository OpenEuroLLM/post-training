"""Runtime patches for upstream bugs that block our training pipeline."""

from post_training.patches.olmo3_swa_rope import (
    install as install_olmo3_swa_rope,
    is_upstream_fixed as is_olmo3_swa_rope_upstream_fixed,
    maybe_install as maybe_install_olmo3_swa_rope,
    uninstall as uninstall_olmo3_swa_rope,
)

__all__ = [
    "install_olmo3_swa_rope",
    "is_olmo3_swa_rope_upstream_fixed",
    "maybe_install_olmo3_swa_rope",
    "uninstall_olmo3_swa_rope",
]
