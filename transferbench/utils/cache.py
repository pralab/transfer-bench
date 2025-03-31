r"""utilities for handling the cache directory."""

import sys
from pathlib import Path


def get_cache_dir() -> Path:
    r"""Get the cache directory."""
    # If inside a venv, use venv/var/mypackage/cache
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        cache_dir = Path(sys.prefix) / "var" / "transferbench" / "cache"
    else:
        # Otherwise, use platform-specific cache directory (e.g., ~/.cache/mypackage on Linux)
        import appdirs

        cache_dir = Path(appdirs.user_cache_dir("transferbench"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
