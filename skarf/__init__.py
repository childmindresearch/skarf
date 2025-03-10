from pathlib import Path

import platformdirs

_CACHE_DIR = Path(platformdirs.user_cache_dir()) / __package__


def get_cache_dir() -> Path:
    """Get a path to the package cache directory."""
    return _CACHE_DIR


def get_default_cache_dir() -> Path:
    """Get the default package cache director."""
    return Path(platformdirs.user_cache_dir()) / __package__


def set_cache_dir(path: Path | None = None) -> None:
    """Override the default cache directory."""
    global _CACHE_DIR
    if path is None:
        path = get_default_cache_dir()
    _CACHE_DIR = Path(path)
