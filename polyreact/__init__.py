"""Polyreactivity prediction package."""

from importlib import metadata

__all__ = ["__version__"]

try:
    __version__ = metadata.version("polyreact")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
