"""Command-line interface entry points and demo generator."""

from .cli import main
from .demo import make_demo

__all__ = ["main", "make_demo"]
