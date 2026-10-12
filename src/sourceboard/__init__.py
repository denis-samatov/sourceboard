"""SourceBoard: local multimodal evidence board built on EmbeddingGemma 2."""

__version__ = "0.1.0"

from .api import Services, create_app
from .board import Board, NotFound
from .core import Database, Project, ProjectRegistry, Settings
from .ingestion import Embedder, Indexer, create_embedder
from .search import SearchEngine

__all__ = [
    "Board",
    "Database",
    "Embedder",
    "Indexer",
    "NotFound",
    "Project",
    "ProjectRegistry",
    "SearchEngine",
    "Services",
    "Settings",
    "__version__",
    "create_app",
    "create_embedder",
]
