"""Search and evaluation: hybrid FTS5 and vector retrieval, RRF fusion, and benchmark evaluation."""

from .evaluate import TaskResult, evaluate, load_tasks, matches
from .search import MODES, SearchEngine, fts_query

__all__ = ["MODES", "SearchEngine", "TaskResult", "evaluate", "fts_query", "load_tasks", "matches"]
