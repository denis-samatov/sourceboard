"""Retrieval evaluation on a JSONL task set.

Each line: {"query": "...", "expected": [{"source": "interviews/a.srt", "page": 3, "t": 125.0}],
            "modalities": ["text"], "task": "SearchQuery"}
`source` is matched as a path suffix; `page` and `t` (seconds) are optional refinements.
A task counts as a hit when any expected item appears in the top k.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .search import SearchEngine

TIME_TOLERANCE_S = 2.0


@dataclass
class TaskResult:
    query: str
    rank: int | None  # 1-based rank of the first matching result, None if not in top k


def matches(result: dict, expected: dict) -> bool:
    source = expected["source"].replace("\\", "/").lstrip("./")
    rel = result["source"].replace("\\", "/")
    if not (rel == source or rel.endswith("/" + source)):
        return False
    if expected.get("page") is not None and result.get("page") != expected["page"]:
        return False
    t = expected.get("t")
    if t is not None:
        start, end = result.get("t_start"), result.get("t_end")
        if start is None:
            return False
        end = start if end is None else end
        if not (start - TIME_TOLERANCE_S <= t <= end + TIME_TOLERANCE_S):
            return False
    return True


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        task = json.loads(line)
        if not task.get("query") or not task.get("expected"):
            raise ValueError(f"{path}:{number}: 'query' and 'expected' are required")
        tasks.append(task)
    return tasks


def evaluate(engine: SearchEngine, tasks: list[dict], k: int = 5, mode: str = "hybrid") -> dict:
    results: list[TaskResult] = []
    for task in tasks:
        hits = engine.search(
            task["query"],
            limit=k,
            modalities=task.get("modalities"),
            task=task.get("task", "SearchQuery"),
            mode=mode,
        )
        rank = next(
            (
                i
                for i, hit in enumerate(hits, start=1)
                if any(matches(hit, exp) for exp in task["expected"])
            ),
            None,
        )
        results.append(TaskResult(task["query"], rank))
    n = len(results) or 1
    return {
        "mode": mode,
        "k": k,
        "tasks": len(results),
        f"hit@{k}": sum(r.rank is not None for r in results) / n,
        # Kept for existing consumers; this is query hit rate, not per-item recall.
        f"recall@{k}": sum(r.rank is not None for r in results) / n,
        f"mrr@{k}": sum(1 / r.rank for r in results if r.rank) / n,
        "misses": [r.query for r in results if r.rank is None],
    }
