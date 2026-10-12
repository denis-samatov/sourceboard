"""Projects: independent indexes (folders, theses, evidence) sharing one loaded model.

Layout under the data dir:
    projects.json                 registry: [{"id", "name", "created_at"}]
    projects/<id>/sourceboard.sqlite3, projects/<id>/media/...
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

from ..ingestion.embed import Embedder
from .config import Settings
from .db import Database, utcnow

log = logging.getLogger(__name__)

DEFAULT_PROJECT = "default"
DB_NAME = "sourceboard.sqlite3"


def rewrite_media_paths(db_path: Path, old_dir: Path, new_dir: Path) -> int:
    """Derived media (frames, page renders) is referenced by absolute path; re-point it after a
    project directory moves. Returns the number of rewritten rows."""
    old_prefix, new_prefix = f"{old_dir}/", f"{new_dir}/"
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE chunks SET media_path = ? || substr(media_path, ?) "
                "WHERE substr(media_path, 1, ?) = ?",
                (new_prefix, len(old_prefix) + 1, len(old_prefix), old_prefix),
            )
            conn.execute(
                "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'index_generation'"
            )
        return cur.rowcount
    finally:
        conn.close()


class Project:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        from ..board.board import Board
        from ..ingestion.indexer import Indexer
        from ..search.search import SearchEngine

        self.settings = settings
        self.db = Database(settings.db_path)
        self.embedder = embedder
        self.indexer = Indexer(self.db, settings, embedder)
        self.search = SearchEngine(self.db, embedder)
        self.board = Board(self.db)


class ProjectRegistry:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self.root = settings.data_dir
        self.registry_path = self.root / "projects.json"
        self._lock = threading.RLock()
        self._open: dict[str, Project] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._migrate_legacy_index()
            if not self._read():
                self._write([self._entry(DEFAULT_PROJECT, "Основной")])

    # ------------------------------------------------------------------ registry file

    def _read(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        return json.loads(self.registry_path.read_text(encoding="utf-8"))["projects"]

    def _write(self, projects: list[dict]) -> None:
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"projects": projects}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.registry_path)

    @staticmethod
    def _entry(project_id: str, name: str) -> dict:
        return {"id": project_id, "name": name, "created_at": utcnow()}

    def project_dir(self, project_id: str) -> Path:
        return self.root / "projects" / project_id

    def _migrate_legacy_index(self) -> None:
        """Versions before projects kept a single index in the data dir root."""
        legacy = self.root / DB_NAME
        if not legacy.exists() or self.registry_path.exists():
            return
        target = self.project_dir(DEFAULT_PROJECT)
        target.mkdir(parents=True, exist_ok=True)
        # Fold the WAL into the main file so only one file has to move.
        conn = sqlite3.connect(legacy)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        for suffix in ("", "-wal", "-shm"):
            part = self.root / f"{DB_NAME}{suffix}"
            if part.exists():
                os.replace(part, target / part.name)
        if (self.root / "media").exists():
            os.replace(self.root / "media", target / "media")
        moved = rewrite_media_paths(target / DB_NAME, self.root, target)
        self._write([self._entry(DEFAULT_PROJECT, "Основной")])
        log.info("Moved the existing index into project %r (%d media paths)", "default", moved)

    # ------------------------------------------------------------------ public API

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self._read()]

    def get(self, project_id: str) -> dict:
        for project in self.list():
            if project["id"] == project_id:
                return project
        from ..board.board import NotFound

        raise NotFound(f"project {project_id}")

    def create(self, name: str) -> dict:
        name = " ".join(name.split())
        if not name:
            raise ValueError("Название проекта не может быть пустым")
        with self._lock:
            projects = self._read()
            if any(p["name"].lower() == name.lower() for p in projects):
                raise ValueError(f"Проект «{name}» уже есть")
            entry = self._entry(f"p-{secrets.token_hex(4)}", name)
            projects.append(entry)
            self._write(projects)
        return entry

    def rename(self, project_id: str, name: str) -> dict:
        name = " ".join(name.split())
        if not name:
            raise ValueError("Название проекта не может быть пустым")
        with self._lock:
            projects = self._read()
            self.get(project_id)
            if any(p["name"].lower() == name.lower() and p["id"] != project_id for p in projects):
                raise ValueError(f"Проект «{name}» уже есть")
            for p in projects:
                if p["id"] == project_id:
                    p["name"] = name
            self._write(projects)
        return self.get(project_id)

    def delete(self, project_id: str) -> None:
        """Deletes the project's index and derived media. Original files are never touched."""
        with self._lock:
            projects = self._read()
            self.get(project_id)
            if len(projects) == 1:
                raise ValueError("Нельзя удалить единственный проект")
            project = self._open.pop(project_id, None)
            if project is not None:
                project.indexer.cancel()
                project.db.close()
            self._write([p for p in projects if p["id"] != project_id])
            shutil.rmtree(self.project_dir(project_id), ignore_errors=True)

    def open(self, project_id: str) -> Project:
        with self._lock:
            project = self._open.get(project_id)
            if project is None:
                self.get(project_id)
                settings = replace(self.settings, data_dir=self.project_dir(project_id))
                project = Project(settings, self.embedder)
                self._open[project_id] = project
            return project

    def default_id(self) -> str:
        return self.list()[0]["id"]

    def resolve(self, ref: str | None) -> str:
        """Accept a project id or (case-insensitive) name; None means the first project."""
        if ref is None:
            return self.default_id()
        for p in self.list():
            if p["id"] == ref or p["name"].lower() == ref.lower():
                return p["id"]
        from ..board.board import NotFound

        raise NotFound(f"project {ref}")
