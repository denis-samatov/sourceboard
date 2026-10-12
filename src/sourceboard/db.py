"""SQLite storage: sources, chunks (with vectors and FTS), theses and evidence links."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at TEXT NOT NULL
);

-- One row per original file. `version` increments every time the content changes, so
-- evidence can tell which revision of the file it was checked against.
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    folder_id INTEGER REFERENCES folders(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    rel_path TEXT NOT NULL,
    kind TEXT NOT NULL,              -- document | image | audio | video | transcript
    sha256 TEXT,
    size INTEGER,
    mtime REAL,
    version INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | indexing | indexed | error | missing
    error TEXT,
    notes TEXT,                      -- non-fatal remarks, e.g. "no transcript"
    sidecar_path TEXT,               -- subtitles used as the transcript of a media file
    duration REAL,
    pages INTEGER,
    indexed_at TEXT
);

-- Searchable fragments. A fragment always points back into its source: a page or a time range.
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    source_version INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    modality TEXT NOT NULL,          -- text | image | audio | video
    page INTEGER,
    t_start REAL,
    t_end REAL,
    text TEXT NOT NULL DEFAULT '',
    extractor TEXT NOT NULL,         -- how the content was obtained (pypdf, srt, whisper, ocr...)
    media_path TEXT,                 -- derived frame / page image used for the embedding
    embed_model TEXT,
    vector BLOB
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_id, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS theses (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL DEFAULT 1,
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- A fragment selected for a thesis. The locator and excerpt are a snapshot, so the link
-- survives re-indexing; chunk_id is re-pointed to the closest new chunk when a source changes.
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    thesis_id INTEGER NOT NULL REFERENCES theses(id) ON DELETE CASCADE,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    source_path TEXT NOT NULL,
    source_rel_path TEXT NOT NULL,
    source_sha256 TEXT,
    source_version INTEGER,
    modality TEXT NOT NULL,
    page INTEGER,
    t_start REAL,
    t_end REAL,
    excerpt TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'context'
        CHECK (role IN ('supports', 'limits', 'contradicts', 'context')),
    context_checked INTEGER NOT NULL DEFAULT 0,
    checked_thesis_revision INTEGER,
    checked_source_sha256 TEXT,
    note TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_by_thesis ON evidence(thesis_id, position);
CREATE INDEX IF NOT EXISTS evidence_by_source ON evidence(source_id);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    """Thin wrapper that gives every thread its own SQLite connection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self._write_lock:
            # executescript manages its own transaction, so it runs outside write().
            self.conn().executescript(SCHEMA)
        with self.write() as conn:
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('index_generation', '0')")

    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Serialized write transaction."""
        conn = self.conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, params).fetchone()

    def generation(self) -> int:
        row = self.one("SELECT value FROM meta WHERE key = 'index_generation'")
        return int(row["value"]) if row else 0

    @staticmethod
    def bump_generation(conn: sqlite3.Connection) -> None:
        """Signal that chunk vectors changed, so in-memory vector caches must reload."""
        conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'index_generation'"
        )

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
