"""Folder scanning and incremental indexing."""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import queue
import tempfile
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from .config import Settings
from .db import Database, utcnow
from .embed import Embedder
from .extract import (
    SUBTITLE_EXT,
    Extraction,
    Extractor,
    IndexCancelled,
    Piece,
    kind_for,
    sha256_file,
)

log = logging.getLogger(__name__)

EMBED_BATCH = 32


@dataclass
class Progress:
    running: bool = False
    total: int = 0
    done: int = 0
    current: str = ""
    detail: str = ""
    errors: int = 0
    cancelled: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **values: object) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "_lock"}


@dataclass
class FileEntry:
    path: Path
    kind: str
    sidecar: Path | None = None


def _subtitle_key(path: Path) -> tuple[Path, str]:
    """`talk.srt` and `talk.ru.srt` both belong to `talk.mp4`."""
    stem = path.stem
    return path.parent, stem.split(".")[0] if "." in stem else stem


# Dependency, build and cache folders: third-party READMEs and assets, not the user's material.
SKIP_DIRS = frozenset(
    {"node_modules", "__pycache__", "site-packages", "venv", "env", "bower_components"}
)


def discover(root: Path, exclude: Path | None = None) -> list[FileEntry]:
    media: dict[tuple[Path, str], FileEntry] = {}
    subtitles: list[Path] = []
    entries: list[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in SKIP_DIRS
            and (exclude is None or (current / d) != exclude)
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = current / name
            kind = kind_for(path)
            if kind is None:
                continue
            if path.suffix.lower() in SUBTITLE_EXT:
                subtitles.append(path)
                continue
            entry = FileEntry(path, kind)
            entries.append(entry)
            if kind in {"audio", "video"}:
                media[(path.parent, path.stem)] = entry
    for sub in subtitles:
        owner = media.get(_subtitle_key(sub)) or media.get((sub.parent, sub.stem))
        if owner is not None and owner.sidecar is None:
            owner.sidecar = sub
        else:
            entries.append(FileEntry(sub, "transcript"))
    return entries


def _fingerprint(entry: FileEntry) -> tuple[int, float]:
    stat = entry.path.stat()
    size, mtime = stat.st_size, stat.st_mtime
    if entry.sidecar is not None:
        side = entry.sidecar.stat()
        size += side.st_size
        mtime = max(mtime, side.st_mtime)
    return size, mtime


def _content_hash(entry: FileEntry) -> str:
    digest = sha256_file(entry.path)
    if entry.sidecar is None:
        return digest
    combined = f"{digest}:{sha256_file(entry.sidecar)}"
    return hashlib.sha256(combined.encode()).hexdigest()


class Indexer:
    def __init__(self, db: Database, settings: Settings, embedder: Embedder) -> None:
        self.db = db
        self.settings = settings
        self.embedder = embedder
        self._cancel = threading.Event()
        self.extractor = Extractor(settings, embedder.modalities - {"text"}, self._cancel)
        self.progress = Progress()
        self._queue: queue.Queue[list[int] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    # ------------------------------------------------------------------ folders

    def add_folder(self, path: str | Path) -> int:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Папка не найдена: {root}")
        with self.db.write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO folders(path, added_at) VALUES (?, ?)", (str(root), utcnow())
            )
            row = conn.execute("SELECT id FROM folders WHERE path = ?", (str(root),)).fetchone()
        return int(row["id"])

    def remove_folder(self, folder_id: int) -> None:
        with self.db.write() as conn:
            conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
            self.db.bump_generation(conn)

    # ------------------------------------------------------------------ background worker

    def request_scan(self, folder_ids: list[int] | None = None) -> None:
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="indexer", daemon=True)
                self._worker.start()
        self._queue.put(folder_ids)

    def cancel(self) -> bool:
        """Stop the running scan after the current step and drop queued scans.
        Returns True if there was something to stop."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        running = self.progress.running
        if running:
            self._cancel.set()
        return running or dropped > 0

    def _run(self) -> None:
        while True:
            folder_ids = self._queue.get()
            try:
                self.scan(folder_ids)
            except Exception:
                log.exception("Indexing run failed")
                self.progress.update(running=False, detail="ошибка индексации, см. журнал")

    # ------------------------------------------------------------------ scanning

    def scan(self, folder_ids: list[int] | None = None) -> dict:
        if folder_ids:
            marks = ",".join("?" * len(folder_ids))
            folders = self.db.query(
                f"SELECT * FROM folders WHERE id IN ({marks})", tuple(folder_ids)
            )
        else:
            folders = self.db.query("SELECT * FROM folders ORDER BY id")

        plan: list[tuple[int, Path, FileEntry]] = []
        for folder in folders:
            root = Path(folder["path"])
            if not root.is_dir():
                # Unplugged drive: keep the catalog searchable, flag originals as unavailable.
                with self.db.write() as conn:
                    conn.execute(
                        "UPDATE sources SET status = 'missing' WHERE folder_id = ?", (folder["id"],)
                    )
                continue
            entries = discover(root, exclude=self.settings.data_dir.resolve())
            seen = {str(e.path) for e in entries}
            with self.db.write() as conn:
                for row in conn.execute(
                    "SELECT id, path FROM sources WHERE folder_id = ?", (folder["id"],)
                ).fetchall():
                    if row["path"] not in seen:
                        conn.execute(
                            "UPDATE sources SET status = 'missing' WHERE id = ?", (row["id"],)
                        )
            plan.extend((folder["id"], root, entry) for entry in entries)

        self._cancel.clear()
        stats = {"indexed": 0, "skipped": 0, "errors": 0, "cancelled": False}
        self.progress.update(
            running=True,
            cancelled=False,
            total=len(plan),
            done=0,
            errors=0,
            current="",
            detail="",
            started_at=utcnow(),
            finished_at=None,
        )
        for folder_id, root, entry in plan:
            if self._cancel.is_set():
                stats["cancelled"] = True
                break
            self.progress.update(current=str(entry.path.relative_to(root)), detail="")
            try:
                changed = self.index_file(folder_id, root, entry)
                stats["indexed" if changed else "skipped"] += 1
            except IndexCancelled:
                stats["cancelled"] = True
                break
            except Exception as exc:
                log.exception("Failed to index %s", entry.path)
                stats["errors"] += 1
                self._mark_error(folder_id, root, entry, exc)
                self.progress.update(errors=stats["errors"])
            self.progress.update(done=self.progress.done + 1)
        self._cancel.clear()
        self.progress.update(
            running=False,
            cancelled=stats["cancelled"],
            current="",
            detail="",
            finished_at=utcnow(),
        )
        return stats

    def _mark_error(self, folder_id: int, root: Path, entry: FileEntry, exc: Exception) -> None:
        with self.db.write() as conn:
            conn.execute(
                """INSERT INTO sources(folder_id, path, rel_path, kind, status, error)
                   VALUES (?, ?, ?, ?, 'error', ?)
                   ON CONFLICT(path) DO UPDATE SET status = 'error', error = excluded.error""",
                (
                    folder_id,
                    str(entry.path),
                    str(entry.path.relative_to(root)),
                    entry.kind,
                    f"{type(exc).__name__}: {exc}",
                ),
            )

    def index_file(self, folder_id: int, root: Path, entry: FileEntry) -> bool:
        """Index one file if it is new or changed. Returns True when chunks were rebuilt."""
        path_str = str(entry.path)
        size, mtime = _fingerprint(entry)
        existing = self.db.one("SELECT * FROM sources WHERE path = ?", (path_str,))
        model_current = existing is not None and self._embeddings_current(existing["id"])
        sidecar = str(entry.sidecar) if entry.sidecar else None
        if (
            existing is not None
            and existing["status"] in {"indexed", "missing"}
            and existing["size"] == size
            and existing["mtime"] == mtime
            and existing["sidecar_path"] == sidecar
            and model_current
        ):
            if existing["status"] == "missing":
                with self.db.write() as conn:
                    conn.execute(
                        "UPDATE sources SET status = 'indexed' WHERE id = ?", (existing["id"],)
                    )
            return False

        sha = _content_hash(entry)
        if (
            existing is not None
            and existing["sha256"] == sha
            and existing["status"] in {"indexed", "missing"}
            and model_current
        ):
            with self.db.write() as conn:  # touched but identical content
                conn.execute(
                    "UPDATE sources SET size = ?, mtime = ?, status = 'indexed' WHERE id = ?",
                    (size, mtime, existing["id"]),
                )
            return False

        rel_path = str(entry.path.relative_to(root))
        with self.db.write() as conn:
            conn.execute(
                """INSERT INTO sources(folder_id, path, rel_path, kind, status)
                   VALUES (?, ?, ?, ?, 'indexing')
                   ON CONFLICT(path) DO UPDATE SET status = 'indexing', error = NULL,
                       folder_id = excluded.folder_id, rel_path = excluded.rel_path,
                       kind = excluded.kind""",
                (folder_id, path_str, rel_path, entry.kind),
            )
            source = conn.execute("SELECT * FROM sources WHERE path = ?", (path_str,)).fetchone()

        try:
            with tempfile.TemporaryDirectory(prefix="sourceboard-") as tmp:
                self.progress.update(detail="извлечение")
                extraction = self.extractor.extract(
                    entry.path, entry.kind, sha, Path(tmp), entry.sidecar
                )
                vectors = self._embed(entry.path, extraction)
        except IndexCancelled:
            # Chunks are only replaced after a full extraction, so the old ones are intact.
            with self.db.write() as conn:
                conn.execute(
                    "UPDATE sources SET status = ? WHERE id = ?",
                    ("indexed" if source["sha256"] else "pending", source["id"]),
                )
            raise

        version = source["version"] + 1 if source["sha256"] != sha else source["version"]
        with self.db.write() as conn:
            conn.execute("DELETE FROM chunks WHERE source_id = ?", (source["id"],))
            for seq, (piece, vector) in enumerate(zip(extraction.pieces, vectors, strict=True)):
                conn.execute(
                    """INSERT INTO chunks(source_id, source_version, seq, modality, page, t_start,
                           t_end, text, extractor, media_path, embed_model, vector)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source["id"],
                        version,
                        seq,
                        piece.modality,
                        piece.page,
                        piece.t_start,
                        piece.t_end,
                        piece.text,
                        piece.extractor,
                        str(piece.media_path) if piece.media_path and piece.keep_media else None,
                        self.embedder.name if vector is not None else None,
                        vector.astype(np.float32).tobytes() if vector is not None else None,
                    ),
                )
            conn.execute(
                """UPDATE sources SET sha256 = ?, size = ?, mtime = ?, version = ?,
                       status = 'indexed', error = NULL, notes = ?, sidecar_path = ?,
                       duration = ?, pages = ?, indexed_at = ? WHERE id = ?""",
                (
                    sha,
                    size,
                    mtime,
                    version,
                    "; ".join(extraction.notes) or None,
                    sidecar,
                    extraction.duration,
                    extraction.pages,
                    utcnow(),
                    source["id"],
                ),
            )
            relink_evidence(conn, source["id"])
            self.db.bump_generation(conn)
        return True

    def _embeddings_current(self, source_id: int) -> bool:
        row = self.db.one(
            """SELECT COUNT(*) AS stale FROM chunks
               WHERE source_id = ? AND vector IS NOT NULL AND embed_model != ?""",
            (source_id, self.embedder.name),
        )
        return row is None or row["stale"] == 0

    def _embed(self, path: Path, extraction: Extraction) -> list[np.ndarray | None]:
        """Embed every piece; a failing modality is reported as a note, not a fatal error."""
        vectors: list[np.ndarray | None] = [None] * len(extraction.pieces)
        groups: dict[str, list[int]] = {}
        for i, piece in enumerate(extraction.pieces):
            if piece.modality == "text":
                if piece.text.strip():
                    groups.setdefault("text", []).append(i)
            elif piece.media_path is not None:
                key = "audio" if piece.modality == "audio" else "image"
                groups.setdefault(key, []).append(i)

        title = path.stem
        done = 0
        total = sum(len(v) for v in groups.values())
        for modality, indices in groups.items():
            for start in range(0, len(indices), EMBED_BATCH):
                if self._cancel.is_set():
                    raise IndexCancelled()
                batch = indices[start : start + EMBED_BATCH]
                pieces: list[Piece] = [extraction.pieces[i] for i in batch]
                self.progress.update(detail=f"эмбеддинги {done}/{total}")
                try:
                    if modality == "text":
                        matrix = self.embedder.embed_documents([(title, p.text) for p in pieces])
                    elif modality == "image":
                        matrix = self.embedder.embed_images([p.media_path for p in pieces])
                    else:
                        matrix = self.embedder.embed_audio([p.media_path for p in pieces])
                except Exception as exc:
                    log.exception("Embedding %s failed for %s", modality, path)
                    note = f"не удалось построить эмбеддинги ({modality}): {exc}"
                    if note not in extraction.notes:
                        extraction.notes.append(note)
                    break
                for i, vector in zip(batch, matrix, strict=True):
                    vectors[i] = vector
                done += len(batch)
        return vectors


def _overlap(
    a_start: float | None, a_end: float | None, b_start: float | None, b_end: float | None
) -> float:
    if None in (a_start, a_end, b_start, b_end):
        return 0.0
    inter = min(a_end, b_end) - max(a_start, b_start)  # type: ignore[operator]
    span = max(a_end, b_end) - min(a_start, b_start)  # type: ignore[operator]
    return max(0.0, inter) / span if span > 0 else 1.0


def relink_evidence(conn, source_id: int) -> None:
    """Point evidence of a rebuilt source at the closest new chunk (same modality and locator,
    most similar text). The original excerpt snapshot stays untouched."""
    links = conn.execute("SELECT * FROM evidence WHERE source_id = ?", (source_id,)).fetchall()
    if not links:
        return
    chunks = conn.execute(
        "SELECT id, modality, page, t_start, t_end, text FROM chunks WHERE source_id = ?",
        (source_id,),
    ).fetchall()
    for link in links:
        best_id, best_score = None, 0.0
        for chunk in chunks:
            if chunk["modality"] != link["modality"]:
                continue
            score = 0.0
            if link["page"] is not None:
                score += 1.0 if chunk["page"] == link["page"] else 0.0
            if link["t_start"] is not None:
                score += _overlap(link["t_start"], link["t_end"], chunk["t_start"], chunk["t_end"])
            if link["excerpt"] and chunk["text"]:
                score += (
                    2.0
                    * difflib.SequenceMatcher(
                        None, link["excerpt"], chunk["text"], autojunk=False
                    ).ratio()
                )
            elif (
                not link["excerpt"]
                and not chunk["text"]
                and link["page"] is None
                and link["t_start"] is None
            ):
                score += 0.5
            if score > best_score:
                best_id, best_score = chunk["id"], score
        conn.execute(
            "UPDATE evidence SET chunk_id = ? WHERE id = ?",
            (best_id if best_score >= 0.5 else None, link["id"]),
        )
