"""HTTP API and static web UI."""

from __future__ import annotations

import mimetypes
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..board.board import ROLES, Board, NotFound
from ..board.export import to_json, to_markdown, to_markers_csv
from ..core.config import Settings
from ..core.db import Database
from ..ingestion.embed import Embedder, create_embedder
from ..ingestion.extract import IMAGE_EXT, asr_available, tool
from ..ingestion.indexer import Indexer
from ..search.search import MODES, SearchEngine

WEB_DIR = Path(__file__).parent / "web"
LOCAL_HOSTS = ["127.0.0.1", "localhost", "[::1]", "::1"]

Modality = Literal["text", "image", "video", "audio"]
Role = Literal["supports", "limits", "contradicts", "context"]


class FolderIn(BaseModel):
    path: str = Field(min_length=1)


class SearchIn(BaseModel):
    q: str = Field(min_length=1, max_length=4000)
    limit: int = Field(20, ge=1, le=100)
    modalities: list[Modality] | None = None
    source_ids: list[int] | None = None
    task: Literal["SearchQuery", "FactChecking", "QuestionAnswering"] = "SearchQuery"
    mode: Literal["hybrid", "lexical", "vector"] = "hybrid"


class ThesisIn(BaseModel):
    title: str = Field(min_length=1, max_length=2000)
    body: str = Field("", max_length=20000)


class ThesisPatch(BaseModel):
    title: str | None = Field(None, max_length=2000)
    body: str | None = Field(None, max_length=20000)
    minor: bool = False


class EvidenceIn(BaseModel):
    chunk_id: int
    role: Role = "context"
    note: str = Field("", max_length=5000)


class EvidencePatch(BaseModel):
    role: Role | None = None
    note: str | None = Field(None, max_length=5000)
    context_checked: bool | None = None


class Services:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.embedder = embedder
        self.indexer = Indexer(self.db, settings, embedder)
        self.search = SearchEngine(self.db, embedder)
        self.board = Board(self.db)


def create_app(
    settings: Settings,
    embedder: Embedder | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    services = Services(settings, embedder or create_embedder(settings))
    app = FastAPI(title="SourceBoard", version=__version__)
    app.state.services = services
    # The API serves local files; refuse requests whose Host is not local (DNS rebinding).
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": f"не найдено: {exc.args[0]}"}, status_code=404)

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def _unavailable(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    s = services

    # ------------------------------------------------------------------ state & sources

    @app.get("/api/state")
    def state() -> dict:
        folders = []
        for row in s.db.query("SELECT * FROM folders ORDER BY id"):
            counts = {
                r["status"]: r["n"]
                for r in s.db.query(
                    "SELECT status, COUNT(*) AS n FROM sources WHERE folder_id = ? GROUP BY status",
                    (row["id"],),
                )
            }
            folders.append(
                {
                    "id": row["id"],
                    "path": row["path"],
                    "available": Path(row["path"]).is_dir(),
                    "sources": counts,
                }
            )
        modalities = {
            r["modality"]: r["n"]
            for r in s.db.query("SELECT modality, COUNT(*) AS n FROM chunks GROUP BY modality")
        }
        return {
            "version": __version__,
            "embedder": {
                "name": s.embedder.name,
                "semantic": s.embedder.semantic,
                "modalities": sorted(s.embedder.modalities),
            },
            "tools": {
                "ffmpeg": bool(tool("ffmpeg")),
                "tesseract": bool(tool("tesseract")),
                "pdftoppm": bool(tool("pdftoppm")),
                "asr": asr_available() and settings.asr_enabled,
            },
            "folders": folders,
            "chunks": modalities,
            "progress": s.indexer.progress.snapshot(),
            "roles": list(ROLES),
            "modes": list(MODES),
        }

    @app.post("/api/folders")
    def add_folder(body: FolderIn) -> dict:
        folder_id = s.indexer.add_folder(body.path)
        s.indexer.request_scan([folder_id])
        return {"id": folder_id}

    @app.delete("/api/folders/{folder_id}")
    def remove_folder(folder_id: int) -> dict:
        s.indexer.remove_folder(folder_id)
        return {"ok": True}

    @app.post("/api/reindex")
    def reindex() -> dict:
        s.indexer.request_scan(None)
        return {"ok": True}

    @app.get("/api/sources")
    def sources(folder_id: int | None = None) -> list[dict]:
        sql = """SELECT s.id, s.folder_id, s.rel_path, s.kind, s.status, s.error, s.notes,
                        s.version, s.duration, s.pages, s.indexed_at, s.sidecar_path,
                        (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) AS chunks
                 FROM sources s"""
        params: tuple = ()
        if folder_id is not None:
            sql += " WHERE s.folder_id = ?"
            params = (folder_id,)
        return [dict(r) for r in s.db.query(sql + " ORDER BY s.rel_path", params)]

    @app.get("/api/sources/{source_id}/file")
    def source_file(source_id: int) -> FileResponse:
        row = s.db.one("SELECT path FROM sources WHERE id = ?", (source_id,))
        if row is None:
            raise NotFound(f"source {source_id}")
        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(410, "Оригинал недоступен: файл перемещён или диск отключён")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, headers={"Content-Disposition": "inline"})

    # ------------------------------------------------------------------ chunks

    @app.get("/api/chunks/{chunk_id}")
    def chunk(chunk_id: int) -> dict:
        row = s.db.one(
            """SELECT c.*, s.rel_path, s.kind, s.status, s.duration, s.pages
               FROM chunks c JOIN sources s ON s.id = c.source_id WHERE c.id = ?""",
            (chunk_id,),
        )
        if row is None:
            raise NotFound(f"chunk {chunk_id}")
        data = {k: row[k] for k in row.keys() if k not in {"vector", "media_path"}}
        data["has_preview"] = bool(row["media_path"]) or row["kind"] == "image"
        data["context"] = [dict(r) for r in _context(s.db, row)]
        data["linked_theses"] = [
            dict(r)
            for r in s.db.query(
                """SELECT e.id AS evidence_id, e.thesis_id, e.role, t.title FROM evidence e
                   JOIN theses t ON t.id = e.thesis_id WHERE e.chunk_id = ?""",
                (chunk_id,),
            )
        ]
        return data

    @app.get("/api/chunks/{chunk_id}/preview")
    def chunk_preview(chunk_id: int) -> FileResponse:
        row = s.db.one(
            """SELECT c.media_path, s.path, s.kind FROM chunks c
               JOIN sources s ON s.id = c.source_id WHERE c.id = ?""",
            (chunk_id,),
        )
        if row is None:
            raise NotFound(f"chunk {chunk_id}")
        candidate = row["media_path"] or (row["path"] if row["kind"] == "image" else None)
        if not candidate or not Path(candidate).is_file():
            raise HTTPException(404, "Нет превью")
        path = Path(candidate)
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "image/jpeg")

    @app.get("/api/chunks/{chunk_id}/similar")
    def chunk_similar(
        chunk_id: int,
        limit: int = Query(20, ge=1, le=100),
        modalities: list[Modality] | None = Query(None),
    ) -> list[dict]:
        return s.search.similar(chunk_id, limit=limit, modalities=modalities)

    # ------------------------------------------------------------------ search

    @app.post("/api/search")
    def search(body: SearchIn) -> list[dict]:
        return s.search.search(
            body.q,
            limit=body.limit,
            modalities=body.modalities,
            source_ids=body.source_ids,
            task=body.task,
            mode=body.mode,
        )

    @app.post("/api/search/image")
    async def search_image(
        file: UploadFile = File(...),
        limit: int = Query(20, ge=1, le=100),
        modalities: list[Modality] | None = Query(None),
    ) -> list[dict]:
        suffix = Path(file.filename or "query.png").suffix.lower()
        if suffix not in IMAGE_EXT:
            raise HTTPException(400, f"Неподдерживаемый формат изображения: {suffix}")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            return s.search.search_by_image(tmp_path, limit=limit, modalities=modalities)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ theses & evidence

    @app.get("/api/theses")
    def theses() -> list[dict]:
        return s.board.list_theses()

    @app.post("/api/theses")
    def create_thesis(body: ThesisIn) -> dict:
        return s.board.create_thesis(body.title, body.body)

    @app.patch("/api/theses/{thesis_id}")
    def update_thesis(thesis_id: int, body: ThesisPatch) -> dict:
        return s.board.update_thesis(thesis_id, title=body.title, body=body.body, minor=body.minor)

    @app.delete("/api/theses/{thesis_id}")
    def delete_thesis(thesis_id: int) -> dict:
        s.board.delete_thesis(thesis_id)
        return {"ok": True}

    @app.get("/api/theses/{thesis_id}/counterpoints")
    def counterpoints(thesis_id: int, limit: int = Query(12, ge=1, le=50)) -> list[dict]:
        try:
            return s.search.counterpoints(thesis_id, limit=limit)
        except KeyError as exc:
            raise NotFound(f"thesis {thesis_id}") from exc

    @app.post("/api/theses/{thesis_id}/evidence")
    def add_evidence(thesis_id: int, body: EvidenceIn) -> dict:
        return s.board.add_evidence(thesis_id, body.chunk_id, body.role, body.note)

    @app.patch("/api/evidence/{evidence_id}")
    def update_evidence(evidence_id: int, body: EvidencePatch) -> dict:
        return s.board.update_evidence(
            evidence_id, role=body.role, note=body.note, context_checked=body.context_checked
        )

    @app.delete("/api/evidence/{evidence_id}")
    def delete_evidence(evidence_id: int) -> dict:
        s.board.delete_evidence(evidence_id)
        return {"ok": True}

    # ------------------------------------------------------------------ export

    @app.get("/api/export/{fmt}")
    def export(fmt: Literal["markdown", "csv", "json"]) -> Response:
        if fmt == "markdown":
            return PlainTextResponse(
                to_markdown(s.db),
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="sourceboard.md"'},
            )
        if fmt == "csv":
            return Response(
                "﻿" + to_markers_csv(s.db),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="sourceboard-markers.csv"'},
            )
        return Response(
            to_json(s.db),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="sourceboard.json"'},
        )

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _context(db: Database, row) -> list:
    """Neighbouring text of a chunk: adjacent paragraphs, the same page, or nearby speech."""
    base = """SELECT id, seq, page, t_start, t_end, text, extractor FROM chunks
              WHERE source_id = ? AND modality = 'text'"""
    if row["modality"] == "text":
        return db.query(
            base
            + """ AND seq BETWEEN
                (SELECT COALESCE(MIN(seq), ?) FROM (SELECT seq FROM chunks WHERE source_id = ?
                    AND modality = 'text' AND seq < ? ORDER BY seq DESC LIMIT 2))
                AND
                (SELECT COALESCE(MAX(seq), ?) FROM (SELECT seq FROM chunks WHERE source_id = ?
                    AND modality = 'text' AND seq > ? ORDER BY seq LIMIT 2))
                ORDER BY seq""",
            (
                row["source_id"],
                row["seq"],
                row["source_id"],
                row["seq"],
                row["seq"],
                row["source_id"],
                row["seq"],
            ),
        )
    if row["page"] is not None:
        return db.query(base + " AND page = ? ORDER BY seq", (row["source_id"], row["page"]))
    if row["t_start"] is not None:
        start, end = row["t_start"] - 30, (row["t_end"] or row["t_start"]) + 30
        return db.query(
            base + " AND t_end >= ? AND t_start <= ? ORDER BY t_start",
            (row["source_id"], start, end),
        )
    return db.query(base + " ORDER BY seq LIMIT 5", (row["source_id"],))
