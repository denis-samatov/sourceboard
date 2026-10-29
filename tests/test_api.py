from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sourceboard.api import create_app
from sourceboard.core import Settings
from sourceboard.ingestion import HashEmbedder


@pytest.fixture
def client(settings: Settings, corpus: Path) -> TestClient:
    app = create_app(settings, HashEmbedder(settings.dim), allowed_hosts=["testserver"])
    services = app.state.services
    services.indexer.scan([services.indexer.add_folder(corpus)])
    return TestClient(app)


def test_rejects_foreign_host(settings: Settings):
    app = create_app(settings, HashEmbedder(settings.dim))  # default: local hosts only
    assert TestClient(app).get("/api/state").status_code == 400
    local = TestClient(app, base_url="http://127.0.0.1:8765")
    assert local.get("/api/state").status_code == 200


def test_state_and_ui(client: TestClient):
    state = client.get("/api/state").json()
    assert state["embedder"]["semantic"] is False
    assert state["folders"][0]["sources"] == {"indexed": 4}
    page = client.get("/")
    assert page.status_code == 200 and "SourceBoard" in page.text
    assert client.get("/app.js").status_code == 200


def test_add_folder_validates_path(client: TestClient, tmp_path: Path):
    response = client.post("/api/folders", json={"path": str(tmp_path / "nope")})
    assert response.status_code == 400


def test_search_open_and_collect(client: TestClient):
    hits = client.post("/api/search", json={"q": "закрыли мост", "limit": 5}).json()
    top = hits[0]
    assert top["source"] == "interviews/ivanova.srt"

    chunk = client.get(f"/api/chunks/{top['chunk_id']}").json()
    assert chunk["context"] and "vector" not in chunk

    thesis = client.post("/api/theses", json={"title": "Транспорт сократился"}).json()
    ev = client.post(
        f"/api/theses/{thesis['id']}/evidence", json={"chunk_id": top["chunk_id"], "role": "limits"}
    ).json()
    assert ev["role"] == "limits"
    ev = client.patch(
        f"/api/evidence/{ev['id']}", json={"context_checked": True, "note": "проверено"}
    ).json()
    assert ev["context_checked"] and ev["note"] == "проверено"
    bad = client.post(
        f"/api/theses/{thesis['id']}/evidence", json={"chunk_id": top["chunk_id"], "role": "proves"}
    )
    assert bad.status_code == 422

    md = client.get("/api/export/markdown")
    assert md.status_code == 200 and "Транспорт сократился" in md.text
    assert client.get("/api/export/csv").text.startswith("﻿thesis,role")
    assert client.delete(f"/api/evidence/{ev['id']}").json() == {"ok": True}
    assert client.delete(f"/api/evidence/{ev['id']}").status_code == 404


def test_source_file_supports_ranges(client: TestClient):
    source = next(s for s in client.get("/api/sources").json() if s["rel_path"].endswith(".pdf"))
    full = client.get(f"/api/sources/{source['id']}/file")
    assert full.status_code == 200 and full.headers["content-type"] == "application/pdf"
    part = client.get(f"/api/sources/{source['id']}/file", headers={"Range": "bytes=0-7"})
    assert part.status_code == 206 and part.content == b"%PDF-1.4"


def test_image_search_needs_vision_model(client: TestClient):
    response = client.post("/api/search/image", files={"file": ("q.png", b"x", "image/png")})
    assert response.status_code == 409


def test_query_image_upload_has_a_byte_limit(client: TestClient):
    response = client.post(
        "/api/search/image",
        files={"file": ("large.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413


def test_chunked_query_body_is_bounded_before_multipart_parsing(client: TestClient):
    response = client.post(
        "/api/search/image",
        content=(b"x" * (1024 * 1024) for _ in range(11)),
        headers={"Content-Type": "multipart/form-data; boundary=oversized"},
    )
    assert response.status_code == 413
