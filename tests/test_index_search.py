from pathlib import Path

from sourceboard.api import Services
from sourceboard.search import evaluate, fts_query, matches


def sources(services: Services) -> dict[str, dict]:
    return {r["rel_path"]: dict(r) for r in services.db.query("SELECT * FROM sources")}


def test_index_extracts_all_supported_files(indexed: Services):
    found = sources(indexed)
    assert set(found) == {
        "docs/minutes.md",
        "docs/report.pdf",
        "docs/transport.docx",
        "interviews/ivanova.srt",
    }
    assert all(s["status"] == "indexed" and s["version"] == 1 for s in found.values())
    assert found["docs/report.pdf"]["pages"] == 2
    pages = indexed.db.query(
        "SELECT page, text FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE s.rel_path = 'docs/report.pdf' ORDER BY page"
    )
    assert [p["page"] for p in pages] == [1, 2]
    assert "61 percent" in pages[1]["text"]


def test_transcript_chunks_keep_timecodes(indexed: Services):
    rows = indexed.db.query(
        "SELECT t_start, t_end, text FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE s.kind = 'transcript' ORDER BY seq"
    )
    assert [(r["t_start"], r["t_end"]) for r in rows] == [(1.0, 9.5), (40.0, 44.0)]


def test_lexical_search_handles_russian_inflection(indexed: Services):
    hits = indexed.search.search("закрытие заводов", mode="lexical")
    assert hits and hits[0]["source"] == "interviews/ivanova.srt"
    assert hits[0]["t_start"] == 1.0
    assert hits[0]["matched_by"] == ["lexical"]


def test_hybrid_search_finds_page_and_respects_filters(indexed: Services):
    hits = indexed.search.search("bus ridership", limit=5)
    assert hits[0]["source"] == "docs/report.pdf"
    assert hits[0]["page"] == 2
    assert hits[0]["similarity"] is not None
    assert indexed.search.search("bus ridership", modalities=["image"]) == []
    only_srt = indexed.search.search(
        "маршрут", source_ids=[sources(indexed)["interviews/ivanova.srt"]["id"]]
    )
    assert {h["source"] for h in only_srt} == {"interviews/ivanova.srt"}


def test_fts_query_escapes_and_drops_stopwords():
    assert fts_query("и в на") is None
    query = fts_query('завода "OR" 2025')
    assert '"завода"' in query and "зав" in query and '"2025"' in query


def test_unchanged_files_are_skipped(indexed: Services):
    stats = indexed.indexer.scan()
    assert stats == {"indexed": 0, "skipped": 4, "errors": 0, "cancelled": False}


def test_changed_file_bumps_version(indexed: Services, corpus: Path):
    path = corpus / "docs" / "minutes.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nНовый абзац.\n", encoding="utf-8")
    stats = indexed.indexer.scan()
    assert stats["indexed"] == 1
    assert sources(indexed)["docs/minutes.md"]["version"] == 2


def test_missing_files_stay_searchable(indexed: Services, corpus: Path):
    (corpus / "interviews" / "ivanova.srt").unlink()
    indexed.indexer.scan()
    assert sources(indexed)["interviews/ivanova.srt"]["status"] == "missing"
    hits = indexed.search.search("мост", mode="lexical")
    assert hits[0]["source_status"] == "missing"


def test_similar_excludes_itself(indexed: Services):
    hit = indexed.search.search("мост осенью", limit=1)[0]
    similar = indexed.search.similar(hit["chunk_id"])
    assert hit["chunk_id"] not in {h["chunk_id"] for h in similar}


def test_evaluate_reports_recall(indexed: Services):
    tasks = [
        {"query": "bus ridership fell", "expected": [{"source": "report.pdf", "page": 2}]},
        {"query": "закрыли мост", "expected": [{"source": "interviews/ivanova.srt", "t": 42}]},
        {"query": "квантовая гравитация", "expected": [{"source": "nowhere.pdf"}]},
    ]
    report = evaluate(indexed.search, tasks, k=3, mode="hybrid")
    assert report["recall@3"] == 2 / 3
    assert report["misses"] == ["квантовая гравитация"]


def test_matches_requires_locator():
    hit = {"source": "a/b.srt", "page": None, "t_start": 10.0, "t_end": 20.0}
    assert matches(hit, {"source": "b.srt", "t": 21})
    assert not matches(hit, {"source": "b.srt", "t": 30})
    assert not matches(hit, {"source": "ab.srt"})


def test_vector_fusion_keeps_strong_hits_of_any_modality(services: Services, tmp_path: Path):
    """Ranks are per modality, but a modality's top hit must not outrank much closer hits of
    another modality (regression: image queries used to return only text)."""
    import numpy as np

    from sourceboard.core import utcnow

    dim = services.embedder.dim
    query = np.zeros(dim, dtype=np.float32)
    query[0] = 1.0

    def vec(sim: float) -> bytes:
        v = np.zeros(dim, dtype=np.float32)
        v[0], v[1] = sim, (1 - sim**2) ** 0.5
        return v.tobytes()

    with services.db.write() as conn:
        conn.execute("INSERT INTO folders(path, added_at) VALUES (?, ?)", (str(tmp_path), utcnow()))
        conn.execute(
            "INSERT INTO sources(id, folder_id, path, rel_path, kind, status) "
            "VALUES (1, 1, 'v.mp4', 'v.mp4', 'video', 'indexed')"
        )
        rows = [("video", 0.97), ("video", 0.77), ("image", 0.92)] + [
            ("text", 0.74 - i / 100) for i in range(5)
        ]
        for seq, (modality, sim) in enumerate(rows):
            conn.execute(
                "INSERT INTO chunks(source_id, source_version, seq, modality, extractor, "
                "embed_model, vector) VALUES (1, 1, ?, ?, 'test', ?, ?)",
                (seq, modality, services.embedder.name, vec(sim)),
            )
        services.db.bump_generation(conn)

    top = services.search.search_by_vector(query, limit=3)
    assert [(h["modality"], h["similarity"]) for h in top][:2] == [("video", 0.97), ("image", 0.92)]


def test_hybrid_damps_word_matches_far_from_meaning(settings, tmp_path: Path):
    """A chunk sharing only a stock phrase with the query must not tie with the best semantic
    hit, but an exact-word match still contributes (floor weight)."""
    import numpy as np

    from sourceboard.core import utcnow
    from sourceboard.ingestion import HashEmbedder

    class AxisEmbedder(HashEmbedder):
        def embed_query(self, text, task="SearchQuery"):
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            return v

    services = Services(settings, AxisEmbedder(settings.dim))

    def vec(sim: float) -> bytes:
        v = np.zeros(settings.dim, dtype=np.float32)
        v[0], v[1] = sim, (1 - sim**2) ** 0.5
        return v.tobytes()

    with services.db.write() as conn:
        conn.execute("INSERT INTO folders(path, added_at) VALUES (?, ?)", (str(tmp_path), utcnow()))
        conn.execute(
            "INSERT INTO sources(id, folder_id, path, rel_path, kind, status) "
            "VALUES (1, 1, 'a.md', 'a.md', 'document', 'indexed')"
        )
        for seq, (text, sim) in enumerate(
            [("граф цитирований не улучшает точность", 0.76), ("рейсов меньше по сравнению", 0.4)]
        ):
            conn.execute(
                "INSERT INTO chunks(source_id, source_version, seq, modality, text, extractor, "
                "embed_model, vector) VALUES (1, 1, ?, 'text', ?, 'test', ?, ?)",
                (seq, text, services.embedder.name, vec(sim)),
            )
        services.db.bump_generation(conn)

    hits = services.search.search("граф по сравнению с поиском", limit=2)
    assert hits[0]["text"].startswith("граф")
    assert hits[1]["similarity"] == 0.4 and hits[1]["matched_by"] == ["lexical", "semantic"]
    assert hits[0]["score"] > 1.5 * hits[1]["score"]
