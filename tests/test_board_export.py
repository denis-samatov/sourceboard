import csv
import io
import json
from pathlib import Path

import pytest

from sourceboard.api import Services
from sourceboard.board import NotFound
from sourceboard.export import to_json, to_markdown, to_markers_csv


def chunk_id(services: Services, rel_path: str, needle: str) -> int:
    row = services.db.one(
        "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE s.rel_path = ? AND c.text LIKE ?",
        (rel_path, f"%{needle}%"),
    )
    assert row is not None
    return row["id"]


@pytest.fixture
def thesis(indexed: Services) -> dict:
    return indexed.board.create_thesis("Закрытие завода сократило транспорт в районе")


def test_add_evidence_snapshots_locator(indexed: Services, thesis: dict):
    cid = chunk_id(indexed, "interviews/ivanova.srt", "мост")
    ev = indexed.board.add_evidence(thesis["id"], cid, "limits", "альтернативная причина")
    assert ev["t_start"] == 40.0 and ev["role"] == "limits"
    assert "мост" in ev["excerpt"]
    assert ev["issues"] == [] and not ev["needs_recheck"]
    again = indexed.board.add_evidence(thesis["id"], cid, "supports")
    assert again["id"] == ev["id"]  # no duplicates per thesis


def test_invalid_role_and_missing_chunk(indexed: Services, thesis: dict):
    with pytest.raises(ValueError):
        indexed.board.add_evidence(thesis["id"], 1, "proves")
    with pytest.raises(NotFound):
        indexed.board.add_evidence(thesis["id"], 999_999)


def test_source_change_requires_recheck_and_relinks(indexed: Services, thesis: dict, corpus: Path):
    cid = chunk_id(indexed, "docs/minutes.md", "рейсов")
    ev = indexed.board.add_evidence(thesis["id"], cid, "supports")
    ev = indexed.board.update_evidence(ev["id"], context_checked=True)
    assert ev["context_checked"] and not ev["needs_recheck"]

    path = corpus / "docs" / "minutes.md"
    path.write_text(path.read_text(encoding="utf-8").replace("до 9", "до 7"), encoding="utf-8")
    indexed.indexer.scan()

    ev = indexed.board.get_evidence(ev["id"])
    assert [i["code"] for i in ev["issues"]] == ["source_changed"]
    assert ev["needs_recheck"]
    assert ev["chunk_id"] is not None and ev["chunk_id"] != cid  # re-pointed to the new chunk
    assert "до 7" in ev["current_excerpt"] and "до 9" in ev["excerpt"]

    ev = indexed.board.update_evidence(ev["id"], context_checked=True)  # editor re-checks
    assert ev["issues"] == [] and "до 7" in ev["excerpt"] and ev["current_excerpt"] is None


def test_thesis_rewrite_invalidates_checks_unless_minor(indexed: Services, thesis: dict):
    cid = chunk_id(indexed, "docs/minutes.md", "рейсов")
    ev = indexed.board.add_evidence(thesis["id"], cid, "supports")
    indexed.board.update_evidence(ev["id"], context_checked=True)

    t = indexed.board.update_thesis(thesis["id"], title=thesis["title"] + " ")  # whitespace only
    assert t["revision"] == 1 and t["needs_recheck"] == 0

    t = indexed.board.update_thesis(
        thesis["id"], title="Закрытие завода сокротило транспорт", minor=True
    )
    assert t["revision"] == 1 and t["needs_recheck"] == 0

    t = indexed.board.update_thesis(thesis["id"], title="Закрытие завода не повлияло на транспорт")
    assert t["revision"] == 2 and t["needs_recheck"] == 1
    assert t["evidence"][0]["issues"][0]["code"] == "thesis_changed"


def test_removed_folder_keeps_snapshot(indexed: Services, thesis: dict):
    cid = chunk_id(indexed, "docs/transport.docx", "61")
    ev = indexed.board.add_evidence(thesis["id"], cid, "supports")
    folder = indexed.db.one("SELECT id FROM folders")
    indexed.indexer.remove_folder(folder["id"])
    ev = indexed.board.get_evidence(ev["id"])
    assert ev["chunk_id"] is None and ev["source_id"] is None
    assert [i["code"] for i in ev["issues"]] == ["source_deleted"]
    assert "61" in ev["excerpt"]


def test_exports(indexed: Services, thesis: dict):
    srt_chunk = chunk_id(indexed, "interviews/ivanova.srt", "мост")
    pdf_chunk = chunk_id(indexed, "docs/report.pdf", "61 percent")
    indexed.board.add_evidence(thesis["id"], srt_chunk, "limits", "уточнить у перевозчика")
    ev = indexed.board.add_evidence(thesis["id"], pdf_chunk, "supports")
    indexed.board.update_evidence(ev["id"], context_checked=True)

    md = to_markdown(indexed.db)
    assert "## 1. Закрытие завода сократило транспорт в районе" in md
    assert "### Подтверждает" in md and "### Ограничивает" in md
    assert "**docs/report.pdf**, стр. 2" in md and "✓ контекст проверен" in md
    assert "00:00:40–00:00:44" in md and "Заметка: уточнить у перевозчика" in md

    rows = list(csv.DictReader(io.StringIO(to_markers_csv(indexed.db))))
    assert len(rows) == 1  # only time-coded evidence
    assert rows[0]["start"] == "00:00:40.000" and rows[0]["end"] == "00:00:44.000"
    assert rows[0]["role"] == "Ограничивает"

    data = json.loads(to_json(indexed.db))
    assert len(data["theses"][0]["evidence"]) == 2
    assert {s["rel_path"] for s in data["sources"]} >= {"docs/report.pdf"}


def test_counterpoints_prefer_unused_sources(indexed: Services):
    thesis = indexed.board.create_thesis("Пассажиропоток снизился на 61 процент")
    used = chunk_id(indexed, "docs/transport.docx", "61")
    indexed.board.add_evidence(thesis["id"], used, "supports")
    results = indexed.search.counterpoints(thesis["id"])
    assert used not in {r["chunk_id"] for r in results}
    assert results and results[0].get("new_source")


def test_markdown_quote_escapes_block_markers():
    from sourceboard.export import _quote

    assert _quote("# Заголовок\n\n- пункт\nтекст") == "> \\# Заголовок\n>\n> \\- пункт\n> текст"


def test_reprints_count_as_one_independent_source(indexed, corpus):
    para = (
        "Перевозчик сообщил, что после остановки завода утренние рейсы маршрута до "
        "промзоны станут убыточными и число рейсов сократят с двадцати двух до девяти."
    )
    (corpus / "docs" / "reprint_a.txt").write_text(para, encoding="utf-8")
    (corpus / "docs" / "reprint_b.txt").write_text("Газета пишет: " + para, encoding="utf-8")
    indexed.indexer.scan()
    thesis = indexed.board.create_thesis("Рейсы сократили")
    for name in ("reprint_a.txt", "reprint_b.txt"):
        indexed.board.add_evidence(
            thesis["id"], chunk_id(indexed, f"docs/{name}", "промзоны"), "supports"
        )
    t = indexed.board.get_thesis(thesis["id"])
    assert t["counts"]["supports"] == 2 and t["independent"]["supports"] == 1
    assert t["evidence"][0]["copies"][0]["source"] == "docs/reprint_b.txt"
    hits = indexed.search.search("утренние рейсы до промзоны убыточны", limit=5)
    assert any(h.get("copy_of") for h in hits)
