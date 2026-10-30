from pathlib import Path

from sourceboard.ingestion import (
    Cue,
    cues_to_srt,
    discover,
    kind_for,
    parse_subtitles,
    parse_timestamp,
    split_text,
    window_cues,
)


def test_split_text_respects_limit_and_keeps_paragraphs():
    text = "Первый абзац.\n\nВторой абзац.\n\n" + "Длинное предложение номер. " * 80
    chunks = split_text(text, 300)
    assert chunks[0].startswith("Первый абзац.\n\nВторой абзац.")
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks).count("Длинное") == 80


def test_split_text_hard_wraps_unbroken_text():
    chunks = split_text("x" * 1000, 300)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) == 1000


def test_parse_timestamps():
    assert parse_timestamp("00:01:02,345") == 62.345
    assert parse_timestamp("01:02.5") == 62.5
    assert parse_timestamp("1:00:00.000") == 3600


def test_parse_srt_and_vtt():
    srt = "1\n00:00:01,000 --> 00:00:02,500\nПривет\n\n2\n00:00:03,000 --> 00:00:04,000\nмир\n"
    vtt = "WEBVTT\n\n00:01.000 --> 00:02.500 align:start\n<v Иван>Привет</v>\n"
    assert parse_subtitles(srt) == [Cue(1.0, 2.5, "Привет"), Cue(3.0, 4.0, "мир")]
    assert parse_subtitles(vtt) == [Cue(1.0, 2.5, "Привет")]


def test_srt_round_trip():
    cues = [Cue(0.5, 2.25, "раз"), Cue(3661.0, 3662.0, "два")]
    assert parse_subtitles(cues_to_srt(cues)) == cues


def test_window_cues_groups_by_time():
    cues = [Cue(i * 10.0, i * 10.0 + 5, f"фраза {i}") for i in range(7)]
    pieces = window_cues(cues, window_s=30, max_chars=1000, extractor="srt")
    assert [(p.t_start, p.t_end) for p in pieces] == [(0, 25), (30, 55), (60, 65)]
    assert pieces[0].text == "фраза 0 фраза 1 фраза 2"


def test_kind_detection():
    assert kind_for(Path("a.PDF")) == "document"
    assert kind_for(Path("a.mov")) == "video"
    assert kind_for(Path("a.vtt")) == "transcript"
    assert kind_for(Path("a.xyz")) is None


def test_discover_pairs_subtitles_with_media(tmp_path: Path):
    (tmp_path / "talk.mp4").write_bytes(b"")
    (tmp_path / "talk.ru.srt").write_text("", encoding="utf-8")
    (tmp_path / "orphan.srt").write_text("", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.txt").write_text("", encoding="utf-8")
    (tmp_path / "app" / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "app" / "node_modules" / "pkg" / "README.md").write_text("", encoding="utf-8")
    entries = {e.path.name: e for e in discover(tmp_path)}
    assert set(entries) == {"talk.mp4", "orphan.srt"}
    assert entries["talk.mp4"].sidecar == tmp_path / "talk.ru.srt"
    assert entries["orphan.srt"].kind == "transcript"


def test_invalid_chunk_and_media_windows_fail_before_extraction(tmp_path):
    import pytest

    from sourceboard.core import Settings

    for field in ("text_chunk_chars", "transcript_window_s", "audio_window_s", "frame_interval_s"):
        for value in (0, -1, float("inf"), float("nan")):
            with pytest.raises(ValueError):
                Settings(data_dir=tmp_path, **{field: value})


def test_docx_expansion_limit_is_checked_before_xml_parser(tmp_path, monkeypatch):
    import zipfile

    import docx
    import pytest

    import sourceboard.ingestion.extract as extraction
    from sourceboard.core import Settings

    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"x" * 4096)
    monkeypatch.setattr(extraction, "MAX_DOCX_EXPANDED_BYTES", 1024, raising=False)
    monkeypatch.setattr(docx, "Document", lambda *args: pytest.fail("XML parser must not run"))
    with pytest.raises(ValueError, match="expanded"):
        extraction.Extractor(Settings(data_dir=tmp_path), frozenset())._docx(path)


def test_image_pixel_limit_precedes_ocr(tmp_path, monkeypatch):
    import pytest
    from PIL import Image

    import sourceboard.ingestion.extract as extraction
    from sourceboard.core import Settings

    path = tmp_path / "large.png"
    Image.new("RGB", (16, 16)).save(path)
    monkeypatch.setattr(extraction, "MAX_IMAGE_PIXELS", 100, raising=False)
    monkeypatch.setattr(extraction, "ocr_image", lambda *args: pytest.fail("OCR must not run"))
    with pytest.raises(ValueError, match="pixel"):
        extraction.Extractor(Settings(data_dir=tmp_path), frozenset({"image"}))._image(path)


def test_image_query_checks_pixels_before_embedding(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest
    from PIL import Image

    import sourceboard.ingestion.extract as extraction
    from sourceboard.search.search import SearchEngine

    path = tmp_path / "query.png"
    Image.new("RGB", (16, 16)).save(path)
    monkeypatch.setattr(extraction, "MAX_IMAGE_PIXELS", 100)
    embedder = SimpleNamespace(
        name="fake",
        modalities={"image"},
        embed_images=lambda *args: pytest.fail("model must not run"),
    )
    with pytest.raises(ValueError, match="pixel"):
        SearchEngine(None, embedder).search_by_image(path)
