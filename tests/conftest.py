from __future__ import annotations

from pathlib import Path

import pytest

from sourceboard.api import Services
from sourceboard.core import Settings
from sourceboard.ingestion import HashEmbedder


def make_pdf(pages: list[str]) -> bytes:
    """Tiny text-layer PDF (Helvetica, ASCII) for extraction tests."""
    objects: list[bytes] = []
    n_pages = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        content_id = 5 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


SRT = """1
00:00:01,000 --> 00:00:04,000
Я двенадцать лет вожу четырнадцатый маршрут.

2
00:00:05,000 --> 00:00:09,500
После закрытия завода утренних рейсов оставили всего несколько.

3
00:00:40,000 --> 00:00:44,000
Пассажиров стало меньше ещё осенью, когда закрыли мост.
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    import docx

    root = tmp_path / "corpus"
    (root / "docs").mkdir(parents=True)
    (root / "interviews").mkdir()
    (root / "docs" / "minutes.md").write_text(
        "# Протокол\n\nЗавод «Северный кабель» прекращает производство с 1 марта 2025 года.\n\n"
        "Маршрут автобуса № 14 сокращается с 22 до 9 рейсов в сутки.\n",
        encoding="utf-8",
    )
    (root / "docs" / "report.pdf").write_bytes(
        make_pdf(["Budget review for the bridge repair", "Bus ridership fell by 61 percent"])
    )
    document = docx.Document()
    document.add_paragraph("Справка перевозчика: пассажиропоток снизился на 61 процент.")
    document.save(str(root / "docs" / "transport.docx"))
    (root / "interviews" / "ivanova.srt").write_text(SRT, encoding="utf-8")
    (root / ".hidden.txt").write_text("should be ignored", encoding="utf-8")
    (root / "unknown.xyz").write_text("ignored", encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        embedder="hash",
        dim=256,
        ocr_enabled=False,
        render_pdf_pages=False,
        asr_enabled=False,
    )


@pytest.fixture
def services(settings: Settings) -> Services:
    return Services(settings, HashEmbedder(settings.dim))


@pytest.fixture
def indexed(services: Services, corpus: Path) -> Services:
    folder_id = services.indexer.add_folder(corpus)
    stats = services.indexer.scan([folder_id])
    assert stats["errors"] == 0, services.db.query("SELECT rel_path, error FROM sources")
    return services
