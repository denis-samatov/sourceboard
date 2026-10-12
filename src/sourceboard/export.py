"""Portable outputs: a Markdown dossier, a CSV of edit markers and a full JSON dump."""

from __future__ import annotations

import csv
import io
import json

from . import __version__
from .board import ROLE_LABELS, ROLES, Board
from .db import Database, utcnow


def timecode(seconds: float | None, millis: bool = False) -> str:
    if seconds is None:
        return ""
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}.{ms:03d}" if millis else base


def locator(item: dict) -> str:
    parts = []
    if item.get("page") is not None:
        parts.append(f"стр. {item['page']}")
    if item.get("t_start") is not None:
        parts.append(f"{timecode(item['t_start'])}–{timecode(item.get('t_end'))}")
    return ", ".join(parts)


def _quote(text: str) -> str:
    """Blockquote with Markdown block markers escaped, so a quoted "# Title" stays plain text."""

    def line_md(line: str) -> str:
        if not line.strip():
            return ">"
        if line.lstrip()[:1] in {"#", ">", "-", "*", "+"}:
            line = "\\" + line.lstrip()
        return f"> {line}"

    return "\n".join(line_md(line) for line in text.strip().splitlines())


MODALITY_LABELS = {"text": "текст", "image": "изображение", "video": "кадр", "audio": "звук"}


def to_markdown(db: Database) -> str:
    board = Board(db)
    theses = board.list_theses()
    lines = [
        "# SourceBoard: подборка источников",
        "",
        f"Сформировано {utcnow()} · SourceBoard {__version__}",
        "",
        "Оценки близости из поиска не являются вероятностью истинности. Роль фрагмента и "
        "проверку контекста отмечает редактор.",
        "",
    ]
    used_sources: dict[str, dict] = {}
    for number, thesis in enumerate(theses, start=1):
        lines += [f"## {number}. {thesis['title']}", ""]
        if thesis["body"]:
            lines += [thesis["body"], ""]
        c = thesis["counts"]
        lines += [
            f"*Фрагментов: {len(thesis['evidence'])} · проверено: {thesis['checked']} · "
            f"требуют повторной проверки: {thesis['needs_recheck']} · "
            f"подтверждает {c['supports']}, ограничивает {c['limits']}, "
            f"противоречит {c['contradicts']}, контекст {c['context']}*",
            "",
        ]
        for role in ROLES:
            items = [e for e in thesis["evidence"] if e["role"] == role]
            if not items:
                continue
            lines += [f"### {ROLE_LABELS[role]}", ""]
            for item in items:
                used_sources.setdefault(item["source"], item)
                where = locator(item)
                head = f"**{item['source']}**" + (f", {where}" if where else "")
                head += f" · {MODALITY_LABELS.get(item['modality'], item['modality'])}"
                head += (
                    " · ✓ контекст проверен"
                    if item["context_checked"]
                    else " · контекст не проверен"
                )
                lines.append(f"- {head}")
                if item["excerpt"].strip():
                    lines += ["", _indent(_quote(item["excerpt"])), ""]
                if item["note"]:
                    lines.append(f"  Заметка: {item['note']}")
                for issue in item["issues"]:
                    lines.append(f"  ⚠ {issue['message']}")
                lines.append("")
    if used_sources:
        lines += ["## Источники", ""]
        for name in sorted(used_sources):
            item = used_sources[name]
            sha = (item.get("source_sha256") or "")[:12]
            lines.append(f"- {name}" + (f" (sha256 {sha}…)" if sha else ""))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


MARKER_FIELDS = [
    "thesis",
    "role",
    "source",
    "start",
    "end",
    "duration_s",
    "excerpt",
    "note",
    "context_checked",
    "needs_recheck",
]


def to_markers_csv(db: Database) -> str:
    """Time-coded evidence (interviews, recordings, video) as an edit-marker table."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MARKER_FIELDS)
    writer.writeheader()
    for thesis in Board(db).list_theses():
        for item in thesis["evidence"]:
            if item["t_start"] is None:
                continue
            end = item["t_end"] if item["t_end"] is not None else item["t_start"]
            writer.writerow(
                {
                    "thesis": thesis["title"],
                    "role": ROLE_LABELS[item["role"]],
                    "source": item["source"],
                    "start": timecode(item["t_start"], millis=True),
                    "end": timecode(end, millis=True),
                    "duration_s": f"{end - item['t_start']:.3f}",
                    "excerpt": " ".join(item["excerpt"].split()),
                    "note": item["note"],
                    "context_checked": "да" if item["context_checked"] else "нет",
                    "needs_recheck": "да" if item["needs_recheck"] else "нет",
                }
            )
    return buffer.getvalue()


def to_json(db: Database) -> str:
    payload = {
        "generator": f"SourceBoard {__version__}",
        "generated_at": utcnow(),
        "theses": Board(db).list_theses(),
        "sources": [
            dict(r)
            for r in db.query(
                "SELECT id, rel_path, path, kind, sha256, version, status, indexed_at "
                "FROM sources ORDER BY rel_path"
            )
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
