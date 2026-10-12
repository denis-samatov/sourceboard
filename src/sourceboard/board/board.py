"""Theses and the evidence linked to them: roles, context checks and re-check signals."""

from __future__ import annotations

import sqlite3

import numpy as np

from ..core.db import Database, utcnow
from ..ingestion.copies import Fragment, copy_pairs, independent_groups

ROLES = ("supports", "limits", "contradicts", "context")
ROLE_LABELS = {
    "supports": "Подтверждает",
    "limits": "Ограничивает",
    "contradicts": "Противоречит",
    "context": "Контекст",
}
ISSUE_MESSAGES = {
    "source_deleted": "источник удалён из индекса",
    "source_missing": "оригинал недоступен: файл перемещён или диск отключён",
    "source_changed": "источник изменился после добавления фрагмента",
    "fragment_lost": "фрагмент не найден в новой версии источника",
    "thesis_changed": "тезис переписан после проверки контекста",
}
# Issues that invalidate an earlier check. A missing original is informational: the indexed
# copy is unchanged, it just cannot be opened right now.
RECHECK_ISSUES = {"source_deleted", "source_changed", "fragment_lost", "thesis_changed"}


def _norm(text: str) -> str:
    return " ".join(text.split())


class NotFound(KeyError):
    pass


class Board:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ theses

    def list_theses(self) -> list[dict]:
        theses = self.db.query("SELECT * FROM theses ORDER BY position, id")
        return [self._thesis_dict(t) for t in theses]

    def get_thesis(self, thesis_id: int) -> dict:
        return self._thesis_dict(self._thesis_row(thesis_id))

    def create_thesis(self, title: str, body: str = "") -> dict:
        title = _norm(title)
        if not title:
            raise ValueError("Тезис не может быть пустым")
        now = utcnow()
        with self.db.write() as conn:
            position = conn.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM theses").fetchone()[
                0
            ]
            cur = conn.execute(
                "INSERT INTO theses(title, body, position, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (title, body.strip(), position, now, now),
            )
        return self.get_thesis(int(cur.lastrowid))

    def update_thesis(
        self,
        thesis_id: int,
        *,
        title: str | None = None,
        body: str | None = None,
        minor: bool = False,
    ) -> dict:
        """Any wording change bumps the revision (and so flags checked evidence for re-check),
        because a single word such as "не" can flip the meaning. `minor=True` lets the editor
        explicitly declare a typo fix that should not reset checks."""
        row = self._thesis_row(thesis_id)
        new_title = _norm(title) if title is not None else row["title"]
        new_body = body.strip() if body is not None else row["body"]
        if not new_title:
            raise ValueError("Тезис не может быть пустым")
        changed = (new_title, _norm(new_body)) != (row["title"], _norm(row["body"]))
        revision = row["revision"] + (1 if changed and not minor else 0)
        with self.db.write() as conn:
            conn.execute(
                "UPDATE theses SET title = ?, body = ?, revision = ?, updated_at = ? WHERE id = ?",
                (new_title, new_body, revision, utcnow(), thesis_id),
            )
            if changed and minor:
                conn.execute(
                    "UPDATE evidence SET checked_thesis_revision = ? "
                    "WHERE thesis_id = ? AND context_checked = 1 AND checked_thesis_revision = ?",
                    (revision, thesis_id, row["revision"]),
                )
        return self.get_thesis(thesis_id)

    def delete_thesis(self, thesis_id: int) -> None:
        self._thesis_row(thesis_id)
        with self.db.write() as conn:
            conn.execute("DELETE FROM theses WHERE id = ?", (thesis_id,))

    # ------------------------------------------------------------------ evidence

    def add_evidence(
        self, thesis_id: int, chunk_id: int, role: str = "context", note: str = ""
    ) -> dict:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self._thesis_row(thesis_id)
        chunk = self.db.one(
            """SELECT c.*, s.path AS source_path, s.rel_path, s.sha256, s.version
               FROM chunks c JOIN sources s ON s.id = c.source_id WHERE c.id = ?""",
            (chunk_id,),
        )
        if chunk is None:
            raise NotFound(f"chunk {chunk_id}")
        existing = self.db.one(
            "SELECT id FROM evidence WHERE thesis_id = ? AND chunk_id = ?", (thesis_id, chunk_id)
        )
        if existing is not None:
            return self.get_evidence(existing["id"])
        now = utcnow()
        with self.db.write() as conn:
            position = conn.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM evidence WHERE thesis_id = ?",
                (thesis_id,),
            ).fetchone()[0]
            cur = conn.execute(
                """INSERT INTO evidence(thesis_id, chunk_id, source_id, source_path,
                       source_rel_path, source_sha256, source_version, modality, page, t_start,
                       t_end, excerpt, role, note, position, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    thesis_id,
                    chunk_id,
                    chunk["source_id"],
                    chunk["source_path"],
                    chunk["rel_path"],
                    chunk["sha256"],
                    chunk["version"],
                    chunk["modality"],
                    chunk["page"],
                    chunk["t_start"],
                    chunk["t_end"],
                    chunk["text"],
                    role,
                    note.strip(),
                    position,
                    now,
                    now,
                ),
            )
        return self.get_evidence(int(cur.lastrowid))

    def get_evidence(self, evidence_id: int) -> dict:
        rows = self._evidence_rows("e.id = ?", (evidence_id,))
        if not rows:
            raise NotFound(f"evidence {evidence_id}")
        return rows[0]

    def update_evidence(
        self,
        evidence_id: int,
        *,
        role: str | None = None,
        note: str | None = None,
        context_checked: bool | None = None,
    ) -> dict:
        link = self.db.one("SELECT * FROM evidence WHERE id = ?", (evidence_id,))
        if link is None:
            raise NotFound(f"evidence {evidence_id}")
        if role is not None and role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        with self.db.write() as conn:
            if role is not None:
                conn.execute("UPDATE evidence SET role = ? WHERE id = ?", (role, evidence_id))
            if note is not None:
                conn.execute(
                    "UPDATE evidence SET note = ? WHERE id = ?", (note.strip(), evidence_id)
                )
            if context_checked is True:
                self._confirm(conn, link)
            elif context_checked is False:
                conn.execute(
                    "UPDATE evidence SET context_checked = 0, checked_thesis_revision = NULL, "
                    "checked_source_sha256 = NULL WHERE id = ?",
                    (evidence_id,),
                )
            conn.execute("UPDATE evidence SET updated_at = ? WHERE id = ?", (utcnow(), evidence_id))
        return self.get_evidence(evidence_id)

    def _confirm(self, conn: sqlite3.Connection, link: sqlite3.Row) -> None:
        """The editor checked the fragment against the *current* source and thesis: re-anchor the
        snapshot to the current version so later changes are detected relative to it."""
        thesis = conn.execute(
            "SELECT revision FROM theses WHERE id = ?", (link["thesis_id"],)
        ).fetchone()
        source = None
        if link["source_id"] is not None:
            source = conn.execute(
                "SELECT sha256, version FROM sources WHERE id = ?", (link["source_id"],)
            ).fetchone()
        chunk = None
        if link["chunk_id"] is not None:
            chunk = conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (link["chunk_id"],)
            ).fetchone()
        sha = source["sha256"] if source else link["source_sha256"]
        conn.execute(
            """UPDATE evidence SET context_checked = 1, checked_thesis_revision = ?,
                   checked_source_sha256 = ?, source_sha256 = ?, source_version = ?
               WHERE id = ?""",
            (
                thesis["revision"],
                sha,
                sha,
                source["version"] if source else link["source_version"],
                link["id"],
            ),
        )
        if chunk is not None:
            conn.execute(
                "UPDATE evidence SET excerpt = ?, page = ?, t_start = ?, t_end = ? WHERE id = ?",
                (chunk["text"], chunk["page"], chunk["t_start"], chunk["t_end"], link["id"]),
            )

    def delete_evidence(self, evidence_id: int) -> None:
        with self.db.write() as conn:
            cur = conn.execute("DELETE FROM evidence WHERE id = ?", (evidence_id,))
            if cur.rowcount == 0:
                raise NotFound(f"evidence {evidence_id}")

    # ------------------------------------------------------------------ helpers

    def _thesis_row(self, thesis_id: int) -> sqlite3.Row:
        row = self.db.one("SELECT * FROM theses WHERE id = ?", (thesis_id,))
        if row is None:
            raise NotFound(f"thesis {thesis_id}")
        return row

    def _thesis_dict(self, row: sqlite3.Row) -> dict:
        evidence = self._evidence_rows("e.thesis_id = ?", (row["id"],))
        counts = {role: 0 for role in ROLES}
        for item in evidence:
            counts[item["role"]] += 1
        independent = self._mark_copies(evidence)
        return {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "revision": row["revision"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "evidence": evidence,
            "counts": counts,
            "independent": independent,
            "checked": sum(1 for e in evidence if e["context_checked"] and not e["needs_recheck"]),
            "needs_recheck": sum(1 for e in evidence if e["needs_recheck"]),
        }

    def _mark_copies(self, evidence: list[dict]) -> dict[str, int]:
        """Annotate possible copies and count independent sources per role."""
        ids = [e["chunk_id"] for e in evidence if e["chunk_id"] is not None]
        vectors = {}
        if ids:
            rows = self.db.query(
                f"SELECT id, vector FROM chunks WHERE id IN ({','.join('?' * len(ids))})",
                tuple(ids),
            )
            vectors = {
                r["id"]: np.frombuffer(r["vector"], dtype=np.float32)
                for r in rows
                if r["vector"] is not None
            }
        fragments = [
            Fragment(e["id"], e["source"], e["modality"], e["excerpt"], vectors.get(e["chunk_id"]))
            for e in evidence
        ]
        for e in evidence:
            e["copies"] = []
        for i, j in copy_pairs(fragments):
            evidence[i]["copies"].append(
                {"evidence_id": evidence[j]["id"], "source": evidence[j]["source"]}
            )
            evidence[j]["copies"].append(
                {"evidence_id": evidence[i]["id"], "source": evidence[i]["source"]}
            )
        result = {}
        for role in ROLES:
            subset = [f for f, e in zip(fragments, evidence, strict=True) if e["role"] == role]
            result[role] = len(set(independent_groups(subset)))
        return result

    def _evidence_rows(self, where: str, params: tuple) -> list[dict]:
        rows = self.db.query(
            f"""SELECT e.*, t.revision AS thesis_revision, s.sha256 AS current_sha256,
                       s.status AS source_status, s.kind AS source_kind,
                       c.text AS current_text, c.media_path AS media_path
                FROM evidence e
                JOIN theses t ON t.id = e.thesis_id
                LEFT JOIN sources s ON s.id = e.source_id
                LEFT JOIN chunks c ON c.id = e.chunk_id
                WHERE {where}
                ORDER BY e.position, e.id""",
            params,
        )
        return [self._evidence_dict(r) for r in rows]

    @staticmethod
    def _evidence_dict(row: sqlite3.Row) -> dict:
        issues: list[str] = []
        if row["source_id"] is None:
            issues.append("source_deleted")
        else:
            if row["source_status"] == "missing":
                issues.append("source_missing")
            if row["current_sha256"] != row["source_sha256"]:
                issues.append("source_changed")
            if row["chunk_id"] is None:
                issues.append("fragment_lost")
        if row["context_checked"] and row["checked_thesis_revision"] != row["thesis_revision"]:
            issues.append("thesis_changed")
        current = row["current_text"]
        return {
            "id": row["id"],
            "thesis_id": row["thesis_id"],
            "chunk_id": row["chunk_id"],
            "source_id": row["source_id"],
            "source": row["source_rel_path"],
            "source_path": row["source_path"],
            "source_kind": row["source_kind"],
            "source_sha256": row["source_sha256"],
            "modality": row["modality"],
            "page": row["page"],
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "excerpt": row["excerpt"],
            "current_excerpt": current
            if current is not None and current != row["excerpt"]
            else None,
            "has_preview": bool(row["media_path"]) or row["source_kind"] == "image",
            "role": row["role"],
            "note": row["note"],
            "context_checked": bool(row["context_checked"]),
            "issues": [{"code": code, "message": ISSUE_MESSAGES[code]} for code in issues],
            "needs_recheck": any(code in RECHECK_ISSUES for code in issues),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
