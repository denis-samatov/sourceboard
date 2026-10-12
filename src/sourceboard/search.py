"""Hybrid search: exact-term FTS5 plus per-modality vector search, fused with RRF.

Vector candidates are ranked separately for each modality before fusion, so a large pile of
video frames cannot push a handful of relevant document pages out of the results. Each vector
candidate is additionally damped by how far its similarity trails the best vector hit overall,
so the top of a weak modality does not outrank strong hits of another one. Scores are
rank-fusion scores, not probabilities and not a measure of truth.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .copies import Fragment, copy_pairs
from .db import Database
from .embed import Embedder

RRF_K = 60
# Candidates whose similarity trails the best vector hit by this much get the minimum weight.
# A heuristic starting point; tune it with `sourceboard eval` on a real archive.
SIMILARITY_MARGIN = 0.15
MIN_VECTOR_WEIGHT = 0.1
MIN_LEXICAL_WEIGHT = 0.3


def _damping(best: float, sim: float) -> float:
    """1.0 near the best vector hit, falling linearly to MIN_VECTOR_WEIGHT over the margin."""
    return min(1.0, max(MIN_VECTOR_WEIGHT, 1.0 - (best - sim) / SIMILARITY_MARGIN))


MODES = ("hybrid", "lexical", "vector")

_STOPWORDS = frozenset(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
    ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть
    был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть
    надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто
    этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
    куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти
    нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше
    чуть том нельзя такой им более всегда конечно всю между
    the a an and or of to in on for with by is are was were be been it this that as at from
    """.split()
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def fts_query(text: str) -> str | None:
    """OR-query with crude prefix matching so Russian inflections still hit (завода → заво*)."""
    terms: list[str] = []
    for token in _WORD_RE.findall(text.lower()):
        if len(token) < 2 or token in _STOPWORDS:
            continue
        terms.append(f'"{token}"')
        if len(token) >= 6 and token.isalpha():
            terms.append(f"{token[: len(token) - 2]}*")
    return " OR ".join(dict.fromkeys(terms)) or None


@dataclass
class Candidate:
    chunk_id: int
    score: float = 0.0
    similarity: float | None = None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    reasons: list[str] = field(default_factory=list)


class VectorIndex:
    """All chunk vectors of the current model, held in memory and reloaded on index changes."""

    def __init__(self, db: Database, model_name: str) -> None:
        self.db = db
        self.model_name = model_name
        self._lock = threading.Lock()
        self._generation = -1
        self.ids = np.zeros(0, dtype=np.int64)
        self.source_ids = np.zeros(0, dtype=np.int64)
        self.modalities = np.zeros(0, dtype=object)
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    def ensure_fresh(self) -> None:
        generation = self.db.generation()
        if generation == self._generation:
            return
        with self._lock:
            if generation == self._generation:
                return
            rows = self.db.query(
                """SELECT id, source_id, modality, vector FROM chunks
                   WHERE vector IS NOT NULL AND embed_model = ? ORDER BY id""",
                (self.model_name,),
            )
            self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
            self.source_ids = np.array([r["source_id"] for r in rows], dtype=np.int64)
            self.modalities = np.array([r["modality"] for r in rows], dtype=object)
            self.matrix = (
                np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, 0), dtype=np.float32)
            )
            self._generation = generation

    def similarities(self, query: np.ndarray, chunk_ids: Iterable[int]) -> dict[int, float]:
        self.ensure_fresh()
        idx = np.nonzero(np.isin(self.ids, list(chunk_ids)))[0]
        if not len(idx):
            return {}
        sims = self.matrix[idx] @ query.astype(np.float32)
        return {int(self.ids[i]): float(sim) for i, sim in zip(idx, sims, strict=True)}

    def vector_for(self, chunk_id: int) -> np.ndarray | None:
        self.ensure_fresh()
        found = np.nonzero(self.ids == chunk_id)[0]
        return self.matrix[found[0]] if len(found) else None

    def search(
        self,
        query: np.ndarray,
        k: int,
        modalities: Iterable[str] | None = None,
        source_ids: Iterable[int] | None = None,
        exclude: Iterable[int] = (),
    ) -> dict[str, list[tuple[int, float]]]:
        self.ensure_fresh()
        if not len(self.ids):
            return {}
        sims = self.matrix @ query.astype(np.float32)
        mask = np.ones(len(self.ids), dtype=bool)
        if source_ids is not None:
            mask &= np.isin(self.source_ids, list(source_ids))
        excluded = list(exclude)
        if excluded:
            mask &= ~np.isin(self.ids, excluded)
        wanted = set(modalities) if modalities else set(self.modalities.tolist())
        result: dict[str, list[tuple[int, float]]] = {}
        for modality in wanted:
            idx = np.nonzero(mask & (self.modalities == modality))[0]
            if not len(idx):
                continue
            top = idx[np.argsort(-sims[idx], kind="stable")[:k]]
            result[modality] = [(int(self.ids[i]), float(sims[i])) for i in top]
        return result


class SearchEngine:
    def __init__(self, db: Database, embedder: Embedder) -> None:
        self.db = db
        self.embedder = embedder
        self.vectors = VectorIndex(db, embedder.name)

    # ------------------------------------------------------------------ public API

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        modalities: list[str] | None = None,
        source_ids: list[int] | None = None,
        exclude: Iterable[int] = (),
        task: str = "SearchQuery",
        mode: str = "hybrid",
    ) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        query = query.strip()
        if not query:
            return []
        excluded = set(exclude)
        pool = max(limit * 3, 50)
        candidates: dict[int, Candidate] = {}
        lexical = []
        if mode in {"hybrid", "lexical"}:
            lexical = self._lexical(query, pool, modalities, source_ids, excluded)
        best, sims = None, {}
        if mode in {"hybrid", "vector"}:
            qvec = self.embedder.embed_query(query, task=task)
            best = self._fuse_vectors(candidates, qvec, pool, modalities, source_ids, excluded)
            if lexical and best is not None:
                sims = self.vectors.similarities(qvec, lexical)
        for rank, chunk_id in enumerate(lexical, start=1):
            # In hybrid mode a word match far from the query's meaning (e.g. a shared stock
            # phrase) is damped, but never below MIN_LEXICAL_WEIGHT: exact names and numbers
            # must stay findable.
            sim = sims.get(chunk_id)
            weight = 1.0 if sim is None else max(MIN_LEXICAL_WEIGHT, _damping(best, sim))
            cand = candidates.setdefault(chunk_id, Candidate(chunk_id))
            cand.lexical_rank = rank
            cand.score += weight / (RRF_K + rank)
            cand.reasons.append("lexical")
            if cand.similarity is None and sim is not None:
                cand.similarity = sim
        return self._finish(candidates, limit)

    def search_by_vector(
        self,
        vector: np.ndarray,
        *,
        limit: int = 20,
        modalities: list[str] | None = None,
        source_ids: list[int] | None = None,
        exclude: Iterable[int] = (),
    ) -> list[dict]:
        candidates: dict[int, Candidate] = {}
        self._fuse_vectors(
            candidates, vector, max(limit * 3, 50), modalities, source_ids, set(exclude)
        )
        return self._finish(candidates, limit)

    def search_by_image(self, image: Path, **kwargs) -> list[dict]:
        if "image" not in self.embedder.modalities:
            raise RuntimeError("Поиск по изображению требует EmbeddingGemma 2 с vision-энкодером")
        return self.search_by_vector(self.embedder.embed_images([image])[0], **kwargs)

    def similar(
        self, chunk_id: int, *, limit: int = 20, modalities: list[str] | None = None
    ) -> list[dict]:
        vector = self.vectors.vector_for(chunk_id)
        if vector is not None:
            return self.search_by_vector(
                vector, limit=limit, modalities=modalities, exclude=[chunk_id]
            )
        row = self.db.one("SELECT text FROM chunks WHERE id = ?", (chunk_id,))
        if row is None or not row["text"].strip():
            return []
        return self.search(
            row["text"][:1000], limit=limit, modalities=modalities, exclude=[chunk_id]
        )

    def counterpoints(self, thesis_id: int, *, limit: int = 12) -> list[dict]:
        """Candidates that may qualify or contradict a thesis: fact-checking query, already
        linked fragments excluded, sources not yet used for this thesis ranked first.
        This is a prompt for review, not an exhaustive search for contradictions."""
        thesis = self.db.one("SELECT * FROM theses WHERE id = ?", (thesis_id,))
        if thesis is None:
            raise KeyError(thesis_id)
        linked = self.db.query(
            "SELECT chunk_id, source_id FROM evidence WHERE thesis_id = ?", (thesis_id,)
        )
        used_chunks = {r["chunk_id"] for r in linked if r["chunk_id"] is not None}
        used_sources = {r["source_id"] for r in linked if r["source_id"] is not None}
        claim = f"{thesis['title']}. {thesis['body']}".strip(". ")
        task = "FactChecking" if self.embedder.semantic else "SearchQuery"
        results = self.search(claim, limit=limit * 2, exclude=used_chunks, task=task)
        fresh = [r for r in results if r["source_id"] not in used_sources]
        known = [r for r in results if r["source_id"] in used_sources]
        for r in fresh:
            r["new_source"] = True
        return (fresh + known)[:limit]

    # ------------------------------------------------------------------ internals

    def _lexical(
        self,
        query: str,
        k: int,
        modalities: list[str] | None,
        source_ids: list[int] | None,
        exclude: set[int],
    ) -> list[int]:
        match = fts_query(query)
        if match is None:
            return []
        sql = """SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id = f.rowid
                 WHERE chunks_fts MATCH ?"""
        params: list = [match]
        if modalities:
            sql += f" AND c.modality IN ({','.join('?' * len(modalities))})"
            params += modalities
        if source_ids:
            sql += f" AND c.source_id IN ({','.join('?' * len(source_ids))})"
            params += source_ids
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(k + len(exclude))
        ids = [row["id"] for row in self.db.query(sql, tuple(params))]
        return [i for i in ids if i not in exclude][:k]

    def _fuse_vectors(
        self,
        candidates: dict[int, Candidate],
        qvec: np.ndarray,
        k: int,
        modalities: list[str] | None,
        source_ids: list[int] | None,
        exclude: set[int],
    ) -> float | None:
        """Add vector candidates; returns the best similarity seen (None if no vectors)."""
        per_modality = self.vectors.search(qvec, k, modalities, source_ids, exclude)
        if not per_modality:
            return None
        best = max(hits[0][1] for hits in per_modality.values() if hits)
        for hits in per_modality.values():
            for rank, (chunk_id, sim) in enumerate(hits, start=1):
                weight = _damping(best, sim)
                cand = candidates.setdefault(chunk_id, Candidate(chunk_id))
                cand.vector_rank = rank
                cand.similarity = sim
                cand.score += weight / (RRF_K + rank)
                cand.reasons.append("semantic")
        return best

    def _finish(self, candidates: dict[int, Candidate], limit: int) -> list[dict]:
        ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)[:limit]
        results = self.hydrate(ranked)
        self._mark_copies(results)
        return results

    def _mark_copies(self, results: list[dict]) -> None:
        """Flag results that look like a copy of a higher-ranked result from another file."""
        fragments = [
            Fragment(
                r["chunk_id"],
                r["source"],
                r["modality"],
                r["text"],
                self.vectors.vector_for(r["chunk_id"]),
            )
            for r in results
        ]
        for i, j in copy_pairs(fragments):
            results[j].setdefault(
                "copy_of", {"chunk_id": results[i]["chunk_id"], "source": results[i]["source"]}
            )

    def hydrate(self, ranked: list[Candidate]) -> list[dict]:
        if not ranked:
            return []
        ids = [c.chunk_id for c in ranked]
        rows = self.db.query(
            f"""SELECT c.id, c.source_id, c.seq, c.modality, c.page, c.t_start, c.t_end, c.text,
                       c.extractor, c.media_path, s.rel_path, s.kind, s.status, s.duration
                FROM chunks c JOIN sources s ON s.id = c.source_id
                WHERE c.id IN ({",".join("?" * len(ids))})""",
            tuple(ids),
        )
        by_id = {row["id"]: row for row in rows}
        out = []
        for cand in ranked:
            row = by_id.get(cand.chunk_id)
            if row is None:
                continue
            out.append(
                {
                    "chunk_id": row["id"],
                    "source_id": row["source_id"],
                    "source": row["rel_path"],
                    "source_kind": row["kind"],
                    "source_status": row["status"],
                    "modality": row["modality"],
                    "page": row["page"],
                    "t_start": row["t_start"],
                    "t_end": row["t_end"],
                    "text": row["text"],
                    "extractor": row["extractor"],
                    "has_preview": bool(row["media_path"]) or row["kind"] == "image",
                    "score": round(cand.score, 6),
                    "similarity": None if cand.similarity is None else round(cand.similarity, 4),
                    "lexical_rank": cand.lexical_rank,
                    "vector_rank": cand.vector_rank,
                    "matched_by": sorted(set(cand.reasons)),
                }
            )
        return out
