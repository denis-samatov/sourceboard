"""Copy detection: "copies are not corroboration".

Three reprints of one statement are one source, not three confirmations. This module flags
*possible* copies among fragments from different files and counts independent sources:

* text: overlap of word 5-gram shingles (a reprinted paragraph inside a longer chunk still
  shares many shingles);
* images, page renders and video frames: near-identical embeddings (the same slide as a PNG
  and as a frame of a recording).

Flags are suggestions for the editor. They detect shared wording or near-identical visuals, not
provenance: two outlets quoting the same press release will be flagged, an independent witness
using similar words usually will not, but neither case is guaranteed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SHINGLE_WORDS = 5
MIN_WORDS = 8  # shorter fragments share too much boilerplate to judge
CONTAINMENT_THRESHOLD = 0.5
SHARED_SHINGLES_THRESHOLD = 20
VISUAL_SIMILARITY_THRESHOLD = 0.97
VISUAL_MODALITIES = frozenset({"image", "video"})

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def shingles(text: str, n: int = SHINGLE_WORDS) -> frozenset[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < MIN_WORDS:
        return frozenset()
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def text_copy(a: frozenset, b: frozenset) -> bool:
    if not a or not b:
        return False
    shared = len(a & b)
    return shared >= SHARED_SHINGLES_THRESHOLD or shared / min(len(a), len(b)) >= (
        CONTAINMENT_THRESHOLD
    )


@dataclass
class Fragment:
    key: int  # caller's identifier (chunk id, evidence id)
    source: str  # file the fragment comes from
    modality: str
    text: str = ""
    vector: np.ndarray | None = None


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def copy_pairs(fragments: Sequence[Fragment]) -> list[tuple[int, int]]:
    """Index pairs (i < j) of fragments from different files that look like copies."""
    sh = [
        shingles(f.text) if f.modality not in VISUAL_MODALITIES else frozenset() for f in fragments
    ]
    pairs = []
    for j, b in enumerate(fragments):
        for i in range(j):
            a = fragments[i]
            if a.source == b.source:
                continue
            visual = a.modality in VISUAL_MODALITIES and b.modality in VISUAL_MODALITIES
            if visual:
                if (
                    a.vector is not None
                    and b.vector is not None
                    and (float(a.vector @ b.vector) >= VISUAL_SIMILARITY_THRESHOLD)
                ):
                    pairs.append((i, j))
            elif text_copy(sh[i], sh[j]):
                pairs.append((i, j))
    return pairs


def independent_groups(fragments: Sequence[Fragment]) -> list[int]:
    """Group id per fragment: fragments of one file and their copies share a group.
    The number of distinct ids is the number of independent sources."""
    uf = _UnionFind(len(fragments))
    first_by_source: dict[str, int] = {}
    for i, f in enumerate(fragments):
        if f.source in first_by_source:
            uf.union(i, first_by_source[f.source])
        else:
            first_by_source[f.source] = i
    for i, j in copy_pairs(fragments):
        uf.union(i, j)
    roots = [uf.find(i) for i in range(len(fragments))]
    renumber: dict[int, int] = {}
    return [renumber.setdefault(r, len(renumber)) for r in roots]
