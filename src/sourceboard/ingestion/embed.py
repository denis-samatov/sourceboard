"""Embedding backends.

`GemmaEmbedder` wraps EmbeddingGemma 2 through SentenceTransformers. `HashEmbedder` is a
dependency-free, text-only lexical fallback used for tests and for machines without the ML
extras; it is not a semantic model and the UI labels it as such.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from ..core.config import Settings

log = logging.getLogger(__name__)

# Query-side prompt names defined in the model's config_sentence_transformers.json.
QUERY_TASKS = ("SearchQuery", "FactChecking", "QuestionAnswering")


class Embedder(Protocol):
    name: str
    dim: int
    modalities: frozenset[str]
    semantic: bool

    def embed_query(self, text: str, task: str = "SearchQuery") -> np.ndarray: ...

    def embed_documents(self, docs: Sequence[tuple[str | None, str]]) -> np.ndarray: ...

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray: ...

    def embed_audio(self, paths: Sequence[Path]) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix / norm if norm > 0 else matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def read_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def format_document(title: str | None, text: str) -> str:
    """Document-side format recommended by the EmbeddingGemma 2 model card."""
    return f"title: {title.strip() if title and title.strip() else 'none'} | text: {text}"


class GemmaEmbedder:
    """EmbeddingGemma 2 via SentenceTransformers (>= 6.1.0)."""

    semantic = True

    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.model_id
        self.dim = settings.dim
        modalities = {"text"}
        if settings.embed_images:
            modalities.add("image")
        if settings.embed_audio:
            modalities.add("audio")
        self.modalities = frozenset(modalities)
        self.name = f"{settings.model_id}@{settings.dim}"
        self._device = settings.device
        self._model = None
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                # The model card forbids float16 (NaN / silent degradation); bf16 or fp32 only.
                cuda_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                dtype = torch.bfloat16 if cuda_bf16 else torch.float32
                config_kwargs: dict[str, None] = {}
                if "image" not in self.modalities:
                    config_kwargs["vision_config"] = None
                if "audio" not in self.modalities:
                    config_kwargs["audio_config"] = None
                log.info("Loading %s (modalities: %s)", self.model_id, sorted(self.modalities))
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self._device,
                    model_kwargs={"torch_dtype": dtype},
                    config_kwargs=config_kwargs,
                )
        return self._model

    def _encode(self, inputs: list, prompt_name: str | None = None) -> np.ndarray:
        if not inputs:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        with self._encode_lock:  # one model shared by every project's indexer and searches
            vectors = model.encode(
                inputs,
                prompt_name=prompt_name,
                batch_size=8,
                truncate_dim=self.dim,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32).reshape(len(inputs), self.dim)

    def embed_query(self, text: str, task: str = "SearchQuery") -> np.ndarray:
        if task not in QUERY_TASKS:
            raise ValueError(f"Unknown query task {task!r}")
        return self._encode([text], prompt_name=task)[0]

    def embed_documents(self, docs: Sequence[tuple[str | None, str]]) -> np.ndarray:
        return self._encode([format_document(title, text) for title, text in docs])

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        self._require("image")
        return self._encode([{"image": str(p)} for p in paths])

    def embed_audio(self, paths: Sequence[Path]) -> np.ndarray:
        """Audio clips must be 16 kHz mono PCM WAV (the indexer extracts them with ffmpeg).
        Passing arrays avoids an extra audio-decoding dependency in transformers."""
        self._require("audio")
        return self._encode([{"audio": read_wav_16k(p)} for p in paths])

    def _require(self, modality: str) -> None:
        if modality not in self.modalities:
            raise RuntimeError(f"{modality} embeddings are disabled in settings")


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashEmbedder:
    """Signed feature hashing over words and character trigrams. Text only, not semantic."""

    semantic = False
    modalities = frozenset({"text"})

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = f"hash-v1@{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in _TOKEN_RE.findall(text.lower()):
            features = [f"w:{word}"]
            padded = f"<{word}>"
            features += [f"c:{padded[i : i + 3]}" for i in range(max(1, len(padded) - 2))]
            for feature in features:
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[index] += sign * (2.0 if feature.startswith("w:") else 1.0)
        return l2_normalize(vec)

    def embed_query(self, text: str, task: str = "SearchQuery") -> np.ndarray:
        return self._vector(text)

    def embed_documents(self, docs: Sequence[tuple[str | None, str]]) -> np.ndarray:
        if not docs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(text) for _, text in docs])

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        raise RuntimeError("HashEmbedder does not support images")

    def embed_audio(self, paths: Sequence[Path]) -> np.ndarray:
        raise RuntimeError("HashEmbedder does not support audio")


def gemma_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def create_embedder(settings: Settings) -> Embedder:
    choice = settings.embedder.lower()
    if choice == "hash":
        return HashEmbedder(settings.dim)
    if choice in {"gemma", "auto"}:
        if gemma_available():
            return GemmaEmbedder(settings)
        if choice == "gemma":
            raise RuntimeError("EmbeddingGemma 2 needs the ML extras: uv sync --extra ml")
        log.warning("ML extras not installed; falling back to the lexical hash embedder")
        return HashEmbedder(settings.dim)
    raise ValueError(f"Unknown embedder {settings.embedder!r} (expected auto, gemma or hash)")
