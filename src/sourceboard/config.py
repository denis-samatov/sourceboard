"""Runtime settings, read from environment variables with CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_MODEL_ID = "google/embeddinggemma-2"
SUPPORTED_DIMS = (768, 512, 256, 128)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    # "auto" uses EmbeddingGemma 2 when its dependencies are installed, otherwise "hash".
    embedder: str = "auto"
    model_id: str = DEFAULT_MODEL_ID
    dim: int = 768
    device: str | None = None
    # Which non-text modalities to embed with the model (images also cover video frames and
    # rendered PDF pages). Disabling one skips loading its encoder.
    embed_images: bool = True
    embed_audio: bool = True
    # Local speech recognition (faster-whisper) for media without sidecar subtitles.
    asr_model: str = "small"
    asr_enabled: bool = True
    ocr_enabled: bool = True
    ocr_langs: str = "rus+eng"
    render_pdf_pages: bool = True
    # Chunking heuristics; starting points to tune against an evaluation set.
    text_chunk_chars: int = 900
    transcript_window_s: float = 30.0
    audio_window_s: float = 30.0
    frame_interval_s: float = 10.0

    def __post_init__(self) -> None:
        if self.dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {self.dim}")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sourceboard.sqlite3"

    @property
    def media_dir(self) -> Path:
        """Derived media (video frames, rendered PDF pages, uploaded query images)."""
        return self.data_dir / "media"

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        data_dir = Path(os.environ.get("SOURCEBOARD_DATA_DIR", "~/.sourceboard")).expanduser()
        settings = cls(
            data_dir=data_dir,
            embedder=os.environ.get("SOURCEBOARD_EMBEDDER", "auto"),
            model_id=os.environ.get("SOURCEBOARD_MODEL", DEFAULT_MODEL_ID),
            dim=int(os.environ.get("SOURCEBOARD_DIM", "768")),
            device=os.environ.get("SOURCEBOARD_DEVICE") or None,
            embed_images=_env_bool("SOURCEBOARD_EMBED_IMAGES", True),
            embed_audio=_env_bool("SOURCEBOARD_EMBED_AUDIO", True),
            asr_model=os.environ.get("SOURCEBOARD_ASR_MODEL", "small"),
            asr_enabled=_env_bool("SOURCEBOARD_ASR", True),
            ocr_enabled=_env_bool("SOURCEBOARD_OCR", True),
            ocr_langs=os.environ.get("SOURCEBOARD_OCR_LANGS", "rus+eng"),
            render_pdf_pages=_env_bool("SOURCEBOARD_RENDER_PDF_PAGES", True),
            transcript_window_s=_env_float("SOURCEBOARD_TRANSCRIPT_WINDOW", 30.0),
            audio_window_s=_env_float("SOURCEBOARD_AUDIO_WINDOW", 30.0),
            frame_interval_s=_env_float("SOURCEBOARD_FRAME_INTERVAL", 10.0),
        )
        clean = {k: v for k, v in overrides.items() if v is not None}
        if "data_dir" in clean:
            clean["data_dir"] = Path(str(clean["data_dir"])).expanduser()
        return replace(settings, **clean)  # type: ignore[arg-type]
