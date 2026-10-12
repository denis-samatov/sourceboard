"""Ingestion pipeline: media extraction, embeddings, incremental indexing, and copy detection."""

from .copies import Fragment, copy_pairs, independent_groups
from .embed import Embedder, GemmaEmbedder, HashEmbedder, create_embedder
from .extract import (
    DOCUMENT_EXT,
    IMAGE_EXT,
    SUBTITLE_EXT,
    Cue,
    Extraction,
    Extractor,
    IndexCancelled,
    Piece,
    asr_available,
    cues_to_srt,
    kind_for,
    parse_subtitles,
    parse_timestamp,
    sha256_file,
    split_text,
    tool,
    window_cues,
)
from .indexer import Indexer, discover

__all__ = [
    "DOCUMENT_EXT",
    "IMAGE_EXT",
    "SUBTITLE_EXT",
    "Cue",
    "Embedder",
    "Extraction",
    "Extractor",
    "Fragment",
    "GemmaEmbedder",
    "HashEmbedder",
    "IndexCancelled",
    "Indexer",
    "Piece",
    "asr_available",
    "copy_pairs",
    "create_embedder",
    "cues_to_srt",
    "discover",
    "independent_groups",
    "kind_for",
    "parse_subtitles",
    "parse_timestamp",
    "sha256_file",
    "split_text",
    "tool",
    "window_cues",
]
