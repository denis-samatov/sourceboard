# SourceBoard Architecture Map

SourceBoard is a local multimodal evidence board designed to discover, track, and verify factual claims against heterogeneous source material (documents, images, audio, and video).

## System Overview

```
Sources (PDF, DOCX, Images, Video, Audio, Subtitles)
                       │
                       ▼
        Content Extraction (`extract.py`)
    (pypdf, python-docx, tesseract OCR, faster-whisper, ffmpeg)
                       │
                       ▼
          Chunking & Location Pinning
         (pages, cue spans, timecodes)
                       │
                       ▼
         Multimodal Embedding (`embed.py`)
       (EmbeddingGemma 2 via sentence-transformers)
                       │
                       ▼
          Storage Engine (`db.py`)
   (SQLite3: sources, chunks, FTS5 index, vectors)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Hybrid Search (`search.py`)     Evidence Board (`board.py`)
(FTS5 + Modality Vectors, RRF)   (Claims, Roles, Verification)
       │                               │
       └───────────────┬───────────────┘
                       ▼
           HTTP API & UI (`api.py`, `web/`)
                       │
                       ▼
             Export (`export.py`)
          (Markdown Dossier, CSV, JSON)
```

## Module Responsibilities

| Module | Core Responsibility |
|---|---|
| `config.py` | Runtime configuration, data directory paths, hardware device selection (`cpu`, `mps`, `cuda`), model thresholds. |
| `db.py` | SQLite connection pooling, schema initialization, migrations, FTS5 virtual tables, transaction management. |
| `extract.py` | Media decoding and content extraction: text layers, OCR with Tesseract, ASR with faster-whisper, frame extraction with ffmpeg. |
| `embed.py` | Embedding generation via EmbeddingGemma 2 (`GemmaEmbedder`) with Matryoshka dimension truncation, fallback `HashEmbedder`. |
| `indexer.py` | Directory traversal, SHA-256 fingerprinting, incremental scanning, chunking, and stale locator re-linking. |
| `copies.py` | Near-duplicate detection to identify syndicated releases and reprints ("copies are not corroboration"). |
| `search.py` | Hybrid search combining SQLite FTS5 lexical matching and in-memory cosine similarity across modalities, fused via RRF. |
| `board.py` | Core domain entities: claims, evidence attachments, epistemic role assignments (`supports`, `limits`, `contradicts`, `context`), and verification staleness detection. |
| `export.py` | Serialization to structured Markdown dossiers, editor cut-lists (CSV with timecodes), and full JSON archives. |
| `projects.py` | Multi-project isolation: separate SQLite databases and media folders under `~/.sourceboard/projects/<id>`. |
| `evaluate.py` | Offline retrieval benchmark evaluation computing Recall@K and MRR on JSONL test suites. |
| `api.py` | FastAPI application exposing search, indexing, project management, and serving static web assets. |
| `cli.py` | Command-line interface commands (`serve`, `index`, `search`, `eval`, `export`, `demo`). |
| `demo.py` | Synthetic demonstration dataset generator with simulated hearing footage, meeting minutes, and audio interviews. |
| `web/` | Vanilla JS / CSS / HTML client interface running directly in the browser with no build step required. |

## Data & Storage Schema

SQLite database (`sourceboard.sqlite3`):
- `sources`: File metadata, absolute path, SHA-256 hash, modification time, extraction version.
- `chunks`: Extracted text/frame/audio snippet, modality, page number, timecode start/end, embedding BLOB.
- `chunks_fts`: FTS5 virtual table for lexical indexing.
- `theses`: Claim title, description body, revision number.
- `evidence`: Link between a chunk and a thesis, assigned role, note, `context_checked` flag, and checked revision snapshot.
