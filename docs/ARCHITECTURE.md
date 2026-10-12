# SourceBoard Architecture Map

SourceBoard is a local multimodal evidence board designed to discover, track, and verify factual claims against heterogeneous source material (documents, images, audio, and video).

## System Overview

```
Sources (PDF, DOCX, Images, Video, Audio, Subtitles)
                       │
                       ▼
    Ingestion Pipeline (`ingestion/extract.py`)
 (pypdf, python-docx, tesseract OCR, faster-whisper, ffmpeg)
                       │
                       ▼
          Chunking & Location Pinning
         (pages, cue spans, timecodes)
                       │
                       ▼
    Multimodal Embedding (`ingestion/embed.py`)
   (EmbeddingGemma 2 via sentence-transformers)
                       │
                       ▼
          Storage Engine (`core/db.py`)
   (SQLite3: sources, chunks, FTS5 index, vectors)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Hybrid Search (`search/search.py`)  Evidence Board (`board/board.py`)
(FTS5 + Modality Vectors, RRF)      (Claims, Roles, Verification)
       │                               │
       └───────────────┬───────────────┘
                       ▼
        HTTP API & Web UI (`api/api.py`, `api/web/`)
                       │
                       ▼
          Export Engine (`board/export.py`)
           (Markdown Dossier, CSV, JSON)
```

## Package Structure & Module Responsibilities

The codebase in `src/sourceboard/` is partitioned into focused, cohesive subpackages:

### `sourceboard.core`
| Module | Responsibility |
|---|---|
| `core/config.py` | Runtime configuration (`Settings`), data directory paths, hardware device selection (`cpu`, `mps`, `cuda`), model thresholds. |
| `core/db.py` | SQLite connection pooling (`Database`), schema initialization, migrations, FTS5 virtual tables, transaction management. |
| `core/projects.py` | Multi-project isolation (`ProjectRegistry`, `Project`): separate SQLite databases and media folders under `~/.sourceboard/projects/<id>`. |

### `sourceboard.ingestion`
| Module | Responsibility |
|---|---|
| `ingestion/extract.py` | Media decoding and content extraction (`Extractor`): text layers, OCR with Tesseract, ASR with faster-whisper, frame extraction with ffmpeg. |
| `ingestion/embed.py` | Embedding generation via EmbeddingGemma 2 (`GemmaEmbedder`) with Matryoshka dimension truncation, fallback `HashEmbedder`. |
| `ingestion/indexer.py` | Directory traversal (`Indexer`), SHA-256 fingerprinting, incremental scanning, chunking, and stale locator re-linking. |
| `ingestion/copies.py` | Near-duplicate detection to identify syndicated releases and reprints ("copies are not corroboration"). |

### `sourceboard.search`
| Module | Responsibility |
|---|---|
| `search/search.py` | Hybrid search engine (`SearchEngine`) combining SQLite FTS5 lexical matching and in-memory cosine similarity across modalities, fused via RRF. |
| `search/evaluate.py` | Offline retrieval benchmark evaluation computing Recall@K and MRR on JSONL test suites (`evaluate`). |

### `sourceboard.board`
| Module | Responsibility |
|---|---|
| `board/board.py` | Core domain entities (`Board`): claims, evidence attachments, epistemic role assignments (`supports`, `limits`, `contradicts`, `context`), and verification staleness detection. |
| `board/export.py` | Serialization to structured Markdown dossiers (`to_markdown`), editor cut-lists (`to_markers_csv`), and full JSON archives (`to_json`). |

### `sourceboard.api`
| Module | Responsibility |
|---|---|
| `api/api.py` | FastAPI application (`create_app`, `Services`) exposing search, indexing, project management, and serving static web assets. |
| `api/web/` | Vanilla JS / CSS / HTML client interface running directly in the browser with no build step required. |

### `sourceboard.cli`
| Module | Responsibility |
|---|---|
| `cli/cli.py` | Command-line interface entrypoint (`main`) providing subcommands: `serve`, `index`, `search`, `eval`, `export`, `demo`. |
| `cli/demo.py` | Synthetic demonstration dataset generator (`make_demo`) with simulated hearing footage, meeting minutes, and audio interviews. |

## Data & Storage Schema

SQLite database (`sourceboard.sqlite3`):
- `sources`: File metadata, absolute path, SHA-256 hash, modification time, extraction version.
- `chunks`: Extracted text/frame/audio snippet, modality, page number, timecode start/end, embedding BLOB.
- `chunks_fts`: FTS5 virtual table for lexical indexing.
- `theses`: Claim title, description body, revision number.
- `evidence`: Link between a chunk and a thesis, assigned role, note, `context_checked` flag, and checked revision snapshot.
