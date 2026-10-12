# SourceBoard

A local tool for gathering verifiable evidence to back claims. Search by question, claim, or image
to pinpoint relevant snippets across documents, interviews, videos, and screenshots. Open found
fragments directly in the original file at the exact page or timecode, link them to claims with
assigned evidence roles and context-verification status, and export an evidence dossier with full citations.

Search is powered by [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2): text, images,
and audio are embedded into a unified vector space. Everything runs entirely on your local machine:
indexing, speech recognition, and OCR run locally, and your source files are never transmitted anywhere.
An Internet connection is only required once to download the model weights.

> Search similarity is not a probability of truth. The role of an evidence snippet ("supports",
> "qualifies", "contradicts", "context") and context verification are evaluated and marked by human judgment.

## What It Does

- **Folder Indexing**: PDF (text layer, scanned OCR, page renderings), DOCX, TXT/MD,
  images (OCR and visual index), audio and video (transcripts, video frames, audio embeddings),
  and SRT/VTT subtitles. Subtitles sharing the base name of a media file (`talk.srt`, `talk.en.srt`
  next to `talk.mp4`) are automatically utilized as its transcript.
- **Incremental Updates**: Unchanged files are skipped during scans. When a file changes, its version
  is bumped. If a file disappears or an external drive is disconnected, the catalog remains searchable
  while original sources are flagged as unavailable.
- **Hybrid Search**: Combines full-text search (SQLite FTS5, with prefix matching for inflected forms)
  and modality-specific vector search, merged via Reciprocal Rank Fusion (RRF). This prevents thousands
  of video frames from crowding out crucial document pages. Includes media type filters and search modes:
  "lexical", "semantic", and "hybrid".
- **Image Search**: Upload via button, drag-and-drop, or paste from clipboard. For example, find the exact
  moment in a presentation video where a specific slide was displayed using its screenshot.
- **Claims & Evidence**: Link snippets with designated roles, notes, and a "context checked" toggle.
  "Find Evidence" queries the claim text (or any selected substring) using the `FactChecking` prompt.
  "Alternative View" suggests candidates from previously unused sources (not an exhaustive contradiction search).
- **Verification Tracking**: If a source file changes (SHA-256 mismatch), a fragment goes missing, or a
  claim is significantly edited, the link is flagged for "re-check". Editors can review both the prior
  quote and the updated text. Minor typo fixes can be explicitly flagged as trivial to retain existing verifications.
- **Exporting**: Export Markdown dossiers organized by claim and role with quotes, page numbers, and timecodes;
  CSV cut-lists (edit markers) for video/audio clips; or complete JSON dumps.
- **Quality Evaluation**: `sourceboard eval` computes recall@k and MRR on custom benchmark tasks across
  lexical, vector, and hybrid modes to quantify performance gains.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra ml --extra asr
```

- `ml`: EmbeddingGemma 2 (sentence-transformers ≥ 6.1, transformers ≥ 5.19, torch, torchvision).
  Without this extra, SourceBoard runs in lexical-only mode (the web UI clearly indicates that semantic search is disabled).
- `asr`: faster-whisper for local transcription of audio and video lacking subtitles. Transcripts are cached.
- Optional system dependencies: `ffmpeg` (frame extraction, audio, duration inspection),
  `tesseract` (OCR), and `pdftoppm` from poppler (PDF page rendering).
  On macOS: `brew install ffmpeg tesseract poppler`.

The model (~1.5 GB) is downloaded from Hugging Face on first run. To store weights in a custom path,
set `HF_HOME`. For fully offline operation after initial download, set `HF_HUB_OFFLINE=1`.

## Quick Start

```bash
uv run sourceboard serve --folder ~/Projects/investigation/sources
```

Open <http://127.0.0.1:8765>. The server binds exclusively to localhost and rejects external
`Host` headers (DNS rebinding protection) because it serves local files.

Try it out on synthetic sample data:

```bash
uv run sourceboard demo ~/sourceboard-demo
```

```bash
uv run sourceboard serve --folder ~/sourceboard-demo
```

The demo corpus depicts a fictional factory closure: meeting minutes, scanned resolution (OCR),
DOCX memo, slide deck, interview with subtitles, recorded public hearing with presentation slides,
and an editorial commentary presenting an alternative perspective. On macOS, interviews are voiced
via `say`. Without `say`, subtitles with estimated timecodes and silent video are generated.

### CLI Commands

```bash
uv run sourceboard index ~/sources
```

```bash
uv run sourceboard search "how many bus routes remain after closure" --limit 5
```

```bash
uv run sourceboard eval ~/sourceboard-demo/eval/tasks.jsonl --k 5
```

```bash
uv run sourceboard export --format markdown -o dossier.md
```

Evaluation task format (JSONL, one task per line):

```json
{"query": "discounted transit fare", "expected": [{"source": "documents/resolution_47.pdf", "page": 2}]}
```

`source` matches against the tail of the path. `page` and `t` (seconds, ±2s tolerance) are optional.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `SOURCEBOARD_DATA_DIR` | `~/.sourceboard` | Index database, extracted frames, page renders, transcript cache |
| `SOURCEBOARD_EMBEDDER` | `auto` | `gemma`, `hash` (lexical), or `auto` |
| `SOURCEBOARD_DIM` | `768` | Embedding dimensionality: 768, 512, 256, or 128 (MRL) |
| `SOURCEBOARD_DEVICE` | auto | `cpu`, `mps`, or `cuda` |
| `SOURCEBOARD_EMBED_IMAGES` / `_AUDIO` | `1` | Set to 0 to disable loading image or audio encoders |
| `SOURCEBOARD_ASR`, `SOURCEBOARD_ASR_MODEL` | `1`, `small` | Automatic speech recognition toggle and model name |
| `SOURCEBOARD_OCR`, `SOURCEBOARD_OCR_LANGS` | `1`, `rus+eng` | OCR toggle and language codes |
| `SOURCEBOARD_FRAME_INTERVAL` | `10` | Video frame sampling interval in seconds |
| `SOURCEBOARD_TRANSCRIPT_WINDOW`, `_AUDIO_WINDOW` | `30` | Speech and audio chunk window size in seconds |

CLI flags `--data-dir`, `--embedder`, `--dim`, and `--device` override these environment variables.
Changing model architecture or embedding dimensionality triggers automatic reindexing on the next scan.

## Architecture

```
files ──► extract.py ──► chunks with locator ──► embed.py ──► SQLite
          (pypdf, docx, OCR,    (page / timecode)   (EmbeddingGemma 2) (chunks + FTS5 + vectors)
           subtitles, whisper,                                                 │
           ffmpeg frames, audio)                                               ▼
browser ◄── api.py (FastAPI) ◄── search.py (FTS5 + per-modality vectors, RRF)
                              ◄── board.py (claims, roles, verification, re-check)
                              ◄── export.py (Markdown, CSV, JSON)
```

- `src/sourceboard/extract.py`: Extraction and chunking. Text paragraphs up to ~900 characters,
  30-second speech windows aligned with cue boundaries, video frames every 10 seconds, and audio
  windows every 30 seconds (16 kHz mono).
- `src/sourceboard/embed.py`: Documents are formatted as `title: … | text: …`; queries use task prompts
  `SearchQuery`, `FactChecking`, or `QuestionAnswering`. Enforces bf16 or fp32 (fp16 is prohibited by the
  model card). Matryoshka dimensionality reduction with re-normalization.
- `src/sourceboard/indexer.py`: Folder traversal, SHA-256 fingerprinting, versioning, and re-linking of
  pinned evidence to updated chunks upon file modifications.
- `src/sourceboard/search.py`: Vectors are held in-memory via NumPy and reloaded on index changes.
  Sufficient for tens of thousands of chunks; scalable to LanceDB if needed.
- `src/sourceboard/board.py`: Evidence links preserve snapshots of the matched quote and locator,
  resilient against re-indexing or source folder deletions.
- `src/sourceboard/web/`: Self-contained vanilla web interface without build pipelines or external dependencies.

Index footprint: ~100k vectors × 768 × float32 ≈ 307 MB (excluding metadata and derived media).

## Limitations

- SHA-256 hashes detect file modifications, but do not prove document authenticity.
- Multiple syndications or reprints of the same press release do not constitute independent corroboration.
  Provenance modeling is not yet supported.
- Chunking windows and thresholds (`SIMILARITY_MARGIN`, window durations) are baseline heuristics; calibrate
  them on real-world archives using `sourceboard eval`.
- Audio tracks from media files are indexed as acoustic audio, not speech. Verifiable quotations require
  transcripts (subtitles or faster-whisper).
- DOCX files lack native pagination: exported citations list only the file name, with broader context viewable in the UI.
- On CPU, initial model loading takes 15–20 seconds. Indexing long videos is bounded by speech transcription throughput.

## Development

```bash
uv run pytest
```

```bash
uv run ruff check src tests
```

Tests use the lexical `HashEmbedder` and require neither model downloads nor network access.
