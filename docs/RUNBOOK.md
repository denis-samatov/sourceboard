# SourceBoard Operations & Runbook

Operational guide for running, maintaining, and benchmarking SourceBoard.

## 1. Local Deployment

Start the service against a target folder:

```bash
uv run sourceboard serve --folder /path/to/archive --port 8765
```

Access the UI at `http://127.0.0.1:8765`.

The service binds only to loopback interfaces (`127.0.0.1`, `localhost`) and rejects requests with unexpected `Host` headers to prevent DNS rebinding attacks.

## 2. Generating Demo Corpus

To evaluate the system without existing documents, generate the bundled fictional demo corpus:

```bash
uv run sourceboard demo ~/sourceboard-demo
uv run sourceboard serve --folder ~/sourceboard-demo
```

## 3. Evaluation & Retrieval Benchmarking

Run automated retrieval benchmarks:

```bash
uv run sourceboard eval ~/sourceboard-demo/eval/tasks.jsonl --k 5
```

Outputs comparative Hit@K and MRR across:
- `lexical` (SQLite FTS5)
- `vector` (EmbeddingGemma 2 cosine similarity)
- `hybrid` (Reciprocal Rank Fusion)

## 4. Troubleshooting System Binaries

| Feature | Dependency | macOS Fix | Linux Fix |
|---|---|---|---|
| Video / Audio Extraction | `ffmpeg` | `brew install ffmpeg` | `sudo apt-get install ffmpeg` |
| Scanned Document OCR | `tesseract` | `brew install tesseract tesseract-lang` | `sudo apt-get install tesseract-ocr tesseract-ocr-rus` |
| PDF Page Rendering | `pdftoppm` (poppler) | `brew install poppler` | `sudo apt-get install poppler-utils` |

## Incomplete embeddings and resource boundaries

An unchanged source is skipped only if vectors required by the current embedder are
present. After a transient embedding failure, another ordinary scan retries that source;
its byte version and retained evidence excerpt do not change. Unsupported modalities and
blank text do not cause endless retries. A failed modality remains visible in source notes.

The image-search endpoint accepts at most 10 MiB of image bytes and bounds the complete
multipart body to that plus 64 KiB before parsing/spooling (including chunked requests).
An upload body must finish within 10 seconds. This bounds that HTTP path, not arbitrary
PDF/OCR/ASR processing. Use the local CLI and trusted archives; do not expose the app as a
remote file-processing service. For hostile archives, separate process/container resource
limits and parser advisory triage are prerequisites, not properties claimed by this tool.

## Reproducible archive acceptance and restore

Before claiming semantic quality:

1. Select an authorized archive and annotate expected pages/timecodes before tuning.
   Keep a final untouched set separate from the development queries.
2. Use `SOURCEBOARD_MODEL` to point at an immutable approved local Hugging Face snapshot;
   record its revision and file hashes. Set `HF_HUB_OFFLINE=1` for acceptance after the
   separately authorized download. The moving default model ID is not an immutable pin.
3. Record Python/locked package versions, OCR language packs and `ffmpeg`, `tesseract`,
   `pdftoppm` versions. Run lexical/vector/hybrid `sourceboard eval` on identical queries.
   Interpret `hit@k` as any-relevant-query success; `recall@k` is its legacy JSON alias.
4. Measure indexing/search latency, RAM/disk and media/OCR/ASR failures at the intended
   archive size. The in-memory dense vector index is not an unlimited-scale service.
5. Stop indexing, make a consistent backup of the complete data directory (including
   SQLite WAL state via SQLite backup tooling), media and source mapping; restore into a
   separate data directory with the same model snapshot and copied test originals.
   Confirm claims, quote snapshots, re-check status, locators and exports before resuming.

Real Gemma/media execution, annotated archive quality, parser isolation and target restore
are external acceptance steps. Offline HashEmbedder tests and a wheel smoke do not prove them.
