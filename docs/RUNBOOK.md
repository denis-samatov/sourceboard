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

Outputs comparative Recall@K and MRR across:
- `lexical` (SQLite FTS5)
- `vector` (EmbeddingGemma 2 cosine similarity)
- `hybrid` (Reciprocal Rank Fusion)

## 4. Troubleshooting System Binaries

| Feature | Dependency | macOS Fix | Linux Fix |
|---|---|---|---|
| Video / Audio Extraction | `ffmpeg` | `brew install ffmpeg` | `sudo apt-get install ffmpeg` |
| Scanned Document OCR | `tesseract` | `brew install tesseract tesseract-lang` | `sudo apt-get install tesseract-ocr tesseract-ocr-rus` |
| PDF Page Rendering | `pdftoppm` (poppler) | `brew install poppler` | `sudo apt-get install poppler-utils` |
