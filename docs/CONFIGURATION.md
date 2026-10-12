# SourceBoard Configuration Guide

SourceBoard is configured via environment variables and CLI flags. CLI flags take precedence over environment variables.

## Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `SOURCEBOARD_DATA_DIR` | Path | `~/.sourceboard` | Root directory for SQLite databases, extracted frames, page renders, and cache. |
| `SOURCEBOARD_EMBEDDER` | String | `auto` | Embedding engine: `auto`, `gemma`, or `hash` (lexical fallback). |
| `SOURCEBOARD_DIM` | Integer | `768` | Embedding dimensionality: `768`, `512`, `256`, or `128` (Matryoshka representation). |
| `SOURCEBOARD_DEVICE` | String | `auto` | Compute device: `auto` (selects `cuda` > `mps` > `cpu`), or explicit `cpu`, `mps`, `cuda`. |
| `SOURCEBOARD_EMBED_IMAGES` | Boolean (0/1) | `1` | Enable or disable image encoder loading. |
| `SOURCEBOARD_EMBED_AUDIO` | Boolean (0/1) | `1` | Enable or disable acoustic audio encoder loading. |
| `SOURCEBOARD_ASR` | Boolean (0/1) | `1` | Enable or disable automated speech recognition. |
| `SOURCEBOARD_ASR_MODEL` | String | `small` | Faster-Whisper model checkpoint (e.g., `tiny`, `base`, `small`, `medium`, `large-v3`). |
| `SOURCEBOARD_OCR` | Boolean (0/1) | `1` | Enable or disable OCR text extraction for scanned documents and images. |
| `SOURCEBOARD_OCR_LANGS` | String | `rus+eng` | Tesseract language pack codes (e.g. `eng`, `rus+eng`). |
| `SOURCEBOARD_FRAME_INTERVAL` | Integer | `10` | Video frame sampling step in seconds. |
| `SOURCEBOARD_TRANSCRIPT_WINDOW` | Integer | `30` | Transcript snippet window length in seconds. |
| `SOURCEBOARD_AUDIO_WINDOW` | Integer | `30` | Acoustic audio snippet window length in seconds. |

## Model Weight Management

- Weights are downloaded from Hugging Face Hub on initial startup (~1.5 GB).
- Use `HF_HOME=/path/to/custom/cache` to designate a custom download folder.
- Set `HF_HUB_OFFLINE=1` to run completely offline once weights have been cached locally.
