# SourceBoard walkthrough

Use the built-in fictional archive to check a claim about a bus route. You will search a transcript, inspect its context, attach a quotation, and export a cited Markdown dossier.

## Set up the local demo

Requires Python 3.12+, Git, and [uv](https://docs.astral.sh/uv/). The walkthrough uses SQLite text search with the hash fallback. It runs without model weights, OCR, or speech recognition. FFmpeg is optional for the generated sample video.

```bash
git clone https://github.com/denis-samatov/sourceboard.git
cd sourceboard
uv sync --locked

sample_root="$(mktemp -d)"
export SOURCEBOARD_DATA_DIR="$sample_root/index"
export SOURCEBOARD_EMBEDDER=hash SOURCEBOARD_OCR=0 SOURCEBOARD_ASR=0
uv run --locked sourceboard demo "$sample_root/materials"
uv run --locked sourceboard index "$sample_root/materials"
uv run --locked sourceboard serve
```

Open <http://127.0.0.1:8765>. The index and the demo files are isolated in the temporary directory printed by `sourceboard demo`.

## Search, review, and export

1. Add the thesis `Снижение пассажиропотока началось до закрытия завода` (ridership began declining before the factory closed).
2. Select **Только слова** and search for `пассажиров стало меньше`. Open the result from `interviews/ivanova_voditel.srt` to read the quote in context.
3. Attach that passage to the thesis as **Подтверждает** and mark its context as checked. The interviewee recalls that passenger numbers dropped when the bridge closed for repairs. The passage gives a reason to look beyond the factory closure. Corroborating the cause would require independent data about ridership and timing.
4. In the **Экспорт** menu, choose **Документ .md**. Look for the citation, its time range, and the context-check flag.

An excerpt from the exported dossier on Linux:

```markdown
### Подтверждает

- **interviews/ivanova_voditel.srt**, 00:00:01–00:00:27 · текст · ✓ контекст проверен
```

Time ranges can differ when the demo generator uses macOS speech synthesis. The corpus is fictional.

## Repeat from the terminal

With the environment variables above set in the current shell:

```bash
uv run --locked sourceboard search "пассажиров стало меньше" --mode lexical --limit 5
uv run --locked sourceboard export --format markdown -o "$sample_root/dossier.md"
```

To try embedding-based retrieval or transcription on your own archive, see [Configuration](CONFIGURATION.md) and the [runbook](RUNBOOK.md).
