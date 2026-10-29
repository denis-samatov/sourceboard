"""Command line entry point: `sourceboard serve | index | search | eval | export | demo`."""

from __future__ import annotations

import argparse
import errno
import json
import logging
import socket
import sys
from pathlib import Path

from ..core.config import Settings


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        data_dir=args.data_dir, embedder=args.embedder, dim=args.dim, device=args.device
    )


def _services(args: argparse.Namespace):
    from ..api.api import Services
    from ..ingestion.embed import create_embedder

    settings = _settings(args)
    return Services(settings, create_embedder(settings))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ..api.api import LOCAL_HOSTS, create_app

    # Bind before loading anything, so a busy port fails fast and nothing starts indexing.
    family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as exc:
        sock.close()
        busy = exc.errno == errno.EADDRINUSE
        reason = "порт уже занят (возможно, SourceBoard уже запущен)" if busy else exc.strerror
        print(f"Не удалось открыть {args.host}:{args.port}: {reason}.", file=sys.stderr)
        if busy:
            print(
                f"Откройте http://{args.host}:{args.port} или укажите другой порт: --port "
                f"{args.port + 1}",
                file=sys.stderr,
            )
        return 1

    settings = _settings(args)
    hosts = LOCAL_HOSTS if args.host in {"127.0.0.1", "localhost", "::1"} else [args.host]
    app = create_app(settings, allowed_hosts=hosts)
    services = app.state.services
    for folder in args.folder or []:
        services.indexer.add_folder(folder)
    services.indexer.request_scan(None)  # pick up changes made while the server was down
    print(
        f"SourceBoard: http://{args.host}:{args.port}  (данные: {settings.data_dir}, "
        f"эмбеддер: {services.embedder.name})"
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    server.run(sockets=[sock])
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    services = _services(args)
    ids = [services.indexer.add_folder(folder) for folder in args.folders]
    stats = services.indexer.scan(ids or None)
    print(json.dumps(stats, ensure_ascii=False))
    for row in services.db.query(
        "SELECT rel_path, status, error, notes FROM sources WHERE status != 'indexed' OR notes "
        "IS NOT NULL ORDER BY rel_path"
    ):
        print(f"  {row['rel_path']}: {row['status']} {row['error'] or ''} {row['notes'] or ''}")
    return 1 if stats["errors"] else 0


def cmd_search(args: argparse.Namespace) -> int:
    from ..board.export import locator

    services = _services(args)
    results = services.search.search(
        args.query, limit=args.limit, modalities=args.modality, mode=args.mode, task=args.task
    )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    for i, hit in enumerate(results, start=1):
        where = locator(hit)
        sim = f" sim={hit['similarity']:.3f}" if hit["similarity"] is not None else ""
        print(
            f"{i:2d}. [{hit['modality']}] {hit['source']}{' · ' + where if where else ''}"
            f"  ({'+'.join(hit['matched_by'])}{sim})"
        )
        if hit["text"]:
            print(f"    {' '.join(hit['text'].split())[:200]}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ..search.evaluate import evaluate, load_tasks

    services = _services(args)
    tasks = load_tasks(Path(args.tasks))
    reports = [evaluate(services.search, tasks, k=args.k, mode=mode) for mode in args.modes]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from ..board.export import to_json, to_markdown, to_markers_csv

    services = _services(args)
    render = {"markdown": to_markdown, "csv": to_markers_csv, "json": to_json}[args.format]
    text = render(services.db)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import make_demo

    print(json.dumps(make_demo(Path(args.target)), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sourceboard", description=__doc__)
    parser.add_argument("--data-dir", help="где хранить индекс (по умолчанию ~/.sourceboard)")
    parser.add_argument("--embedder", choices=["auto", "gemma", "hash"])
    parser.add_argument("--dim", type=int, choices=[768, 512, 256, 128])
    parser.add_argument("--device", help="cpu, mps или cuda (по умолчанию выбирает torch)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="запустить веб-интерфейс")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--folder", action="append", help="добавить папку при запуске")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("index", help="добавить папки и проиндексировать")
    p.add_argument("folders", nargs="*")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="поиск из терминала")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--modality", action="append", choices=["text", "image", "video", "audio"])
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "lexical", "vector"])
    p.add_argument(
        "--task",
        default="SearchQuery",
        choices=["SearchQuery", "FactChecking", "QuestionAnswering"],
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval", help="hit@k и MRR на наборе заданий JSONL")
    p.add_argument("tasks")
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["lexical", "vector", "hybrid"],
        choices=["lexical", "vector", "hybrid"],
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("export", help="выгрузить подборку")
    p.add_argument("--format", default="markdown", choices=["markdown", "csv", "json"])
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("demo", help="создать вымышленный демо-корпус")
    p.add_argument("target")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
