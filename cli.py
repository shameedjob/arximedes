import argparse
import logging
from pathlib import Path

from platformdirs import user_data_dir

from . import config

DEFAULT_DB = Path(user_data_dir("arximedes")) / "papers.db"


def cmd_scan(args: argparse.Namespace) -> None:
    from . import embedder

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-PDF warnings
    args.db.parent.mkdir(parents=True, exist_ok=True)  # sqlite won't create missing dirs
    con = embedder.connect(args.db)
    embedder.index_dir(con, args.path)


def cmd_search(args: argparse.Namespace) -> None:
    from . import embedder
    from . import query

    if not args.db.exists():
        raise SystemExit(f"no index at {args.db}; run `arximedes scan <dir>` first")
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-PDF warnings
    con = embedder.connect(args.db)
    for sentence, path, page, score in query.search(con, args.query, k=args.k, y=args.y):
        print(f"{sentence}\n    {path}  p.{page + 1}  ({score:.3f})\n")  # loader pages are 0-based


def cmd_model(args: argparse.Namespace) -> None:
    if args.reset:
        config.set_model(None)
    elif args.name:
        config.set_model(args.name)
    name, source = config.get_model()
    print(f"{name}  (from {source})")
    if source == f"${config.MODEL_ENV}" and (args.name or args.reset):
        print(f"note: ${config.MODEL_ENV} is set and overrides the saved model")


def build_parser() -> argparse.ArgumentParser:
    # options shared by every subcommand, accepted after the subcommand name
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help=f"path to the index database (default: {DEFAULT_DB})")

    parser = argparse.ArgumentParser(prog="arximedes")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", parents=[common], help="index a directory of PDFs into the database")
    scan.add_argument("path", help="directory of PDFs")
    scan.set_defaults(func=cmd_scan)

    search = sub.add_parser("search", parents=[common], help="find the passages that best match a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10, help="chunks retrieved to pick candidate papers (default: 10)")
    search.add_argument("-y", type=int, default=5, help="passages to show (default: 5)")
    search.set_defaults(func=cmd_search)

    model = sub.add_parser(
        "model", help="show or set the embedding model",
        description=f"Show the embedding model, or save a new one. ${config.MODEL_ENV} overrides "
                    f"the saved choice; the default is {config.DEFAULT_MODEL}.",
    )
    choice = model.add_mutually_exclusive_group()
    choice.add_argument("name", nargs="?", help="sentence-transformers model name to save")
    choice.add_argument("--reset", action="store_true", help="forget the saved model and use the default")
    model.set_defaults(func=cmd_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0
