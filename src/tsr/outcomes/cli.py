from __future__ import annotations

import argparse

from ..config import load_yaml
from .build import build, publish, verify


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Candidate outcome and label engine"
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build outcome shards")
    build_parser.add_argument("--config", required=True)
    build_parser.add_argument("--resume")
    build_parser.add_argument("--dry-run", action="store_true")
    build_parser.add_argument("--max-new", type=int)

    verify_parser = subparsers.add_parser("verify", help="Verify outcome shards and IDs")
    verify_parser.add_argument("--reference", required=True)
    verify_parser.add_argument("--output", required=True)

    publish_parser = subparsers.add_parser("publish", help="Publish a completed outcome run")
    publish_parser.add_argument("--run", required=True)
    publish_parser.add_argument("--target", required=True)


def run(args: argparse.Namespace) -> int:
    if args.command == "build":
        result = build(
            load_yaml(args.config),
            resume=args.resume,
            dry_run=args.dry_run,
            max_new=args.max_new,
        )
    elif args.command == "verify":
        result = verify(args.reference, args.output)
    else:
        result = publish(args.run, args.target)
    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsr outcomes")
    configure_parser(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
