from __future__ import annotations

import argparse

from ..config import load_yaml
from .baseline import compare as compare_baselines
from .baseline import publish as publish_baselines
from .baseline import run_baselines
from .build import build, publish, verify


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Module 06 specialist research dataset and baseline gate"
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build selection-safe research shards")
    build_parser.add_argument("--config", required=True)
    build_parser.add_argument("--resume")
    build_parser.add_argument("--dry-run", action="store_true")
    build_parser.add_argument("--max-new", type=int)

    verify_parser = subparsers.add_parser("verify", help="Verify research shards")
    verify_parser.add_argument("--reference", required=True)
    verify_parser.add_argument("--output", required=True)

    publish_parser = subparsers.add_parser("publish", help="Publish completed research build")
    publish_parser.add_argument("--run", required=True)
    publish_parser.add_argument("--target", required=True)

    baseline_parser = subparsers.add_parser("baselines", help="Run fixed walk-forward baselines")
    baseline_parser.add_argument("--config", required=True)
    baseline_parser.add_argument("--research-root", required=True)
    baseline_parser.add_argument("--resume")

    compare_parser = subparsers.add_parser("compare-baselines", help="Compare two baseline runs")
    compare_parser.add_argument("--first", required=True)
    compare_parser.add_argument("--second", required=True)
    compare_parser.add_argument("--output", required=True)

    publish_baseline_parser = subparsers.add_parser("publish-baselines", help="Publish baseline run")
    publish_baseline_parser.add_argument("--run", required=True)
    publish_baseline_parser.add_argument("--target", required=True)


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
    elif args.command == "publish":
        result = publish(args.run, args.target)
    elif args.command == "baselines":
        result = run_baselines(
            load_yaml(args.config),
            research_root=args.research_root,
            resume=args.resume,
        )
    elif args.command == "compare-baselines":
        result = compare_baselines(args.first, args.second, args.output)
    else:
        result = publish_baselines(args.run, args.target)
    print(result)
    return 0
