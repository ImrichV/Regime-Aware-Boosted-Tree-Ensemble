from __future__ import annotations

import argparse

from ..config import load_yaml
from .audit import audit_stooq_archive
from .report import compare_module_01_runs, write_module_01_report
from .stooq import StooqArchive


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Canonical Stooq data module"
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit every classified member in the Stooq ZIP")
    audit.add_argument("--config", required=True)
    audit.add_argument("--resume", default=None, help="Existing run directory to resume")
    audit.add_argument("--dry-run", action="store_true")

    compare = subparsers.add_parser("compare", help="Compare two complete Module 01 runs")
    compare.add_argument("--reference-run", required=True)
    compare.add_argument("--candidate-run", required=True)
    compare.add_argument("--output", required=True)

    report = subparsers.add_parser("report", help="Create a Module 01 acceptance report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--output", required=True)

    show = subparsers.add_parser("show", help="Load and display one ZIP member")
    show.add_argument("--archive", required=True)
    show.add_argument("--member", required=True)
    show.add_argument("--rows", type=int, default=5)
    show.add_argument("--strict", action="store_true")


def run(args: argparse.Namespace) -> int:
    if args.command == "audit":
        run_dir = audit_stooq_archive(
            load_yaml(args.config),
            resume_dir=args.resume,
            dry_run=args.dry_run,
        )
        print(f"Completed run: {run_dir}")
        return 0
    if args.command == "compare":
        output = compare_module_01_runs(
            args.reference_run, args.candidate_run, args.output
        )
        print(f"Wrote comparison: {output}")
        return 0
    if args.command == "report":
        output = write_module_01_report(args.run_dir, args.output)
        print(f"Wrote report: {output}")
        return 0
    if args.command == "show":
        archive = StooqArchive(args.archive)
        frame = archive.load_member(
            args.member,
            invalid_policy="raise" if args.strict else "drop",
        )
        print(frame.head(args.rows).to_string(index=False))
        print(f"Rows: {len(frame):,}")
        return 0
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsr data")
    configure_parser(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
