from __future__ import annotations

import argparse

from . import __version__
from .candidates.cli import configure_parser as configure_candidates
from .candidates.cli import run as run_candidates
from .data.cli import configure_parser as configure_data
from .data.cli import run as run_data
from .doctor import configure_parser as configure_doctor
from .doctor import run as run_doctor
from .features.cli import configure_parser as configure_features
from .features.cli import run as run_features
from .outcomes.cli import configure_parser as configure_outcomes
from .outcomes.cli import run as run_outcomes
from .research.cli import configure_parser as configure_research
from .research.cli import run as run_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsr",
        description="Modular Stooq-only trading-systems research platform",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="area", required=True)

    data_parser = subparsers.add_parser("data", help="Module 01 data commands")
    configure_data(data_parser)
    data_parser.set_defaults(_runner=run_data)

    features_parser = subparsers.add_parser("features", help="Module 02 feature commands")
    configure_features(features_parser)
    features_parser.set_defaults(_runner=run_features)

    candidates_parser = subparsers.add_parser(
        "candidates",
        help="Module 03 framework and Module 04 playbook commands",
    )
    configure_candidates(candidates_parser)
    candidates_parser.set_defaults(_runner=run_candidates)

    outcomes_parser = subparsers.add_parser("outcomes", help="Module 05 outcome and label commands")
    configure_outcomes(outcomes_parser)
    outcomes_parser.set_defaults(_runner=run_outcomes)

    research_parser = subparsers.add_parser("research", help="Module 06 research and baseline commands")
    configure_research(research_parser)
    research_parser.set_defaults(_runner=run_research)

    doctor_parser = subparsers.add_parser("doctor", help="Check repository readiness")
    configure_doctor(doctor_parser)
    doctor_parser.set_defaults(_runner=run_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args._runner(args))


if __name__ == "__main__":
    raise SystemExit(main())
