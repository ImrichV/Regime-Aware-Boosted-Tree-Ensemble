from __future__ import annotations

import json
from pathlib import Path

from tsr.data.report import (
    compare_module_01_runs,
    module_01_acceptance,
    write_module_01_report,
)


def valid_summary() -> dict:
    return {
        "dry_run": False,
        "audited_member_count": 2,
        "classified_member_count_in_inventory": 2,
        "file_error_count": 0,
        "duplicate_instrument_id_count": 0,
        "archive_sha256": "a",
        "dataset_fingerprint_sha256": "b",
        "module_version": "v1",
        "archive_member_count": 2,
        "global_first_date": "2000-01-01",
        "global_last_date": "2026-01-01",
        "exchange_counts": {"NASDAQ": 2},
        "instrument_class_counts": {"stocks": 2},
        "row_count": 10,
        "valid_row_count": 9,
        "invalid_row_count": 1,
        "empty_file_count": 0,
        "invalid_header_file_count": 0,
        "zero_volume_row_count": 0,
        "invalid_reason_counts": {"high_below_ohlc": 1},
        "duplicate_ticker_count": 0,
        "survivor_window_days": 10,
        "survivor_like_instrument_count": 2,
        "survivor_like_fraction": 1.0,
    }


def test_acceptance_passes_valid_full_run() -> None:
    accepted, failures = module_01_acceptance(
        valid_summary(), {"deterministic_match": True}
    )
    assert accepted is True
    assert failures == []


def test_acceptance_rejects_dry_run() -> None:
    summary = valid_summary()
    summary["dry_run"] = True
    accepted, failures = module_01_acceptance(
        summary, {"deterministic_match": True}
    )
    assert accepted is False
    assert any("dry run" in failure for failure in failures)


def test_writes_human_and_machine_reports(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps(valid_summary()))
    (run_dir / "determinism_check.json").write_text(
        json.dumps({"deterministic_match": True, "reference_run": "ref"})
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "abc",
                "config_hash": "cfg",
                "source_tree_sha256": "src",
                "git_commit": "git",
            }
        )
    )
    output = write_module_01_report(run_dir, tmp_path / "report.md")
    assert "**ACCEPTED**" in output.read_text()
    machine = json.loads(output.with_suffix(".json").read_text())
    assert machine["accepted"] is True


def test_compare_identical_run_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    summary = valid_summary()
    for run in [reference, candidate]:
        (run / "summary.json").write_text(json.dumps(summary))
        for name in [
            "archive_inventory.csv",
            "symbol_manifest.csv",
            "invalid_rows.csv",
            "file_errors.csv",
        ]:
            (run / name).write_text("same\n")
    output = compare_module_01_runs(reference, candidate, tmp_path / "compare.json")
    result = json.loads(output.read_text())
    assert result["deterministic_match"] is True


def test_acceptance_rejects_missing_determinism() -> None:
    accepted, failures = module_01_acceptance(valid_summary(), None)
    assert accepted is False
    assert any("deterministic" in failure for failure in failures)
