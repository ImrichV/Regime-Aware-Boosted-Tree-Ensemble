from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _fmt_int(value: int | float | None) -> str:
    return f"{int(value or 0):,}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compare_module_01_runs(
    reference_run_dir: str | Path,
    candidate_run_dir: str | Path,
    output_path: str | Path,
) -> Path:
    reference = Path(reference_run_dir).expanduser().resolve()
    candidate = Path(candidate_run_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    files = [
        "archive_inventory.csv",
        "symbol_manifest.csv",
        "invalid_rows.csv",
        "file_errors.csv",
    ]
    file_comparisons: dict[str, Any] = {}
    all_match = True
    for name in files:
        ref_path = reference / name
        candidate_path = candidate / name
        ref_hash = _sha256_path(ref_path) if ref_path.exists() else None
        candidate_hash = _sha256_path(candidate_path) if candidate_path.exists() else None
        match = ref_hash is not None and ref_hash == candidate_hash
        all_match = all_match and match
        file_comparisons[name] = {
            "reference_sha256": ref_hash,
            "candidate_sha256": candidate_hash,
            "match": match,
        }

    ref_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
    fingerprint_match = (
        ref_summary.get("dataset_fingerprint_sha256")
        == candidate_summary.get("dataset_fingerprint_sha256")
    )
    archive_match = ref_summary.get("archive_sha256") == candidate_summary.get("archive_sha256")
    core_counts = [
        "archive_member_count",
        "classified_member_count_in_inventory",
        "audited_member_count",
        "row_count",
        "valid_row_count",
        "invalid_row_count",
        "file_error_count",
        "empty_file_count",
    ]
    count_comparisons = {
        key: {
            "reference": ref_summary.get(key),
            "candidate": candidate_summary.get(key),
            "match": ref_summary.get(key) == candidate_summary.get(key),
        }
        for key in core_counts
    }
    counts_match = all(item["match"] for item in count_comparisons.values())
    deterministic = all_match and fingerprint_match and archive_match and counts_match
    payload = {
        "deterministic_match": deterministic,
        "reference_run": str(reference),
        "candidate_run": str(candidate),
        "archive_sha256_match": archive_match,
        "dataset_fingerprint_match": fingerprint_match,
        "file_comparisons": file_comparisons,
        "core_count_comparisons": count_comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def module_01_acceptance(
    summary: dict[str, Any], determinism: dict[str, Any] | None = None
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if summary.get("dry_run"):
        failures.append("The run is a dry run, not a full archive audit.")
    if summary.get("audited_member_count") != summary.get(
        "classified_member_count_in_inventory"
    ):
        failures.append("Not every classified archive member was audited.")
    if summary.get("file_error_count") != 0:
        failures.append("One or more member-level operational errors occurred.")
    if summary.get("duplicate_instrument_id_count") != 0:
        failures.append("Canonical instrument IDs are not unique.")
    if not summary.get("archive_sha256"):
        failures.append("Archive SHA-256 is missing.")
    if not summary.get("dataset_fingerprint_sha256"):
        failures.append("Dataset fingerprint is missing.")
    if determinism is None:
        failures.append("A deterministic comparison with another full run is missing.")
    elif not determinism.get("deterministic_match"):
        failures.append("The independent full-run comparison did not reproduce identical outputs.")
    return not failures, failures


def write_module_01_report(run_dir: str | Path, output_path: str | Path) -> Path:
    run_path = Path(run_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
    determinism_path = run_path / "determinism_check.json"
    determinism = (
        json.loads(determinism_path.read_text(encoding="utf-8"))
        if determinism_path.exists()
        else None
    )
    accepted, failures = module_01_acceptance(summary, determinism)

    invalid_reasons = summary.get("invalid_reason_counts", {})
    reason_lines = (
        "\n".join(f"- {reason}: {_fmt_int(count)}" for reason, count in invalid_reasons.items())
        if invalid_reasons
        else "- None"
    )
    exchange_lines = "\n".join(
        f"- {name}: {_fmt_int(count)}"
        for name, count in summary.get("exchange_counts", {}).items()
    )
    class_lines = "\n".join(
        f"- {name}: {_fmt_int(count)}"
        for name, count in summary.get("instrument_class_counts", {}).items()
    )
    failure_section = (
        "\n".join(f"- {failure}" for failure in failures)
        if failures
        else "- No hard acceptance failure."
    )

    text = f"""# Module 01 Acceptance Report

## Verdict

**{'ACCEPTED' if accepted else 'NOT ACCEPTED'}** as the canonical Stooq data module.

This verdict covers data ingestion, structural validation, lineage, reproducibility, and downstream access. It does not validate any trading strategy.

## Run identity

- Run ID: `{manifest.get('run_id')}`
- Module version: `{summary.get('module_version')}`
- Configuration hash: `{manifest.get('config_hash')}`
- Source-tree SHA-256: `{manifest.get('source_tree_sha256')}`
- Git commit: `{manifest.get('git_commit')}`
- Archive SHA-256: `{summary.get('archive_sha256')}`
- Dataset fingerprint: `{summary.get('dataset_fingerprint_sha256')}`

## Coverage

- Archive members: {_fmt_int(summary.get('archive_member_count'))}
- Classified members: {_fmt_int(summary.get('classified_member_count_in_inventory'))}
- Audited members: {_fmt_int(summary.get('audited_member_count'))}
- First valid date: {summary.get('global_first_date')}
- Last valid date: {summary.get('global_last_date')}

### Exchanges

{exchange_lines}

### Instrument classes

{class_lines}

## Row audit

- Source rows: {_fmt_int(summary.get('row_count'))}
- Valid rows: {_fmt_int(summary.get('valid_row_count'))}
- Quarantined rows: {_fmt_int(summary.get('invalid_row_count'))}
- Empty files: {_fmt_int(summary.get('empty_file_count'))}
- Invalid-header files: {_fmt_int(summary.get('invalid_header_file_count'))}
- Member-level operational errors: {_fmt_int(summary.get('file_error_count'))}
- Zero-volume rows: {_fmt_int(summary.get('zero_volume_row_count'))}

### Quarantine reasons

{reason_lines}

## Identity integrity

- Duplicate canonical instrument IDs: {_fmt_int(summary.get('duplicate_instrument_id_count'))}
- Ticker texts appearing in multiple canonical locations: {_fmt_int(summary.get('duplicate_ticker_count'))}

The latter is not automatically an error because ticker text is not the primary key. Downstream joins must use `instrument_id`.

## Survivor-panel diagnostic

- Instruments ending within {summary.get('survivor_window_days')} calendar days of the archive maximum date: {_fmt_int(summary.get('survivor_like_instrument_count'))}
- Fraction: {_fmt_pct(summary.get('survivor_like_fraction'))}

This diagnostic does not prove complete or incomplete delisting coverage by itself. It is a strong warning that later strategy estimates may be materially affected by survivor-panel bias.

## Deterministic reproduction

- Comparison available: {"yes" if determinism is not None else "no"}
- Identical deterministic outputs: {"yes" if determinism and determinism.get("deterministic_match") else "no"}
- Reference run: {determinism.get("reference_run") if determinism else "n/a"}

## Hard acceptance checks

{failure_section}

## Accepted downstream interface

Module 02 must consume the published `symbol_manifest.csv` and load bars through `StooqCatalog`. It may not independently reinterpret ZIP paths, ticker aliases, fractional volume, or invalid-row rules.

## Scope boundary

No feature engine, setup definition, machine-learning model, ranker, exit policy, or portfolio simulator is accepted by this report.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    acceptance_json = output.with_suffix(".json")
    acceptance_json.write_text(
        json.dumps(
            {
                "accepted": accepted,
                "failures": failures,
                "run_id": manifest.get("run_id"),
                "module_version": summary.get("module_version"),
                "dataset_fingerprint_sha256": summary.get("dataset_fingerprint_sha256"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
