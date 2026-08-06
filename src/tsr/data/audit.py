from __future__ import annotations

import csv
import hashlib
import json
import shutil
import signal
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..run import RunContext, atomic_json, utc_now
from .stooq import StooqArchive, classify_member_path, sha256_file


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, sort_keys=True)
            writer.writerow(row)
    temp.replace(path)


def _dedupe(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        deduped[tuple(record.get(field) for field in key_fields)] = record
    return [deduped[key] for key in sorted(deduped, key=lambda item: tuple(str(v) for v in item))]


def _archive_inventory(archive_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    inventory: list[dict[str, Any]] = []
    classified: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            identity = classify_member_path(info.filename)
            record = {
                "source_member": info.filename,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "is_txt": info.filename.lower().endswith(".txt"),
                "path_classified": identity is not None,
                "instrument_id": identity.instrument_id if identity else None,
                "ticker": identity.ticker if identity else None,
                "exchange": identity.exchange if identity else None,
                "instrument_class": identity.instrument_class if identity else None,
            }
            inventory.append(record)
            if identity is not None:
                classified.append(info.filename)
    return inventory, classified


def audit_stooq_archive(
    config: dict[str, Any],
    *,
    resume_dir: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    module = config.get("module", {})
    data = config.get("data", {})
    execution = config.get("execution", {})
    publication = config.get("publication", {})

    module_name = str(module.get("name", "module_01_stooq_data"))
    module_version = str(module.get("version", "v1"))
    archive_path = Path(data["archive_path"]).expanduser().resolve()
    runs_root = Path(execution.get("runs_root", "runs")).expanduser().resolve()
    checkpoint_every = int(execution.get("checkpoint_every_files", 100))
    progress_every = int(execution.get("progress_every_files", 100))
    dry_run_max_files = int(execution.get("dry_run_max_files", 25))
    compute_archive_sha256 = bool(data.get("compute_archive_sha256", True))
    survivor_window_days = int(data.get("survivor_window_days", 10))

    context = RunContext.create(
        module_name=module_name,
        module_version=module_version,
        config=config,
        runs_root=runs_root,
        resume_dir=resume_dir,
    )
    assert context.run_dir is not None

    partial_dir = context.run_dir / "partial"
    partial_dir.mkdir(exist_ok=True)
    file_results_path = partial_dir / "file_results.jsonl"
    invalid_rows_path = partial_dir / "invalid_rows.jsonl"
    file_errors_path = partial_dir / "file_errors.jsonl"

    previous_signal_handlers: dict[int, Any] = {}

    def _raise_interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Received signal {signum}; progress is checkpointed for resume.")

    for signal_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _raise_interrupted)

    try:
        context.log("audit_started", archive_path=str(archive_path), dry_run=dry_run)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        inventory, members = _archive_inventory(archive_path)
        if dry_run:
            members = members[:dry_run_max_files]
        inventory_csv = context.run_dir / "archive_inventory.csv"
        _write_csv(
            inventory_csv,
            inventory,
            [
                "source_member",
                "compressed_size",
                "uncompressed_size",
                "crc32",
                "is_txt",
                "path_classified",
                "instrument_id",
                "ticker",
                "exchange",
                "instrument_class",
            ],
        )
        context.record_output("archive_inventory", inventory_csv)

        checkpoint = context.load_checkpoint() or {}
        next_index = int(checkpoint.get("next_index", 0))
        if next_index > len(members):
            raise ValueError("Checkpoint points beyond the current member list.")

        archive = StooqArchive(archive_path)
        started = time.monotonic()
        for index in range(next_index, len(members)):
            member = members[index]
            try:
                parsed = archive.parse_member(member, keep_rows=False)
                _append_jsonl(file_results_path, parsed.summary)
                for invalid in parsed.invalid_rows:
                    _append_jsonl(invalid_rows_path, invalid)
            except (KeyboardInterrupt, InterruptedError):
                raise
            except Exception as exc:  # operational failure is isolated to the file and recorded
                _append_jsonl(
                    file_errors_path,
                    {
                        "source_member": member,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                context.log(
                    "file_failed",
                    level="ERROR",
                    source_member=member,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            processed = index + 1
            if processed % checkpoint_every == 0 or processed == len(members):
                context.save_checkpoint(
                    {
                        "next_index": processed,
                        "member_count": len(members),
                        "last_member": member,
                        "dry_run": dry_run,
                    }
                )
            if processed % progress_every == 0 or processed == len(members):
                elapsed = time.monotonic() - started
                rate = (processed - next_index) / elapsed if elapsed > 0 else 0.0
                context.set_progress(
                    processed_files=processed,
                    total_files=len(members),
                    percent=round(100.0 * processed / max(1, len(members)), 2),
                    files_per_second=round(rate, 3),
                    last_member=member,
                )
                print(
                    f"Module 01 audit: {processed:,}/{len(members):,} files "
                    f"({100.0 * processed / max(1, len(members)):.1f}%)",
                    flush=True,
                )

        file_results = _dedupe(_read_jsonl(file_results_path), ("source_member",))
        invalid_rows = _dedupe(
            _read_jsonl(invalid_rows_path),
            ("source_member", "source_row_number", "reason"),
        )
        file_errors = _dedupe(_read_jsonl(file_errors_path), ("source_member", "error_type"))

        manifest_fields = [
            "source_member",
            "instrument_id",
            "ticker",
            "exchange",
            "instrument_class",
            "compressed_size",
            "uncompressed_size",
            "crc32",
            "header_valid",
            "empty_file",
            "row_count",
            "valid_row_count",
            "invalid_row_count",
            "invalid_reason_counts",
            "first_date",
            "last_date",
            "zero_volume_count",
            "extreme_return_50_count",
            "extreme_return_100_count",
            "max_abs_return",
            "min_close",
            "max_close",
        ]
        invalid_fields = [
            "source_member",
            "instrument_id",
            "ticker",
            "source_row_number",
            "date_raw",
            "reason",
            "raw_preview",
        ]
        error_fields = ["source_member", "error_type", "error_message", "traceback"]

        symbol_manifest_csv = context.run_dir / "symbol_manifest.csv"
        invalid_rows_csv = context.run_dir / "invalid_rows.csv"
        file_errors_csv = context.run_dir / "file_errors.csv"
        _write_csv(symbol_manifest_csv, file_results, manifest_fields)
        _write_csv(invalid_rows_csv, invalid_rows, invalid_fields)
        _write_csv(file_errors_csv, file_errors, error_fields)

        global_last_date = max(
            (record["last_date"] for record in file_results if record.get("last_date")),
            default=None,
        )
        last_date_counts = Counter(record.get("last_date") for record in file_results)
        class_counts = Counter(record.get("instrument_class") for record in file_results)
        exchange_counts = Counter(record.get("exchange") for record in file_results)
        invalid_reason_counts: Counter[str] = Counter()
        for record in file_results:
            invalid_reason_counts.update(record.get("invalid_reason_counts") or {})

        instrument_id_counts = Counter(record["instrument_id"] for record in file_results)
        ticker_locations: defaultdict[str, set[str]] = defaultdict(set)
        for record in file_results:
            ticker_locations[record["ticker"]].add(record["instrument_id"])
        duplicate_instrument_ids = {
            key: count for key, count in instrument_id_counts.items() if count > 1
        }
        duplicate_tickers = {
            ticker: sorted(values) for ticker, values in ticker_locations.items() if len(values) > 1
        }

        # Calendar-day window is intentionally conservative and transparent.
        survivor_like_count = 0
        dated_instrument_count = sum(1 for record in file_results if record.get("last_date"))
        if global_last_date:
            from datetime import date, timedelta

            max_date = date.fromisoformat(global_last_date)
            threshold = max_date - timedelta(days=survivor_window_days)
            survivor_like_count = sum(
                1
                for record in file_results
                if record.get("last_date") and date.fromisoformat(record["last_date"]) >= threshold
            )

        archive_sha256 = sha256_file(archive_path) if compute_archive_sha256 else None
        fingerprint_payload = "\n".join(
            f"{record['source_member']}|{record['crc32']}|{record['valid_row_count']}|"
            f"{record.get('first_date')}|{record.get('last_date')}"
            for record in file_results
        )
        dataset_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()

        summary = {
            "module_name": module_name,
            "module_version": module_version,
            "completed_at": utc_now(),
            "dry_run": dry_run,
            "archive_path": str(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "dataset_fingerprint_sha256": dataset_fingerprint,
            "archive_member_count": len(inventory),
            "classified_member_count_in_inventory": sum(
                1 for record in inventory if record["path_classified"]
            ),
            "unclassified_member_count_in_inventory": sum(
                1 for record in inventory if not record["path_classified"]
            ),
            "audited_member_count": len(file_results),
            "file_error_count": len(file_errors),
            "empty_file_count": sum(bool(record["empty_file"]) for record in file_results),
            "invalid_header_file_count": sum(
                (not bool(record["header_valid"])) and (not bool(record["empty_file"]))
                for record in file_results
            ),
            "row_count": sum(int(record["row_count"]) for record in file_results),
            "valid_row_count": sum(int(record["valid_row_count"]) for record in file_results),
            "invalid_row_count": sum(int(record["invalid_row_count"]) for record in file_results),
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "zero_volume_row_count": sum(
                int(record["zero_volume_count"]) for record in file_results
            ),
            "extreme_return_50_count": sum(
                int(record["extreme_return_50_count"]) for record in file_results
            ),
            "extreme_return_100_count": sum(
                int(record["extreme_return_100_count"]) for record in file_results
            ),
            "instrument_class_counts": dict(sorted(class_counts.items())),
            "exchange_counts": dict(sorted(exchange_counts.items())),
            "global_first_date": min(
                (record["first_date"] for record in file_results if record.get("first_date")),
                default=None,
            ),
            "global_last_date": global_last_date,
            "top_last_dates": last_date_counts.most_common(20),
            "survivor_window_days": survivor_window_days,
            "dated_instrument_count": dated_instrument_count,
            "survivor_like_instrument_count": survivor_like_count,
            "survivor_like_fraction": (
                survivor_like_count / dated_instrument_count
                if dated_instrument_count
                else None
            ),
            "duplicate_instrument_id_count": len(duplicate_instrument_ids),
            "duplicate_ticker_count": len(duplicate_tickers),
            "duplicate_instrument_ids": duplicate_instrument_ids,
            "duplicate_tickers": duplicate_tickers,
        }
        summary_json = context.run_dir / "summary.json"
        fingerprint_json = context.run_dir / "dataset_fingerprint.json"
        atomic_json(summary_json, summary)
        atomic_json(
            fingerprint_json,
            {
                "archive_sha256": archive_sha256,
                "dataset_fingerprint_sha256": dataset_fingerprint,
                "config_hash": context.manifest["config_hash"],
                "module_version": module_version,
            },
        )

        for name, path in {
            "symbol_manifest": symbol_manifest_csv,
            "invalid_rows": invalid_rows_csv,
            "file_errors": file_errors_csv,
            "summary": summary_json,
            "dataset_fingerprint": fingerprint_json,
        }.items():
            context.record_output(name, path)

        configured_publish_root = Path(
            publication.get(
                "artifact_root",
                Path("artifacts") / f"{module_name}_{module_version}",
            )
        ).expanduser().resolve()
        # A dry run is diagnostic only and must never overwrite the accepted full-run artifacts.
        publish_root = (
            context.run_dir / "dry_run_artifacts" if dry_run else configured_publish_root
        )
        publish_root.mkdir(parents=True, exist_ok=True)
        for source in [
            inventory_csv,
            symbol_manifest_csv,
            invalid_rows_csv,
            file_errors_csv,
            summary_json,
            fingerprint_json,
            context.manifest_path,
        ]:
            shutil.copy2(source, publish_root / source.name)
        latest_pointer = publish_root / "SOURCE_RUN.txt"
        latest_pointer.write_text(str(context.run_dir) + "\n", encoding="utf-8")

        context.complete(
            audited_member_count=len(file_results),
            valid_row_count=summary["valid_row_count"],
            invalid_row_count=summary["invalid_row_count"],
            published_artifact_root=str(publish_root),
            publication_mode="dry_run_isolated" if dry_run else "accepted_full_run",
        )
        return context.run_dir
    except (KeyboardInterrupt, InterruptedError) as exc:
        context.interrupt(str(exc))
        raise
    except Exception as exc:
        context.fail(exc)
        raise
    finally:
        for signum, previous_handler in previous_signal_handlers.items():
            signal.signal(signum, previous_handler)
