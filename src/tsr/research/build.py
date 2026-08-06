from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..candidates.storage import CandidateStore
from ..config import config_hash
from ..features.schema import FEATURE_NAMES
from ..features.storage import FeatureStore
from ..outcomes.schema import ticker_split
from ..outcomes.storage import OutcomeStore
from ..run import RunContext, atomic_json
from .schema import (
    MODULE_NAME,
    MODULE_VERSION,
    OUTCOME_LABEL_COLUMNS,
    RESEARCH_COLUMNS,
    schema_payload,
)
from .storage import ResearchStore, file_sha256, shard_relative_path, write_research_shard

MANIFEST_COLUMNS = (
    "instrument_id",
    "ticker",
    "exchange",
    "research_count",
    "family_counts_json",
    "shard_relative_path",
    "shard_sha256",
)


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _append_row(path: Path, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    frame = pd.DataFrame([row], columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")


def _load_done(path: Path) -> set[str]:
    if not path.exists() or not path.stat().st_size:
        return set()
    return set(pd.read_csv(path)["instrument_id"].astype(str))


def _upstream(config: dict[str, Any]):
    upstream = config["upstream"]
    feature_root = Path(upstream["feature_root"])
    outcome_root = Path(upstream["outcome_root"])
    feature_store = FeatureStore(feature_root)
    outcome_store = OutcomeStore(outcome_root)
    feature_fingerprint = _json(feature_root / "dataset_fingerprint.json")
    outcome_fingerprint = _json(outcome_root / "dataset_fingerprint.json")
    candidate_catalog = _json(outcome_root / "candidate_catalog.json")

    stores: dict[str, CandidateStore] = {}
    index: dict[str, list[tuple[str, str]]] = {}
    source_rows: list[dict[str, Any]] = []
    for root_value in upstream["candidate_roots"]:
        root = Path(root_value)
        store = CandidateStore(root)
        keys = sorted(set(store.manifest["generator_key"].astype(str)))
        if len(keys) != 1:
            raise ValueError(f"candidate root must contain one generator key: {root}")
        key = keys[0]
        if key in stores:
            raise ValueError(f"duplicate candidate generator key: {key}")
        stores[key] = store
        nonzero = store.manifest[store.manifest["candidate_count"] > 0]
        for row in nonzero.itertuples(index=False):
            index.setdefault(str(row.instrument_id), []).append((key, str(row.instrument_id)))
        source_rows.append(
            {
                "generator_key": key,
                "candidate_count": int(store.manifest["candidate_count"].sum()),
                "instrument_rows": int(len(store.manifest)),
            }
        )
    expected = {
        (str(item["generator_key"]), int(item["candidate_count"]), int(item["instrument_rows"]))
        for item in candidate_catalog["sources"]
    }
    actual = {
        (item["generator_key"], item["candidate_count"], item["instrument_rows"])
        for item in source_rows
    }
    if actual != expected:
        raise ValueError("configured candidate roots do not match Module 05 candidate catalog")
    return feature_store, outcome_store, stores, index, feature_fingerprint, outcome_fingerprint, candidate_catalog


def _feature_frame(store: FeatureStore, instrument_id: str, verify: bool) -> pd.DataFrame:
    arrays = store.load_arrays(instrument_id, verify=verify)
    frame = pd.DataFrame({
        "signal_date": arrays["date"].astype(np.int64),
        "signal_source_row_number": arrays["source_row_number"].astype(np.int64),
        "feature_segment_id": arrays["segment_id"].astype(np.int64),
        **{name: arrays[name].astype(np.float64) for name in FEATURE_NAMES},
    })
    if frame.duplicated(["signal_source_row_number"]).any():
        raise ValueError(f"duplicate source row numbers in features: {instrument_id}")
    return frame


def _load_candidates(
    stores: dict[str, CandidateStore],
    index: dict[str, list[tuple[str, str]]],
    instrument_id: str,
    verify: bool,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for key, _ in sorted(index.get(instrument_id, [])):
        frame = stores[key].load_shard(key, instrument_id, verify=verify)
        if len(frame):
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    if result["candidate_id"].duplicated().any():
        raise ValueError(f"duplicate candidate IDs for {instrument_id}")
    return result.sort_values(["signal_date", "setup_family", "candidate_id"]).reset_index(drop=True)


def _research_rows(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    *,
    excluded_temporal: set[str],
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=RESEARCH_COLUMNS)
    outcome_columns = [
        "candidate_id",
        "ticker_split",
        "temporal_partition",
        "entry_status",
        *OUTCOME_LABEL_COLUMNS,
    ]
    joined = candidates.merge(
        outcomes[outcome_columns],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if joined["ticker_split"].isna().any():
        raise ValueError("candidate missing Module 05 outcome row")
    joined = joined[
        (joined["ticker_split"] == "development")
        & (~joined["temporal_partition"].isin(excluded_temporal))
    ].copy()
    if joined.empty:
        return pd.DataFrame(columns=RESEARCH_COLUMNS)
    joined = joined.merge(
        features,
        on=["signal_date", "signal_source_row_number"],
        how="left",
        validate="many_to_one",
    )
    if joined["feature_segment_id"].isna().any():
        raise ValueError("candidate signal row missing from feature store")
    if not (
        joined["feature_segment_id"].astype("int64")
        == joined["signal_segment_id"].astype("int64")
    ).all():
        raise ValueError("candidate/feature segment mismatch")
    joined = joined.drop(columns=["feature_segment_id"])
    if (joined["ticker_split"] != "development").any():
        raise ValueError("protected ticker split entered research rows")
    if joined["temporal_partition"].isin(excluded_temporal).any():
        raise ValueError("excluded temporal partition entered research rows")
    return joined.reindex(columns=RESEARCH_COLUMNS).sort_values(
        ["signal_date", "setup_family", "candidate_id"]
    ).reset_index(drop=True)


def _dataset_fingerprint(
    *,
    schema: dict[str, Any],
    manifest: pd.DataFrame,
    feature_fingerprint: dict[str, Any],
    outcome_fingerprint: dict[str, Any],
    candidate_catalog: dict[str, Any],
) -> dict[str, Any]:
    manifest_payload = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    payload = {
        "research_schema_sha256": schema["schema_sha256"],
        "feature_dataset_fingerprint_sha256": feature_fingerprint.get(
            "feature_dataset_fingerprint_sha256"
        ),
        "outcome_dataset_fingerprint_sha256": outcome_fingerprint.get(
            "outcome_dataset_fingerprint_sha256"
        ),
        "candidate_catalog_sha256": candidate_catalog["candidate_catalog_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "instrument_count": int(len(manifest)),
        "research_count": int(manifest["research_count"].sum()),
    }
    payload["research_dataset_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def build(
    config: dict[str, Any],
    *,
    resume: str | Path | None = None,
    dry_run: bool = False,
    max_new: int | None = None,
) -> Path:
    execution = config.get("execution") or {}
    context = RunContext.create(
        MODULE_NAME,
        MODULE_VERSION,
        config,
        execution["runs_root"],
        resume_dir=resume,
    )
    assert context.run_dir is not None
    run_dir = context.run_dir
    partial = run_dir / "research_manifest.partial.csv"
    error_path = run_dir / "file_errors.csv"
    try:
        (
            feature_store,
            outcome_store,
            candidate_stores,
            candidate_index,
            feature_fingerprint,
            outcome_fingerprint,
            candidate_catalog,
        ) = _upstream(config)
        schema = schema_payload()
        atomic_json(run_dir / "research_schema.json", schema)
        atomic_json(run_dir / "feature_dataset_fingerprint.json", feature_fingerprint)
        atomic_json(run_dir / "outcome_dataset_fingerprint.json", outcome_fingerprint)
        atomic_json(run_dir / "candidate_catalog.json", candidate_catalog)

        selection = config.get("selection") or {}
        if selection.get("ticker_split", "development") != "development":
            raise ValueError("Module 06 v1 permits only development ticker split")
        excluded_temporal = set(selection.get("excluded_temporal_partitions") or [])
        if "pseudo_lockbox_2024_2026" not in excluded_temporal:
            raise ValueError("Module 06 must exclude pseudo_lockbox_2024_2026")

        universe = outcome_store.manifest.copy().sort_values("instrument_id")
        universe = universe[
            universe["instrument_id"].astype(str).map(ticker_split) == "development"
        ].reset_index(drop=True)
        if dry_run:
            universe = universe.head(int(execution.get("dry_run_max_instruments", 25)))
        done = _load_done(partial)
        completed_this_call = 0
        verify = bool(execution.get("verify_upstream_shards", True))
        for row in universe.itertuples(index=False):
            instrument_id = str(row.instrument_id)
            if instrument_id in done:
                continue
            if max_new is not None and completed_this_call >= int(max_new):
                context.interrupt("max_new reached")
                return run_dir
            try:
                candidates = _load_candidates(candidate_stores, candidate_index, instrument_id, verify)
                outcomes = outcome_store.load_instrument(instrument_id, verify=verify)
                features = _feature_frame(feature_store, instrument_id, verify)
                research = _research_rows(
                    candidates,
                    outcomes,
                    features,
                    excluded_temporal=excluded_temporal,
                )
                relative = shard_relative_path(instrument_id)
                digest = write_research_shard(run_dir / relative, research)
                family_counts = (
                    research.groupby("setup_family").size().sort_index().astype(int).to_dict()
                    if len(research)
                    else {}
                )
                _append_row(
                    partial,
                    {
                        "instrument_id": instrument_id,
                        "ticker": str(row.ticker),
                        "exchange": str(row.exchange),
                        "research_count": int(len(research)),
                        "family_counts_json": json.dumps(family_counts, sort_keys=True, separators=(",", ":")),
                        "shard_relative_path": relative.as_posix(),
                        "shard_sha256": digest,
                    },
                    MANIFEST_COLUMNS,
                )
                done.add(instrument_id)
                completed_this_call += 1
                context.set_progress(
                    completed_instruments=len(done),
                    total_instruments=len(universe),
                )
                context.save_checkpoint(
                    {"last_instrument_id": instrument_id, "completed_instruments": len(done)}
                )
            except Exception as error:
                _append_row(
                    error_path,
                    {
                        "instrument_id": instrument_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                    ("instrument_id", "error_type", "error_message"),
                )
                raise

        manifest = pd.read_csv(partial).sort_values("instrument_id").reset_index(drop=True)
        if set(manifest["instrument_id"].astype(str)) != set(universe["instrument_id"].astype(str)):
            raise ValueError("research manifest does not account for development universe")
        final_manifest = run_dir / "research_manifest.csv"
        manifest.to_csv(final_manifest, index=False, lineterminator="\n")
        fingerprint = _dataset_fingerprint(
            schema=schema,
            manifest=manifest,
            feature_fingerprint=feature_fingerprint,
            outcome_fingerprint=outcome_fingerprint,
            candidate_catalog=candidate_catalog,
        )
        atomic_json(run_dir / "dataset_fingerprint.json", fingerprint)
        family_totals: dict[str, int] = {}
        for text in manifest["family_counts_json"].fillna("{}"):
            for family, count in json.loads(text).items():
                family_totals[family] = family_totals.get(family, 0) + int(count)
        summary = {
            "module_name": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "dry_run": bool(dry_run),
            "instrument_count": int(len(manifest)),
            "research_count": int(manifest["research_count"].sum()),
            "family_counts": dict(sorted(family_totals.items())),
            "protected_ticker_rows": 0,
            "pseudo_lockbox_rows": 0,
            "operational_errors": 0 if not error_path.exists() else int(len(pd.read_csv(error_path))),
            "research_dataset_fingerprint_sha256": fingerprint[
                "research_dataset_fingerprint_sha256"
            ],
        }
        atomic_json(run_dir / "summary.json", summary)
        context.record_output("research_manifest", final_manifest)
        context.record_output("research_schema", run_dir / "research_schema.json")
        context.record_output("dataset_fingerprint", run_dir / "dataset_fingerprint.json")
        context.complete(**summary)
        return run_dir
    except KeyboardInterrupt:
        context.interrupt("keyboard interrupt")
        raise
    except Exception as error:
        context.fail(error)
        raise


def verify(reference: str | Path, output: str | Path) -> Path:
    root = Path(reference).expanduser().resolve()
    store = ResearchStore(root)
    failures: list[dict[str, Any]] = []
    row_count = 0
    ids: set[str] = set()
    family_counts: dict[str, int] = {}
    for row in store.manifest.itertuples(index=False):
        path = root / row.shard_relative_path
        actual = file_sha256(path)
        if actual != row.shard_sha256:
            failures.append({"instrument_id": row.instrument_id, "reason": "hash_mismatch"})
            continue
        frame = store.load_instrument(row.instrument_id)
        row_count += len(frame)
        if len(frame) and (
            (frame["ticker_split"] != "development").any()
            or (frame["temporal_partition"] == "pseudo_lockbox_2024_2026").any()
        ):
            failures.append({"instrument_id": row.instrument_id, "reason": "protected_row"})
        duplicate = ids.intersection(set(frame["candidate_id"].astype(str)))
        if duplicate:
            failures.append({"instrument_id": row.instrument_id, "reason": "duplicate_candidate_id"})
        ids.update(frame["candidate_id"].astype(str))
        for family, count in frame.groupby("setup_family").size().items():
            family_counts[str(family)] = family_counts.get(str(family), 0) + int(count)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "instrument_count": int(len(store.manifest)),
        "research_count": int(row_count),
        "unique_candidate_ids": int(len(ids)),
        "family_counts": dict(sorted(family_counts.items())),
        "failures": failures,
    }
    output_path = Path(output).expanduser().resolve()
    atomic_json(output_path, report)
    if failures:
        raise ValueError(f"research verification failed: {len(failures)} failures")
    return output_path


def publish(run: str | Path, target: str | Path) -> Path:
    source = Path(run).expanduser().resolve()
    destination = Path(target).expanduser().resolve()
    manifest = json.loads((source / "run_manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("only completed research runs may be published")
    if destination.exists():
        raise FileExistsError(f"publication target already exists: {destination}")
    shutil.copytree(source, destination)
    return destination
