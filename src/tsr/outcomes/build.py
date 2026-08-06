from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..candidates.schema import CANDIDATE_SCHEMA_VERSION
from ..candidates.storage import CandidateStore, file_sha256 as candidate_file_sha256
from ..data.stooq import StooqCatalog
from ..features.storage import FeatureStore
from ..run import RunContext, atomic_json
from .engine import compute_outcomes
from .schema import MODULE_NAME, MODULE_VERSION, OUTCOME_SCHEMA_VERSION, schema_payload
from .storage import OutcomeStore, file_sha256, shard_relative_path, write_outcome_shard

MANIFEST_COLUMNS = (
    "instrument_id",
    "ticker",
    "exchange",
    "instrument_class",
    "candidate_count",
    "outcome_count",
    "entry_available_count",
    "no_entry_count",
    "complete_60_count",
    "shard_relative_path",
    "shard_sha256",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _append_row(path: Path, row: Mapping[str, Any], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=columns).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        lineterminator="\n",
    )


def _load_done(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(path)["instrument_id"].astype(str))


def _candidate_catalog(config: Mapping[str, Any]) -> tuple[list[CandidateStore], dict[str, Any], dict[str, list[tuple[int, str]]]]:
    roots = [Path(value).expanduser().resolve() for value in config["upstream"]["candidate_roots"]]
    if not roots:
        raise ValueError("at least one candidate root is required")
    stores: list[CandidateStore] = []
    catalog_rows: list[dict[str, Any]] = []
    instrument_index: dict[str, list[tuple[int, str]]] = {}
    seen_generator_keys: set[str] = set()
    for store_index, root in enumerate(roots):
        store = CandidateStore(root)
        schema_version = store.schema.get("schema_version")
        if schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError(f"candidate schema mismatch at {root}: {schema_version}")
        generator_keys = sorted(set(store.manifest["generator_key"].astype(str)))
        overlap = seen_generator_keys.intersection(generator_keys)
        if overlap:
            raise ValueError(f"duplicate generator keys across candidate roots: {sorted(overlap)}")
        seen_generator_keys.update(generator_keys)
        stores.append(store)
        manifest_rows = store.manifest[
            ["generator_key", "instrument_id", "candidate_count", "shard_sha256"]
        ].sort_values(["generator_key", "instrument_id"])
        manifest_sha256 = _canonical_hash(manifest_rows.to_dict("records"))
        for generator_key in generator_keys:
            subset = store.manifest[store.manifest["generator_key"] == generator_key]
            catalog_rows.append(
                {
                    "generator_key": generator_key,
                    "candidate_schema_version": schema_version,
                    "instrument_rows": int(len(subset)),
                    "candidate_count": int(subset["candidate_count"].sum()),
                    "manifest_content_sha256": manifest_sha256,
                }
            )
        nonempty = store.manifest[store.manifest["candidate_count"] > 0]
        for row in nonempty.itertuples(index=False):
            instrument_index.setdefault(str(row.instrument_id), []).append(
                (store_index, str(row.generator_key))
            )
    payload = {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "sources": sorted(catalog_rows, key=lambda item: item["generator_key"]),
    }
    payload["candidate_catalog_sha256"] = _canonical_hash(payload)
    return stores, payload, instrument_index


def _upstream(config: Mapping[str, Any]):
    upstream = config["upstream"]
    data_fingerprint = _read_json(upstream["data_dataset_fingerprint_path"])
    feature_fingerprint = _read_json(upstream["feature_dataset_fingerprint_path"])
    catalog = StooqCatalog(
        upstream["archive_path"],
        upstream["symbol_manifest_path"],
        invalid_policy="drop",
    )
    features = FeatureStore(upstream["feature_root"])
    catalog_ids = set(catalog.manifest["instrument_id"].astype(str))
    feature_ids = set(features.manifest["instrument_id"].astype(str))
    if catalog_ids != feature_ids:
        raise ValueError("Module 01 and Module 02 instrument universes differ")
    return catalog, features, data_fingerprint, feature_fingerprint


def _load_candidates_for_instrument(
    stores: list[CandidateStore],
    instrument_index: dict[str, list[tuple[int, str]]],
    instrument_id: str,
    *,
    verify: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for store_index, generator_key in instrument_index.get(instrument_id, []):
        frame = stores[store_index].load_shard(generator_key, instrument_id, verify=verify)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["signal_date", "setup_family", "candidate_id"]).reset_index(drop=True)
    if result["candidate_id"].duplicated().any():
        duplicate = result.loc[result["candidate_id"].duplicated(), "candidate_id"].iloc[0]
        raise ValueError(f"duplicate candidate ID across catalog: {duplicate}")
    return result


def _dataset_fingerprint(
    *,
    schema: dict[str, Any],
    candidate_catalog: dict[str, Any],
    manifest: pd.DataFrame,
    data_fingerprint: dict[str, Any],
    feature_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    manifest_payload = manifest[
        ["instrument_id", "candidate_count", "outcome_count", "shard_sha256"]
    ].sort_values("instrument_id").to_dict("records")
    payload = {
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "schema_sha256": schema["schema_sha256"],
        "candidate_catalog_sha256": candidate_catalog["candidate_catalog_sha256"],
        "data_dataset_fingerprint_sha256": data_fingerprint.get("dataset_fingerprint_sha256"),
        "feature_dataset_fingerprint_sha256": feature_fingerprint.get(
            "feature_dataset_fingerprint_sha256"
        ),
        "manifest_content_sha256": _canonical_hash(manifest_payload),
        "candidate_count": int(manifest["candidate_count"].sum()),
        "outcome_count": int(manifest["outcome_count"].sum()),
    }
    payload["outcome_dataset_fingerprint_sha256"] = _canonical_hash(payload)
    return payload


def build(
    config: dict[str, Any],
    resume: str | Path | None = None,
    dry_run: bool = False,
    max_new: int | None = None,
) -> Path:
    execution = config["execution"]
    context = RunContext.create(
        MODULE_NAME,
        MODULE_VERSION,
        config,
        execution["runs_root"],
        resume_dir=resume,
    )
    run_dir = context.run_dir
    assert run_dir is not None
    partial_manifest = run_dir / "outcome_manifest.partial.csv"
    error_path = run_dir / "file_errors.csv"
    try:
        catalog, feature_store, data_fingerprint, feature_fingerprint = _upstream(config)
        stores, candidate_catalog, instrument_index = _candidate_catalog(config)
        schema = schema_payload()
        atomic_json(run_dir / "outcome_schema.json", schema)
        atomic_json(run_dir / "candidate_catalog.json", candidate_catalog)
        atomic_json(run_dir / "data_dataset_fingerprint.json", data_fingerprint)
        atomic_json(run_dir / "feature_dataset_fingerprint.json", feature_fingerprint)

        selection = config.get("selection") or {}
        universe = catalog.list_instruments(
            instrument_class=selection.get("instrument_class", "stocks"),
            require_rows=bool(selection.get("require_rows", True)),
        ).sort_values("instrument_id")
        explicit = selection.get("instrument_ids")
        if explicit:
            order = {str(value): index for index, value in enumerate(explicit)}
            universe = universe[universe["instrument_id"].isin(order)].copy()
            universe["_order"] = universe["instrument_id"].map(order)
            universe = universe.sort_values("_order").drop(columns="_order")
        if selection.get("max_instruments") is not None:
            universe = universe.head(int(selection["max_instruments"]))
        if dry_run:
            universe = universe.head(int(execution.get("dry_run_max_instruments", 25)))
        universe = universe.reset_index(drop=True)

        universe_ids = set(universe["instrument_id"].astype(str))
        unknown_candidate_ids = set(instrument_index).difference(set(catalog.manifest["instrument_id"].astype(str)))
        if unknown_candidate_ids:
            raise ValueError(f"candidate catalog contains unknown instruments: {sorted(unknown_candidate_ids)[:5]}")

        done = _load_done(partial_manifest)
        if partial_manifest.exists() and partial_manifest.stat().st_size:
            existing_partial = pd.read_csv(partial_manifest)
            candidates_processed = int(existing_partial["candidate_count"].sum())
        else:
            candidates_processed = 0
        completed_this_call = 0
        verify_inputs = bool(execution.get("verify_candidate_shards", True))
        for identity in universe.itertuples(index=False):
            instrument_id = str(identity.instrument_id)
            if instrument_id in done:
                continue
            if max_new is not None and completed_this_call >= int(max_new):
                context.interrupt("max_new reached")
                return run_dir
            try:
                candidates = _load_candidates_for_instrument(
                    stores,
                    instrument_index,
                    instrument_id,
                    verify=verify_inputs,
                )
                if candidates.empty:
                    outcomes = pd.DataFrame(columns=schema["columns"])
                else:
                    bars = catalog.load_instrument(instrument_id)
                    features = feature_store.load_instrument(instrument_id, verify=False)
                    outcomes = compute_outcomes(
                        candidates,
                        bars,
                        features,
                        candidate_catalog_sha256=candidate_catalog["candidate_catalog_sha256"],
                    )
                relative = shard_relative_path(instrument_id)
                digest = write_outcome_shard(run_dir / relative, outcomes)
                row = {
                    "instrument_id": instrument_id,
                    "ticker": str(identity.ticker),
                    "exchange": str(identity.exchange),
                    "instrument_class": str(identity.instrument_class),
                    "candidate_count": int(len(candidates)),
                    "outcome_count": int(len(outcomes)),
                    "entry_available_count": int(
                        (outcomes["entry_status"] == "available").sum() if len(outcomes) else 0
                    ),
                    "no_entry_count": int(
                        (outcomes["entry_status"] != "available").sum() if len(outcomes) else 0
                    ),
                    "complete_60_count": int(
                        (outcomes["evaluation_termination"] == "complete_60").sum()
                        if len(outcomes)
                        else 0
                    ),
                    "shard_relative_path": relative.as_posix(),
                    "shard_sha256": digest,
                }
                _append_row(partial_manifest, row, MANIFEST_COLUMNS)
                done.add(instrument_id)
                completed_this_call += 1
                candidates_processed += int(len(candidates))
                context.set_progress(
                    completed_instruments=len(done),
                    total_instruments=len(universe),
                    candidates_processed=candidates_processed,
                )
                context.save_checkpoint(
                    {
                        "last_instrument_id": instrument_id,
                        "completed_instruments": len(done),
                    }
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

        manifest = pd.read_csv(partial_manifest).sort_values("instrument_id").reset_index(drop=True)
        if set(manifest["instrument_id"].astype(str)) != universe_ids:
            raise ValueError("outcome manifest does not account for the selected universe")
        if not (manifest["candidate_count"] == manifest["outcome_count"]).all():
            raise ValueError("candidate/outcome row-count mismatch")
        final_manifest = run_dir / "outcome_manifest.csv"
        manifest.to_csv(final_manifest, index=False, lineterminator="\n")
        fingerprint = _dataset_fingerprint(
            schema=schema,
            candidate_catalog=candidate_catalog,
            manifest=manifest,
            data_fingerprint=data_fingerprint,
            feature_fingerprint=feature_fingerprint,
        )
        atomic_json(run_dir / "dataset_fingerprint.json", fingerprint)
        summary = {
            "module_name": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "dry_run": bool(dry_run),
            "instrument_count": int(len(manifest)),
            "candidate_count": int(manifest["candidate_count"].sum()),
            "outcome_count": int(manifest["outcome_count"].sum()),
            "entry_available_count": int(manifest["entry_available_count"].sum()),
            "no_entry_count": int(manifest["no_entry_count"].sum()),
            "complete_60_count": int(manifest["complete_60_count"].sum()),
            "candidate_catalog_sha256": candidate_catalog["candidate_catalog_sha256"],
            "outcome_dataset_fingerprint_sha256": fingerprint[
                "outcome_dataset_fingerprint_sha256"
            ],
            "operational_errors": 0 if not error_path.exists() else int(len(pd.read_csv(error_path))),
            "all_candidates_accounted_for": bool(
                (manifest["candidate_count"] == manifest["outcome_count"]).all()
            ),
        }
        atomic_json(run_dir / "summary.json", summary)
        context.record_output("outcome_manifest", final_manifest)
        context.record_output("outcome_schema", run_dir / "outcome_schema.json")
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
    store = OutcomeStore(root)
    failures: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    outcome_count = 0
    for row in store.manifest.itertuples(index=False):
        path = root / row.shard_relative_path
        actual_hash = file_sha256(path)
        if actual_hash != row.shard_sha256:
            failures.append(
                {
                    "instrument_id": row.instrument_id,
                    "reason": "shard_hash_mismatch",
                    "expected": row.shard_sha256,
                    "actual": actual_hash,
                }
            )
            continue
        frame = store.load_instrument(row.instrument_id, verify=False)
        if len(frame) != int(row.candidate_count):
            failures.append(
                {
                    "instrument_id": row.instrument_id,
                    "reason": "candidate_outcome_count_mismatch",
                }
            )
        overlap = candidate_ids.intersection(set(frame["candidate_id"].astype(str)))
        if overlap:
            failures.append(
                {
                    "instrument_id": row.instrument_id,
                    "reason": "duplicate_candidate_id",
                    "candidate_id": sorted(overlap)[0],
                }
            )
        candidate_ids.update(frame["candidate_id"].astype(str))
        outcome_count += len(frame)
    summary = _read_json(root / "summary.json")
    if outcome_count != int(summary["outcome_count"]):
        failures.append(
            {
                "reason": "summary_outcome_count_mismatch",
                "expected": summary["outcome_count"],
                "actual": outcome_count,
            }
        )
    result = {
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "reference": str(root),
        "passed": not failures,
        "shards_verified": int(len(store.manifest)),
        "outcomes_verified": int(outcome_count),
        "unique_candidate_ids": int(len(candidate_ids)),
        "failures": failures,
    }
    output_path = Path(output).expanduser().resolve()
    atomic_json(output_path, result)
    return output_path


def publish(run: str | Path, target: str | Path) -> Path:
    run_root = Path(run).expanduser().resolve()
    target_root = Path(target).expanduser().resolve()
    manifest = _read_json(run_root / "run_manifest.json")
    summary = _read_json(run_root / "summary.json")
    if manifest.get("status") != "completed":
        raise ValueError("only completed outcome runs may be published")
    if summary.get("dry_run"):
        raise ValueError("dry-run outcomes may not be published")
    if not summary.get("all_candidates_accounted_for"):
        raise ValueError("candidate accounting gate failed")
    if target_root.exists():
        raise FileExistsError(f"publication target already exists: {target_root}")
    shutil.copytree(run_root, target_root)
    (target_root / "SOURCE_RUN.txt").write_text(str(run_root) + "\n", encoding="utf-8")
    return target_root
