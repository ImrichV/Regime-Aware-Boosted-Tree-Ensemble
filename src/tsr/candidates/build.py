from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..data.stooq import StooqCatalog
from ..features.storage import FeatureStore
from ..run import RunContext, atomic_json
from .base import CandidateContext, CandidateGenerator, GeneratorRegistry
from .diagnostics import candidate_diagnostics
from .engine import combine_candidate_frames, materialize_candidates
from .schema import MODULE_NAME, MODULE_VERSION, schema_payload
from .storage import (
    CandidateStore,
    file_sha256,
    shard_relative_path,
    write_candidate_shard,
)

MANIFEST_COLUMNS = (
    "generator_key",
    "setup_family",
    "setup_version",
    "direction",
    "instrument_id",
    "ticker",
    "exchange",
    "instrument_class",
    "source_bar_count",
    "candidate_count",
    "first_signal_date",
    "last_signal_date",
    "shard_relative_path",
    "shard_sha256",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _load_generator(spec: Mapping[str, Any]) -> CandidateGenerator:
    import_path = str(spec["class"])
    if ":" not in import_path:
        raise ValueError("generator class must use module:Class format")
    module_name, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    generator = cls(spec.get("config") or {})
    if not isinstance(generator, CandidateGenerator):
        raise TypeError(f"{import_path} is not a CandidateGenerator")
    return generator


def load_registry(config: Mapping[str, Any]) -> GeneratorRegistry:
    registry = GeneratorRegistry()
    for spec in config.get("generators", []):
        registry.register(_load_generator(spec))
    if not len(registry):
        raise ValueError("at least one candidate generator is required")
    return registry


def _upstream(config: Mapping[str, Any]) -> tuple[StooqCatalog, FeatureStore, dict[str, Any], dict[str, Any]]:
    upstream = config["upstream"]
    data_fingerprint = _read_json(upstream["data_dataset_fingerprint_path"])
    feature_fingerprint = _read_json(upstream["feature_dataset_fingerprint_path"])
    catalog = StooqCatalog(
        upstream["archive_path"],
        upstream["symbol_manifest_path"],
        invalid_policy="drop",
    )
    features = FeatureStore(upstream["feature_root"])
    feature_schema_version = features.schema["schema_version"]
    if feature_fingerprint.get("feature_schema_version") not in {None, feature_schema_version}:
        raise ValueError("feature fingerprint/schema mismatch")
    manifest_ids = set(catalog.manifest["instrument_id"])
    feature_ids = set(features.manifest["instrument_id"])
    if manifest_ids != feature_ids:
        raise ValueError("Module 01 and Module 02 instrument universes differ")
    return catalog, features, data_fingerprint, feature_fingerprint


def _instrument_universe(config: Mapping[str, Any], catalog: StooqCatalog) -> pd.DataFrame:
    selection = config.get("selection") or {}
    universe = catalog.list_instruments(
        exchange=selection.get("exchange"),
        instrument_class=selection.get("instrument_class"),
        require_rows=bool(selection.get("require_rows", True)),
    )
    explicit = selection.get("instrument_ids")
    if explicit:
        order = {instrument_id: index for index, instrument_id in enumerate(explicit)}
        universe = universe[universe["instrument_id"].isin(order)].copy()
        universe["_order"] = universe["instrument_id"].map(order)
        universe = universe.sort_values("_order").drop(columns="_order")
    else:
        universe = universe.sort_values("instrument_id")
    limit = selection.get("max_instruments")
    if limit is not None:
        universe = universe.head(int(limit))
    return universe.reset_index(drop=True)


def _context(
    catalog: StooqCatalog,
    features: FeatureStore,
    identity: Any,
    feature_dataset_sha256: str,
) -> CandidateContext:
    bars = catalog.load_instrument(identity.instrument_id)
    feature_frame = features.load_instrument(identity.instrument_id, verify=False)
    context = CandidateContext(
        instrument_id=identity.instrument_id,
        ticker=identity.ticker,
        exchange=identity.exchange,
        instrument_class=identity.instrument_class,
        bars=bars,
        features=feature_frame,
        feature_schema_version=features.schema["schema_version"],
        feature_dataset_sha256=feature_dataset_sha256,
    )
    context.validate()
    return context


def _append_row(path: Path, row: Mapping[str, Any], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row], columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")


def _load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    return set(zip(frame["generator_key"], frame["instrument_id"], strict=False))


def _generator_manifest(registry: GeneratorRegistry, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs_by_class = {str(spec["class"]): spec for spec in config.get("generators", [])}
    rows = []
    for generator in registry.list():
        import_path = f"{generator.__class__.__module__}:{generator.__class__.__name__}"
        spec = specs_by_class.get(import_path, {})
        rows.append(
            {
                "generator_key": generator.metadata.generator_key,
                "class": import_path,
                "metadata": {
                    "setup_family": generator.metadata.setup_family,
                    "setup_version": generator.metadata.setup_version,
                    "direction": generator.metadata.direction,
                    "required_features": list(generator.metadata.required_features),
                    "description": generator.metadata.description,
                    "permissive": generator.metadata.permissive,
                    "overlap_policy": generator.metadata.overlap_policy,
                    "decision_time": generator.metadata.decision_time,
                    "earliest_entry_rule": generator.metadata.earliest_entry_rule,
                },
                "config": generator.config,
                "class_spec": spec,
            }
        )
    return rows


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
    partial_manifest = run_dir / "candidate_manifest.partial.csv"
    try:
        catalog, feature_store, data_fingerprint, feature_fingerprint = _upstream(config)
        registry = load_registry(config)
        generator_manifest = _generator_manifest(registry, config)
        allow_test = bool((config.get("framework") or {}).get("allow_test_generators", False))
        if not allow_test:
            for row in generator_manifest:
                if row["class"].startswith("tsr.candidates.testing:"):
                    raise ValueError("test generators require framework.allow_test_generators=true")
        universe = _instrument_universe(config, catalog)
        if dry_run:
            universe = universe.head(int(execution.get("dry_run_max_files", 25)))
        feature_dataset_sha256 = str(
            feature_fingerprint.get("dataset_sha256")
            or feature_fingerprint.get("fingerprint_sha256")
            or feature_fingerprint.get("dataset_fingerprint_sha256")
            or feature_fingerprint.get("feature_dataset_fingerprint_sha256")
            or _canonical_hash(feature_fingerprint)
        )
        schema = schema_payload()
        atomic_json(run_dir / "candidate_schema.json", schema)
        atomic_json(run_dir / "generator_manifest.json", {"generators": generator_manifest})
        done = _load_done(partial_manifest)
        generated = 0
        operational_errors: list[dict[str, Any]] = []
        expected_pairs = len(universe) * len(registry)
        for generator in registry.list():
            for identity in universe.itertuples(index=False):
                pair = (generator.metadata.generator_key, identity.instrument_id)
                if pair in done:
                    continue
                try:
                    candidate_context = _context(
                        catalog, feature_store, identity, feature_dataset_sha256
                    )
                    frame = materialize_candidates(generator, candidate_context)
                    relative = shard_relative_path(
                        generator.metadata.generator_key, identity.instrument_id
                    )
                    shard_hash = write_candidate_shard(run_dir / relative, frame)
                    first_signal = None if frame.empty else int(frame["signal_date"].min())
                    last_signal = None if frame.empty else int(frame["signal_date"].max())
                    row = {
                        "generator_key": generator.metadata.generator_key,
                        "setup_family": generator.metadata.setup_family,
                        "setup_version": generator.metadata.setup_version,
                        "direction": generator.metadata.direction,
                        "instrument_id": identity.instrument_id,
                        "ticker": identity.ticker,
                        "exchange": identity.exchange,
                        "instrument_class": identity.instrument_class,
                        "source_bar_count": int(len(candidate_context.bars)),
                        "candidate_count": int(len(frame)),
                        "first_signal_date": first_signal,
                        "last_signal_date": last_signal,
                        "shard_relative_path": str(relative.as_posix()),
                        "shard_sha256": shard_hash,
                    }
                    _append_row(partial_manifest, row, MANIFEST_COLUMNS)
                    done.add(pair)
                    generated += 1
                    context.set_progress(
                        completed_pairs=len(done),
                        expected_pairs=expected_pairs,
                        current_generator=generator.metadata.generator_key,
                        current_instrument=identity.instrument_id,
                    )
                    context.save_checkpoint(
                        {
                            "completed_pairs": len(done),
                            "expected_pairs": expected_pairs,
                            "last_generator_key": generator.metadata.generator_key,
                            "last_instrument_id": identity.instrument_id,
                        }
                    )
                    if max_new is not None and generated >= int(max_new):
                        context.interrupt("max_new reached")
                        return run_dir
                except KeyboardInterrupt:
                    context.interrupt("keyboard interrupt")
                    raise
                except Exception as error:  # operational record; scientific violations still fail publication
                    error_row = {
                        "generator_key": generator.metadata.generator_key,
                        "instrument_id": identity.instrument_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                    operational_errors.append(error_row)
                    context.log("candidate_pair_failed", level="ERROR", **error_row)
        manifest = pd.read_csv(partial_manifest) if partial_manifest.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
        manifest = manifest.sort_values(["generator_key", "instrument_id"], kind="mergesort")
        manifest.to_csv(run_dir / "candidate_manifest.csv", index=False, lineterminator="\n")
        errors = pd.DataFrame(
            operational_errors,
            columns=("generator_key", "instrument_id", "error_type", "error_message"),
        )
        errors.to_csv(run_dir / "file_errors.csv", index=False, lineterminator="\n")
        all_frames = []
        for row in manifest.itertuples(index=False):
            path = run_dir / row.shard_relative_path
            if int(row.candidate_count):
                all_frames.append(pd.read_csv(path, keep_default_na=False))
        all_candidates = combine_candidate_frames(all_frames)
        diagnostics = candidate_diagnostics(
            all_candidates,
            eligible_bar_count=int(manifest["source_bar_count"].sum()) if len(manifest) else 0,
            instrument_count=int(manifest["instrument_id"].nunique()) if len(manifest) else 0,
        )
        atomic_json(run_dir / "candidate_diagnostics.json", diagnostics)
        completed_pairs = int(len(manifest))
        candidate_count = int(manifest["candidate_count"].sum()) if len(manifest) else 0
        dataset_fingerprint = {
            "module_name": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "candidate_schema_version": schema["schema_version"],
            "candidate_schema_sha256": schema["schema_sha256"],
            "data_dataset_fingerprint": data_fingerprint,
            "feature_dataset_fingerprint": feature_fingerprint,
            "generator_manifest_sha256": _canonical_hash(generator_manifest),
            "candidate_manifest_sha256": file_sha256(run_dir / "candidate_manifest.csv"),
            "candidate_count": candidate_count,
            "completed_pairs": completed_pairs,
            "expected_pairs": expected_pairs,
            "dry_run": bool(dry_run),
        }
        dataset_fingerprint["dataset_sha256"] = _canonical_hash(dataset_fingerprint)
        atomic_json(run_dir / "dataset_fingerprint.json", dataset_fingerprint)
        summary = {
            "module_name": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "dry_run": bool(dry_run),
            "generator_count": len(registry),
            "instrument_count": int(len(universe)),
            "expected_pairs": expected_pairs,
            "completed_pairs": completed_pairs,
            "candidate_count": candidate_count,
            "operational_errors": int(len(errors)),
            "all_pairs_accounted_for": completed_pairs == expected_pairs,
            "test_generators_present": any(
                row["class"].startswith("tsr.candidates.testing:")
                for row in generator_manifest
            ),
            "candidate_dataset_sha256": dataset_fingerprint["dataset_sha256"],
        }
        atomic_json(run_dir / "summary.json", summary)
        for name in (
            "candidate_schema.json",
            "generator_manifest.json",
            "candidate_manifest.csv",
            "candidate_diagnostics.json",
            "file_errors.csv",
            "dataset_fingerprint.json",
            "summary.json",
        ):
            context.record_output(name, run_dir / name)
        context.complete(
            candidate_count=candidate_count,
            completed_pairs=completed_pairs,
            expected_pairs=expected_pairs,
        )
        return run_dir
    except KeyboardInterrupt:
        raise
    except Exception as error:
        context.fail(error)
        raise


def verify(reference: str | Path, output: str | Path) -> Path:
    root = Path(reference).expanduser().resolve()
    store = CandidateStore(root)
    failures: list[dict[str, Any]] = []
    verified = 0
    candidate_ids: set[str] = set()
    for row in store.manifest.itertuples(index=False):
        try:
            frame = store.load_shard(row.generator_key, row.instrument_id, verify=True)
            duplicates = set(frame["candidate_id"]).intersection(candidate_ids)
            if duplicates:
                raise ValueError(f"candidate IDs duplicated across shards: {sorted(duplicates)[:3]}")
            candidate_ids.update(frame["candidate_id"])
            verified += 1
        except Exception as error:
            failures.append(
                {
                    "generator_key": row.generator_key,
                    "instrument_id": row.instrument_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    result = {
        "reference": str(root),
        "shards_expected": int(len(store.manifest)),
        "shards_verified": verified,
        "candidate_ids_verified": len(candidate_ids),
        "failure_count": len(failures),
        "passed": not failures and verified == len(store.manifest),
        "failures": failures[:100],
    }
    output_path = Path(output).expanduser().resolve()
    atomic_json(output_path, result)
    return output_path


def publish(run: str | Path, target: str | Path) -> Path:
    source = Path(run).expanduser().resolve()
    summary = _read_json(source / "summary.json")
    manifest = _read_json(source / "run_manifest.json")
    if summary["dry_run"]:
        raise ValueError("dry-run candidate output cannot be published")
    if summary["operational_errors"]:
        raise ValueError("candidate run has operational errors")
    if not summary["all_pairs_accounted_for"]:
        raise ValueError("candidate run is incomplete")
    if summary["test_generators_present"]:
        raise ValueError("test-generator candidate output cannot be published")
    if manifest["status"] != "completed":
        raise ValueError("candidate run did not complete")
    destination = Path(target).expanduser().resolve()
    temporary = Path(str(destination) + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    for name in (
        "candidate_schema.json",
        "generator_manifest.json",
        "candidate_manifest.csv",
        "candidate_diagnostics.json",
        "file_errors.csv",
        "dataset_fingerprint.json",
        "summary.json",
        "run_manifest.json",
    ):
        shutil.copy2(source / name, temporary / name)
    candidate_manifest = pd.read_csv(source / "candidate_manifest.csv")
    for row in candidate_manifest.itertuples(index=False):
        src = source / row.shard_relative_path
        dst = temporary / row.shard_relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    (temporary / "SOURCE_RUN.txt").write_text(str(source) + "\n")
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(temporary, destination)
    return destination
