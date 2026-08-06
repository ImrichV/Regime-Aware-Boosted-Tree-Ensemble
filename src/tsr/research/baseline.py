from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import load_yaml
from ..run import RunContext, atomic_json
from .schema import (
    BASELINE_POLICY_VERSION,
    FOLD_DEFINITIONS,
    MODULE_NAME,
    MODULE_VERSION,
    PREDICTOR_COLUMNS,
)
from .storage import ResearchStore

READ_COLUMNS = (
    "candidate_id",
    "instrument_id",
    "setup_family",
    "direction",
    "signal_date",
    "ticker_split",
    "temporal_partition",
    *PREDICTOR_COLUMNS,
    "h20_end_date",
    "h20_directional_close_return",
    "barrier_2r_20_result",
)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rank_metrics(y_return: np.ndarray, y_target: np.ndarray, score: np.ndarray, top_fraction: float) -> dict[str, Any]:
    n = len(score)
    if n == 0:
        return {"n": 0}
    order = np.argsort(score, kind="mergesort")
    count = max(1, int(math.ceil(n * top_fraction)))
    bottom = order[:count]
    top = order[-count:]
    corr = spearmanr(score, y_return).statistic
    try:
        auc = roc_auc_score(y_target, score) if len(np.unique(y_target)) == 2 else None
    except ValueError:
        auc = None
    base_rate = float(np.mean(y_target))
    top_rate = float(np.mean(y_target[top]))
    return {
        "n": int(n),
        "spearman": _safe_float(corr),
        "mean_return": float(np.mean(y_return)),
        "top_return": float(np.mean(y_return[top])),
        "bottom_return": float(np.mean(y_return[bottom])),
        "top_minus_mean": float(np.mean(y_return[top]) - np.mean(y_return)),
        "top_minus_bottom": float(np.mean(y_return[top]) - np.mean(y_return[bottom])),
        "base_target_rate": base_rate,
        "top_target_rate": top_rate,
        "top_target_rate_lift": (top_rate / base_rate) if base_rate > 0 else None,
        "auc": _safe_float(auc),
    }


def _prediction_bytes(frame: pd.DataFrame) -> bytes:
    text = io.StringIO(newline="")
    frame.to_csv(text, index=False, lineterminator="\n", float_format="%.12g", na_rep="")
    return gzip.compress(text.getvalue().encode("utf-8"), compresslevel=9, mtime=0)


def _write_prediction(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _prediction_bytes(frame)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _load_safe_data(store: ResearchStore, verify: bool) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for frame in store.iter_frames(verify=verify, usecols=READ_COLUMNS):
        if len(frame):
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame(columns=READ_COLUMNS)
    result = pd.concat(pieces, ignore_index=True)
    if (result["ticker_split"] != "development").any():
        raise ValueError("protected ticker split entered baseline input")
    if (result["temporal_partition"] == "pseudo_lockbox_2024_2026").any():
        raise ValueError("pseudo-lockbox row entered baseline input")
    if result["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate ID in baseline input")
    for column in ("signal_date", "h20_end_date", *PREDICTOR_COLUMNS, "h20_directional_close_return"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _preprocessing_steps() -> list[tuple[str, Any]]:
    # Return fresh estimator instances so independently fitted pipelines never share state.
    return [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]


def _pipelines(cfg: dict[str, Any]) -> tuple[Pipeline, Pipeline]:
    ridge = Pipeline(
        [*_preprocessing_steps(), ("model", Ridge(alpha=float(cfg["ridge_alpha"])))]
    )
    logistic = Pipeline(
        [
            *_preprocessing_steps(),
            (
                "model",
                LogisticRegression(
                    C=float(cfg["logistic_c"]),
                    solver="lbfgs",
                    max_iter=int(cfg["logistic_max_iter"]),
                    random_state=int(cfg["random_seed"]),
                ),
            ),
        ]
    )
    return ridge, logistic


def _top_coefficients(pipe: Pipeline, limit: int = 20) -> list[dict[str, Any]]:
    try:
        names = pipe[:-1].get_feature_names_out(PREDICTOR_COLUMNS)
        coef = np.asarray(pipe.named_steps["model"].coef_).reshape(-1)
        order = np.argsort(np.abs(coef))[::-1][:limit]
        return [
            {"feature": str(names[index]), "coefficient": float(coef[index])}
            for index in order
        ]
    except Exception:
        return []


def _evaluate_family(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    valid = frame[
        frame["h20_directional_close_return"].notna()
        & frame["h20_end_date"].notna()
        & frame["barrier_2r_20_result"].isin(["target", "stop", "timeout"])
    ].copy()
    valid["y_target"] = (valid["barrier_2r_20_result"] == "target").astype(int)
    fold_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for fold in FOLD_DEFINITIONS:
        train = valid[
            (valid["signal_date"] < fold["test_start"])
            & (valid["h20_end_date"] < fold["test_start"])
        ].copy()
        test = valid[
            (valid["signal_date"] >= fold["test_start"])
            & (valid["signal_date"] <= fold["test_end"])
            & (valid["h20_end_date"] <= fold["test_end"])
        ].copy()
        if len(train) < int(cfg["minimum_train_rows"]) or len(test) < int(cfg["minimum_test_rows"]):
            fold_rows.append(
                {
                    "fold": fold["name"],
                    "status": "insufficient_rows",
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                }
            )
            continue
        if train["y_target"].nunique() < 2:
            fold_rows.append(
                {
                    "fold": fold["name"],
                    "status": "single_class_train",
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                }
            )
            continue
        x_train = train.loc[:, PREDICTOR_COLUMNS].replace([np.inf, -np.inf], np.nan)
        x_test = test.loc[:, PREDICTOR_COLUMNS].replace([np.inf, -np.inf], np.nan)
        y_return_train = train["h20_directional_close_return"].to_numpy(float)
        y_target_train = train["y_target"].to_numpy(int)
        ridge, logistic = _pipelines(cfg)
        ridge.fit(x_train, y_return_train)
        logistic.fit(x_train, y_target_train)
        fold_prediction = pd.DataFrame(
            {
                "candidate_id": test["candidate_id"].astype(str).to_numpy(),
                "instrument_id": test["instrument_id"].astype(str).to_numpy(),
                "setup_family": test["setup_family"].astype(str).to_numpy(),
                "direction": test["direction"].astype(str).to_numpy(),
                "signal_date": test["signal_date"].astype(int).to_numpy(),
                "h20_end_date": test["h20_end_date"].astype(int).to_numpy(),
                "fold": fold["name"],
                "y_return": test["h20_directional_close_return"].to_numpy(float),
                "y_target": test["y_target"].to_numpy(int),
                "raw_score": test["raw_setup_strength"].to_numpy(float),
                "ridge_score": ridge.predict(x_test),
                "logistic_score": logistic.predict_proba(x_test)[:, 1],
            }
        )
        predictions.append(fold_prediction)
        fold_entry: dict[str, Any] = {
            "fold": fold["name"],
            "status": "valid",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start": int(train["signal_date"].min()),
            "train_last_label_date": int(train["h20_end_date"].max()),
            "test_start": int(test["signal_date"].min()),
            "test_last_label_date": int(test["h20_end_date"].max()),
            "test_target_rate": float(test["y_target"].mean()),
        }
        for score_name in ("raw_score", "ridge_score", "logistic_score"):
            metrics = _rank_metrics(
                fold_prediction["y_return"].to_numpy(float),
                fold_prediction["y_target"].to_numpy(int),
                fold_prediction[score_name].to_numpy(float),
                float(cfg["top_fraction"]),
            )
            for key, value in metrics.items():
                fold_entry[f"{score_name}_{key}"] = value
        fold_entry["logistic_brier"] = float(
            brier_score_loss(
                fold_prediction["y_target"], fold_prediction["logistic_score"]
            )
        )
        fold_rows.append(fold_entry)
        metadata.append(
            {
                "fold": fold["name"],
                "ridge_top_coefficients": _top_coefficients(ridge),
                "logistic_top_coefficients": _top_coefficients(logistic),
            }
        )
    if predictions:
        oof = pd.concat(predictions, ignore_index=True).sort_values(
            ["signal_date", "instrument_id", "candidate_id"]
        ).reset_index(drop=True)
    else:
        oof = pd.DataFrame(
            columns=(
                "candidate_id",
                "instrument_id",
                "setup_family",
                "direction",
                "signal_date",
                "h20_end_date",
                "fold",
                "y_return",
                "y_target",
                "raw_score",
                "ridge_score",
                "logistic_score",
            )
        )
    return oof, fold_rows, metadata


def _gate_score(
    aggregate: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    score_name: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    valid = [row for row in fold_rows if row.get("status") == "valid"]
    regression = gate["regression"]
    classification = gate["classification"]
    nonnegative_spearman = sum(
        1
        for row in valid
        if (row.get(f"{score_name}_spearman") is not None)
        and row[f"{score_name}_spearman"] >= 0
    )
    positive_top_lift = sum(
        1 for row in valid if (row.get(f"{score_name}_top_minus_mean") or 0) > 0
    )
    auc_ge_half = sum(
        1
        for row in valid
        if (row.get(f"{score_name}_auc") is not None)
        and row[f"{score_name}_auc"] >= 0.5
    )
    top_above_base = sum(
        1
        for row in valid
        if (row.get(f"{score_name}_top_target_rate") is not None)
        and (row.get(f"{score_name}_base_target_rate") is not None)
        and row[f"{score_name}_top_target_rate"] >= row[f"{score_name}_base_target_rate"]
    )
    regression_pass = bool(
        aggregate.get("spearman") is not None
        and aggregate["spearman"] >= float(regression["min_spearman"])
        and aggregate.get("top_return", 0) > 0
        and aggregate.get("top_minus_mean", -1) >= float(regression["min_top_minus_mean"])
        and positive_top_lift >= int(regression["min_positive_top_lift_folds"])
        and nonnegative_spearman >= int(regression["min_nonnegative_spearman_folds"])
    )
    classification_pass = bool(
        aggregate.get("auc") is not None
        and aggregate["auc"] >= float(classification["min_auc"])
        and aggregate.get("top_target_rate_lift") is not None
        and aggregate["top_target_rate_lift"] >= float(classification["min_top_rate_lift"])
        and top_above_base >= int(classification["min_top_above_base_folds"])
        and auc_ge_half >= int(classification["min_auc_ge_half_folds"])
    )
    return {
        "score": score_name,
        "regression_pass": regression_pass,
        "classification_pass": classification_pass,
        "nonnegative_spearman_folds": nonnegative_spearman,
        "positive_top_lift_folds": positive_top_lift,
        "auc_ge_half_folds": auc_ge_half,
        "top_above_base_folds": top_above_base,
    }


def run_baselines(
    config: dict[str, Any],
    *,
    research_root: str | Path,
    resume: str | Path | None = None,
) -> Path:
    baseline_cfg = config["baseline"]
    context = RunContext.create(
        MODULE_NAME,
        MODULE_VERSION,
        config,
        config["execution"]["runs_root"],
        resume_dir=resume,
    )
    assert context.run_dir is not None
    run_dir = context.run_dir
    try:
        store = ResearchStore(research_root)
        research_fingerprint = json.loads(
            (Path(research_root) / "dataset_fingerprint.json").read_text()
        )
        data = _load_safe_data(store, verify=bool(config["execution"].get("verify_upstream_shards", True)))
        families = sorted(data["setup_family"].dropna().unique())
        all_fold_rows: list[dict[str, Any]] = []
        family_rows: list[dict[str, Any]] = []
        gate_report: dict[str, Any] = {
            "baseline_policy_version": BASELINE_POLICY_VERSION,
            "research_dataset_fingerprint_sha256": research_fingerprint[
                "research_dataset_fingerprint_sha256"
            ],
            "protected_data_access": {
                "integration_holdout_rows": 0,
                "final_ticker_holdout_rows": 0,
                "pseudo_lockbox_rows": 0,
            },
            "families": {},
        }
        model_metadata: dict[str, Any] = {}
        prediction_manifest: list[dict[str, Any]] = []
        for index, family in enumerate(families, start=1):
            family_frame = data[data["setup_family"] == family].copy()
            oof, fold_rows, metadata = _evaluate_family(family_frame, baseline_cfg)
            for row in fold_rows:
                all_fold_rows.append({"setup_family": family, **row})
            aggregate_by_score: dict[str, dict[str, Any]] = {}
            for score_name in ("raw_score", "ridge_score", "logistic_score"):
                aggregate_by_score[score_name] = _rank_metrics(
                    oof["y_return"].to_numpy(float),
                    oof["y_target"].to_numpy(int),
                    oof[score_name].to_numpy(float),
                    float(baseline_cfg["top_fraction"]),
                ) if len(oof) else {"n": 0}
            valid_fold_count = sum(1 for row in fold_rows if row.get("status") == "valid")
            score_gates = {
                score: _gate_score(metrics, fold_rows, score, baseline_cfg["gate"])
                for score, metrics in aggregate_by_score.items()
            }
            sufficient_sample = (
                len(oof) >= int(baseline_cfg["minimum_oof_rows"])
                and valid_fold_count >= int(baseline_cfg["minimum_valid_folds"])
            )
            evidence_pass = any(
                item["regression_pass"] or item["classification_pass"]
                for item in score_gates.values()
            )
            decision = (
                "PROCEED_TO_SPECIALIST_RESEARCH"
                if sufficient_sample and evidence_pass
                else "BASELINE_INCONCLUSIVE"
            )
            prediction_path = run_dir / "oof_predictions" / f"{family}.csv.gz"
            prediction_hash = _write_prediction(prediction_path, oof)
            prediction_manifest.append(
                {
                    "setup_family": family,
                    "oof_rows": int(len(oof)),
                    "relative_path": str(prediction_path.relative_to(run_dir)),
                    "sha256": prediction_hash,
                }
            )
            family_entry = {
                "setup_family": family,
                "development_rows": int(len(family_frame)),
                "oof_rows": int(len(oof)),
                "valid_folds": int(valid_fold_count),
                "decision": decision,
                "sufficient_sample": bool(sufficient_sample),
                "evidence_pass": bool(evidence_pass),
                "scores": aggregate_by_score,
                "score_gates": score_gates,
            }
            gate_report["families"][family] = family_entry
            flat = {
                "setup_family": family,
                "development_rows": int(len(family_frame)),
                "oof_rows": int(len(oof)),
                "valid_folds": int(valid_fold_count),
                "decision": decision,
            }
            for score_name, metrics in aggregate_by_score.items():
                for key, value in metrics.items():
                    flat[f"{score_name}_{key}"] = value
            family_rows.append(flat)
            model_metadata[family] = metadata
            context.set_progress(completed_families=index, total_families=len(families))
        fold_frame = pd.DataFrame(all_fold_rows)
        family_frame = pd.DataFrame(family_rows)
        prediction_manifest_frame = pd.DataFrame(prediction_manifest)
        fold_frame.to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n", float_format="%.12g")
        family_frame.to_csv(run_dir / "family_metrics.csv", index=False, lineterminator="\n", float_format="%.12g")
        prediction_manifest_frame.to_csv(
            run_dir / "oof_prediction_manifest.csv", index=False, lineterminator="\n"
        )
        atomic_json(run_dir / "family_gate.json", gate_report)
        atomic_json(run_dir / "model_metadata.json", model_metadata)
        summary = {
            "module_name": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "baseline_policy_version": BASELINE_POLICY_VERSION,
            "family_count": int(len(families)),
            "development_rows": int(len(data)),
            "total_oof_rows": int(family_frame["oof_rows"].sum()),
            "proceed_family_count": int(
                (family_frame["decision"] == "PROCEED_TO_SPECIALIST_RESEARCH").sum()
            ),
            "inconclusive_family_count": int(
                (family_frame["decision"] == "BASELINE_INCONCLUSIVE").sum()
            ),
            "protected_rows_accessed": 0,
            "research_dataset_fingerprint_sha256": research_fingerprint[
                "research_dataset_fingerprint_sha256"
            ],
        }
        atomic_json(run_dir / "baseline_summary.json", summary)
        context.complete(**summary)
        return run_dir
    except Exception as error:
        context.fail(error)
        raise


def compare(first: str | Path, second: str | Path, output: str | Path) -> Path:
    a = Path(first).expanduser().resolve()
    b = Path(second).expanduser().resolve()
    files = [
        "family_metrics.csv",
        "fold_metrics.csv",
        "family_gate.json",
        "oof_prediction_manifest.csv",
        "baseline_summary.json",
    ]
    differences: list[str] = []
    for relative in files:
        if (a / relative).read_bytes() != (b / relative).read_bytes():
            differences.append(relative)
    first_manifest = pd.read_csv(a / "oof_prediction_manifest.csv")
    second_manifest = pd.read_csv(b / "oof_prediction_manifest.csv")
    if not first_manifest.equals(second_manifest):
        differences.append("oof_prediction_manifest_values")
    report = {
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
        "first": str(a),
        "second": str(b),
    }
    output_path = Path(output).expanduser().resolve()
    atomic_json(output_path, report)
    if differences:
        raise ValueError(f"baseline determinism comparison failed: {differences}")
    return output_path


def publish(run: str | Path, target: str | Path) -> Path:
    source = Path(run).expanduser().resolve()
    destination = Path(target).expanduser().resolve()
    manifest = json.loads((source / "run_manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("only completed baseline runs may be published")
    if destination.exists():
        raise FileExistsError(f"baseline publication target exists: {destination}")
    shutil.copytree(source, destination)
    return destination
