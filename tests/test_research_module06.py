from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsr.features.schema import FEATURE_NAMES
from tsr.research.baseline import _evaluate_family, _gate_score, _rank_metrics
from tsr.research.schema import RESEARCH_COLUMNS
from tsr.research.storage import read_research_shard, write_research_shard


def _research_frame(rows: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(260805)
    dates = np.concatenate(
        [
            np.linspace(20070102, 20121220, 400, dtype=int),
            np.linspace(20130102, 20141201, 200, dtype=int),
            np.linspace(20150102, 20161201, 200, dtype=int),
            np.linspace(20170102, 20181201, 200, dtype=int),
            np.linspace(20190102, 20201201, 200, dtype=int),
            np.linspace(20210102, 20221201, 200, dtype=int),
            np.linspace(20230102, 20231101, 200, dtype=int),
        ]
    )[:rows]
    signal = rng.normal(size=rows)
    target_prob = 1 / (1 + np.exp(-0.5 * signal))
    y_target = rng.binomial(1, target_prob)
    y_return = 0.02 * signal + rng.normal(scale=0.06, size=rows)
    frame = pd.DataFrame(index=np.arange(rows), columns=RESEARCH_COLUMNS)
    frame["candidate_id"] = [f"cand_{i:06d}" for i in range(rows)]
    frame["instrument_id"] = [f"US:NYSE:STOCKS:T{i%100:03d}" for i in range(rows)]
    frame["ticker"] = [f"T{i%100:03d}" for i in range(rows)]
    frame["exchange"] = "NYSE"
    frame["setup_family"] = "unit_family"
    frame["setup_version"] = "v1.0.0"
    frame["direction"] = "LONG"
    frame["signal_date"] = dates
    frame["signal_source_row_number"] = np.arange(rows) + 2
    frame["signal_segment_id"] = 0
    frame["raw_setup_strength"] = signal
    frame["generator_payload_json"] = "{}"
    frame["ticker_split"] = "development"
    frame["temporal_partition"] = "development_pre2013"
    frame["entry_status"] = "available"
    for name in FEATURE_NAMES:
        frame[name] = rng.normal(size=rows)
    frame["return_20"] = signal + rng.normal(scale=0.2, size=rows)
    frame["h20_end_date"] = frame["signal_date"] + 60
    frame["h20_directional_close_return"] = y_return
    frame["barrier_2r_20_result"] = np.where(y_target == 1, "target", "stop")
    for column in RESEARCH_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.reindex(columns=RESEARCH_COLUMNS)


def test_research_storage_is_deterministic(tmp_path: Path) -> None:
    frame = _research_frame(20)
    first = tmp_path / "a.csv.gz"
    second = tmp_path / "b.csv.gz"
    hash_a = write_research_shard(first, frame)
    hash_b = write_research_shard(second, frame)
    assert hash_a == hash_b
    restored = read_research_shard(first)
    assert list(restored.columns) == list(RESEARCH_COLUMNS)
    assert len(restored) == 20


def test_rank_metrics_detects_ordering() -> None:
    y_return = np.linspace(-0.1, 0.1, 100)
    y_target = (y_return > 0.02).astype(int)
    score = y_return.copy()
    metrics = _rank_metrics(y_return, y_target, score, 0.1)
    assert metrics["spearman"] > 0.99
    assert metrics["top_minus_mean"] > 0
    assert metrics["auc"] > 0.99


def test_fixed_walkforward_baselines_generate_oof_predictions() -> None:
    frame = _research_frame()
    cfg = {
        "minimum_train_rows": 300,
        "minimum_test_rows": 100,
        "ridge_alpha": 10.0,
        "logistic_c": 0.1,
        "logistic_max_iter": 1000,
        "random_seed": 260805,
        "top_fraction": 0.1,
    }
    oof, folds, metadata = _evaluate_family(frame, cfg)
    assert len(oof) >= 1000
    assert sum(row["status"] == "valid" for row in folds) >= 5
    assert oof["candidate_id"].is_unique
    assert oof["ridge_score"].notna().all()
    assert oof["logistic_score"].between(0, 1).all()
    assert metadata


def test_gate_score_uses_frozen_thresholds() -> None:
    aggregate = {
        "spearman": 0.03,
        "top_return": 0.04,
        "top_minus_mean": 0.01,
        "auc": 0.53,
        "top_target_rate_lift": 1.12,
    }
    folds = []
    for _ in range(6):
        folds.append(
            {
                "status": "valid",
                "raw_score_spearman": 0.02,
                "raw_score_top_minus_mean": 0.01,
                "raw_score_auc": 0.52,
                "raw_score_top_target_rate": 0.3,
                "raw_score_base_target_rate": 0.25,
            }
        )
    gate = {
        "regression": {
            "min_spearman": 0.015,
            "min_top_minus_mean": 0.005,
            "min_positive_top_lift_folds": 4,
            "min_nonnegative_spearman_folds": 3,
        },
        "classification": {
            "min_auc": 0.515,
            "min_top_rate_lift": 1.08,
            "min_top_above_base_folds": 4,
            "min_auc_ge_half_folds": 3,
        },
    }
    result = _gate_score(aggregate, folds, "raw_score", gate)
    assert result["regression_pass"]
    assert result["classification_pass"]
