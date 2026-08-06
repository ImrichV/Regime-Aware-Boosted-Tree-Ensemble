from __future__ import annotations

import hashlib
import json
from typing import Any

from ..features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

MODULE_NAME = "module_06_specialist_baseline_gate"
MODULE_VERSION = "v1.0.0"
RESEARCH_SCHEMA_VERSION = "research_schema_v1.0.0"
BASELINE_POLICY_VERSION = "baseline_policy_v1.0.0"

IDENTITY_COLUMNS = (
    "candidate_id",
    "instrument_id",
    "ticker",
    "exchange",
    "setup_family",
    "setup_version",
    "direction",
    "signal_date",
    "signal_source_row_number",
    "signal_segment_id",
    "raw_setup_strength",
    "generator_payload_json",
    "ticker_split",
    "temporal_partition",
    "entry_status",
)

HORIZONS = (1, 3, 5, 10, 20, 40, 60)
OUTCOME_LABEL_COLUMNS = tuple(
    value
    for horizon in HORIZONS
    for value in (
        f"h{horizon}_end_date",
        f"h{horizon}_directional_close_return",
        f"h{horizon}_directional_mfe",
        f"h{horizon}_directional_mae",
    )
) + (
    "barrier_1r_20_result",
    "barrier_1r_20_bars_to_event",
    "barrier_2r_20_result",
    "barrier_2r_20_bars_to_event",
    "barrier_3r_20_result",
    "barrier_3r_20_bars_to_event",
    "barrier_5r_40_result",
    "barrier_5r_40_bars_to_event",
)

PREDICTOR_COLUMNS = ("raw_setup_strength", *FEATURE_NAMES)
RESEARCH_COLUMNS = (*IDENTITY_COLUMNS, *FEATURE_NAMES, *OUTCOME_LABEL_COLUMNS)

FOLD_DEFINITIONS = (
    {"name": "2013_2014", "test_start": 20130101, "test_end": 20141231},
    {"name": "2015_2016", "test_start": 20150101, "test_end": 20161231},
    {"name": "2017_2018", "test_start": 20170101, "test_end": 20181231},
    {"name": "2019_2020", "test_start": 20190101, "test_end": 20201231},
    {"name": "2021_2022", "test_start": 20210101, "test_end": 20221231},
    {"name": "2023", "test_start": 20230101, "test_end": 20231231},
)


def schema_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "baseline_policy_version": BASELINE_POLICY_VERSION,
        "row_key": ["candidate_id"],
        "columns": list(RESEARCH_COLUMNS),
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selection_safe_universe": {
            "ticker_split": "development",
            "excluded_temporal_partitions": ["pseudo_lockbox_2024_2026"],
        },
        "baseline_targets": {
            "regression": "h20_directional_close_return",
            "classification": "barrier_2r_20_result == target; stop/timeout == 0",
        },
        "walk_forward_folds": list(FOLD_DEFINITIONS),
        "prohibited": [
            "integration_holdout outcomes",
            "final_ticker_holdout outcomes",
            "2024-2026 pseudo-lockbox outcomes",
            "same-close execution",
            "hyperparameter search",
            "portfolio decision",
        ],
    }
    payload["schema_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload
