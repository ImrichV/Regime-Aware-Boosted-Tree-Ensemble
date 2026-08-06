from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

MODULE_NAME = "module_05_candidate_outcomes"
MODULE_VERSION = "v1.0.0"
OUTCOME_SCHEMA_VERSION = "outcome_schema_v1.0.0"
SPLIT_POLICY_VERSION = "split_policy_v1.0.0"

HORIZONS = (1, 3, 5, 10, 20, 40, 60)
TARGET_R_MULTIPLES = (1, 2, 3, 5)
STOP_ATR_MULTIPLE = 1.5
BARRIER_SPECS = (
    (1, 20),
    (2, 20),
    (3, 20),
    (5, 40),
)

CORE_COLUMNS = (
    "candidate_id",
    "outcome_schema_version",
    "split_policy_version",
    "candidate_schema_version",
    "candidate_catalog_sha256",
    "instrument_id",
    "ticker",
    "exchange",
    "setup_family",
    "setup_version",
    "direction",
    "signal_date",
    "ticker_split",
    "temporal_partition",
    "entry_status",
    "entry_date",
    "entry_source_row_number",
    "entry_segment_id",
    "signal_close",
    "signal_atr_14_abs",
    "entry_open",
    "entry_gap_return_raw",
    "same_segment_bars_available",
    "evaluation_termination",
)

HORIZON_COLUMNS = tuple(
    column
    for horizon in HORIZONS
    for column in (
        f"h{horizon}_end_date",
        f"h{horizon}_directional_close_return",
        f"h{horizon}_directional_mfe",
        f"h{horizon}_directional_mae",
    )
)

FIRST_TOUCH_COLUMNS = (
    "stop_1p5atr_first_touch_bar",
    "stop_1p5atr_first_touch_date",
    *tuple(
        column
        for target in TARGET_R_MULTIPLES
        for column in (
            f"target_{target}r_first_touch_bar",
            f"target_{target}r_first_touch_date",
        )
    ),
)

BARRIER_COLUMNS = tuple(
    column
    for target, holding in BARRIER_SPECS
    for column in (
        f"barrier_{target}r_{holding}_result",
        f"barrier_{target}r_{holding}_bars_to_event",
    )
)

OUTCOME_COLUMNS = CORE_COLUMNS + HORIZON_COLUMNS + FIRST_TOUCH_COLUMNS + BARRIER_COLUMNS

STRING_COLUMNS = (
    "candidate_id",
    "outcome_schema_version",
    "split_policy_version",
    "candidate_schema_version",
    "candidate_catalog_sha256",
    "instrument_id",
    "ticker",
    "exchange",
    "setup_family",
    "setup_version",
    "direction",
    "ticker_split",
    "temporal_partition",
    "entry_status",
    "evaluation_termination",
    *tuple(f"barrier_{target}r_{holding}_result" for target, holding in BARRIER_SPECS),
)

INT_COLUMNS = ("signal_date", "same_segment_bars_available")
NULLABLE_INT_COLUMNS = (
    "entry_date",
    "entry_source_row_number",
    "entry_segment_id",
    *tuple(f"h{horizon}_end_date" for horizon in HORIZONS),
    "stop_1p5atr_first_touch_bar",
    "stop_1p5atr_first_touch_date",
    *tuple(
        column
        for target in TARGET_R_MULTIPLES
        for column in (
            f"target_{target}r_first_touch_bar",
            f"target_{target}r_first_touch_date",
        )
    ),
    *tuple(f"barrier_{target}r_{holding}_bars_to_event" for target, holding in BARRIER_SPECS),
)

FLOAT_COLUMNS = (
    "signal_close",
    "signal_atr_14_abs",
    "entry_open",
    "entry_gap_return_raw",
    *tuple(
        column
        for horizon in HORIZONS
        for column in (
            f"h{horizon}_directional_close_return",
            f"h{horizon}_directional_mfe",
            f"h{horizon}_directional_mae",
        )
    ),
)

ENTRY_STATUS_VALUES = {"available", "no_later_bar", "next_bar_new_segment"}
EVALUATION_TERMINATION_VALUES = {
    "no_entry",
    "complete_60",
    "end_of_history",
    "segment_boundary",
}
BARRIER_RESULT_VALUES = {"target", "stop", "timeout", "incomplete", "unavailable_atr", "no_entry"}
TICKER_SPLIT_VALUES = {"development", "integration_holdout", "final_ticker_holdout"}
TEMPORAL_PARTITION_VALUES = {
    "development_pre2013",
    "walkforward_2013_2014",
    "walkforward_2015_2016",
    "walkforward_2017_2018",
    "walkforward_2019_2020",
    "walkforward_2021_2022",
    "walkforward_2023",
    "pseudo_lockbox_2024_2026",
    "future_out_of_policy",
}


def ticker_split(instrument_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_POLICY_VERSION}|{instrument_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 7_000:
        return "development"
    if bucket < 8_500:
        return "integration_holdout"
    return "final_ticker_holdout"


def temporal_partition(signal_date: int) -> str:
    value = int(signal_date)
    if value <= 20121231:
        return "development_pre2013"
    if value <= 20141231:
        return "walkforward_2013_2014"
    if value <= 20161231:
        return "walkforward_2015_2016"
    if value <= 20181231:
        return "walkforward_2017_2018"
    if value <= 20201231:
        return "walkforward_2019_2020"
    if value <= 20221231:
        return "walkforward_2021_2022"
    if value <= 20231231:
        return "walkforward_2023"
    if value <= 20261231:
        return "pseudo_lockbox_2024_2026"
    return "future_out_of_policy"


def schema_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "row_key": ["candidate_id"],
        "columns": list(OUTCOME_COLUMNS),
        "timing": {
            "entry": "next valid session open after signal close",
            "horizon_convention": "h=1 is entry-session close; the entry session is included in MFE and MAE",
            "cross_segment_evaluation": "forbidden",
            "daily_bar_ambiguity": "stop wins when stop and target are first touched on the same daily bar",
        },
        "directional_metrics": {
            "long": "positive means price moved upward in the trade direction",
            "short": "positive means price moved downward in the trade direction",
            "mfe": "nonnegative maximum favourable return from entry open",
            "mae": "nonpositive maximum adverse return from entry open",
        },
        "horizons": list(HORIZONS),
        "barriers": {
            "risk_unit": "1.5 times signal-close ATR(14)",
            "stop_atr_multiple": STOP_ATR_MULTIPLE,
            "target_R_multiples": list(TARGET_R_MULTIPLES),
            "derived_barrier_specs": [
                {"target_R": target, "maximum_holding_bars": holding}
                for target, holding in BARRIER_SPECS
            ],
            "fill_model": "touch labels only; no execution price or P&L is claimed",
        },
        "ticker_split": {
            "hash": "SHA-256(split_policy_version|instrument_id), first 64 bits modulo 10000",
            "development": "0-6999",
            "integration_holdout": "7000-8499",
            "final_ticker_holdout": "8500-9999",
        },
        "temporal_partitions": sorted(TEMPORAL_PARTITION_VALUES),
        "forbidden_uses": [
            "candidate generation",
            "same-row feature construction",
            "protected-period repair of labels",
            "portfolio P&L claims",
        ],
        "limitations": [
            "Stooq is survivor dominated.",
            "Daily OHLC bars cannot reveal intraday path ordering.",
            "Barrier labels use pessimistic stop-first resolution on ambiguous bars.",
            "Short-side outcomes do not include borrow availability, locate fees, recalls, or borrow cost.",
        ],
    }
    payload["schema_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def validate_outcome_mapping(row: Mapping[str, Any]) -> None:
    missing = set(OUTCOME_COLUMNS).difference(row)
    extra = set(row).difference(OUTCOME_COLUMNS)
    if missing:
        raise ValueError(f"outcome row missing columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"outcome row has unknown columns: {sorted(extra)}")
    if row["outcome_schema_version"] != OUTCOME_SCHEMA_VERSION:
        raise ValueError("outcome schema version mismatch")
    if row["split_policy_version"] != SPLIT_POLICY_VERSION:
        raise ValueError("split policy version mismatch")
    if row["direction"] not in {"LONG", "SHORT"}:
        raise ValueError("invalid direction")
    if row["ticker_split"] not in TICKER_SPLIT_VALUES:
        raise ValueError("invalid ticker split")
    if row["temporal_partition"] not in TEMPORAL_PARTITION_VALUES:
        raise ValueError("invalid temporal partition")
    if row["entry_status"] not in ENTRY_STATUS_VALUES:
        raise ValueError("invalid entry status")
    if row["evaluation_termination"] not in EVALUATION_TERMINATION_VALUES:
        raise ValueError("invalid evaluation termination")
    if row["entry_status"] == "available":
        for column in ("entry_date", "entry_source_row_number", "entry_segment_id", "entry_open", "signal_close"):
            if _is_null(row[column]):
                raise ValueError(f"available outcome requires {column}")
        if int(row["entry_date"]) <= int(row["signal_date"]):
            raise ValueError("entry date must follow signal date")
    else:
        if row["evaluation_termination"] != "no_entry":
            raise ValueError("unavailable entry requires no_entry termination")
        if int(row["same_segment_bars_available"]) != 0:
            raise ValueError("unavailable entry cannot have future bars")
    for column in FLOAT_COLUMNS:
        value = row[column]
        if not _is_null(value) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite value in {column}")
    for target, holding in BARRIER_SPECS:
        result = row[f"barrier_{target}r_{holding}_result"]
        if result not in BARRIER_RESULT_VALUES:
            raise ValueError(f"invalid barrier result: {result}")
        bars = row[f"barrier_{target}r_{holding}_bars_to_event"]
        if result in {"target", "stop", "timeout"} and _is_null(bars):
            raise ValueError("completed barrier result requires bars_to_event")
        if result in {"incomplete", "unavailable_atr", "no_entry"} and not _is_null(bars):
            raise ValueError("incomplete barrier result requires null bars_to_event")
