from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    BARRIER_SPECS,
    HORIZONS,
    OUTCOME_COLUMNS,
    OUTCOME_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
    STOP_ATR_MULTIPLE,
    TARGET_R_MULTIPLES,
    temporal_partition,
    ticker_split,
    validate_outcome_mapping,
)


def _date_int(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="raise").to_numpy(np.int64)
        if len(numeric) == 0 or np.all((numeric >= 19000101) & (numeric <= 29991231)):
            return numeric.astype(np.int32)
    dates = pd.to_datetime(series, errors="raise")
    return (
        dates.dt.year.to_numpy(np.int32) * 10000
        + dates.dt.month.to_numpy(np.int32) * 100
        + dates.dt.day.to_numpy(np.int32)
    )


def _nullable_int(value: int | np.integer | None) -> int | None:
    return None if value is None else int(value)


def _first_touch_date(offset: int | None, dates: np.ndarray) -> int | None:
    if offset is None:
        return None
    return int(dates[offset - 1])


def _barrier_result(
    *,
    stop_touch: int | None,
    target_touch: int | None,
    maximum_holding_bars: int,
    bars_available: int,
    atr_available: bool,
    entry_available: bool,
) -> tuple[str, int | None]:
    if not entry_available:
        return "no_entry", None
    if not atr_available:
        return "unavailable_atr", None
    stop = stop_touch if stop_touch is not None and stop_touch <= maximum_holding_bars else None
    target = target_touch if target_touch is not None and target_touch <= maximum_holding_bars else None
    if stop is not None and target is not None:
        if stop <= target:  # pessimistic stop-first when both occur on one daily bar
            return "stop", int(stop)
        return "target", int(target)
    if stop is not None:
        return "stop", int(stop)
    if target is not None:
        return "target", int(target)
    if bars_available >= maximum_holding_bars:
        return "timeout", int(maximum_holding_bars)
    return "incomplete", None


def _empty_metrics(row: dict[str, Any], entry_status: str) -> dict[str, Any]:
    row.update(
        {
            "entry_status": entry_status,
            "entry_date": None,
            "entry_source_row_number": None,
            "entry_segment_id": None,
            "signal_close": None,
            "signal_atr_14_abs": None,
            "entry_open": None,
            "entry_gap_return_raw": None,
            "same_segment_bars_available": 0,
            "evaluation_termination": "no_entry",
        }
    )
    for horizon in HORIZONS:
        row[f"h{horizon}_end_date"] = None
        row[f"h{horizon}_directional_close_return"] = None
        row[f"h{horizon}_directional_mfe"] = None
        row[f"h{horizon}_directional_mae"] = None
    row["stop_1p5atr_first_touch_bar"] = None
    row["stop_1p5atr_first_touch_date"] = None
    for target in TARGET_R_MULTIPLES:
        row[f"target_{target}r_first_touch_bar"] = None
        row[f"target_{target}r_first_touch_date"] = None
    for target, holding in BARRIER_SPECS:
        row[f"barrier_{target}r_{holding}_result"] = "no_entry"
        row[f"barrier_{target}r_{holding}_bars_to_event"] = None
    return row


def compute_outcomes(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    candidate_catalog_sha256: str,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    required_bars = {"date", "open", "high", "low", "close", "source_row_number"}
    missing_bars = required_bars.difference(bars.columns)
    if missing_bars:
        raise ValueError(f"bars missing columns: {sorted(missing_bars)}")
    required_features = {"date", "source_row_number", "segment_id", "atr_14_pct"}
    missing_features = required_features.difference(features.columns)
    if missing_features:
        raise ValueError(f"features missing columns: {sorted(missing_features)}")
    if len(bars) != len(features):
        raise ValueError("bar/feature row-count mismatch")

    bar_dates = _date_int(bars["date"])
    feature_dates = _date_int(features["date"])
    if not np.array_equal(bar_dates, feature_dates):
        raise ValueError("bar/feature date mismatch")
    bar_source_rows = bars["source_row_number"].to_numpy(np.int64)
    feature_source_rows = features["source_row_number"].to_numpy(np.int64)
    if not np.array_equal(bar_source_rows, feature_source_rows):
        raise ValueError("bar/feature source-row mismatch")

    opens = bars["open"].to_numpy(np.float64)
    highs = bars["high"].to_numpy(np.float64)
    lows = bars["low"].to_numpy(np.float64)
    closes = bars["close"].to_numpy(np.float64)
    segments = features["segment_id"].to_numpy(np.int64)
    atr_pct = features["atr_14_pct"].to_numpy(np.float64)
    date_to_index = {int(value): index for index, value in enumerate(bar_dates)}

    records: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        signal_date = int(candidate["signal_date"])
        instrument_id = str(candidate["instrument_id"])
        base: dict[str, Any] = {
            "candidate_id": str(candidate["candidate_id"]),
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "candidate_schema_version": str(candidate["candidate_schema_version"]),
            "candidate_catalog_sha256": candidate_catalog_sha256,
            "instrument_id": instrument_id,
            "ticker": str(candidate["ticker"]),
            "exchange": str(candidate["exchange"]),
            "setup_family": str(candidate["setup_family"]),
            "setup_version": str(candidate["setup_version"]),
            "direction": str(candidate["direction"]),
            "signal_date": signal_date,
            "ticker_split": ticker_split(instrument_id),
            "temporal_partition": temporal_partition(signal_date),
        }
        entry_status = str(candidate["historical_entry_status"])
        if entry_status != "available":
            records.append(_empty_metrics(base, entry_status))
            continue

        entry_date = int(candidate["earliest_entry_date"])
        signal_index = date_to_index.get(signal_date)
        entry_index = date_to_index.get(entry_date)
        if signal_index is None:
            raise ValueError(f"candidate signal date missing from bars: {candidate['candidate_id']}")
        if entry_index is None:
            raise ValueError(f"candidate entry date missing from bars: {candidate['candidate_id']}")
        if entry_index != signal_index + 1:
            raise ValueError(f"candidate entry is not next valid bar: {candidate['candidate_id']}")
        if int(segments[signal_index]) != int(candidate["signal_segment_id"]):
            raise ValueError(f"candidate signal segment mismatch: {candidate['candidate_id']}")
        if int(segments[entry_index]) != int(segments[signal_index]):
            raise ValueError(f"available candidate crosses a segment boundary: {candidate['candidate_id']}")
        if int(bar_source_rows[signal_index]) != int(candidate["signal_source_row_number"]):
            raise ValueError(f"candidate source-row mismatch: {candidate['candidate_id']}")

        entry_segment = int(segments[entry_index])
        later_reset = np.flatnonzero(segments[entry_index:] != entry_segment)
        segment_end = entry_index + int(later_reset[0]) if len(later_reset) else len(bars)
        bars_available = int(segment_end - entry_index)
        maximum = min(bars_available, max(HORIZONS))
        evaluation_end = entry_index + maximum
        if bars_available >= max(HORIZONS):
            termination = "complete_60"
        elif segment_end < len(bars):
            termination = "segment_boundary"
        else:
            termination = "end_of_history"

        entry_open = float(opens[entry_index])
        signal_close = float(closes[signal_index])
        signal_atr_abs = float(signal_close * atr_pct[signal_index]) if np.isfinite(atr_pct[signal_index]) else None
        if signal_atr_abs is not None and signal_atr_abs <= 0:
            signal_atr_abs = None
        direction_sign = 1.0 if candidate["direction"] == "LONG" else -1.0

        eval_high = highs[entry_index:evaluation_end]
        eval_low = lows[entry_index:evaluation_end]
        eval_close = closes[entry_index:evaluation_end]
        eval_dates = bar_dates[entry_index:evaluation_end]
        if direction_sign > 0:
            favourable = eval_high / entry_open - 1.0
            adverse = eval_low / entry_open - 1.0
        else:
            favourable = 1.0 - eval_low / entry_open
            adverse = 1.0 - eval_high / entry_open
        cumulative_mfe = np.maximum.accumulate(favourable) if maximum else np.array([], dtype=float)
        cumulative_mae = np.minimum.accumulate(adverse) if maximum else np.array([], dtype=float)

        base.update(
            {
                "entry_status": "available",
                "entry_date": entry_date,
                "entry_source_row_number": int(bar_source_rows[entry_index]),
                "entry_segment_id": entry_segment,
                "signal_close": signal_close,
                "signal_atr_14_abs": signal_atr_abs,
                "entry_open": entry_open,
                "entry_gap_return_raw": entry_open / signal_close - 1.0,
                "same_segment_bars_available": bars_available,
                "evaluation_termination": termination,
            }
        )
        for horizon in HORIZONS:
            if bars_available >= horizon:
                index = horizon - 1
                base[f"h{horizon}_end_date"] = int(eval_dates[index])
                base[f"h{horizon}_directional_close_return"] = float(
                    direction_sign * (eval_close[index] / entry_open - 1.0)
                )
                base[f"h{horizon}_directional_mfe"] = float(cumulative_mfe[index])
                base[f"h{horizon}_directional_mae"] = float(cumulative_mae[index])
            else:
                base[f"h{horizon}_end_date"] = None
                base[f"h{horizon}_directional_close_return"] = None
                base[f"h{horizon}_directional_mfe"] = None
                base[f"h{horizon}_directional_mae"] = None

        stop_touch: int | None = None
        target_touches: dict[int, int | None] = {target: None for target in TARGET_R_MULTIPLES}
        atr_available = signal_atr_abs is not None
        if atr_available and maximum:
            stop_distance = STOP_ATR_MULTIPLE * float(signal_atr_abs)
            if direction_sign > 0:
                stop_mask = eval_low <= entry_open - stop_distance
            else:
                stop_mask = eval_high >= entry_open + stop_distance
            stop_indices = np.flatnonzero(stop_mask)
            if len(stop_indices):
                stop_touch = int(stop_indices[0] + 1)
            for target in TARGET_R_MULTIPLES:
                distance = target * stop_distance
                if direction_sign > 0:
                    target_mask = eval_high >= entry_open + distance
                else:
                    target_mask = eval_low <= entry_open - distance
                target_indices = np.flatnonzero(target_mask)
                if len(target_indices):
                    target_touches[target] = int(target_indices[0] + 1)

        base["stop_1p5atr_first_touch_bar"] = stop_touch
        base["stop_1p5atr_first_touch_date"] = _first_touch_date(stop_touch, eval_dates)
        for target in TARGET_R_MULTIPLES:
            touch = target_touches[target]
            base[f"target_{target}r_first_touch_bar"] = touch
            base[f"target_{target}r_first_touch_date"] = _first_touch_date(touch, eval_dates)
        for target, holding in BARRIER_SPECS:
            result, bars_to_event = _barrier_result(
                stop_touch=stop_touch,
                target_touch=target_touches[target],
                maximum_holding_bars=holding,
                bars_available=bars_available,
                atr_available=atr_available,
                entry_available=True,
            )
            base[f"barrier_{target}r_{holding}_result"] = result
            base[f"barrier_{target}r_{holding}_bars_to_event"] = bars_to_event
        validate_outcome_mapping(base)
        records.append(base)

    result = pd.DataFrame(records, columns=OUTCOME_COLUMNS)
    if result["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate_id in outcome output")
    return result
