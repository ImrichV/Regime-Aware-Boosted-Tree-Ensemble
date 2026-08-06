from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .base import CandidateContext, CandidateGenerator, CandidateSignals
from .schema import (
    CANDIDATE_COLUMNS,
    CANDIDATE_SCHEMA_VERSION,
    canonical_json,
    candidate_id,
    validate_candidate_mapping,
)


@dataclass(frozen=True)
class PrefixCausalityResult:
    checkpoints_tested: int
    rows_compared: int
    passed: bool


def configuration_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _date_int(series: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(series)
    return (
        dates.dt.year.to_numpy(np.int32) * 10000
        + dates.dt.month.to_numpy(np.int32) * 100
        + dates.dt.day.to_numpy(np.int32)
    )


def _entry_alignment(features: pd.DataFrame, dates: np.ndarray, index: int) -> tuple[int | None, str]:
    if index + 1 >= len(features):
        return None, "no_later_bar"
    current_segment = int(features.iloc[index]["segment_id"])
    next_segment = int(features.iloc[index + 1]["segment_id"])
    if current_segment != next_segment:
        return None, "next_bar_new_segment"
    return int(dates[index + 1]), "available"


def materialize_candidates(
    generator: CandidateGenerator,
    context: CandidateContext,
) -> pd.DataFrame:
    generator.validate_context(context)
    signals = generator.generate(context)
    signals.validate(len(context.bars))
    metadata = generator.metadata
    cfg_hash = configuration_hash(generator.config)
    dates = _date_int(context.bars["date"])
    source_rows = context.features["source_row_number"].to_numpy(np.int64)
    segments = context.features["segment_id"].to_numpy(np.int64)
    strengths = (
        np.asarray(signals.raw_setup_strength, dtype=np.float64)
        if signals.raw_setup_strength is not None
        else None
    )
    payload_by_row = signals.payload_by_row or {}
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(signals.mask):
        entry_date, entry_status = _entry_alignment(context.features, dates, int(index))
        row: dict[str, Any] = {
            "candidate_id": candidate_id(
                setup_family=metadata.setup_family,
                setup_version=metadata.setup_version,
                instrument_id=context.instrument_id,
                signal_date=int(dates[index]),
                direction=metadata.direction,
            ),
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "instrument_id": context.instrument_id,
            "ticker": context.ticker,
            "exchange": context.exchange,
            "instrument_class": context.instrument_class,
            "signal_date": int(dates[index]),
            "signal_source_row_number": int(source_rows[index]),
            "signal_segment_id": int(segments[index]),
            "setup_family": metadata.setup_family,
            "setup_version": metadata.setup_version,
            "direction": metadata.direction,
            "decision_time": metadata.decision_time,
            "earliest_entry_rule": metadata.earliest_entry_rule,
            "earliest_entry_date": entry_date,
            "historical_entry_status": entry_status,
            "raw_setup_strength": None if strengths is None else float(strengths[index]),
            "generator_payload_json": canonical_json(payload_by_row.get(int(index))),
            "generator_config_sha256": cfg_hash,
            "feature_schema_version": context.feature_schema_version,
            "feature_dataset_sha256": context.feature_dataset_sha256,
        }
        validate_candidate_mapping(row)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if frame.empty:
        return frame
    if frame["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate_id generated within an instrument")
    frame = frame.sort_values(
        ["signal_date", "setup_family", "setup_version", "direction", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame


def combine_candidate_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    collected = [frame for frame in frames if not frame.empty]
    if not collected:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    result = pd.concat(collected, ignore_index=True)
    if result["candidate_id"].duplicated().any():
        duplicates = result.loc[
            result["candidate_id"].duplicated(keep=False), "candidate_id"
        ].head(10)
        raise ValueError(f"duplicate candidate_id across generators: {duplicates.tolist()}")
    return result.sort_values(
        ["signal_date", "instrument_id", "setup_family", "setup_version", "direction"],
        kind="mergesort",
    ).reset_index(drop=True)


def audit_prefix_causality(
    generator: CandidateGenerator,
    context: CandidateContext,
    checkpoints: Iterable[int],
    *,
    atol: float = 1e-12,
) -> PrefixCausalityResult:
    generator.validate_context(context)
    full = generator.generate(context)
    full.validate(len(context.bars))
    tested = 0
    rows_compared = 0
    for length in sorted(set(int(x) for x in checkpoints)):
        if length <= 0 or length > len(context.bars):
            continue
        prefix_context = context.prefix(length)
        prefix = generator.generate(prefix_context)
        prefix.validate(length)
        if not np.array_equal(prefix.mask, full.mask[:length]):
            raise AssertionError(
                f"prefix causality failed for {generator.metadata.generator_key} at length {length}: mask changed"
            )
        if full.raw_setup_strength is None:
            if prefix.raw_setup_strength is not None:
                raise AssertionError("prefix unexpectedly returned strengths")
        else:
            if prefix.raw_setup_strength is None:
                raise AssertionError("prefix omitted strengths")
            selected = full.mask[:length]
            if not np.allclose(
                np.asarray(prefix.raw_setup_strength)[selected],
                np.asarray(full.raw_setup_strength)[:length][selected],
                rtol=0.0,
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(
                    f"prefix causality failed for {generator.metadata.generator_key} at length {length}: strength changed"
                )
        full_payload = full.payload_by_row or {}
        prefix_payload = prefix.payload_by_row or {}
        expected_payload = {k: v for k, v in full_payload.items() if k < length}
        if canonical_json(prefix_payload) != canonical_json(expected_payload):
            raise AssertionError(
                f"prefix causality failed for {generator.metadata.generator_key} at length {length}: payload changed"
            )
        tested += 1
        rows_compared += length
    return PrefixCausalityResult(tested, rows_compared, True)
