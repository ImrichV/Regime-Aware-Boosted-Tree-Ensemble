from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsr.candidates.schema import CANDIDATE_COLUMNS, CANDIDATE_SCHEMA_VERSION, candidate_id, schema_payload as candidate_schema_payload
from tsr.candidates.storage import shard_relative_path as candidate_shard_path
from tsr.candidates.storage import write_candidate_shard
from tsr.features.engine import compute_features
from tsr.features.schema import payload as feature_schema_payload
from tsr.features.storage import shard_rel as feature_shard_path
from tsr.features.storage import write_npz
from tsr.outcomes.build import build, publish, verify
from tsr.outcomes.engine import compute_outcomes
from tsr.outcomes.schema import OUTCOME_COLUMNS, ticker_split
from tsr.outcomes.storage import read_outcome_shard, write_outcome_shard


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": ["US:NYSE:STOCKS:ABC"] * 5,
            "ticker": ["ABC"] * 5,
            "exchange": ["NYSE"] * 5,
            "instrument_class": ["stocks"] * 5,
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]),
            "open": [100.0, 100.0, 102.0, 103.0, 104.0],
            "high": [101.0, 104.0, 105.0, 106.0, 107.0],
            "low": [99.0, 98.0, 101.0, 102.0, 103.0],
            "close": [100.0, 102.0, 104.0, 105.0, 106.0],
            "volume": [1000.0] * 5,
            "source_member": ["abc"] * 5,
            "source_row_number": [2, 3, 4, 5, 6],
        }
    )


def _features(bars: pd.DataFrame, segments: list[int] | None = None) -> pd.DataFrame:
    result = compute_features(bars).arrays
    frame = pd.DataFrame(result)
    frame["atr_14_pct"] = np.array([0.01] * len(frame), dtype=np.float32)
    if segments is not None:
        frame["segment_id"] = np.array(segments, dtype=np.int32)
    return frame


def _candidate(direction: str = "LONG", status: str = "available") -> pd.DataFrame:
    instrument_id = "US:NYSE:STOCKS:ABC"
    family = "unit_test_setup"
    signal_date = 20200102
    row = {
        "candidate_id": candidate_id(
            setup_family=family,
            setup_version="v1.0.0",
            instrument_id=instrument_id,
            signal_date=signal_date,
            direction=direction,
        ),
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "instrument_id": instrument_id,
        "ticker": "ABC",
        "exchange": "NYSE",
        "instrument_class": "stocks",
        "signal_date": signal_date,
        "signal_source_row_number": 2,
        "signal_segment_id": 0,
        "setup_family": family,
        "setup_version": "v1.0.0",
        "direction": direction,
        "decision_time": "signal_close",
        "earliest_entry_rule": "next_valid_session_open",
        "earliest_entry_date": 20200103 if status == "available" else None,
        "historical_entry_status": status,
        "raw_setup_strength": 0.5,
        "generator_payload_json": "{}",
        "generator_config_sha256": "a" * 64,
        "feature_schema_version": "feature_schema_v1.0.0",
        "feature_dataset_sha256": "f" * 64,
    }
    return pd.DataFrame([row], columns=CANDIDATE_COLUMNS)


def test_long_outcome_uses_next_open_and_pessimistic_ambiguity() -> None:
    result = compute_outcomes(_candidate("LONG"), _bars(), _features(_bars()), candidate_catalog_sha256="c" * 64)
    row = result.iloc[0]
    assert row.entry_date == 20200103
    assert row.entry_open == 100.0
    assert row.h1_directional_close_return == pytest.approx(0.02)
    assert row.h1_directional_mfe == pytest.approx(0.04)
    assert row.h1_directional_mae == pytest.approx(-0.02)
    assert row.stop_1p5atr_first_touch_bar == 1
    assert row.target_1r_first_touch_bar == 1
    assert row.barrier_1r_20_result == "stop"
    assert row.barrier_1r_20_bars_to_event == 1


def test_short_outcome_is_direction_adjusted() -> None:
    bars = _bars()
    bars.loc[1, ["open", "high", "low", "close"]] = [100.0, 102.0, 96.0, 97.0]
    result = compute_outcomes(_candidate("SHORT"), bars, _features(bars), candidate_catalog_sha256="c" * 64)
    row = result.iloc[0]
    assert row.h1_directional_close_return == pytest.approx(0.03)
    assert row.h1_directional_mfe == pytest.approx(0.04)
    assert row.h1_directional_mae == pytest.approx(-0.02)
    assert row.barrier_1r_20_result == "stop"


def test_horizons_do_not_cross_segment_boundary() -> None:
    features = _features(_bars(), segments=[0, 0, 0, 1, 1])
    result = compute_outcomes(_candidate(), _bars(), features, candidate_catalog_sha256="c" * 64)
    row = result.iloc[0]
    assert row.same_segment_bars_available == 2
    assert row.evaluation_termination == "segment_boundary"
    assert row.h1_end_date == 20200103
    assert row.h3_end_date is None or pd.isna(row.h3_end_date)
    assert row.barrier_3r_20_result in {"stop", "target", "incomplete"}


def test_no_entry_candidate_has_no_future_metrics() -> None:
    result = compute_outcomes(
        _candidate(status="no_later_bar"),
        _bars(),
        _features(_bars()),
        candidate_catalog_sha256="c" * 64,
    )
    row = result.iloc[0]
    assert row.entry_status == "no_later_bar"
    assert row.evaluation_termination == "no_entry"
    assert row.same_segment_bars_available == 0
    assert row.barrier_1r_20_result == "no_entry"
    assert pd.isna(row.h1_directional_close_return)


def test_ticker_split_is_deterministic() -> None:
    value = ticker_split("US:NYSE:STOCKS:ABC")
    assert value == ticker_split("US:NYSE:STOCKS:ABC")
    assert value in {"development", "integration_holdout", "final_ticker_holdout"}


def test_outcome_storage_roundtrip(tmp_path: Path) -> None:
    frame = compute_outcomes(_candidate(), _bars(), _features(_bars()), candidate_catalog_sha256="c" * 64)
    path = tmp_path / "outcomes.csv"
    write_outcome_shard(path, frame)
    restored = read_outcome_shard(path)
    assert tuple(restored.columns) == OUTCOME_COLUMNS
    assert restored.loc[0, "candidate_id"] == frame.loc[0, "candidate_id"]
    assert float(restored.loc[0, "h1_directional_close_return"]) == pytest.approx(0.02)


def _make_module05_upstream(tmp_path: Path) -> dict[str, str]:
    archive_path = tmp_path / "stooq.zip"
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    header = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    lines = []
    dates = pd.bdate_range("2019-01-02", periods=80)
    close = 100.0
    for index, date in enumerate(dates):
        close += 0.2
        lines.append(
            f"ABC.US,D,{date:%Y%m%d},000000,{close-0.1:.4f},{close+1:.4f},{close-1:.4f},{close:.4f},{100000+index},0\n"
        )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, header + "".join(lines))

    instrument_id = "US:NYSE:STOCKS:ABC"
    symbol_manifest = tmp_path / "symbol_manifest.csv"
    pd.DataFrame(
        [
            {
                "source_member": member,
                "instrument_id": instrument_id,
                "ticker": "ABC",
                "exchange": "NYSE",
                "instrument_class": "stocks",
                "valid_row_count": len(dates),
                "invalid_row_count": 0,
                "first_date": str(dates[0].date()),
                "last_date": str(dates[-1].date()),
            }
        ]
    ).to_csv(symbol_manifest, index=False)

    from tsr.data.stooq import StooqCatalog

    catalog = StooqCatalog(archive_path, symbol_manifest)
    bars = catalog.load_instrument(instrument_id)
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    (feature_root / "feature_schema.json").write_text(json.dumps(feature_schema_payload()))
    arrays = compute_features(bars).arrays
    relative = feature_shard_path(instrument_id)
    digest = write_npz(feature_root / relative, arrays)
    pd.DataFrame(
        [
            {
                "instrument_id": instrument_id,
                "ticker": "ABC",
                "exchange": "NYSE",
                "instrument_class": "stocks",
                "row_count": len(bars),
                "shard_relative_path": relative.as_posix(),
                "shard_sha256": digest,
            }
        ]
    ).to_csv(feature_root / "feature_manifest.csv", index=False)

    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    (candidate_root / "candidate_schema.json").write_text(json.dumps(candidate_schema_payload()))
    signal_index = 30
    signal_date = int(dates[signal_index].strftime("%Y%m%d"))
    entry_date = int(dates[signal_index + 1].strftime("%Y%m%d"))
    candidate = _candidate().iloc[0].to_dict()
    candidate.update(
        {
            "candidate_id": candidate_id(
                setup_family="unit_test_setup",
                setup_version="v1.0.0",
                instrument_id=instrument_id,
                signal_date=signal_date,
                direction="LONG",
            ),
            "signal_date": signal_date,
            "signal_source_row_number": int(bars.iloc[signal_index].source_row_number),
            "signal_segment_id": 0,
            "earliest_entry_date": entry_date,
        }
    )
    candidate_frame = pd.DataFrame([candidate], columns=CANDIDATE_COLUMNS)
    generator_key = "unit_test_setup|v1.0.0|LONG"
    candidate_relative = candidate_shard_path(generator_key, instrument_id)
    candidate_digest = write_candidate_shard(candidate_root / candidate_relative, candidate_frame)
    pd.DataFrame(
        [
            {
                "generator_key": generator_key,
                "setup_family": "unit_test_setup",
                "setup_version": "v1.0.0",
                "direction": "LONG",
                "instrument_id": instrument_id,
                "ticker": "ABC",
                "exchange": "NYSE",
                "instrument_class": "stocks",
                "source_bar_count": len(bars),
                "candidate_count": 1,
                "first_signal_date": signal_date,
                "last_signal_date": signal_date,
                "shard_relative_path": candidate_relative.as_posix(),
                "shard_sha256": candidate_digest,
            }
        ]
    ).to_csv(candidate_root / "candidate_manifest.csv", index=False)

    data_fp = tmp_path / "data_fp.json"
    data_fp.write_text(json.dumps({"dataset_fingerprint_sha256": "d" * 64}))
    feature_fp = tmp_path / "feature_fp.json"
    feature_fp.write_text(json.dumps({"feature_dataset_fingerprint_sha256": "f" * 64}))
    return {
        "archive_path": str(archive_path),
        "symbol_manifest_path": str(symbol_manifest),
        "data_dataset_fingerprint_path": str(data_fp),
        "feature_root": str(feature_root),
        "feature_dataset_fingerprint_path": str(feature_fp),
        "candidate_root": str(candidate_root),
    }


def test_module05_build_verify_and_publish(tmp_path: Path) -> None:
    upstream = _make_module05_upstream(tmp_path)
    config = {
        "module": {"name": "module_05_candidate_outcomes", "version": "v1.0.0"},
        "upstream": {
            "archive_path": upstream["archive_path"],
            "symbol_manifest_path": upstream["symbol_manifest_path"],
            "data_dataset_fingerprint_path": upstream["data_dataset_fingerprint_path"],
            "feature_root": upstream["feature_root"],
            "feature_dataset_fingerprint_path": upstream["feature_dataset_fingerprint_path"],
            "candidate_roots": [upstream["candidate_root"]],
        },
        "selection": {"instrument_class": "stocks", "require_rows": True},
        "execution": {
            "runs_root": str(tmp_path / "runs"),
            "dry_run_max_instruments": 1,
            "verify_candidate_shards": True,
        },
    }
    run = build(config)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["candidate_count"] == 1
    assert summary["outcome_count"] == 1
    verification = verify(run, tmp_path / "verification.json")
    assert json.loads(verification.read_text())["passed"]
    target = tmp_path / "published"
    publish(run, target)
    assert (target / "outcome_manifest.csv").exists()
