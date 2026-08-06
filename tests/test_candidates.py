from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsr.candidates.base import CandidateContext
from tsr.candidates.engine import (
    audit_prefix_causality,
    combine_candidate_frames,
    materialize_candidates,
)
from tsr.candidates.schema import (
    CANDIDATE_COLUMNS,
    CANDIDATE_SCHEMA_VERSION,
    candidate_id,
    schema_payload,
    validate_candidate_mapping,
)
from tsr.candidates.storage import (
    CandidateStore,
    deterministic_csv_bytes,
    shard_relative_path,
    write_candidate_shard,
)
from tsr.candidates.testing import FrameworkProbeGenerator, FutureLeakingProbeGenerator


def make_context(n: int = 220) -> CandidateContext:
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 + np.arange(n, dtype=float) * 0.2
    bars = pd.DataFrame(
        {
            "instrument_id": "US:NASDAQ:STOCKS:AAA",
            "ticker": "AAA",
            "exchange": "NASDAQ",
            "instrument_class": "stocks",
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
            "source_member": "data/daily/us/nasdaq stocks/1/aaa.us.txt",
            "source_row_number": np.arange(2, n + 2),
        }
    )
    ret20 = np.full(n, np.nan)
    ret20[20:] = close[20:] / close[:-20] - 1.0
    ret1 = np.full(n, np.nan)
    ret1[1:] = close[1:] / close[:-1] - 1.0
    features = pd.DataFrame(
        {
            "instrument_id": bars.instrument_id,
            "date": dates,
            "source_row_number": bars.source_row_number,
            "segment_id": np.zeros(n, dtype=np.int32),
            "segment_reset": np.r_[1, np.zeros(n - 1)].astype(np.uint8),
            "history_bars": np.arange(1, n + 1),
            "return_20": ret20,
            "return_1": ret1,
        }
    )
    return CandidateContext(
        instrument_id="US:NASDAQ:STOCKS:AAA",
        ticker="AAA",
        exchange="NASDAQ",
        instrument_class="stocks",
        bars=bars,
        features=features,
        feature_schema_version="feature_schema_v1.0.0",
        feature_dataset_sha256="f" * 64,
    )


def test_candidate_id_is_deterministic_and_component_sensitive():
    base = candidate_id(
        setup_family="framework_probe",
        setup_version="v1.0.0",
        instrument_id="US:NASDAQ:STOCKS:AAA",
        signal_date=20200131,
        direction="LONG",
    )
    assert base == candidate_id(
        setup_family="framework_probe",
        setup_version="v1.0.0",
        instrument_id="US:NASDAQ:STOCKS:AAA",
        signal_date=20200131,
        direction="LONG",
    )
    changed = candidate_id(
        setup_family="framework_probe",
        setup_version="v1.0.0",
        instrument_id="US:NASDAQ:STOCKS:AAA",
        signal_date=20200203,
        direction="LONG",
    )
    assert base != changed
    assert base.startswith("cand_") and len(base) == 69


def test_schema_payload_is_stable_and_forbids_outcomes():
    payload = schema_payload()
    assert payload["schema_version"] == CANDIDATE_SCHEMA_VERSION
    assert "future_return" in payload["forbidden_fields"]
    assert payload["timing"]["same_close_entry"] == "forbidden"
    assert len(payload["schema_sha256"]) == 64


def test_materialize_candidates_preserves_lineage_and_next_open_timing():
    context = make_context(220)
    generator = FrameworkProbeGenerator({"modulus": 50, "offset": 0})
    frame = materialize_candidates(generator, context)
    assert tuple(frame.columns) == CANDIDATE_COLUMNS
    assert len(frame) == 4
    assert frame.candidate_id.is_unique
    assert (frame.earliest_entry_date > frame.signal_date).all()
    assert set(frame.historical_entry_status) == {"available"}
    for row in frame.to_dict("records"):
        validate_candidate_mapping(row)
        source_row = int(row["signal_source_row_number"])
        index = source_row - 2
        assert row["signal_date"] == int(context.bars.iloc[index].date.strftime("%Y%m%d"))
        assert row["raw_setup_strength"] == pytest.approx(
            float(context.features.iloc[index].return_20)
        )
        assert json.loads(row["generator_payload_json"])["probe_modulus"] == 50


def test_last_bar_candidate_has_no_historical_entry_date():
    context = make_context(97)
    generator = FrameworkProbeGenerator({"modulus": 97, "offset": 0})
    frame = materialize_candidates(generator, context)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.historical_entry_status == "no_later_bar"
    assert pd.isna(row.earliest_entry_date)


def test_segment_break_blocks_historical_entry_alignment():
    context = make_context(100)
    context.features.loc[97, "segment_id"] = 1
    context.features.loc[97, "segment_reset"] = 1
    context.features.loc[98:, "segment_id"] = 1
    generator = FrameworkProbeGenerator({"modulus": 97, "offset": 0})
    frame = materialize_candidates(generator, context)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.signal_source_row_number == 98
    assert row.historical_entry_status == "next_bar_new_segment"
    assert pd.isna(row.earliest_entry_date)


def test_prefix_causality_audit_passes_valid_generator():
    context = make_context(220)
    result = audit_prefix_causality(
        FrameworkProbeGenerator({"modulus": 31}),
        context,
        checkpoints=[25, 60, 120, 180, 220],
    )
    assert result.passed
    assert result.checkpoints_tested == 5
    assert result.rows_compared == 605


def test_prefix_causality_audit_detects_future_leakage():
    context = make_context(100)
    with pytest.raises(AssertionError, match="prefix causality failed"):
        audit_prefix_causality(
            FutureLeakingProbeGenerator(),
            context,
            checkpoints=[40, 60, 80],
        )


def test_combine_rejects_duplicate_candidate_ids():
    frame = materialize_candidates(FrameworkProbeGenerator({"modulus": 50}), make_context())
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        combine_candidate_frames([frame, frame.copy()])


def test_candidate_csv_is_deterministic_and_store_roundtrips(tmp_path: Path):
    frame = materialize_candidates(FrameworkProbeGenerator({"modulus": 50}), make_context())
    assert deterministic_csv_bytes(frame) == deterministic_csv_bytes(frame.copy())
    key = "framework_probe|v1.0.0|LONG"
    rel = shard_relative_path(key, "US:NASDAQ:STOCKS:AAA")
    digest = write_candidate_shard(tmp_path / rel, frame)
    pd.DataFrame(
        [
            {
                "generator_key": key,
                "instrument_id": "US:NASDAQ:STOCKS:AAA",
                "candidate_count": len(frame),
                "shard_relative_path": str(rel.as_posix()),
                "shard_sha256": digest,
            }
        ]
    ).to_csv(tmp_path / "candidate_manifest.csv", index=False)
    (tmp_path / "candidate_schema.json").write_text(json.dumps(schema_payload()))
    store = CandidateStore(tmp_path)
    loaded = store.load_shard(key, "US:NASDAQ:STOCKS:AAA", verify=True)
    assert list(loaded.candidate_id) == list(frame.candidate_id)
    assert list(loaded.signal_date) == list(frame.signal_date)
