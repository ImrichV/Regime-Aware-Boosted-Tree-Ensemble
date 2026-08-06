from __future__ import annotations

import numpy as np
import pandas as pd

from tsr.candidates.base import CandidateContext
from tsr.candidates.engine import audit_prefix_causality, materialize_candidates
from tsr.features.engine import compute_features
from tsr.playbooks.public_demo import PublicTrendPullbackDemoGenerator


def _context(rows: int = 300) -> CandidateContext:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    trend = 100.0 * np.exp(np.linspace(0, 0.35, rows))
    pullback = np.ones(rows)
    pullback[-30:] = np.linspace(1.0, 0.94, 30)
    close = trend * pullback
    bars = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 2_000_000.0),
            "source_row_number": np.arange(2, rows + 2),
        }
    )
    arrays = compute_features(bars).arrays
    features = pd.DataFrame({name: value for name, value in arrays.items()})
    features["date"] = pd.to_datetime(features["date"].astype(str))
    return CandidateContext(
        instrument_id="US:NASDAQ:STOCKS:DEMO",
        ticker="DEMO",
        exchange="NASDAQ",
        instrument_class="stocks",
        bars=bars,
        features=features,
        feature_schema_version="feature_schema_v1.0.0",
        feature_dataset_sha256="f" * 64,
    )


def test_public_demo_is_causal_and_materializes_candidates():
    context = _context()
    generator = PublicTrendPullbackDemoGenerator(
        {
            "minimum_log_dollar_volume": 0.0,
            "minimum_return_20": -0.10,
            "pullback_from_high_min": -0.20,
            "pullback_from_high_max": -0.005,
        }
    )
    result = audit_prefix_causality(generator, context, checkpoints=[140, 200, 260, 300])
    assert result.passed
    frame = materialize_candidates(generator, context)
    assert len(frame) > 0
    assert frame.candidate_id.is_unique
    assert set(frame.setup_family) == {"public_trend_pullback_demo"}
