from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsr.ensemble import EnsembleConfig, RegimeAwareBoostedTreeEnsemble


def _frame(rows_per_family: int = 350, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for family_index, family in enumerate(("pullback", "breakout", "reversal")):
        n = rows_per_family
        trend = rng.normal(0, 1, n)
        volatility = rng.lognormal(-2.8, 0.35, n)
        relative_strength = rng.normal(0, 1, n)
        breadth = rng.normal(0, 1, n)
        market_trend = rng.normal(0, 1, n)
        family_edge = (family_index - 1) * 0.12
        latent = (
            0.8 * trend
            + 0.7 * relative_strength
            + 0.35 * breadth
            + 0.45 * market_trend
            - 4.0 * volatility
            + family_edge
            + rng.normal(0, 0.8, n)
        )
        probability = 1.0 / (1.0 + np.exp(-latent))
        target = rng.binomial(1, probability)
        returns = 0.025 * latent + rng.normal(0, 0.04, n)
        parts.append(
            pd.DataFrame(
                {
                    "candidate_id": [f"{family}_{i}" for i in range(n)],
                    "setup_family": family,
                    "trend": trend,
                    "volatility": volatility,
                    "relative_strength": relative_strength,
                    "breadth": breadth,
                    "market_trend": market_trend,
                    "y_target": target,
                    "y_return": returns,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _config() -> EnsembleConfig:
    return EnsembleConfig(
        feature_columns=("trend", "volatility", "relative_strength", "breadth", "market_trend"),
        regime_columns=("breadth", "market_trend", "volatility"),
        minimum_family_rows=200,
        max_iter=60,
        min_samples_leaf=25,
    )


def test_ensemble_fits_specialists_and_returns_bounded_probabilities():
    frame = _frame()
    model = RegimeAwareBoostedTreeEnsemble(_config()).fit(frame)
    assert model.families == ("breakout", "pullback", "reversal")
    prediction = model.predict(frame.iloc[:120])
    assert (prediction.model_status == "scored").all()
    assert prediction.regime_gate_probability.between(0, 1).all()
    assert prediction.specialist_target_probability.between(0, 1).all()
    assert prediction.ensemble_score.between(0, 1).all()
    assert len(model.specialist_metadata()) == 3


def test_unseen_family_is_not_silently_scored():
    frame = _frame()
    model = RegimeAwareBoostedTreeEnsemble(_config()).fit(frame)
    probe = frame.iloc[:4].copy()
    probe["setup_family"] = "unknown"
    prediction = model.predict(probe)
    assert (prediction.model_status == "family_not_fitted").all()
    assert prediction.ensemble_score.isna().all()


def test_fit_rejects_missing_contract_columns():
    with pytest.raises(ValueError, match="missing columns"):
        RegimeAwareBoostedTreeEnsemble(_config()).fit(pd.DataFrame({"setup_family": ["x"]}))
