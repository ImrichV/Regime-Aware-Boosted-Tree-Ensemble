"""Run the public regime-aware specialist ensemble on synthetic data.

This example proves the software contract only.  It is not a market-performance
claim and it contains no production trading rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsr.ensemble import EnsembleConfig, RegimeAwareBoostedTreeEnsemble


def make_synthetic(rows_per_family: int = 500, seed: int = 260806) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for family in ("trend_pullback", "breakout", "failed_move"):
        n = rows_per_family
        trend = rng.normal(size=n)
        relative_strength = rng.normal(size=n)
        breadth = rng.normal(size=n)
        market_trend = rng.normal(size=n)
        volatility = rng.lognormal(-2.8, 0.4, size=n)
        latent = (
            0.7 * trend
            + 0.6 * relative_strength
            + 0.4 * breadth
            + 0.5 * market_trend
            - 3.5 * volatility
            + rng.normal(scale=0.9, size=n)
        )
        p = 1.0 / (1.0 + np.exp(-latent))
        frames.append(
            pd.DataFrame(
                {
                    "candidate_id": [f"{family}-{i}" for i in range(n)],
                    "setup_family": family,
                    "trend": trend,
                    "relative_strength": relative_strength,
                    "breadth": breadth,
                    "market_trend": market_trend,
                    "volatility": volatility,
                    "y_target": rng.binomial(1, p),
                    "y_return": 0.02 * latent + rng.normal(scale=0.04, size=n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    frame = make_synthetic()
    train = frame.sample(frac=0.75, random_state=1)
    test = frame.drop(train.index)
    config = EnsembleConfig(
        feature_columns=(
            "trend",
            "relative_strength",
            "breadth",
            "market_trend",
            "volatility",
        ),
        regime_columns=("breadth", "market_trend", "volatility"),
        minimum_family_rows=200,
        max_iter=100,
    )
    model = RegimeAwareBoostedTreeEnsemble(config).fit(train)
    print(model.specialist_metadata().to_string(index=False))
    print(model.predict(test.head(12)).to_string(index=False))


if __name__ == "__main__":
    main()
