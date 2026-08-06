from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..candidates.base import CandidateContext, CandidateGenerator, CandidateSignals
from ..candidates.schema import GeneratorMetadata


class PublicTrendPullbackDemoGenerator(CandidateGenerator):
    """Illustrative causal candidate generator for synthetic/public examples.

    This class is deliberately simple and configurable.  It demonstrates the
    candidate contract, signal-close timing, payload lineage, and prefix causality.
    It is not one of the private accepted setup definitions and is not presented as
    a profitable trading strategy.
    """

    metadata = GeneratorMetadata(
        setup_family="public_trend_pullback_demo",
        setup_version="v1.0.0",
        direction="LONG",
        required_features=(
            "history_bars",
            "return_20",
            "distance_to_prior_high_60",
            "atr_14_pct",
            "log_dollar_volume_20_prior",
        ),
        description="Public causal trend/pullback example for framework demonstration.",
        permissive=True,
        overlap_policy="allow",
    )

    DEFAULTS: Mapping[str, float | int] = {
        "minimum_history_bars": 120,
        "minimum_return_20": 0.02,
        "pullback_from_high_min": -0.12,
        "pullback_from_high_max": -0.02,
        "maximum_atr_14_pct": 0.08,
        "minimum_log_dollar_volume": 12.0,
    }

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        merged = dict(self.DEFAULTS)
        merged.update(dict(config or {}))
        super().__init__(merged)

    def generate(self, context: CandidateContext) -> CandidateSignals:
        self.validate_context(context)
        n = len(context.features)
        if n == 0 or context.instrument_class.lower() != "stocks":
            return CandidateSignals(mask=np.zeros(n, dtype=bool))

        f = context.features
        history = f["history_bars"].to_numpy(np.int64)
        ret20 = f["return_20"].to_numpy(np.float64)
        from_high = f["distance_to_prior_high_60"].to_numpy(np.float64)
        atr = f["atr_14_pct"].to_numpy(np.float64)
        liquidity = f["log_dollar_volume_20_prior"].to_numpy(np.float64)

        finite = (
            np.isfinite(ret20)
            & np.isfinite(from_high)
            & np.isfinite(atr)
            & np.isfinite(liquidity)
        )
        cfg = self.config
        mask = (
            finite
            & (history >= int(cfg["minimum_history_bars"]))
            & (ret20 >= float(cfg["minimum_return_20"]))
            & (from_high >= float(cfg["pullback_from_high_min"]))
            & (from_high <= float(cfg["pullback_from_high_max"]))
            & (atr <= float(cfg["maximum_atr_14_pct"]))
            & (liquidity >= float(cfg["minimum_log_dollar_volume"]))
        )

        # A bounded descriptive score for diagnostics only.
        trend_component = np.clip(ret20 / 0.20, 0.0, 1.0)
        pullback_center = (
            float(cfg["pullback_from_high_min"]) + float(cfg["pullback_from_high_max"])
        ) / 2.0
        half_width = max(
            1e-9,
            (float(cfg["pullback_from_high_max"]) - float(cfg["pullback_from_high_min"]))
            / 2.0,
        )
        pullback_component = np.clip(1.0 - np.abs(from_high - pullback_center) / half_width, 0.0, 1.0)
        strength = 0.6 * trend_component + 0.4 * pullback_component

        payload = {
            int(i): {
                "public_demo": True,
                "return_20": float(ret20[i]),
                "distance_to_prior_high_60": float(from_high[i]),
                "atr_14_pct": float(atr[i]),
            }
            for i in np.flatnonzero(mask)
        }
        return CandidateSignals(
            mask=mask.astype(bool),
            raw_setup_strength=strength.astype(np.float64),
            payload_by_row=payload,
        )
