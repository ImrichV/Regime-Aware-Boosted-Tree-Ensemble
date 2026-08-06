from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def candidate_diagnostics(
    candidates: pd.DataFrame,
    *,
    eligible_bar_count: int,
    instrument_count: int,
) -> dict[str, Any]:
    if candidates.empty:
        return {
            "candidate_count": 0,
            "eligible_bar_count": int(eligible_bar_count),
            "candidates_per_1000_eligible_bars": 0.0,
            "instrument_count": int(instrument_count),
            "instruments_with_candidates": 0,
            "dates_with_candidates": 0,
            "max_candidates_on_one_date": 0,
            "median_candidates_on_active_date": 0.0,
            "entry_status_counts": {},
            "by_year": {},
            "by_exchange": {},
            "by_instrument_class": {},
        }
    active_date_counts = candidates.groupby("signal_date").size()
    by_year = candidates.assign(year=candidates["signal_date"] // 10000).groupby("year").size()
    return {
        "candidate_count": int(len(candidates)),
        "eligible_bar_count": int(eligible_bar_count),
        "candidates_per_1000_eligible_bars": float(
            len(candidates) * 1000.0 / eligible_bar_count if eligible_bar_count else 0.0
        ),
        "instrument_count": int(instrument_count),
        "instruments_with_candidates": int(candidates["instrument_id"].nunique()),
        "dates_with_candidates": int(candidates["signal_date"].nunique()),
        "max_candidates_on_one_date": int(active_date_counts.max()),
        "median_candidates_on_active_date": float(active_date_counts.median()),
        "entry_status_counts": {
            str(k): int(v)
            for k, v in candidates["historical_entry_status"].value_counts().sort_index().items()
        },
        "by_year": {str(int(k)): int(v) for k, v in by_year.sort_index().items()},
        "by_exchange": {
            str(k): int(v)
            for k, v in candidates["exchange"].value_counts().sort_index().items()
        },
        "by_instrument_class": {
            str(k): int(v)
            for k, v in candidates["instrument_class"].value_counts().sort_index().items()
        },
    }
