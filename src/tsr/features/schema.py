from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

MODULE_NAME = "module_02_features"
MODULE_VERSION = "v1.0.0"
FEATURE_SCHEMA_VERSION = "feature_schema_v1.0.0"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    lookback_bars: int
    group: str
    formula: str
    volume_dependent: bool = False


DEFINITIONS = (
    FeatureDefinition("intraday_return", 1, "bar", "close/open-1"),
    FeatureDefinition("gap_return", 2, "bar", "open/previous_close-1"),
    FeatureDefinition("high_low_range_pct", 1, "bar", "(high-low)/close"),
    FeatureDefinition("true_range_pct", 1, "bar", "true_range/previous_close; first segment bar uses high-low over close"),
    FeatureDefinition("close_location", 1, "bar", "(close-low)/(high-low); NaN when range is zero"),
    FeatureDefinition("body_to_range", 1, "bar", "(close-open)/(high-low); NaN when range is zero"),
    *tuple(FeatureDefinition(f"return_{w}", w + 1, "return", f"close/close[t-{w}]-1") for w in (1, 5, 10, 20, 60, 120, 252)),
    *tuple(FeatureDefinition(f"sma_distance_{w}", w, "trend", f"close/mean(close,{w})-1") for w in (10, 20, 50, 100, 200)),
    *tuple(FeatureDefinition(f"realized_vol_{w}", w + 1, "volatility", f"sample std of log returns over {w} bars") for w in (10, 20, 60)),
    FeatureDefinition("atr_14_pct", 14, "volatility", "mean(true_range,14)/close"),
    FeatureDefinition("volume_ratio_20_prior", 21, "volume", "volume/mean(previous 20 volumes)", True),
    FeatureDefinition("log_dollar_volume_20_prior", 21, "volume", "log1p(mean(previous 20 close*volume))", True),
    *tuple(FeatureDefinition(f"distance_to_prior_high_{w}", w + 1, "location", f"close/max(previous {w} highs)-1") for w in (20, 60, 252)),
    *tuple(FeatureDefinition(f"distance_to_prior_low_{w}", w + 1, "location", f"close/min(previous {w} lows)-1") for w in (20, 60, 252)),
    *tuple(FeatureDefinition(f"efficiency_ratio_{w}", w + 1, "path", f"absolute {w}-bar displacement/sum absolute bar changes") for w in (20, 60)),
    *tuple(FeatureDefinition(f"up_fraction_{w}", w + 1, "path", f"fraction positive one-bar returns over {w} bars") for w in (20, 60)),
    FeatureDefinition("range_ratio_5_20", 20, "contraction", "mean(range_pct,5)/mean(range_pct,20)"),
    FeatureDefinition("realized_vol_ratio_10_60", 61, "contraction", "realized_vol_10/realized_vol_60"),
)
FEATURE_NAMES = tuple(item.name for item in DEFINITIONS)
META_NAMES = ("date", "source_row_number", "segment_id", "history_bars", "source_gap_rows", "calendar_gap_days", "segment_reset")


def payload() -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "row_key": ["instrument_id", "date"],
        "feature_dtype": "float32",
        "date_encoding": "int32 YYYYMMDD",
        "information_cutoff": "after current daily bar closes",
        "earliest_permitted_entry": "next valid session open",
        "missing_history": "NaN; never imputed",
        "continuity": {
            "source_row_discontinuity": "start new segment",
            "calendar_gap_above_configured_threshold": "start new segment",
            "cross_segment_rolling_state": "forbidden",
        },
        "features": [asdict(x) for x in DEFINITIONS],
        "forbidden": ["future outcome", "target", "label", "candidate flag", "rank", "prediction", "trade result"],
        "limitations": [
            "Volume features inherit uncertainty in Stooq adjustment methodology.",
            "The source archive is survivor dominated.",
            "Daily bars cannot reveal intraday event ordering.",
        ],
    }
    data["schema_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return data
