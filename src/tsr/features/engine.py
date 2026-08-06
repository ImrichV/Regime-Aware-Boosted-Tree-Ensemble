from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .schema import FEATURE_NAMES


@dataclass(frozen=True)
class FeatureResult:
    arrays: Mapping[str, np.ndarray]
    segment_count: int
    source_gap_reset_count: int
    calendar_gap_reset_count: int


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x); out = np.full(n, np.nan, dtype=np.float64)
    if n < w: return out
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    out[w-1:] = (cs[w:] - cs[:-w]) / w
    return out


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x); out = np.full(n, np.nan, dtype=np.float64)
    if n < w or w < 2: return out
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    cs2 = np.concatenate(([0.0], np.cumsum(x*x, dtype=np.float64)))
    sums = cs[w:] - cs[:-w]; sums2 = cs2[w:] - cs2[:-w]
    var = (sums2 - sums*sums/w) / (w-1)
    out[w-1:] = np.sqrt(np.maximum(var, 0.0))
    return out


def _safe(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.full_like(n, np.nan, dtype=np.float64)
    np.divide(n, d, out=out, where=(d != 0) & np.isfinite(d))
    out[~np.isfinite(out)] = np.nan
    return out


def _segment_features(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    n = len(c); f = {name: np.full(n, np.nan, dtype=np.float64) for name in FEATURE_NAMES}
    f["intraday_return"] = c/o - 1.0
    if n > 1: f["gap_return"][1:] = o[1:]/c[:-1]-1.0
    rng = h-l
    f["high_low_range_pct"] = rng/c
    tr = rng.copy()
    if n > 1: tr[1:] = np.maximum.reduce((rng[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr_den = c.copy()
    if n > 1: tr_den[1:] = c[:-1]
    f["true_range_pct"] = tr/tr_den
    f["close_location"] = _safe(c-l, rng)
    f["body_to_range"] = _safe(c-o, rng)
    for w in (1,5,10,20,60,120,252):
        if n > w: f[f"return_{w}"][w:] = c[w:]/c[:-w]-1.0
    for w in (10,20,50,100,200):
        f[f"sma_distance_{w}"] = _safe(c, _roll_mean(c,w)) - 1.0
    lr = np.diff(np.log(c))
    for w in (10,20,60):
        rv = np.full(n, np.nan)
        if len(lr): rv[1:] = _roll_std(lr,w)
        f[f"realized_vol_{w}"] = rv
    f["atr_14_pct"] = _roll_mean(tr,14)/c
    if n > 20:
        win_v = sliding_window_view(v[:-1],20)
        f["volume_ratio_20_prior"][20:] = v[20:]/win_v.mean(axis=1)
        win_dv = sliding_window_view((c*v)[:-1],20)
        f["log_dollar_volume_20_prior"][20:] = np.log1p(win_dv.mean(axis=1))
    for w in (20,60,252):
        if n > w:
            f[f"distance_to_prior_high_{w}"][w:] = c[w:]/sliding_window_view(h[:-1],w).max(axis=1)-1.0
            f[f"distance_to_prior_low_{w}"][w:] = c[w:]/sliding_window_view(l[:-1],w).min(axis=1)-1.0
    changes = np.abs(np.diff(c)); ups = (np.diff(c)>0).astype(np.float64)
    for w in (20,60):
        if n > w:
            path = sliding_window_view(changes,w).sum(axis=1)
            f[f"efficiency_ratio_{w}"][w:] = _safe(np.abs(c[w:]-c[:-w]), path)
            f[f"up_fraction_{w}"][w:] = sliding_window_view(ups,w).mean(axis=1)
    m5 = _roll_mean(f["high_low_range_pct"],5); m20 = _roll_mean(f["high_low_range_pct"],20)
    f["range_ratio_5_20"] = _safe(m5,m20)
    f["realized_vol_ratio_10_60"] = _safe(f["realized_vol_10"],f["realized_vol_60"])
    for k in f: f[k][~np.isfinite(f[k])] = np.nan
    return f


def compute_features(bars: pd.DataFrame, max_calendar_gap_days: int = 30) -> FeatureResult:
    req={"date","open","high","low","close","volume","source_row_number"}
    miss=req-set(bars.columns)
    if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    n=len(bars)
    if n==0:
        arrays={"date":np.array([],np.int32),"source_row_number":np.array([],np.int32),"segment_id":np.array([],np.int32),"history_bars":np.array([],np.int32),"source_gap_rows":np.array([],np.int32),"calendar_gap_days":np.array([],np.int32),"segment_reset":np.array([],np.uint8)}
        arrays.update({k:np.array([],np.float32) for k in FEATURE_NAMES})
        return FeatureResult(arrays,0,0,0)
    dates=pd.to_datetime(bars.date)
    if not dates.is_monotonic_increasing or dates.duplicated().any(): raise ValueError("Dates must be unique and increasing")
    sr=bars.source_row_number.to_numpy(np.int64)
    source_gap=np.zeros(n,np.int32); cal_gap=np.zeros(n,np.int32)
    if n>1:
        source_gap[1:]=np.maximum(sr[1:]-sr[:-1]-1,0).astype(np.int32)
        cal_gap[1:]=dates.diff().dt.days.fillna(0).to_numpy(np.int32)[1:]
    sreset=source_gap>0; creset=cal_gap>max_calendar_gap_days; reset=sreset|creset; reset[0]=True
    seg=np.cumsum(reset,dtype=np.int32)-1; hist=np.empty(n,np.int32)
    allf={k:np.full(n,np.nan,np.float32) for k in FEATURE_NAMES}
    o=bars.open.to_numpy(float); h=bars.high.to_numpy(float); l=bars.low.to_numpy(float); c=bars.close.to_numpy(float); v=bars.volume.to_numpy(float)
    for sid in range(int(seg[-1])+1):
        idx=np.flatnonzero(seg==sid); a,b=idx[0],idx[-1]+1; hist[a:b]=np.arange(1,b-a+1,dtype=np.int32)
        sf=_segment_features(o[a:b],h[a:b],l[a:b],c[a:b],v[a:b])
        for k in FEATURE_NAMES: allf[k][a:b]=sf[k].astype(np.float32)
    dateint=(dates.dt.year.to_numpy(np.int32)*10000+dates.dt.month.to_numpy(np.int32)*100+dates.dt.day.to_numpy(np.int32))
    arrays={"date":dateint,"source_row_number":sr.astype(np.int32),"segment_id":seg,"history_bars":hist,"source_gap_rows":source_gap,"calendar_gap_days":cal_gap,"segment_reset":reset.astype(np.uint8),**allf}
    return FeatureResult(arrays,int(seg[-1])+1,int(sreset.sum()),int(creset.sum()))
