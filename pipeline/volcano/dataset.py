"""Merge seismic, thermal and labels into one daily feature table per case.

Every feature on day D uses data from day D and earlier only. tests/test_dataset.py
checks this by changing the future and confirming the past does not move.
"""
import numpy as np
import pandas as pd

from .labels import label_days

SEISMIC_BASE = ["rsam_mean", "rsam_median", "rsam_max", "mf_mean", "hf_mean", "dsar_median"]
THERMAL_BASE = ["frp_max", "frp_sum", "n_det"]


def _slope(y: np.ndarray) -> float:
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return np.nan
    return np.polyfit(np.arange(len(y))[ok], y[ok], 1)[0]


def add_rolling(df: pd.DataFrame, cols: list[str], windows: tuple[int, ...]) -> pd.DataFrame:
    """Trailing mean and trend (slope per day) for each column and window length."""
    out = {}
    for c in cols:
        for w in windows:
            r = df[c].rolling(w, min_periods=max(2, w // 2))
            out[f"{c}_mean{w}"] = r.mean()
            if w >= 3:
                out[f"{c}_trend{w}"] = r.apply(_slope, raw=True)
            else:
                out[f"{c}_trend{w}"] = df[c].diff(w - 1)
    return df.join(pd.DataFrame(out, index=df.index))


def seismic_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Scale-free seismic features.

    Raw amplitude depends on the instrument and how far the station is from the
    vent, so it cannot be compared between volcanoes. We take log10 and subtract
    each station's own typical level so far (median of all earlier days). A value
    of 0.3 then means "about twice this station's normal", at any volcano.
    """
    log = np.log10(daily[SEISMIC_BASE])
    baseline = log.expanding(min_periods=7).median().shift(1)
    rel = (log - baseline).add_suffix("_rel")
    return add_rolling(rel, list(rel.columns), (2, 7, 14))


def thermal_features(daily: pd.DataFrame) -> pd.DataFrame:
    keep = daily[[*THERMAL_BASE, "no_detection", "days_since_det"]]
    return add_rolling(keep, THERMAL_BASE, (7, 14, 30))


def build_case(case: dict, seismic_daily: pd.DataFrame, thermal_daily: pd.DataFrame | None,
               horizons: tuple[int, ...] = (2, 7)) -> pd.DataFrame:
    dates = pd.date_range(case["window_start"], case["window_end"], freq="D", name="date")
    seis = seismic_daily.reindex(dates)
    parts = [seis[["rsam_median", "seis_coverage"]], seismic_features(seis)]
    if thermal_daily is not None:
        # thermal history starts before the window so 30 day features are full from day one
        parts.append(thermal_features(thermal_daily).reindex(dates))
    df = pd.concat(parts, axis=1)
    onset = case.get("onset")
    for n in horizons:
        df[f"label_{n}d"], df["pre_onset"] = label_days(dates, onset, n)
    df.insert(0, "case", case["id"])
    return df
