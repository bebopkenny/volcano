"""NASA FIRMS VIIRS active fire detections and daily thermal features."""
import io
import os
import time

import numpy as np
import pandas as pd
import requests

from .config import ROOT

RAW = ROOT / "data" / "raw" / "firms"
API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SOURCES = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
CHUNK_DAYS = 5      # the area API allows at most 5 days per request
HISTORY_DAYS = 30   # extra days fetched before the window for 30 day features
RADIUS_KM = 5.0
MAX_DAYS_SINCE = 60


def map_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY")
    env = ROOT.parent / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FIRMS_MAP_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("FIRMS_MAP_KEY is not set. Put it in .env at the repo root (see .env.example).")
    return key


def _get(url: str) -> str:
    for attempt in range(5):
        r = requests.get(url, timeout=60)
        text = r.text
        # FIRMS reports bad keys and quota errors as plain text, sometimes with status 200
        if r.ok and (text.startswith("latitude") or not text.strip()):
            return text
        if r.status_code in (400, 401, 403) or "Invalid" in text:
            raise RuntimeError(f"FIRMS refused the request: {text[:200]}")
        time.sleep(2 ** attempt * 5)
    raise RuntimeError(f"FIRMS request failed after retries: HTTP {r.status_code} {text[:200]}")


def fetch_detections(case: dict, log=print) -> pd.DataFrame:
    """All VIIRS detections in a box around the vent. Responses are cached on disk."""
    cache = RAW / case["id"]
    cache.mkdir(parents=True, exist_ok=True)
    pad = 0.1  # degrees, about 11 km; the exact radius filter happens later
    box = f"{case['lon'] - pad},{case['lat'] - pad},{case['lon'] + pad},{case['lat'] + pad}"
    start = pd.Timestamp(case["window_start"]) - pd.Timedelta(days=HISTORY_DAYS)
    end = pd.Timestamp(case["window_end"])
    key = None
    frames = []
    for source in SOURCES:
        day = start
        while day <= end:
            n = min(CHUNK_DAYS, (end - day).days + 1)
            path = cache / f"{source}_{day.date()}_{n}.csv"
            if not path.exists():
                key = key or map_key()
                path.write_text(_get(f"{API}/{key}/{source}/{box}/{n}/{day.date()}"))
                log(f"{case['id']} {source} {day.date()} +{n}d")
                time.sleep(0.5)  # far below the 5000 transactions per 10 minutes limit
            if path.stat().st_size:
                frames.append(pd.read_csv(path).assign(source=source))
            day += pd.Timedelta(days=n)
    cols = ["latitude", "longitude", "acq_date", "frp", "source"]
    return pd.concat(frames)[cols] if frames else pd.DataFrame(columns=cols)


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def daily_thermal(det: pd.DataFrame, lat: float, lon: float, dates: pd.DatetimeIndex,
                  radius_km: float = RADIUS_KM) -> pd.DataFrame:
    """One row per UTC day from detections within `radius_km` of the vent.

    A day with no detection gets FRP 0 and no_detection 1. That is "nothing seen",
    not "no heat": cloud and the wildfire tuned detection threshold both hide weak heat.
    """
    near = det[_km(lat, lon, det["latitude"].to_numpy(float), det["longitude"].to_numpy(float)) <= radius_km]
    g = near.groupby(pd.to_datetime(near["acq_date"]))["frp"]
    out = pd.DataFrame({"frp_max": g.max(), "frp_sum": g.sum(), "n_det": g.count()}).reindex(dates).fillna(0.0)
    out["no_detection"] = (out["n_det"] == 0).astype(int)
    # days since the most recent detection, counting today as 0, capped when none is known
    last = pd.Series(np.where(out["n_det"] > 0, np.arange(len(out)), np.nan), index=dates).ffill()
    out["days_since_det"] = (np.arange(len(out)) - last).fillna(MAX_DAYS_SINCE).clip(upper=MAX_DAYS_SINCE)
    out.index.name = "date"
    return out
