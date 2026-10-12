"""Seismic download and feature extraction.

Band definitions follow Dempsey et al. 2020 (Whakaari forecaster): band-passed
ground velocity, mean absolute amplitude over 10 minute windows.
  rsam: 2 to 5 Hz      mf: 4.5 to 8 Hz      hf: 8 to 16 Hz
  dsar: ratio of mf to hf amplitude after integrating velocity to displacement
"""
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from .config import INTERIM

WINDOW_S = 600
BANDS = {"rsam": (2.0, 5.0), "mf": (4.5, 8.0), "hf": (8.0, 16.0)}
MIN_COVERAGE = 0.5  # a window or day with less data than this is NaN


def tenmin_features(st: Stream, day: UTCDateTime, sensitivity: float = 1.0) -> pd.DataFrame:
    """10 minute band amplitudes for one UTC day. `st` is one channel, gaps allowed."""
    n_win = 86400 // WINDOW_S
    cols = [*BANDS, "mf_disp", "hf_disp"]
    sums = {c: np.zeros(n_win) for c in cols}
    counts = np.zeros(n_win)
    fs = 0.0
    # Each gap-free segment is filtered on its own so gaps never get interpolated.
    for tr in st.split():
        fs = tr.stats.sampling_rate
        if tr.stats.npts < 60 * fs:
            continue
        tr = tr.copy()
        tr.data = tr.data.astype(np.float64) / sensitivity
        tr.detrend("linear")
        tr.taper(0.01, max_length=30)
        win = np.floor((tr.times() + (tr.stats.starttime - day)) / WINDOW_S).astype(int)
        ok = (win >= 0) & (win < n_win)
        counts += np.bincount(win[ok], minlength=n_win)
        disp = tr.copy().integrate()
        for name, (lo, hi) in BANDS.items():
            for col, src in ((name, tr), (f"{name}_disp", disp)):
                if col not in sums:
                    continue
                amp = np.abs(src.copy().filter("bandpass", freqmin=lo, freqmax=hi, corners=4, zerophase=True).data)
                sums[col] += np.bincount(win[ok], weights=amp[ok], minlength=n_win)
    index = pd.date_range(day.datetime, periods=n_win, freq=f"{WINDOW_S}s", name="time")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({c: sums[c] / counts for c in cols}, index=index)
        out["dsar"] = out.pop("mf_disp") / out.pop("hf_disp")
    if fs:
        out[counts < MIN_COVERAGE * WINDOW_S * fs] = np.nan
    return out


def daily_features(tenmin: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 10 minute windows to one row per UTC day."""
    g = tenmin.groupby(tenmin.index.floor("D"))
    out = pd.DataFrame({
        "rsam_mean": g["rsam"].mean(),
        # the median ignores short earthquake spikes, the max keeps them
        "rsam_median": g["rsam"].median(),
        "rsam_max": g["rsam"].max(),
        "mf_mean": g["mf"].mean(),
        "hf_mean": g["hf"].mean(),
        "dsar_median": g["dsar"].median(),
        "seis_coverage": g["rsam"].count() / (86400 // WINDOW_S),
    })
    out.loc[out["seis_coverage"] < MIN_COVERAGE, out.columns != "seis_coverage"] = np.nan
    out.index.name = "date"
    return out


def fetch_case(case: dict, log=print) -> pd.DataFrame:
    """Download a case window day by day and return 10 minute features.

    Each finished day is cached as a small CSV, so reruns only fetch what is missing.
    shortcut: raw waveforms are not kept (about 1 GB per case), so changing the
    band definitions means downloading again. Cache miniSEED if that gets painful.
    """
    cache = INTERIM / case["id"] / "tenmin"
    cache.mkdir(parents=True, exist_ok=True)
    client = Client(case["fdsn_url"])
    net, sta, cha = case["network"], case["station"], case["channel"]
    start, end = UTCDateTime(case["window_start"]), UTCDateTime(case["window_end"])
    inv = client.get_stations(network=net, station=sta, channel=cha, level="response",
                              starttime=start, endtime=end + 86400)
    frames = []
    day = start
    while day <= end:
        path = cache / f"{day.date}.csv"
        if not path.exists():
            try:
                # 60 s of padding keeps filter edge effects out of the day
                st = client.get_waveforms(net, sta, "*", cha, day - 60, day + 86400 + 60)
            except FDSNNoDataException:
                st = Stream()
            # sensitivity is looked up per day in case the instrument changed
            sens = inv.get_response(st[0].id, day + 43200).instrument_sensitivity.value if st else 1.0
            df = tenmin_features(st, day, sens)
            df.to_csv(path)
            log(f"{case['id']} {day.date}: {df['rsam'].count()}/{len(df)} windows")
        frames.append(pd.read_csv(path, index_col="time", parse_dates=True))
        day += 86400
    return pd.concat(frames)
