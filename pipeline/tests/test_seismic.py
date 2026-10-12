import numpy as np
from obspy import Stream, Trace, UTCDateTime

from volcano.seismic import daily_features, tenmin_features

DAY = UTCDateTime("2022-01-01")
FS = 50.0


def sine(freq, amp, start, seconds):
    t = np.arange(int(seconds * FS)) / FS
    return Trace(amp * np.sin(2 * np.pi * freq * t), {"sampling_rate": FS, "starttime": start})


def test_band_amplitudes_pick_the_right_band():
    df = tenmin_features(Stream([sine(3.0, 2.0, DAY, 86400)]), DAY)
    assert len(df) == 144
    mid = df.iloc[10:-10]
    # mean |A sin| is 2A/pi
    assert np.allclose(mid["rsam"], 2 * 2.0 / np.pi, rtol=0.02)
    assert (mid["hf"] < 0.01 * mid["rsam"]).all()


def test_sensitivity_scales_amplitude():
    st = Stream([sine(3.0, 2.0, DAY, 3600)])
    a = tenmin_features(st, DAY)["rsam"].iloc[2]
    b = tenmin_features(st, DAY, sensitivity=10.0)["rsam"].iloc[2]
    assert np.isclose(a, 10 * b)


def test_dsar_is_higher_for_lower_frequency_signal():
    low = tenmin_features(Stream([sine(6.0, 1.0, DAY, 3600)]), DAY)["dsar"].iloc[2]
    high = tenmin_features(Stream([sine(12.0, 1.0, DAY, 3600)]), DAY)["dsar"].iloc[2]
    assert low > 10 * high


def test_gaps_become_nan_not_zero():
    # data for hours 0-1 and 3-4 only
    st = Stream([sine(3.0, 1.0, DAY, 3600), sine(3.0, 1.0, DAY + 3 * 3600, 3600)])
    df = tenmin_features(st, DAY)
    assert df["rsam"].iloc[2] > 0.5
    assert np.isnan(df["rsam"].iloc[9])  # 01:30, inside the gap
    assert df["rsam"].iloc[20] > 0.5
    daily = daily_features(df)
    assert np.isnan(daily["rsam_mean"].iloc[0])  # 12 of 144 windows is under 50%
    assert np.isclose(daily["seis_coverage"].iloc[0], 12 / 144)


def test_empty_stream_gives_all_nan_day():
    df = tenmin_features(Stream(), DAY)
    assert len(df) == 144 and df["rsam"].isna().all()
