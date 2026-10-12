import numpy as np
import pandas as pd

from volcano.dataset import build_case, seismic_features, thermal_features
from volcano.labels import label_days
from volcano.thermal import daily_thermal

DATES = pd.date_range("2022-01-01", periods=60, freq="D")
RNG = np.random.default_rng(0)


def fake_seismic():
    cols = ["rsam_mean", "rsam_median", "rsam_max", "mf_mean", "hf_mean", "dsar_median"]
    df = pd.DataFrame(RNG.lognormal(size=(60, 6)), index=DATES, columns=cols)
    df["seis_coverage"] = 1.0
    return df


def fake_thermal():
    det = pd.DataFrame({
        "latitude": 10.0, "longitude": 20.0,
        "acq_date": RNG.choice(DATES.strftime("%Y-%m-%d"), 40),
        "frp": RNG.uniform(1, 50, 40),
    })
    return daily_thermal(det, 10.0, 20.0, DATES)


def test_features_never_look_forward():
    """Changing days after `cut` must not change any feature on or before `cut`."""
    cut = 40
    for make, feats in ((fake_seismic, seismic_features), (fake_thermal, thermal_features)):
        base = make()
        changed = base.copy()
        changed.iloc[cut + 1:] = changed.iloc[cut + 1:] * 100 + 7
        a, b = feats(base).iloc[:cut + 1], feats(changed).iloc[:cut + 1]
        pd.testing.assert_frame_equal(a, b)
        # and the test has teeth: the future rows do differ
        assert not feats(base).iloc[cut + 1:].equals(feats(changed).iloc[cut + 1:])


def test_seismic_features_ignore_station_scale():
    base = fake_seismic()
    a = seismic_features(base)
    b = seismic_features(base.assign(**{c: base[c] * 1000 for c in base.columns if c != "seis_coverage"}))
    pd.testing.assert_frame_equal(a, b, atol=1e-9)


def test_labels_around_onset():
    dates = pd.date_range("2022-11-20", "2022-12-01", freq="D")
    label, pre = label_days(dates, "2022-11-28T09:30:00Z", 2)
    assert label[label == 1].index.strftime("%m-%d").tolist() == ["11-26", "11-27"]
    assert pre["2022-11-27"] and not pre["2022-11-28"] and not pre["2022-12-01"]
    label7, _ = label_days(dates, "2022-11-28T09:30:00Z", 7)
    assert label7.sum() == 7


def test_unrest_case_is_all_negative():
    label, pre = label_days(DATES, None, 2)
    assert label.sum() == 0 and pre.all()


def test_thermal_radius_and_gaps():
    det = pd.DataFrame({
        "latitude": [10.0, 10.0, 10.5], "longitude": [20.0, 20.01, 20.0],
        "acq_date": ["2022-01-03", "2022-01-03", "2022-01-04"], "frp": [5.0, 9.0, 99.0],
    })
    out = daily_thermal(det, 10.0, 20.0, DATES[:6])
    assert out.loc["2022-01-03", ["frp_max", "frp_sum", "n_det"]].tolist() == [9.0, 14.0, 2.0]
    assert out.loc["2022-01-04", "n_det"] == 0  # 55 km away, outside the radius
    assert out["no_detection"].tolist() == [1, 1, 0, 1, 1, 1]
    assert out["days_since_det"].tolist() == [60, 60, 0, 1, 2, 3]


def test_build_case_shape():
    case = {"id": "x", "window_start": "2022-01-10", "window_end": "2022-02-20", "onset": "2022-02-15T12:00:00Z"}
    df = build_case(case, fake_seismic(), fake_thermal())
    assert len(df) == 42 and df["label_2d"].sum() == 2 and df["label_7d"].sum() == 7
    assert not df.loc["2022-02-15":, "pre_onset"].any()
