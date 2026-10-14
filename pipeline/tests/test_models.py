import numpy as np
import pandas as pd

from volcano.models import RUNS, best_threshold, case_metrics, predict


def test_best_threshold():
    y = pd.Series([0, 0, 1, 1])
    assert best_threshold(y, pd.Series([0.1, 0.2, 0.6, 0.9])) == 0.6
    # 0.2 (both hits, two false alarms) and 0.9 (one hit, none) tie on F1; take the one raising fewer alarms
    assert best_threshold(pd.Series([1, 0, 0, 1]), pd.Series([0.2, 0.3, 0.4, 0.9])) == 0.9


def test_case_metrics_lead_time_and_false_alarms():
    dates = pd.date_range("2022-01-01", periods=5)
    p = pd.DataFrame({"date": dates, "label_2d": [0, 0, 0, 1, 1], "score_A": [0.9, 0, 0, 0.8, 0.9],
                      "thr_A": 0.5, "alarm_A": [True, False, False, True, True]})
    m = case_metrics(p, "label_2d", "2022-01-06T12:00:00Z", "A")
    # first alarm on a positive day is Jan 4, issued at midnight starting Jan 5: 1.5 days before onset
    assert m["detected"] and m["lead_days"] == 1.5
    assert m["false_alarm_days"] == 1


def test_held_out_case_never_seen_in_training():
    """Scores for a case must not change when that case's labels change."""
    rng = np.random.default_rng(0)
    rows = []
    for case in "abc":
        y = np.zeros(30, int)
        y[-3:] = 1
        rows.append(pd.DataFrame({"case": case, "date": pd.date_range("2022-01-01", periods=30),
                                  "rsam_median_rel": rng.normal(size=30) + y, "frp_max": rng.normal(size=30),
                                  "label_2d": y}))
    df = pd.concat(rows, ignore_index=True)
    for c in ("rsam_median_rel_trend7", "dsar_median_rel", "dsar_median_rel_trend7", "frp_max_trend30"):
        df[c] = rng.normal(size=len(df))
    df["frp_max"] = df["frp_max"].abs()
    flipped = df.copy()
    flipped.loc[flipped["case"] == "a", "label_2d"] = 1 - flipped["label_2d"]
    a = df["case"] == "a"
    for run in RUNS:
        before, after = predict(df, "label_2d", run), predict(flipped, "label_2d", run)
        for k in ("A", "B"):
            pd.testing.assert_series_equal(before.loc[a, f"score_{k}"], after.loc[a, f"score_{k}"])
            pd.testing.assert_series_equal(before.loc[a, f"thr_{k}"], after.loc[a, f"thr_{k}"])
