"""Build one daily feature table per case from the cached downloads.

Usage (from pipeline/): python -m scripts.build [case-id ...]
Writes data/interim/{case}/daily.csv and a sanity plot next to it.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from volcano.config import INTERIM, load_cases
from volcano.dataset import build_case
from volcano.seismic import daily_features, fetch_case
from volcano.thermal import HISTORY_DAYS, RAW, daily_thermal, fetch_detections


def build(case: dict) -> pd.DataFrame:
    seis = daily_features(fetch_case(case))
    thermal = None
    if (RAW / case["id"]).exists():
        dates = pd.date_range(pd.Timestamp(case["window_start"]) - pd.Timedelta(days=HISTORY_DAYS),
                              case["window_end"], freq="D")
        thermal = daily_thermal(fetch_detections(case), case["lat"], case["lon"], dates)
    else:
        print(f"{case['id']}: no FIRMS download yet, building without thermal features")
    df = build_case(case, seis, thermal)
    out = INTERIM / case["id"]
    df.to_csv(out / "daily.csv")

    panels = ["rsam_median"] + (["frp_max"] if thermal is not None else [])
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)), sharex=True, squeeze=False)
    for ax, col in zip(axes[:, 0], panels):
        ax.plot(df.index, df[col])
        ax.set_ylabel(col)
        if col == "rsam_median":
            ax.set_yscale("log")
        if case.get("onset"):
            ax.axvline(pd.Timestamp(case["onset"]).tz_localize(None), color="red")
    axes[0, 0].set_title(f"{case['id']} (red line: eruption onset)")
    fig.savefig(out / "sanity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return df


if __name__ == "__main__":
    cases = load_cases()
    for case_id in sys.argv[1:] or cases:
        df = build(cases[case_id])
        print(f"{case_id}: {len(df)} days, {df.shape[1]} columns, "
              f"{int(df['label_2d'].sum())} positive days (2d), {int(df['label_7d'].sum())} (7d)")
