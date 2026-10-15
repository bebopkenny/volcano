"""Export small JSON files for the web app (data contract in PROJECT_BRIEF.md section 7).

Usage (from pipeline/): python -m scripts.export
Reads daily tables and data/results (run scripts.build and scripts.evaluate first) and
writes public/data/cases.json, public/data/metrics.json and public/data/timeseries/*.json.
"""
import json
import math

import pandas as pd

from scripts.evaluate import RESULTS
from volcano.config import INTERIM, ROOT, load_cases
from volcano.models import RUNS

OUT = ROOT.parent / "public" / "data"
PRIMARY = ("rf", 2)   # the brief's planned model and horizon; drives the timeseries
VARIANTS = ("", "_by_volcano")


def _clean(values, digits: int) -> list:
    # JSON has no NaN; round so the files stay small
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits) for v in values]


def export_cases(cases: dict) -> list[dict]:
    return [{"id": c["id"], "volcano": c["volcano"], "region": c["region"], "lat": c["lat"], "lon": c["lon"],
             "type": c["type"], "onset": c.get("onset"), "windowStart": c["window_start"],
             "windowEnd": c["window_end"], "station": f"{c['network']}.{c['station']}", "sources": c["sources"]}
            for c in cases.values()]


def export_timeseries(case_id: str, pred: pd.DataFrame) -> dict:
    daily = pd.read_csv(INTERIM / case_id / "daily.csv", parse_dates=["date"])
    # predictions exist only for pre-onset days; later days get null
    p = daily[["date"]].merge(pred[pred["case"] == case_id], on="date", how="left")
    assert len(p) == len(daily), f"{case_id}: duplicate prediction dates"
    run, horizon = PRIMARY
    return {"run": run, "horizonDays": horizon,
            "dates": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
            "rsam": [None if math.isnan(v) else float(f"{v:.3e}") for v in daily["rsam_median"]],
            "frpMax": _clean(daily["frp_max"], 2),
            "probSeismic": _clean(p["score_A"], 4),
            "probSeismicThermal": _clean(p["score_B"], 4),
            # significant figures, not decimals: some thresholds are below 0.0001
            "thresholdSeismic": float(f"{p['thr_A'].dropna().iloc[0]:.3g}"),
            "thresholdSeismicThermal": float(f"{p['thr_B'].dropna().iloc[0]:.3g}")}


if __name__ == "__main__":
    cases = load_cases()
    (OUT / "timeseries").mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(export_cases(cases), indent=1))

    metrics = {"primary": {"run": PRIMARY[0], "horizonDays": PRIMARY[1]}, "runs": {}}
    for run in RUNS:
        for v in VARIANTS:
            for n in (2, 7):
                metrics["runs"][f"{run}{v}_{n}d"] = json.loads((RESULTS / f"{run}{v}_metrics_{n}d.json").read_text())
    (OUT / "metrics.json").write_text(json.dumps(metrics, separators=(",", ":")))

    pred = pd.read_csv(RESULTS / f"{PRIMARY[0]}_predictions_{PRIMARY[1]}d.csv", parse_dates=["date"])
    for cid in cases:
        (OUT / "timeseries" / f"{cid}.json").write_text(json.dumps(export_timeseries(cid, pred), separators=(",", ":")))
    size = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"Wrote {len(cases)} cases to {OUT} ({size / 1024:.0f} KB)")
