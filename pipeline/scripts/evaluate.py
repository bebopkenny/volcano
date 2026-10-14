"""Leave-one-case-out evaluation of Baseline 0, Model A and Model B.

Usage (from pipeline/): python -m scripts.evaluate
Reads data/interim/{case}/daily.csv (run scripts.build first) and writes
data/results/predictions_{N}d.csv and metrics_{N}d.json for N = 2 and 7.
"""
import json

import pandas as pd

from volcano.config import INTERIM, ROOT, load_cases
from volcano.models import MODELS, evaluate

RESULTS = ROOT / "data" / "results"


def load_days(cases: dict) -> pd.DataFrame:
    df = pd.concat(pd.read_csv(INTERIM / cid / "daily.csv", parse_dates=["date"]) for cid in cases)
    # onset day and later are not forecasting targets
    return df[df["pre_onset"]].reset_index(drop=True)


def report(m: dict) -> None:
    print(f"\n=== horizon {m['horizon_days']} days: {m['n_days']} days, {m['n_positive_days']} positive ===")
    print(pd.DataFrame(m["overall"]).T.to_string())
    print("\n95% CI (bootstrap over cases):", m["bootstrap_95ci"])
    rows = {(c, k): v for c, per in m["per_case"].items() for k, v in per.items()}
    print("\n" + pd.DataFrame(rows).T.drop(columns="threshold").to_string())
    print("\nModel B thermal share of importance:", m["model_b_importance"]["thermal_share"])
    print("top thermal:", m["model_b_importance"]["top_thermal"])


if __name__ == "__main__":
    cases = load_cases()
    df = load_days(cases)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for n in (2, 7):
        pred, metrics = evaluate(df, cases, n)
        pred.to_csv(RESULTS / f"predictions_{n}d.csv", index=False)
        (RESULTS / f"metrics_{n}d.json").write_text(json.dumps(metrics, indent=2))
        report(metrics)
    print(f"\nModels: {', '.join(MODELS)}. Results in {RESULTS}")
