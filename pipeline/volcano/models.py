"""Baseline 0, Model A (seismic) and Model B (seismic + thermal), scored leave-one-case-out.

Each case is scored by models that never saw it, and the alert threshold for that
case is chosen from the other cases only. Days are never shuffled across cases.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

THERMAL_PREFIX = ("frp_", "n_det", "no_detection", "days_since_det")
BASELINE_COL = "rsam_median_rel"   # log10 of today's RSAM over the station's normal
SEED = 0
MODELS = ("0", "A", "B")


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    # only station-relative seismic features: raw amplitude is not comparable between volcanoes
    seismic = [c for c in df.columns if "_rel" in c]
    thermal = [c for c in df.columns if c.startswith(THERMAL_PREFIX)]
    return {"A": seismic, "B": seismic + thermal}


def make_model() -> RandomForestClassifier:
    # identical for A and B; balanced weights because only about 1 day in 100 is positive
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)


def baseline_score(df: pd.DataFrame) -> pd.Series:
    # the first week has no station baseline yet; -10 (a 10^-10 ratio) never alarms
    return df[BASELINE_COL].fillna(-10.0)


def best_threshold(y: pd.Series, score: pd.Series) -> float:
    """Threshold with the best F1. Ties go to the highest threshold, which raises fewer alarms."""
    p, r, t = precision_recall_curve(y, score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)[:-1]
    return float(t[len(f1) - 1 - np.argmax(f1[::-1])])


def _loco_proba(df: pd.DataFrame, cols: list[str], label: str) -> pd.Series:
    """Probability for every row from a model trained on all other cases."""
    out = pd.Series(np.nan, index=df.index)
    for case in df["case"].unique():
        test = df["case"] == case
        m = make_model().fit(df.loc[~test, cols], df.loc[~test, label])
        out[test] = m.predict_proba(df.loc[test, cols])[:, 1]
    return out


def predict(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Held-out score, threshold and alarm for each model on every row.

    `df` holds pre-onset days of all cases with a fresh RangeIndex.
    """
    feats = feature_sets(df)
    out = df[["case", "date", label]].copy()
    for case in df["case"].unique():
        test = df["case"] == case
        train = df[~test]
        s = baseline_score(df)
        out.loc[test, "score_0"] = s[test]
        out.loc[test, "thr_0"] = best_threshold(train[label], s[~test])
        for name, cols in feats.items():
            # threshold from scores the training cases got while held out themselves
            inner = _loco_proba(train, cols, label)
            out.loc[test, f"thr_{name}"] = best_threshold(train[label], inner)
            m = make_model().fit(train[cols], train[label])
            out.loc[test, f"score_{name}"] = m.predict_proba(df.loc[test, cols])[:, 1]
    for name in MODELS:
        out[f"alarm_{name}"] = out[f"score_{name}"] >= out[f"thr_{name}"]
    return out


def case_metrics(p: pd.DataFrame, label: str, onset: str | None, model: str) -> dict:
    y, score, alarm = p[label], p[f"score_{model}"], p[f"alarm_{model}"]
    neg = y == 0
    m = {"threshold": round(float(p[f"thr_{model}"].iloc[0]), 4),
         "false_alarm_days": int((alarm & neg).sum()),
         "false_alarms_per_year": round(float((alarm & neg).sum() / neg.sum() * 365), 1)}
    if onset:
        hit = alarm & (y == 1)
        # a forecast for day D is issued at the end of D (midnight UTC)
        issued = pd.Timestamp(p.loc[hit, "date"].min()) + pd.Timedelta(days=1) if hit.any() else None
        m |= {"detected": bool(hit.any()),
              "lead_days": round((pd.Timestamp(onset).tz_localize(None) - issued) / pd.Timedelta(days=1), 2)
              if issued is not None else None,
              "roc_auc": round(float(roc_auc_score(y, score)), 3),
              "pr_auc": round(float(average_precision_score(y, score)), 3)}
    return m


def bootstrap_auc(pred: pd.DataFrame, label: str, n: int = 2000) -> dict:
    """95% intervals for pooled ROC/PR AUC by resampling whole cases, plus B minus A."""
    rng = np.random.default_rng(SEED)
    groups = [g for _, g in pred.groupby("case")]
    draws = {k: [] for k in ("roc_0", "roc_A", "roc_B", "pr_A", "pr_B", "roc_B_minus_A", "pr_B_minus_A")}
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if s[label].nunique() < 2:
            continue
        for k in ("0", "A", "B"):
            draws[f"roc_{k}"].append(roc_auc_score(s[label], s[f"score_{k}"]))
        for k in ("A", "B"):
            draws[f"pr_{k}"].append(average_precision_score(s[label], s[f"score_{k}"]))
        draws["roc_B_minus_A"].append(draws["roc_B"][-1] - draws["roc_A"][-1])
        draws["pr_B_minus_A"].append(draws["pr_B"][-1] - draws["pr_A"][-1])
    return {k: [round(float(q), 3) for q in np.percentile(v, [2.5, 97.5])] for k, v in draws.items()}


def evaluate(df: pd.DataFrame, cases: dict, horizon: int) -> tuple[pd.DataFrame, dict]:
    label = f"label_{horizon}d"
    pred = predict(df, label)
    per_case = {cid: {k: case_metrics(p, label, cases[cid].get("onset"), k) for k in MODELS}
                for cid, p in pred.groupby("case", sort=False)}
    eruptions = [c for c in per_case if cases[c].get("onset")]
    unrest = pred[pred["case"].map(lambda c: not cases[c].get("onset"))]
    overall = {}
    for k in MODELS:
        leads = [per_case[c][k]["lead_days"] for c in eruptions if per_case[c][k]["detected"]]
        overall[k] = {
            "roc_auc": round(float(roc_auc_score(pred[label], pred[f"score_{k}"])), 3),
            "pr_auc": round(float(average_precision_score(pred[label], pred[f"score_{k}"])), 3),
            "eruptions_detected": f"{len(leads)}/{len(eruptions)}",
            "mean_lead_days": round(float(np.mean(leads)), 2) if leads else None,
            "unrest_false_alarms_per_year": round(float(unrest[f"alarm_{k}"].sum() / len(unrest) * 365), 1),
        }
    # importance from Model B fit on every case; impurity based, so treat it as a rough ranking
    cols = feature_sets(df)["B"]
    imp = pd.Series(make_model().fit(df[cols], df[label]).feature_importances_, index=cols)
    thermal = imp[[c for c in cols if c.startswith(THERMAL_PREFIX)]]
    importance = {"thermal_share": round(float(thermal.sum()), 3),
                  "top": {k: round(float(v), 4) for k, v in imp.nlargest(10).items()},
                  "top_thermal": {k: round(float(v), 4) for k, v in thermal.nlargest(5).items()}}
    metrics = {"horizon_days": horizon, "overall": overall, "bootstrap_95ci": bootstrap_auc(pred, label),
               "per_case": per_case, "model_b_importance": importance,
               "n_days": len(pred), "n_positive_days": int(pred[label].sum())}
    return pred, metrics
