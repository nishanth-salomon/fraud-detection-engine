"""End-to-end run: generate -> features -> chronological split -> train -> choose threshold on
validation -> report on the untouched test period."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .evaluate import (best_threshold, chronological_split, cost_at_threshold,
                       precision_recall_at, summarise)
from .features import FEATURES, compute_features
from .models import FraudModel
from .synth import generate

REVIEW_COST = 150.0


def novel_pattern_experiment(train, test, seed, budget=0.01):
    """Stress test: what happens on a fraud pattern the supervised model has never been shown?

    For each fraud type, its labelled rows in the TRAINING period are relabelled as normal, the model is
    retrained, and recall is measured on that pattern in the test period at a fixed review budget
    (the top `budget` share of transactions). The isolation forest and the rules never use labels, so
    this shows how much protection they add when the labelled history is incomplete.
    """
    rows = {}
    for ft in sorted(train.loc[train.is_fraud == 1, "fraud_type"].unique()):
        tr = train.copy()
        tr.loc[tr.fraud_type == ft, "is_fraud"] = 0
        model = FraudModel(seed=seed).fit(tr)
        sub = test[(test.is_fraud == 0) | (test.fraud_type == ft)]
        sc = model.score(sub)
        y = sub.is_fraud.to_numpy()
        rows[ft] = {name: precision_recall_at(y, sc[col], budget)[1]
                    for name, col in (("rules", "rules"), ("isolation forest", "iforest"),
                                      ("gradient boosting", "gbm"), ("blend", "score"),
                                      ("max-fusion", "score_max"))}
        rows[ft]["n_test_fraud"] = int(y.sum())
    return rows


def run(n_customers=600, days=90, seed=7, out_dir="artifacts", review_cost=REVIEW_COST, verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    raw = generate(n_customers=n_customers, days=days, seed=seed)
    feat = compute_features(raw)
    train, val, test = chronological_split(feat)
    log(f"rows: train={len(train):,}  val={len(val):,}  test={len(test):,} | "
        f"fraud rate: train={train.is_fraud.mean():.3%} test={test.is_fraud.mean():.3%}")

    model = FraudModel(seed=seed).fit(train)
    val_s, test_s = model.score(val), model.score(test)

    # threshold chosen on validation only; the test period is never used for tuning
    cut = best_threshold(val.is_fraud, val_s["score"], val.amount, review_cost)
    test_cost = cost_at_threshold(test.is_fraud, test_s["score"], test.amount, cut["threshold"], review_cost)

    components = {
        "rules only": test_s["rules"], "isolation forest only": test_s["iforest"],
        "gradient boosting only": test_s["gbm"], "blend (final)": test_s["score"],
        "max-fusion (alternative)": test_s["score_max"],
    }
    metrics = {name: summarise(test.is_fraud, s) for name, s in components.items()}

    perm = permutation_importance(
        model.gbm, val[FEATURES].to_numpy(dtype=float), val.is_fraud.to_numpy(dtype=int),
        scoring="average_precision", n_repeats=3, random_state=seed)
    importance = sorted(zip(FEATURES, perm.importances_mean), key=lambda x: -x[1])

    by_type = {}
    for ft, g in test[test.is_fraud == 1].groupby("fraud_type"):
        flagged = test_s.loc[g.index, "score"] >= cut["threshold"]
        by_type[ft] = {"count": int(len(g)), "caught": int(flagged.sum()),
                       "recall": float(flagged.mean())}

    novel = novel_pattern_experiment(train, test, seed)

    alerts = test.join(test_s).sort_values("score", ascending=False).head(50).copy()
    alerts["outliers"] = model.outlier_features(alerts)
    alerts[["txn_id", "ts", "customer_id", "merchant_cat", "amount", "channel", "score", "gbm", "iforest",
            "rules", "rule_reasons", "outliers", "is_fraud", "fraud_type"]].to_csv(out / "top_alerts.csv", index=False)

    result = {
        "config": {"n_customers": n_customers, "days": days, "seed": seed, "review_cost": review_cost},
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "threshold_from_validation": cut,
        "test_at_threshold": test_cost,
        "test_metrics": metrics,
        "recall_by_fraud_type": by_type,
        "novel_pattern_recall_at_1pct": novel,
        "top_features": [{"feature": f, "importance": float(v)} for f, v in importance[:8]],
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    model.save(out / "model.joblib")
    (out / "report.md").write_text(render_report(result))
    return result


def render_report(r: dict) -> str:
    t, m = r["test_at_threshold"], r["test_metrics"]
    lines = [
        "# Fraud engine run report", "",
        "> All data is **synthetic**. Numbers describe how the engine behaves on generated traffic, "
        "not how it would perform on a real bank's data.", "",
        f"- Test period rows: {r['rows']['test']:,} (train {r['rows']['train']:,}, validation {r['rows']['val']:,})",
        f"- Alert threshold: {r['threshold_from_validation']['threshold']:.3f} (chosen on validation, "
        f"review cost Rs {r['config']['review_cost']:.0f} per alert)", "",
        "## Business result on the test period", "",
        f"- Alerts raised: {t['alerts']:,} | fraud caught: {t['fraud_caught']} of {t['fraud_total']} "
        f"(recall {t['recall']:.1%}, precision {t['precision']:.1%})",
        f"- Fraud loss with no system: Rs {t['baseline_cost']:,.0f}",
        f"- Loss plus review cost with this system: Rs {t['cost']:,.0f}",
        f"- **Net saving: Rs {t['saved']:,.0f}**", "",
        "## Ranking quality (test period)", "",
        "| Signal | PR-AUC | ROC-AUC | Precision@1% | Recall@1% |", "|---|---|---|---|---|",
    ]
    for name, s in m.items():
        lines.append(f"| {name} | {s['pr_auc']:.3f} | {s['roc_auc']:.3f} | "
                     f"{s['precision@1.0%']:.1%} | {s['recall@1.0%']:.1%} |")
    lines += ["", f"Base rate (random guessing PR-AUC) = {m['blend (final)']['base_rate']:.3%}", "",
              "## Recall by fraud pattern (at the chosen threshold)", "",
              "| Pattern | Count | Caught | Recall |", "|---|---|---|---|"]
    for ft, v in r["recall_by_fraud_type"].items():
        lines.append(f"| {ft} | {v['count']} | {v['caught']} | {v['recall']:.1%} |")
    lines += ["", "## Stress test: a fraud pattern the labels never showed", "",
              "Each row retrains with that pattern's training labels hidden, then measures recall in the test "
              "period when only the top 1% of transactions can be reviewed.", "",
              "| Hidden pattern | Rules | Isolation forest | Gradient boosting | Blend | Max-fusion |",
              "|---|---|---|---|---|---|"]
    for ft, v in r["novel_pattern_recall_at_1pct"].items():
        lines.append(f"| {ft} | {v['rules']:.1%} | {v['isolation forest']:.1%} | "
                     f"{v['gradient boosting']:.1%} | {v['blend']:.1%} | {v['max-fusion']:.1%} |")
    lines += ["", "## What the gradient-boosting model relies on", ""]
    lines += [f"- `{x['feature']}`: {x['importance']:.4f}" for x in r["top_features"]]
    return "\n".join(lines) + "\n"


def score_csv(model_path, input_csv, output_csv, threshold=None):
    """Score a CSV of transactions. History for each customer comes from the file itself."""
    model = FraudModel.load(model_path)
    raw = pd.read_csv(input_csv, parse_dates=["ts"])
    feat = compute_features(raw)
    scored = model.score(feat)
    result = feat[["txn_id", "ts", "customer_id", "amount"]].join(scored)
    result["outliers"] = model.outlier_features(feat)
    if threshold is not None:
        result["flag"] = result["score"] >= threshold
    result.sort_values("score", ascending=False).to_csv(output_csv, index=False)
    return result
