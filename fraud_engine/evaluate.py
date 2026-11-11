"""Evaluation that matches how fraud teams work: a limited review budget and unequal error costs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def chronological_split(df, train_frac=0.70, val_frac=0.10):
    """Split by time, never at random, so the test set is genuinely 'the future'."""
    ts = df["ts"].sort_values().reset_index(drop=True)
    t_train = ts.iloc[int(len(ts) * train_frac) - 1]
    t_val = ts.iloc[int(len(ts) * (train_frac + val_frac)) - 1]
    train = df[df["ts"] <= t_train]
    val = df[(df["ts"] > t_train) & (df["ts"] <= t_val)]
    test = df[df["ts"] > t_val]
    return train, val, test


def precision_recall_at(y, score, fraction):
    """Review the top `fraction` of transactions by score: what share of them is fraud, what share of fraud is caught."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(len(y) * fraction)))
    top = np.argsort(-score, kind="mergesort")[:k]
    caught = y[top].sum()
    return float(caught / k), float(caught / max(y.sum(), 1)), k


def best_threshold(y, score, amount, review_cost=150.0):
    """Pick the alert cut-off that minimises money lost.

    Cost of a miss  = the fraudulent amount that goes through.
    Cost of an alert = a fixed analyst review cost (applies to false and true alerts alike).
    Baseline        = flag nothing (lose every fraudulent amount).
    """
    y, score, amount = np.asarray(y), np.asarray(score), np.asarray(amount, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y_s, a_s, s_s = y[order], amount[order], score[order]
    total_fraud = float((a_s * y_s).sum())
    caught_amt = np.concatenate([[0.0], np.cumsum(a_s * y_s)])
    k = np.arange(len(y_s) + 1)
    cost = (total_fraud - caught_amt) + review_cost * k
    best = int(np.argmin(cost))
    threshold = float(s_s[best - 1]) if best > 0 else float(s_s[0]) + 1e-9
    return {
        "threshold": threshold, "alerts": best, "cost": float(cost[best]),
        "baseline_cost": total_fraud, "saved": float(total_fraud - cost[best]),
    }


def cost_at_threshold(y, score, amount, threshold, review_cost=150.0):
    y, score, amount = np.asarray(y), np.asarray(score), np.asarray(amount, dtype=float)
    flagged = score >= threshold
    missed = float((amount * y * (~flagged)).sum())
    caught = float((amount * y * flagged).sum())
    cost = missed + review_cost * int(flagged.sum())
    baseline = float((amount * y).sum())
    return {
        "alerts": int(flagged.sum()), "fraud_caught": int((y * flagged).sum()),
        "fraud_total": int(y.sum()), "amount_caught": caught, "amount_missed": missed,
        "cost": cost, "baseline_cost": baseline, "saved": baseline - cost,
        "precision": float((y * flagged).sum() / max(flagged.sum(), 1)),
        "recall": float((y * flagged).sum() / max(y.sum(), 1)),
    }


def summarise(y, score):
    y = np.asarray(y)
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "base_rate": float(y.mean()),
    }
    for frac in (0.005, 0.01, 0.02):
        p, r, k = precision_recall_at(y, score, frac)
        out[f"precision@{frac:.1%}"] = p
        out[f"recall@{frac:.1%}"] = r
    return out
