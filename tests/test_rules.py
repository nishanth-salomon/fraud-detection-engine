import pandas as pd

from fraud_engine.features import compute_features
from fraud_engine.rules import RULES, apply_rules

COLS = ["txn_id", "ts", "customer_id", "merchant_id", "merchant_cat", "amount", "lat", "lon",
        "channel", "is_fraud", "fraud_type"]


def test_impossible_travel_rule_fires_with_reason():
    t0 = pd.Timestamp("2026-03-01 12:00")
    rows = [["T0", t0, "C1", "M1", "dining", 400.0, 13.0827, 80.2707, "pos", 0, ""],
            ["T1", t0 + pd.Timedelta(minutes=45), "C1", "M2", "dining", 400.0, 28.6139, 77.2090, "pos", 1, "x"]]
    r = apply_rules(compute_features(pd.DataFrame(rows, columns=COLS)))
    assert bool(r.loc[1, "R_IMPOSSIBLE_TRAVEL"]) and not bool(r.loc[0, "R_IMPOSSIBLE_TRAVEL"])
    assert "R_IMPOSSIBLE_TRAVEL" in r.loc[1, "rule_reasons"]
    assert r.loc[1, "rule_score"] >= RULES["R_IMPOSSIBLE_TRAVEL"][0] - 1e-9


def test_quiet_customer_triggers_nothing():
    t0 = pd.Timestamp("2026-03-01 12:00")
    rows = [[f"T{i}", t0 + pd.Timedelta(days=i), "C1", "M1", "grocery", 300.0, 13.0, 80.0, "pos", 0, ""]
            for i in range(8)]
    r = apply_rules(compute_features(pd.DataFrame(rows, columns=COLS)))
    assert (r["rule_score"] == 0).all() and (r["rule_reasons"] == "").all()


def test_rule_score_is_bounded(feat):
    r = apply_rules(feat)
    assert r["rule_score"].between(0, 1).all()
