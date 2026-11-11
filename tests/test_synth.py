import pandas as pd

from fraud_engine.synth import FRAUD_TYPES, generate


def test_deterministic_for_a_seed():
    a = generate(n_customers=40, days=20, seed=11)
    b = generate(n_customers=40, days=20, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data():
    a = generate(n_customers=40, days=20, seed=11)
    b = generate(n_customers=40, days=20, seed=12)
    assert not a["amount"].equals(b["amount"])


def test_fraud_rate_and_types(raw):
    rate = raw["is_fraud"].mean()
    assert 0.004 < rate < 0.02
    assert set(raw.loc[raw.is_fraud == 1, "fraud_type"]) <= set(FRAUD_TYPES)
    assert (raw.loc[raw.is_fraud == 0, "fraud_type"] == "").all()


def test_sorted_by_time_and_unique_ids(raw):
    assert raw["ts"].is_monotonic_increasing
    assert raw["txn_id"].is_unique
    assert (raw["amount"] > 0).all()
