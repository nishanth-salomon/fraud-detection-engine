import numpy as np
import pandas as pd

from fraud_engine.features import FEATURES, compute_features, haversine_km

COLS = ["txn_id", "ts", "customer_id", "merchant_id", "merchant_cat",
        "amount", "lat", "lon", "channel", "is_fraud", "fraud_type"]


def _tiny(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_haversine_chennai_mumbai():
    d = haversine_km(13.0827, 80.2707, 19.0760, 72.8777)
    assert 1000 < d < 1060                     # about 1,030 km


def test_no_future_leakage(raw):
    """Features of earlier rows must not change when later rows are added."""
    cut = int(len(raw) * 0.6)
    prefix = compute_features(raw.iloc[:cut].copy())
    full = compute_features(raw.copy())
    np.testing.assert_allclose(prefix[FEATURES].to_numpy(),
                               full.iloc[:cut][FEATURES].to_numpy(), rtol=1e-9, atol=1e-9)


def test_future_edits_do_not_move_past_features(raw):
    cut = int(len(raw) * 0.6)
    tampered = raw.copy()
    tampered.loc[tampered.index >= cut, "amount"] *= 1000     # corrupt only the future
    before = compute_features(raw)[FEATURES].iloc[:cut].to_numpy()
    after = compute_features(tampered)[FEATURES].iloc[:cut].to_numpy()
    np.testing.assert_allclose(before, after)


def test_row_order_preserved_and_no_nans(raw, feat):
    assert list(feat["txn_id"]) == list(raw["txn_id"])
    assert feat[FEATURES].notna().all().all()


def test_velocity_probe_and_novelty_counts():
    t0 = pd.Timestamp("2026-02-01 10:00")
    rows = [[f"T{i}", t0 + pd.Timedelta(minutes=i), "C1", f"M{i}", "fuel", 10.0, 13.0, 80.0, "online", 0, ""]
            for i in range(6)]
    f = compute_features(_tiny(rows))
    assert list(f["count_1h"]) == [0, 1, 2, 3, 4, 5]
    assert list(f["small_count_1h"]) == [0, 1, 2, 3, 4, 5]
    assert list(f["new_merchant"]) == [1.0] * 6


def test_impossible_travel_speed():
    t0 = pd.Timestamp("2026-02-01 10:00")
    rows = [["T0", t0, "C1", "M1", "fuel", 500.0, 13.0827, 80.2707, "pos", 0, ""],
            ["T1", t0 + pd.Timedelta(hours=1), "C1", "M2", "fuel", 500.0, 19.0760, 72.8777, "pos", 0, ""]]
    f = compute_features(_tiny(rows))
    assert f.loc[1, "dist_prev_km"] > 1000
    assert f.loc[1, "speed_kmh"] > 900
    assert f.loc[0, "speed_kmh"] == 0


def test_baseline_excludes_the_transaction_itself():
    """A spike must be judged against the customer's PAST, not against a baseline that already contains it."""
    t0 = pd.Timestamp("2026-02-01 10:00")
    rows = [[f"T{i}", t0 + pd.Timedelta(days=i), "C1", "M1", "grocery", 100.0, 13.0, 80.0, "pos", 0, ""]
            for i in range(5)]
    rows.append(["T5", t0 + pd.Timedelta(days=5), "C1", "M1", "grocery", 5000.0, 13.0, 80.0, "pos", 1, "x"])
    f = compute_features(_tiny(rows))
    assert abs(f.loc[5, "amt_ratio"] - 5000.0 / 101.0) < 1e-9      # prior mean is exactly 100
    assert f.loc[5, "amt_z"] > 4


def test_cold_start_amount_signals_are_neutral():
    t0 = pd.Timestamp("2026-02-01 10:00")
    f = compute_features(_tiny([["T0", t0, "C1", "M1", "fuel", 99999.0, 13.0, 80.0, "pos", 0, ""]]))
    assert f.loc[0, "amt_z"] == 0 and f.loc[0, "amt_ratio"] == 1
