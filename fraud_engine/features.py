"""Leakage-safe feature engineering.

Every feature for a transaction is computed from that transaction and the customer's *earlier*
transactions only. Appending later rows to the table must never change an earlier row's features
(`tests/test_features.py::test_no_future_leakage` enforces it). That is the property that makes
offline metrics a fair estimate of how the model behaves when it scores live traffic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 3            # below this many prior transactions, behaviour stats are not trusted
SMALL_AMOUNT = 100.0       # "probe-sized" transaction for the card-testing signal
FEATURES = [
    "log_amount", "hour_sin", "hour_cos", "is_night", "is_online", "is_atm",
    "n_prior", "amt_z", "amt_ratio", "secs_since_prev", "count_1h", "count_24h",
    "sum_24h_ratio", "small_count_1h", "new_merchant", "new_category",
    "dist_prev_km", "speed_kmh", "dist_home_km",
]
EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _window_counts(t: np.ndarray, window_s: float, cum: np.ndarray | None = None):
    """For each i, count (or sum) over earlier rows j < i with t[j] >= t[i] - window_s."""
    left = np.searchsorted(t, t - window_s, side="left")
    i = np.arange(len(t))
    left = np.minimum(left, i)                       # never reach past the current row
    if cum is None:
        return (i - left).astype(float)
    return cum[i] - cum[left]                        # cum has a leading 0, so cum[k] = sum of first k


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` (original row order) with FEATURES columns added."""
    work = df.copy()
    work["_ts"] = pd.to_datetime(work["ts"])
    work["_pos"] = np.arange(len(work))
    work = work.sort_values(["customer_id", "_ts", "_pos"], kind="mergesort").reset_index(drop=True)

    n = len(work)
    feats = {name: np.zeros(n) for name in FEATURES}
    secs = (work["_ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    amount = work["amount"].to_numpy(dtype=float)
    lat = work["lat"].to_numpy(dtype=float)
    lon = work["lon"].to_numpy(dtype=float)
    merch = work["merchant_id"].to_numpy()
    cats = work["merchant_cat"].to_numpy()

    for _, idx in work.groupby("customer_id", sort=False).indices.items():
        idx = np.asarray(idx)
        t, a = secs[idx], amount[idx]
        k = np.arange(len(idx), dtype=float)
        cs0 = np.concatenate([[0.0], np.cumsum(a)])
        csq = np.concatenate([[0.0], np.cumsum(a * a)])
        prior_n = k
        with np.errstate(divide="ignore", invalid="ignore"):
            prior_mean = np.where(k > 0, cs0[:-1] / np.maximum(k, 1), 0.0)
            prior_var = np.where(k > 0, csq[:-1] / np.maximum(k, 1) - prior_mean ** 2, 0.0)
        prior_std = np.sqrt(np.clip(prior_var, 0, None))
        trusted = prior_n >= MIN_HISTORY
        z = np.where(trusted, (a - prior_mean) / (prior_std + 0.25 * prior_mean + 1.0), 0.0)
        ratio = np.where(trusted, a / (prior_mean + 1.0), 1.0)

        prev_gap = np.full(len(idx), 7 * 86400.0)
        prev_gap[1:] = t[1:] - t[:-1]
        small_cum = np.concatenate([[0.0], np.cumsum((a < SMALL_AMOUNT).astype(float))])
        sum24 = _window_counts(t, 86400, cs0)

        seen_m, seen_c = set(), set()
        new_m = np.zeros(len(idx))
        new_c = np.zeros(len(idx))
        for j, row in enumerate(idx):
            m, c = merch[row], cats[row]
            new_m[j] = 0.0 if m in seen_m else 1.0
            new_c[j] = 0.0 if c in seen_c else 1.0
            seen_m.add(m)
            seen_c.add(c)

        la, lo = lat[idx], lon[idx]
        dist_prev = np.zeros(len(idx))
        dist_prev[1:] = haversine_km(la[:-1], lo[:-1], la[1:], lo[1:])
        hours_gap = np.maximum(prev_gap / 3600.0, 1.0 / 60.0)
        speed = np.where(k > 0, dist_prev / hours_gap, 0.0)
        cla, clo = np.concatenate([[0.0], np.cumsum(la)]), np.concatenate([[0.0], np.cumsum(lo)])
        home_lat = np.where(k > 0, cla[:-1] / np.maximum(k, 1), la)
        home_lon = np.where(k > 0, clo[:-1] / np.maximum(k, 1), lo)
        dist_home = np.where(k > 0, haversine_km(home_lat, home_lon, la, lo), 0.0)

        feats["n_prior"][idx] = prior_n
        feats["amt_z"][idx] = z
        feats["amt_ratio"][idx] = ratio
        feats["secs_since_prev"][idx] = prev_gap
        feats["count_1h"][idx] = _window_counts(t, 3600)
        feats["count_24h"][idx] = _window_counts(t, 86400)
        feats["sum_24h_ratio"][idx] = np.where(trusted, sum24 / (prior_mean + 1.0), 0.0)
        feats["small_count_1h"][idx] = _window_counts(t, 3600, small_cum)
        feats["new_merchant"][idx] = new_m
        feats["new_category"][idx] = new_c
        feats["dist_prev_km"][idx] = dist_prev
        feats["speed_kmh"][idx] = np.clip(speed, 0, 5000)
        feats["dist_home_km"][idx] = dist_home

    hour = work["_ts"].dt.hour.to_numpy() + work["_ts"].dt.minute.to_numpy() / 60.0
    feats["log_amount"] = np.log1p(amount)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["is_night"] = (hour < 5).astype(float)
    feats["is_online"] = (work["channel"].to_numpy() == "online").astype(float)
    feats["is_atm"] = (work["channel"].to_numpy() == "atm").astype(float)

    out = work.drop(columns=["_ts"])
    for name in FEATURES:
        out[name] = feats[name]
    out = out.sort_values("_pos", kind="mergesort").drop(columns=["_pos"]).reset_index(drop=True)
    return out
