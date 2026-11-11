"""Synthetic card-transaction generator.

No real financial data is used or needed. The generator produces:
  * normal behaviour: each customer has a home city, favourite merchants, a typical spend level
    and preferred hours of the day;
  * benign oddities: large purchases, trips to other cities, late-night online orders. These are
    legitimate but look suspicious, so a model cannot win by flagging every outlier;
  * four fraud patterns with different signatures (card testing, account takeover, impossible
    travel, ATM spree).

Everything is seeded, so the same seed gives byte-identical data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = {
    "Chennai": (13.0827, 80.2707),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
}
CATEGORIES = ["grocery", "fuel", "dining", "electronics", "travel",
              "fashion", "utilities", "pharmacy", "entertainment", "jewellery"]
FRAUD_TYPES = ["card_testing", "account_takeover", "impossible_travel", "atm_spree"]
START = pd.Timestamp("2026-01-01")
COLUMNS = ["txn_id", "ts", "customer_id", "merchant_id", "merchant_cat",
           "amount", "lat", "lon", "channel", "is_fraud", "fraud_type"]


def _make_merchants(rng: np.random.Generator, n: int) -> pd.DataFrame:
    city_names = list(CITIES)
    cities = rng.choice(city_names, size=n)
    return pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in range(n)],
        "merchant_cat": rng.choice(CATEGORIES, size=n),
        "city": cities,
        "lat": [CITIES[c][0] for c in cities],
        "lon": [CITIES[c][1] for c in cities],
    })


def _row(ts, cust, merch, amount, lat, lon, channel, fraud=0, ftype=""):
    return (ts, cust, merch["merchant_id"], merch["merchant_cat"], float(amount),
            float(lat), float(lon), channel, int(fraud), ftype)


def generate(n_customers: int = 600, days: int = 90, seed: int = 7,
             fraud_rate: float = 0.007, n_merchants: int = 400) -> pd.DataFrame:
    """Return a chronologically sorted transaction table. `fraud_rate` is the target share of fraud rows."""
    rng = np.random.default_rng(seed)
    merchants = _make_merchants(rng, n_merchants)
    mrecs = merchants.to_dict("records")
    by_city = {c: [m for m in mrecs if m["city"] == c] for c in CITIES}
    city_names = list(CITIES)
    horizon = days * 86400

    customers = []
    for i in range(n_customers):
        home = str(rng.choice(city_names))
        fav = list(rng.choice(by_city[home], size=min(12, len(by_city[home])), replace=False))
        customers.append({
            "id": f"C{i:05d}", "home": home, "fav": fav,
            "mu": np.log(rng.uniform(150, 2500)), "sigma": rng.uniform(0.45, 0.8),
            "rate": rng.uniform(0.8, 2.6),                      # transactions per day
            "pref_hour": rng.uniform(9, 20),
        })

    rows = []
    # ---------------- normal behaviour + benign oddities ----------------
    for c in customers:
        hlat, hlon = CITIES[c["home"]]
        n = rng.poisson(c["rate"] * days)
        days_idx = rng.integers(0, days, size=n)
        hours = (rng.normal(c["pref_hour"], 4.0, size=n)) % 24
        secs = days_idx * 86400 + hours * 3600 + rng.uniform(0, 600, size=n)
        for s in secs:
            ts = START + pd.Timedelta(seconds=float(s))
            channel = rng.choice(["pos", "online", "atm"], p=[0.55, 0.35, 0.10])
            odd = rng.random()
            if channel == "online":
                m = mrecs[int(rng.integers(0, len(mrecs)))]
                lat, lon = hlat + rng.normal(0, 0.03), hlon + rng.normal(0, 0.03)
            else:
                pool = c["fav"] if rng.random() < 0.8 else by_city[c["home"]]
                m = pool[int(rng.integers(0, len(pool)))]
                lat, lon = m["lat"] + rng.normal(0, 0.03), m["lon"] + rng.normal(0, 0.03)
            amount = float(np.clip(rng.lognormal(c["mu"], c["sigma"]), 10, None))
            if channel == "atm":
                amount = max(500.0, round(amount / 500) * 500)
            if odd < 0.012:                                      # benign: big one-off purchase
                amount *= rng.uniform(3, 6)
            elif odd < 0.020 and channel == "pos":              # benign: legitimate trip elsewhere
                other = [x for x in city_names if x != c["home"]]
                city = str(rng.choice(other))
                m = by_city[city][int(rng.integers(0, len(by_city[city])))]
                lat, lon = m["lat"] + rng.normal(0, 0.03), m["lon"] + rng.normal(0, 0.03)
            elif odd < 0.026 and channel == "online":           # benign: late-night order
                ts = ts.normalize() + pd.Timedelta(hours=float(rng.uniform(0.5, 3.5)))
            rows.append(_row(ts, c["id"], m, amount, lat, lon, channel))

    # ---------------- fraud injection ----------------
    target = int(len(rows) * fraud_rate)
    injected = 0
    while injected < target:
        c = customers[int(rng.integers(0, n_customers))]
        hlat, hlon = CITIES[c["home"]]
        mean_amt = float(np.exp(c["mu"] + c["sigma"] ** 2 / 2))
        ftype = str(rng.choice(FRAUD_TYPES, p=[0.30, 0.30, 0.20, 0.20]))
        t0 = START + pd.Timedelta(seconds=float(rng.uniform(86400, horizon - 86400)))
        batch = []
        if ftype == "card_testing":
            n = int(rng.integers(5, 12))
            t = t0
            for k in range(n):
                m = mrecs[int(rng.integers(0, len(mrecs)))]
                amt = rng.uniform(1, 40) if k < n - 1 else mean_amt * rng.uniform(4, 10)
                batch.append(_row(t, c["id"], m, amt, hlat, hlon, "online", 1, ftype))
                t = t + pd.Timedelta(seconds=float(rng.uniform(20, 120)))
        elif ftype == "account_takeover":
            n = int(rng.integers(2, 5))
            t = t0.normalize() + pd.Timedelta(hours=float(rng.uniform(0.2, 4.5)))
            pool = [m for m in mrecs if m["merchant_cat"] in ("electronics", "jewellery", "travel")]
            for _ in range(n):
                m = pool[int(rng.integers(0, len(pool)))]
                batch.append(_row(t, c["id"], m, mean_amt * rng.uniform(5, 15), hlat, hlon, "online", 1, ftype))
                t = t + pd.Timedelta(minutes=float(rng.uniform(5, 40)))
        elif ftype == "impossible_travel":
            far = [x for x in city_names if x != c["home"]
                   and abs(CITIES[x][0] - hlat) + abs(CITIES[x][1] - hlon) > 8]
            city = str(rng.choice(far))
            anchor = c["fav"][int(rng.integers(0, len(c["fav"])))]
            batch.append(_row(t0, c["id"], anchor, mean_amt * rng.uniform(0.5, 1.5),
                              anchor["lat"], anchor["lon"], "pos", 0, ""))     # genuine anchor purchase
            t = t0 + pd.Timedelta(seconds=float(rng.uniform(1800, 7200)))
            for _ in range(int(rng.integers(1, 3))):
                m = by_city[city][int(rng.integers(0, len(by_city[city])))]
                batch.append(_row(t, c["id"], m, mean_amt * rng.uniform(3, 8),
                                  m["lat"], m["lon"], "pos", 1, ftype))
                t = t + pd.Timedelta(minutes=float(rng.uniform(10, 40)))
        else:  # atm_spree
            far = [x for x in city_names if x != c["home"]]
            city = str(rng.choice(far))
            n = int(rng.integers(3, 7))
            t = t0
            for _ in range(n):
                m = by_city[city][int(rng.integers(0, len(by_city[city])))]
                batch.append(_row(t, c["id"], m, float(rng.choice([10000, 15000, 20000])),
                                  m["lat"], m["lon"], "atm", 1, ftype))
                t = t + pd.Timedelta(minutes=float(rng.uniform(3, 10)))
        rows.extend(batch)
        injected += sum(r[8] for r in batch)

    df = pd.DataFrame(rows, columns=COLUMNS[1:])
    df = df.sort_values(["ts", "customer_id"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    df["amount"] = df["amount"].round(2)
    return df
