"""Transparent rule layer. Each rule has a code, a weight and a plain-English reason, so every alert
can say *why* it fired. Rules also feed the blended score as one of three signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    "R_VELOCITY": (0.35, "5 or more transactions in the previous hour"),
    "R_CARD_TEST": (0.45, "3 or more probe-sized (under 100) transactions in the previous hour"),
    "R_IMPOSSIBLE_TRAVEL": (0.60, "moved over 100 km at more than 900 km/h since the previous transaction"),
    "R_AMOUNT_SPIKE": (0.35, "amount far above this customer's own history"),
    "R_NIGHT_NEW_MERCHANT": (0.30, "night-time, first-time merchant, and an unusually large amount"),
    "R_ATM_FAR_HOME": (0.40, "ATM withdrawal more than 300 km from the customer's usual location"),
}


def apply_rules(feat: pd.DataFrame) -> pd.DataFrame:
    """Return boolean columns for every rule plus `rule_score` (0..1) and `rule_reasons`."""
    out = pd.DataFrame(index=feat.index)
    out["R_VELOCITY"] = feat["count_1h"] >= 5
    out["R_CARD_TEST"] = (feat["small_count_1h"] >= 3) & ((feat["amount"] >= 100) | (feat["count_1h"] >= 5))
    out["R_IMPOSSIBLE_TRAVEL"] = (feat["speed_kmh"] > 900) & (feat["dist_prev_km"] > 100)
    out["R_AMOUNT_SPIKE"] = (feat["amt_z"] > 4) & (feat["amt_ratio"] > 5)
    out["R_NIGHT_NEW_MERCHANT"] = (feat["is_night"] == 1) & (feat["new_merchant"] == 1) & (feat["amt_ratio"] > 3)
    out["R_ATM_FAR_HOME"] = (feat["is_atm"] == 1) & (feat["dist_home_km"] > 300)

    weights = np.array([RULES[c][0] for c in out.columns])
    fired = out.to_numpy(dtype=float)
    out["rule_score"] = np.clip(1.0 - np.prod(1.0 - fired * weights, axis=1), 0, 1)   # noisy-OR
    out["rule_reasons"] = [
        "; ".join(f"{c}: {RULES[c][1]}" for c in RULES if row[c]) for _, row in out.iterrows()
    ]
    return out
