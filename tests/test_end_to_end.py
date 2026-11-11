import json

import numpy as np

from fraud_engine import pipeline
from fraud_engine.features import compute_features
from fraud_engine.models import FraudModel
from fraud_engine.synth import generate


def test_pipeline_beats_chance_and_writes_artifacts(tmp_path):
    res = pipeline.run(n_customers=200, days=60, seed=5, out_dir=tmp_path, verbose=False)
    blend = res["test_metrics"]["blend (final)"]
    assert blend["pr_auc"] > 10 * blend["base_rate"]
    assert res["test_at_threshold"]["saved"] > 0
    for name in ("metrics.json", "report.md", "top_alerts.csv", "model.joblib"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["rows"]["test"] > 0
    assert set(res["novel_pattern_recall_at_1pct"]) <= {
        "card_testing", "account_takeover", "impossible_travel", "atm_spree"}


def test_model_roundtrip_and_csv_scoring(tmp_path):
    raw = generate(n_customers=80, days=40, seed=9)
    feat = compute_features(raw)
    model = FraudModel(seed=9).fit(feat)
    model.save(tmp_path / "m.joblib")
    loaded = FraudModel.load(tmp_path / "m.joblib")
    a, b = model.score(feat.head(500)), loaded.score(feat.head(500))
    np.testing.assert_array_equal(a["score"].to_numpy(), b["score"].to_numpy())
    raw.to_csv(tmp_path / "tx.csv", index=False)
    out = pipeline.score_csv(tmp_path / "m.joblib", tmp_path / "tx.csv", tmp_path / "scored.csv", threshold=0.6)
    assert len(out) == len(raw) and "flag" in out.columns and (tmp_path / "scored.csv").exists()
