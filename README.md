# fraud-detection-engine

Explainable transaction-fraud detection in Python. Three signals (rules, an isolation forest and gradient
boosting) are blended into one score, evaluated the way a fraud team would: chronological splits, a limited
review budget, and money saved instead of accuracy.

> **All data is synthetic.** The generator produces card transactions with realistic structure and four fraud
> patterns. Results describe how the engine behaves on that traffic. They are **not** claims about performance
> on a real bank's data, where fraud is subtler and labels arrive late.

## Quick start

```bash
pip install -r requirements.txt
python -m fraud_engine demo            # generate -> train -> evaluate -> artifacts/report.md
python -m pytest -q                    # 21 tests
python -m fraud_engine generate --out data/transactions.csv
python -m fraud_engine score --input data/transactions.csv --threshold 0.67
```

## What is in it

| Piece | File | Idea |
|---|---|---|
| Data generator | `synth.py` | Customers with a home city, favourite merchants and spend habits. Includes **benign oddities** (big one-off purchases, trips, late-night orders) so outliers are not automatically fraud. Four fraud patterns: card testing, account takeover, impossible travel, ATM spree. Seeded and reproducible. |
| Features | `features.py` | 19 behavioural features per transaction: amount vs the customer's own history, velocity in the last hour and day, probe-sized transaction counts, first-time merchant and category, distance and speed since the previous transaction (haversine), distance from usual location. **Every feature uses only earlier rows.** |
| Rules | `rules.py` | Six named rules, each with a weight and a plain-English reason, combined by noisy-OR. Works with no training data. |
| Models | `models.py` | Gradient boosting (learns known patterns), isolation forest (flags rarity without labels), rules. Blend or max-fusion. |
| Evaluation | `evaluate.py` | Chronological train/validation/test split, precision/recall at a review budget, and a **cost-based alert threshold** chosen on validation only. |
| Pipeline and CLI | `pipeline.py`, `__main__.py` | One command to produce `report.md`, `metrics.json`, `top_alerts.csv` (with reasons) and a saved model. |

## Design decisions worth knowing

- **No future leakage, and tested.** Two tests append or corrupt *later* rows and assert that earlier rows' features do not move. I confirmed they fail when a leak is deliberately introduced.
- **Time-based split.** Training is the first 70% of the timeline, threshold selection uses the next 10%, and the final 20% is the untouched test period.
- **Money, not accuracy.** The alert cut-off minimises `fraud amount missed + review cost x alerts`. With a 0.7% fraud rate, accuracy is meaningless: flagging nothing is 99.3% accurate.
- **Explainable alerts.** Each flagged transaction lists the rules that fired and the features that sit more than 3 standard deviations from normal.

## What the first run shows (seed 7, 600 customers, 90 days)

Test period: 18,126 transactions, 114 fraud.

| Signal | PR-AUC | Recall at 1% review budget |
|---|---|---|
| Rules only | 0.59 | 70% |
| Isolation forest only | 0.76 | 78% |
| Gradient boosting only | 0.999 | 100% |
| Blend | 0.998 | 100% |

At the validation-chosen threshold the engine raised 112 alerts and caught 111 of 114 fraud cases.

**Read that honestly:** 0.999 means the synthetic fraud is easy to separate once the model has labelled examples
of it. A real portfolio would not look like this. That is why the report includes a harder test.

### Stress test: a fraud pattern the labels never showed

Each pattern's training labels are hidden, the model is retrained, and recall is measured at a 1% review budget:

| Hidden pattern | Rules | Isolation forest | Gradient boosting | Blend | Max-fusion |
|---|---|---|---|---|---|
| account_takeover | 86% | 97% | 56% | 100% | 97% |
| atm_spree | 100% | 100% | 100% | 100% | 100% |
| card_testing | 34% | 71% | 9% | 34% | 71% |
| impossible_travel | 50% | 100% | 92% | 100% | 100% |

What it shows:
- Supervised boosting collapses on card testing it has never seen (9%).
- The unsupervised isolation forest still catches 71% of it, so labels are not the only line of defence.
- The default **blend** lets the confident-but-blind boosting model drown out the isolation forest on card testing (34%). **Max-fusion** fixes that (71%) at a cost of about 1.5 PR-AUC points on known patterns (0.983 vs 0.998). Use max-fusion when labelled history is thin or fraud tactics change quickly.

## Known limits

- Synthetic data only. Real data brings label delay, concept drift, merchant-level effects and far noisier ground truth.
- The blend weights (0.65 / 0.20 / 0.15) and the max-fusion exponent are fixed by hand, not tuned.
- Scoring a CSV takes each customer's history from that file, so a short file means less reliable behaviour features. A production system would keep a rolling customer state store.
- No per-alert model explanation beyond rules and feature outliers (no SHAP).
- Gradient-boosting probabilities are not calibrated; the cost-based threshold works on ranking, not on a true probability.

## Ideas for next steps

Calibrate scores, add a streaming customer-state store, add graph features (shared devices or merchants), and run a drift monitor on score distributions.
