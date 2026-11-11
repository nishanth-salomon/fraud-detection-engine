# Fraud engine run report

> All data is **synthetic**. Numbers describe how the engine behaves on generated traffic, not how it would perform on a real bank's data.

- Test period rows: 18,126 (train 63,438, validation 9,063)
- Alert threshold: 0.669 (chosen on validation, review cost Rs 150 per alert)

## Business result on the test period

- Alerts raised: 112 | fraud caught: 111 of 114 (recall 97.4%, precision 99.1%)
- Fraud loss with no system: Rs 1,274,447
- Loss plus review cost with this system: Rs 59,746
- **Net saving: Rs 1,214,702**

## Ranking quality (test period)

| Signal | PR-AUC | ROC-AUC | Precision@1% | Recall@1% |
|---|---|---|---|---|
| rules only | 0.590 | 0.849 | 44.2% | 70.2% |
| isolation forest only | 0.759 | 0.994 | 49.2% | 78.1% |
| gradient boosting only | 0.999 | 1.000 | 63.0% | 100.0% |
| blend (final) | 0.998 | 1.000 | 63.0% | 100.0% |
| max-fusion (alternative) | 0.983 | 1.000 | 61.9% | 98.2% |

Base rate (random guessing PR-AUC) = 0.629%

## Recall by fraud pattern (at the chosen threshold)

| Pattern | Count | Caught | Recall |
|---|---|---|---|
| account_takeover | 36 | 33 | 91.7% |
| atm_spree | 31 | 31 | 100.0% |
| card_testing | 35 | 35 | 100.0% |
| impossible_travel | 12 | 12 | 100.0% |

## Stress test: a fraud pattern the labels never showed

Each row retrains with that pattern's training labels hidden, then measures recall in the test period when only the top 1% of transactions can be reviewed.

| Hidden pattern | Rules | Isolation forest | Gradient boosting | Blend | Max-fusion |
|---|---|---|---|---|---|
| account_takeover | 86.1% | 97.2% | 55.6% | 100.0% | 97.2% |
| atm_spree | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| card_testing | 34.3% | 71.4% | 8.6% | 34.3% | 71.4% |
| impossible_travel | 50.0% | 100.0% | 91.7% | 100.0% | 100.0% |

## What the gradient-boosting model relies on

- `dist_prev_km`: 0.0630
- `log_amount`: 0.0405
- `secs_since_prev`: 0.0214
- `amt_ratio`: 0.0133
- `new_merchant`: 0.0031
- `is_online`: 0.0025
- `is_night`: 0.0014
- `dist_home_km`: 0.0006
