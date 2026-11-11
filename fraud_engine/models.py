"""Three complementary signals blended into one score:

  * gradient boosting   - supervised, learns known fraud patterns from labelled history;
  * isolation forest    - unsupervised, flags behaviour that is rare for the whole population, so it
                          can still react to a pattern the labels have not seen yet;
  * rule layer          - hand-written, explainable, works from day one with no training data.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import FEATURES
from .rules import apply_rules

DEFAULT_WEIGHTS = (0.65, 0.20, 0.15)      # gbm, isolation forest, rules


class FraudModel:
    def __init__(self, seed: int = 7, weights=DEFAULT_WEIGHTS):
        self.seed = seed
        self.weights = tuple(weights)
        self.scaler: StandardScaler | None = None
        self.iforest: IsolationForest | None = None
        self.gbm: HistGradientBoostingClassifier | None = None
        self._train_anomaly: np.ndarray | None = None

    # -------------------------------------------------------------- training
    def _scaled(self, feat: pd.DataFrame) -> np.ndarray:
        return np.clip(self.scaler.transform(feat[FEATURES].to_numpy(dtype=float)), -10, 10)

    def fit(self, feat_train: pd.DataFrame) -> "FraudModel":
        X = feat_train[FEATURES].to_numpy(dtype=float)
        y = feat_train["is_fraud"].to_numpy(dtype=int)
        self.scaler = StandardScaler().fit(X)
        Xs = np.clip(self.scaler.transform(X), -10, 10)
        self.iforest = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(Xs)
        self._train_anomaly = np.sort(-self.iforest.score_samples(Xs))
        self.gbm = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=200, l2_regularization=1.0,
            class_weight="balanced", random_state=self.seed,
        ).fit(X, y)
        return self

    # --------------------------------------------------------------- scoring
    def components(self, feat: pd.DataFrame) -> pd.DataFrame:
        gbm = self.gbm.predict_proba(feat[FEATURES].to_numpy(dtype=float))[:, 1]
        raw = -self.iforest.score_samples(self._scaled(feat))
        # percentile against the training population: 0.99 means "more unusual than 99% of training traffic"
        iforest = np.searchsorted(self._train_anomaly, raw, side="right") / len(self._train_anomaly)
        rules = apply_rules(feat)
        out = pd.DataFrame({"gbm": gbm, "iforest": iforest, "rules": rules["rule_score"].to_numpy()},
                           index=feat.index)
        out["rule_reasons"] = rules["rule_reasons"].to_numpy()
        return out

    def score(self, feat: pd.DataFrame) -> pd.DataFrame:
        """`score` is the weighted blend (best when labelled history covers the fraud you face).
        `score_max` takes the strongest single signal (the isolation forest percentile is sharpened to the
        8th power so only extreme rarity competes). It gives up about 1.5 PR-AUC points on known patterns
        but roughly doubles recall on a pattern the labels never showed: see the stress test in the report."""
        comp = self.components(feat)
        wg, wi, wr = self.weights
        comp["score"] = wg * comp["gbm"] + wi * comp["iforest"] + wr * comp["rules"]
        comp["score_max"] = np.maximum.reduce(
            [comp["gbm"].to_numpy(), comp["iforest"].to_numpy() ** 8, comp["rules"].to_numpy()])
        return comp

    def outlier_features(self, feat: pd.DataFrame, z: float = 3.0, top: int = 3) -> list[str]:
        """Per-row text listing the features that sit more than `z` standard deviations from normal."""
        scaled = self.scaler.transform(feat[FEATURES].to_numpy(dtype=float))
        names = np.array(FEATURES)
        out = []
        for row in scaled:
            hits = np.argsort(-np.abs(row))[:top]
            out.append(", ".join(f"{names[h]}={row[h]:+.1f}sd" for h in hits if abs(row[h]) >= z))
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FraudModel":
        return joblib.load(path)
