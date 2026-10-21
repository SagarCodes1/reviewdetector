"""Baseline Detection Systems (Table 3: S1, S2, S3, S4).

Isolates individual evidence channels to quantify the performance gain
of our fused hybrid model.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib

try:
    from xgboost import XGBClassifier
    import xgboost
    _ = xgboost.core._LIB
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


class SentimentOnlyBaseline:
    """S1: Sentiment-only baseline using VADER compound score and star rating."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.model = LogisticRegression(C=C, random_state=random_state, class_weight="balanced")

    def fit(self, X_sentiment: np.ndarray, y: np.ndarray):
        """X_sentiment is an (N, 2) array containing [c(t), s_hat]."""
        self.model.fit(X_sentiment, y)
        return self

    def predict(self, X_sentiment: np.ndarray) -> np.ndarray:
        return self.model.predict(X_sentiment)

    def predict_proba(self, X_sentiment: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_sentiment)[:, 1]


class TfidfLogisticBaseline:
    """S2: TF-IDF + Logistic Regression text baseline."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=random_state, class_weight="balanced")

    def fit(self, X_tfidf: np.ndarray, y: np.ndarray):
        self.model.fit(X_tfidf, y)
        return self

    def predict(self, X_tfidf: np.ndarray) -> np.ndarray:
        return self.model.predict(X_tfidf)

    def predict_proba(self, X_tfidf: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_tfidf)[:, 1]


class TfidfRandomForestBaseline:
    """S3: TF-IDF + Random Forest non-linear text baseline."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 25, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, class_weight="balanced", n_jobs=-1
        )

    def fit(self, X_tfidf: np.ndarray, y: np.ndarray):
        self.model.fit(X_tfidf, y)
        return self

    def predict(self, X_tfidf: np.ndarray) -> np.ndarray:
        return self.model.predict(X_tfidf)

    def predict_proba(self, X_tfidf: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_tfidf)[:, 1]


class BehavioralXGBoostBaseline:
    """S4: Behavioral-only gradient boosted decision trees isolating metadata signals."""

    def __init__(self, max_depth: int = 4, learning_rate: float = 0.05, n_estimators: int = 200, random_state: int = 42):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state

        if HAS_XGBOOST:
            self.model = XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=n_estimators,
                class_weight="balanced",
                random_state=random_state,
            )

    def fit(self, X_beh: np.ndarray, y: np.ndarray):
        if HAS_XGBOOST:
            pos_ratio = (len(y) - sum(y)) / max(sum(y), 1)
            self.model.set_params(scale_pos_weight=pos_ratio)
        self.model.fit(X_beh, y)
        return self

    def predict(self, X_beh: np.ndarray) -> np.ndarray:
        return self.model.predict(X_beh)

    def predict_proba(self, X_beh: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_beh)[:, 1]
