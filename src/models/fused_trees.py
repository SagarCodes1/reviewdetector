"""Fused Tree Ensemble Models (Table 3: S6, S7).

Combines linguistic, stylometric, and behavioral features with class weighting
and early stopping.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib

try:
    from xgboost import XGBClassifier
    import xgboost
    _ = xgboost.core._LIB
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


class FusedRandomForestClassifier:
    """S6: Fused Linguistic + Behavioral Random Forest Classifier."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: Optional[int] = 15,
        min_samples_leaf: int = 4,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        return self


class FusedXGBoostClassifier:
    """S7: Fused Linguistic + Behavioral Gradient Boosted Classifier (Champion Model)."""

    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        if HAS_XGBOOST:
            self.model = XGBClassifier(
                max_depth=min(max_depth, 5),
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                eval_metric="logloss",
                reg_lambda=1.5,
                reg_alpha=0.5,
                n_jobs=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=min(max_depth, 5),
                learning_rate=learning_rate,
                max_iter=min(n_estimators, 250),
                min_samples_leaf=10,
                l2_regularization=1.5,
                early_stopping=True,
                n_iter_no_change=15,
                class_weight="balanced",
                random_state=random_state,
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
    ):
        if HAS_XGBOOST:
            pos_ratio = float((len(y_train) - sum(y_train)) / max(sum(y_train), 1))
            self.model.set_params(scale_pos_weight=pos_ratio)

            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)

            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> np.ndarray:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        # Fallback for HistGradientBoostingClassifier
        return np.ones(self.model.n_features_in_) / self.model.n_features_in_

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        return self
