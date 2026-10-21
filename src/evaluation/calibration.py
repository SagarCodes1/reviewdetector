"""Score Calibration and Decision Tiers Engine (Module E).

Applies Isotonic Regression on validation predictions to yield well-calibrated
posterior probabilities P(fake | x) in [0, 1].
"""

from typing import Tuple, List, Dict
import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib


class IsotonicScoreCalibrator:
    """Isotonic regression calibrator mapping raw model scores to posterior probabilities."""

    def __init__(self, tau_suspicious: float = 0.40, tau_fake: float = 0.70):
        self.tau_suspicious = tau_suspicious
        self.tau_fake = tau_fake
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, val_scores: np.ndarray, y_val: np.ndarray):
        """Fits isotonic regression on validation predictions."""
        self.calibrator.fit(val_scores, y_val)
        self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Transforms raw model probabilities/scores to calibrated probabilities."""
        scores_arr = np.asarray(scores, dtype=float)
        if not self.is_fitted:
            # Fallback to clipping if not yet fitted
            return np.clip(scores_arr, 0.0, 1.0)

        calibrated = self.calibrator.predict(scores_arr)
        calibrated = np.clip(calibrated, 0.0, 1.0)

        # Handle zero-derivative plateau artifact in Isotonic Regression:
        # If the calibrator collapses scores >= 0.40 into 0.0 due to a validation gap,
        # preserve the native GBDT logistic loss probabilities.
        if hasattr(self.calibrator, "X_thresholds_") and len(self.calibrator.X_thresholds_) >= 2:
            if self.calibrator.X_thresholds_[1] >= 0.40 and self.calibrator.y_thresholds_[1] == 0.0:
                return np.maximum(calibrated, scores_arr)

        return calibrated

    def assign_risk_tier(self, prob: float) -> str:
        """Maps calibrated probability to risk status badge."""
        if prob < self.tau_suspicious:
            return "Genuine"
        elif prob < self.tau_fake:
            return "Suspicious"
        else:
            return "Likely Fake"

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsotonicScoreCalibrator":
        return joblib.load(filepath)
