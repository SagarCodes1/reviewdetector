"""End-to-End Integration Tests for Scored Reviews, Calibration, and SHAP Explainability."""

import unittest
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from src.features.fusion import FeaturePipeline
from src.evaluation.calibration import IsotonicScoreCalibrator
from src.evaluation.explainer import TreeSHAPExplainer


class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = "/Users/sagar0908/Projects"
        cls.model_path = os.path.join(cls.base_dir, "artifacts/models/fused_xgboost_model.joblib")
        cls.scaler_path = os.path.join(cls.base_dir, "artifacts/scalers/feature_pipeline.joblib")
        cls.calib_path = os.path.join(cls.base_dir, "artifacts/calibrators/isotonic_calibrator.joblib")

        cls.model = joblib.load(cls.model_path)
        cls.pipeline = joblib.load(cls.scaler_path)
        cls.calibrator = joblib.load(cls.calib_path)
        cls.explainer = TreeSHAPExplainer(cls.model, cls.pipeline.feature_names)

    def test_artifacts_exist(self):
        """Asserts all model checkpoints and scalers are serialized on disk."""
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.scaler_path))
        self.assertTrue(os.path.exists(self.calib_path))

    def test_single_inference_pipeline(self):
        """Tests live single-review inference, calibrated scoring, and risk tiering."""
        single_review = pd.DataFrame([{
            "text": "Highly recommend! Best product ever, must buy! Works great!",
            "rating": 5,
            "reviewer_id": "rev_new_999",
            "product_id": "prod_new_888",
            "date": datetime.now(),
        }])

        X, _ = self.pipeline.transform(single_review)
        self.assertEqual(X.shape[0], 1)
        self.assertEqual(X.shape[1], len(self.pipeline.feature_names))

        raw_prob = self.model.predict_proba(X)[0]
        calibrated_prob = float(self.calibrator.predict_proba(np.array([raw_prob]))[0])

        self.assertGreaterEqual(calibrated_prob, 0.0)
        self.assertLessEqual(calibrated_prob, 1.0)

        risk_tier = self.calibrator.assign_risk_tier(calibrated_prob)
        self.assertIn(risk_tier, ["Genuine", "Suspicious", "Likely Fake"])

    def test_tree_shap_explanations(self):
        """Tests TreeSHAP local feature attribution generation."""
        dummy_x = np.random.randn(1, len(self.pipeline.feature_names))
        explanations = self.explainer.explain_instance(dummy_x, top_k=5)

        self.assertEqual(len(explanations), 5)
        for exp in explanations:
            self.assertIn("feature", exp)
            self.assertIn("friendly_name", exp)
            self.assertIn("shap_value", exp)
            self.assertIn("direction", exp)


if __name__ == "__main__":
    unittest.main()
