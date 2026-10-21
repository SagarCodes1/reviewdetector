"""Unit Tests for Linguistic, Stylometric, Behavioral and Fusion Pipelines."""

import unittest
from datetime import datetime
import numpy as np
import pandas as pd
from src.features.stylometric import StylometricEngine
from src.features.behavioral import BehavioralEngine, RobustSentimentAnalyzer
from src.features.fusion import FeaturePipeline


class TestFeaturesPipeline(unittest.TestCase):

    def setUp(self):
        self.sty_engine = StylometricEngine()
        self.sentiment_analyzer = RobustSentimentAnalyzer()
        self.beh_engine = BehavioralEngine()

    def test_stylometric_metrics(self):
        """Verifies stylometric metrics extraction."""
        text = "Highly recommend! Best product ever, must buy! Works great! Model 1080p."
        feats = self.sty_engine.extract_features(text)

        self.assertIn("generic_ratio", feats)
        self.assertIn("readability", feats)
        self.assertIn("specificity", feats)
        self.assertGreater(feats["generic_ratio"], 0.20, "Generic phrase ratio should detect stock phrases")
        self.assertGreater(feats["specificity"], 0.0, "Specificity should detect model numbers/digits")

    def test_sentiment_and_mismatch(self):
        """Verifies that rating-text mismatch correctly flags contradictions."""
        # Positive rating with strongly negative text
        pos_rating = 5
        neg_text = "Terrible scam, broken on day one. Total waste of money and cheap garbage."
        c_t = self.sentiment_analyzer.polarity_score(neg_text)
        self.assertLess(c_t, -0.4, "Negative text should produce negative compound score")

        s_hat = (pos_rating - 3) / 2.0  # +1.0
        mismatch = abs(s_hat - c_t)
        self.assertGreater(mismatch, 0.60, "Contradiction should exceed default mismatch threshold 0.60")

    def test_feature_pipeline_fit_transform(self):
        """Verifies end-to-end feature extraction and dimensionality."""
        records = [
            {"text": "Solid stainless steel build with sharp cutting edge.", "rating": 5, "reviewer_id": "u1", "product_id": "p1", "date": datetime(2025, 1, 1)},
            {"text": "Highly recommend! Best product ever, works great!", "rating": 5, "reviewer_id": "u2", "product_id": "p1", "date": datetime(2025, 1, 2)},
            {"text": "Broke immediately upon arrival. Terrible cheap plastic.", "rating": 1, "reviewer_id": "u3", "product_id": "p2", "date": datetime(2025, 1, 3)},
            {"text": "Decent quality for the low price. Nothing special.", "rating": 3, "reviewer_id": "u1", "product_id": "p2", "date": datetime(2025, 1, 4)},
        ]
        df = pd.DataFrame(records)

        pipeline = FeaturePipeline(svd_dim=10, pca_dim=5, tau_m=0.60)
        X, names = pipeline.fit_transform(df)

        self.assertEqual(X.shape[0], len(df))
        self.assertEqual(X.shape[1], len(names))
        self.assertFalse(np.isnan(X).any(), "Feature matrix contains NaNs")


if __name__ == "__main__":
    unittest.main()
