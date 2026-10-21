"""Feature Fusion Subsystem (Module D).

Integrates linguistic, stylometric, and behavioral features into a unified,
normalized feature vector x in R^389.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features.linguistic import LinguisticEngine
from src.features.stylometric import StylometricEngine
from src.features.behavioral import BehavioralEngine


class FeaturePipeline:
    """End-to-end pipeline extracting and fusing linguistic and behavioral features."""

    def __init__(
        self,
        svd_dim: int = 64,
        pca_dim: int = 32,
        tau_m: float = 0.60,
        random_state: int = 42,
    ):
        self.svd_dim = svd_dim
        self.pca_dim = pca_dim
        self.tau_m = tau_m
        self.random_state = random_state

        self.linguistic_engine = LinguisticEngine(
            svd_dim=svd_dim, pca_dim=pca_dim, random_state=random_state
        )
        self.stylometric_engine = StylometricEngine()
        self.behavioral_engine = BehavioralEngine(tau_m=tau_m)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    @property
    def feature_slices(self) -> Dict[str, slice]:
        """Returns standard slice ranges for ablation and sub-baseline models."""
        svd_end = self.svd_dim
        pca_end = svd_end + self.pca_dim
        sty_end = pca_end + 12
        return {
            "tfidf": slice(0, svd_end),
            "bert": slice(svd_end, pca_end),
            "linguistic": slice(0, pca_end),
            "stylometric": slice(pca_end, sty_end),
            "behavioral": slice(sty_end, None),
        }

    def fit(self, df: pd.DataFrame):
        """Fits all feature transformers strictly on the training partition."""
        # 1. Fit linguistic engine
        self.linguistic_engine.fit(df["text"].tolist())

        # 2. Fit behavioral engine (indexes reviewer and product aggregates)
        self.behavioral_engine.fit(df)

        # 3. Fit scaler on stylometric + behavioral numeric features
        sty_beh_df = self._extract_raw_tabular_features(df)
        self.scaler.fit(sty_beh_df.values)

        # Build feature names
        svd_names = [f"tfidf_svd_{i}" for i in range(self.svd_dim)]
        pca_names = [f"bert_pca_{i}" for i in range(self.pca_dim)]
        tabular_names = list(sty_beh_df.columns)
        self.feature_names = svd_names + pca_names + tabular_names
        self.is_fitted = True

    def transform(self, df: pd.DataFrame, update_context: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Transforms a DataFrame into a fused feature matrix X and returns feature names."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transforming.")

        # Update batch behavioral context for live scraped reviews or unindexed products
        if update_context:
            self.behavioral_engine.update_product_batch(df)

        # 1. Linguistic matrices
        svd_mat, pca_mat = self.linguistic_engine.transform(df["text"].tolist())

        # 2. Stylometric + Behavioral tabular features
        sty_beh_df = self._extract_raw_tabular_features(df)
        scaled_tabular = self.scaler.transform(sty_beh_df.values)

        # 3. Concatenate into fused feature vector
        fused_matrix = np.hstack([svd_mat, pca_mat, scaled_tabular])
        return fused_matrix, self.feature_names

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fits pipeline and transforms training data."""
        self.fit(df)
        return self.transform(df)

    def _extract_raw_tabular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts the 12 stylometric and 15 behavioral features for each record."""
        rows = []
        for _, row in df.iterrows():
            sty = self.stylometric_engine.extract_features(row["text"])
            beh = self.behavioral_engine.compute_review_features(
                text=row["text"],
                rating=int(row["rating"]),
                reviewer_id=str(row["reviewer_id"]),
                product_id=str(row["product_id"]),
                date=pd.to_datetime(row["date"]),
            )
            combined = {**sty, **beh}
            rows.append(combined)

        return pd.DataFrame(rows)
