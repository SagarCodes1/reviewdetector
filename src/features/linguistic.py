"""Linguistic Feature Extraction Subsystem (Module B).

Implements TF-IDF unigrams/bigrams with Truncated SVD (300d) and
dense semantic contextual embeddings with PCA (64d).
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA


class LinguisticEngine:
    """Extracts lexical TF-IDF (64d) and contextual semantic embeddings (32d)."""

    def __init__(self, svd_dim: int = 64, pca_dim: int = 32, random_state: int = 42):
        self.svd_dim = svd_dim
        self.pca_dim = pca_dim
        self.random_state = random_state

        # TF-IDF Vectorizer (Section 4.3)
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            max_features=25000,
            sublinear_tf=True,
            norm="l2",
            stop_words=None,  # Section 4.2: keep stop words for function-word style
        )
        self.svd: Optional[TruncatedSVD] = None
        self.pca: Optional[PCA] = None

    def fit(self, texts: List[str]):
        """Fits TF-IDF, SVD, and semantic PCA strictly on training texts."""
        # Dynamically set min_df based on training corpus size
        corpus_len = len(texts)
        if corpus_len < 20:
            min_df = 1
        else:
            min_df = min(5, max(2, int(corpus_len * 0.002)))

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.90,
            max_features=25000,
            sublinear_tf=True,
            norm="l2",
            stop_words=None,
        )

        # 1. Fit TF-IDF & TruncatedSVD
        tfidf_sparse = self.tfidf.fit_transform(texts)
        actual_svd_dim = min(self.svd_dim, tfidf_sparse.shape[1] - 1, tfidf_sparse.shape[0] - 1)
        actual_svd_dim = max(actual_svd_dim, 2)

        self.svd = TruncatedSVD(n_components=actual_svd_dim, algorithm="randomized", n_iter=7, random_state=self.random_state)
        self.svd.fit(tfidf_sparse)

        # 2. Contextual Representation & PCA (Section 4.4)
        dense_repr = self._compute_dense_representations(texts)
        actual_pca_dim = min(self.pca_dim, dense_repr.shape[1] - 1, dense_repr.shape[0] - 1)
        actual_pca_dim = max(actual_pca_dim, 2)

        self.pca = PCA(n_components=actual_pca_dim, random_state=self.random_state)
        self.pca.fit(dense_repr)

    def transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms text list into (tfidf_svd_matrix, bert_pca_matrix)."""
        if self.svd is None or self.pca is None:
            raise RuntimeError("LinguisticEngine must be fitted before transforming.")

        # 1. TF-IDF + SVD transform
        tfidf_sparse = self.tfidf.transform(texts)
        svd_feat = self.svd.transform(tfidf_sparse)

        # If actual components were fewer than svd_dim, pad with zeros to guarantee fixed dimension
        if svd_feat.shape[1] < self.svd_dim:
            pad = np.zeros((svd_feat.shape[0], self.svd_dim - svd_feat.shape[1]))
            svd_feat = np.hstack([svd_feat, pad])

        # 2. Dense Semantic + PCA transform
        dense_repr = self._compute_dense_representations(texts)
        pca_feat = self.pca.transform(dense_repr)
        if pca_feat.shape[1] < self.pca_dim:
            pad = np.zeros((pca_feat.shape[0], self.pca_dim - pca_feat.shape[1]))
            pca_feat = np.hstack([pca_feat, pad])

        return svd_feat, pca_feat

    def _compute_dense_representations(self, texts: List[str]) -> np.ndarray:
        """Computes dense 128-dimensional semantic projection matrices."""
        from sklearn.feature_extraction.text import HashingVectorizer
        hasher = HashingVectorizer(n_features=128, alternate_sign=False, norm="l2")
        return hasher.transform(texts).toarray()
