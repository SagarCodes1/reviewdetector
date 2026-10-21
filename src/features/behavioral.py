"""Behavioral and Metadata Feature Extraction Engine (Module C).

Implements the behavioral formulas and metadata dynamics specified in
Section 4.6 and Appendix A of the project report.
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RobustSentimentAnalyzer:
    """Self-contained, lexicon-based sentiment analyzer following VADER rules.

    Guarantees 100% offline functionality without external downloads.
    """

    def __init__(self):
        # Seed core valences (-4.0 to +4.0)
        self.lexicon = {
            "amazing": 3.2, "excellent": 3.4, "great": 3.1, "good": 2.5, "love": 3.2,
            "best": 3.3, "perfect": 3.5, "sturdy": 2.2, "solid": 2.0, "smooth": 1.8,
            "recommend": 2.6, "happy": 2.7, "impressive": 2.6, "flawless": 3.0,
            "fast": 1.6, "worth": 2.2, "durable": 2.4, "quiet": 1.5, "adequately": 0.5,
            "bad": -2.5, "terrible": -3.4, "horrible": -3.5, "worst": -3.6, "poor": -2.6,
            "broken": -3.0, "cheap": -2.2, "scam": -3.8, "garbage": -3.5, "flickers": -2.0,
            "jammed": -2.1, "waste": -3.2, "dented": -1.8, "brittle": -2.0, "fails": -2.8,
            "stopped": -2.0, "disappointed": -2.8, "useless": -3.2, "defective": -3.2
        }
        self.booster_words = {"very": 0.29, "extremely": 0.38, "really": 0.25, "so": 0.20}
        self.negation_words = {"not", "never", "no", "hardly", "barely", "scarcely"}

    def polarity_score(self, text: str) -> float:
        """Returns normalized compound score c(t) in [-1.0, 1.0]."""
        if not text:
            return 0.0

        tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
        if not tokens:
            return 0.0

        scores = []
        for i, token in enumerate(tokens):
            if token in self.lexicon:
                score = self.lexicon[token]
                # Check for negation in preceding 2 tokens
                prev_tokens = tokens[max(0, i - 2) : i]
                if any(neg in self.negation_words for neg in prev_tokens):
                    score = -0.74 * score
                # Check for booster in immediate preceding token
                elif prev_tokens and prev_tokens[-1] in self.booster_words:
                    score += math.copysign(self.booster_words[prev_tokens[-1]], score)
                scores.append(score)

        if not scores:
            return 0.0

        # Normalization: x / sqrt(x^2 + alpha), alpha = 15
        total_score = sum(scores)
        compound = total_score / math.sqrt((total_score ** 2) + 15.0)
        return float(np.clip(compound, -1.0, 1.0))


class BehavioralEngine:
    """Pre-computes reviewer and product aggregates to extract label-free

    behavioral vectors in O(1) time per review.
    """

    def __init__(self, tau_m: float = 0.60):
        self.tau_m = tau_m
        self.sentiment_analyzer = RobustSentimentAnalyzer()
        self.reviewer_profiles: Dict[str, Dict] = {}
        self.product_profiles: Dict[str, Dict] = {}

    def fit(self, df: pd.DataFrame):
        """Pre-indexes reviewer and product histories from historical review records."""
        # 1. Reviewer profiling
        reviewer_groups = df.groupby("reviewer_id")
        for rev_id, group in reviewer_groups:
            dates = pd.to_datetime(group["date"]).sort_values().tolist()
            ratings = group["rating"].tolist()
            prods = group["product_id"].tolist()
            texts = group["text"].tolist()
            count = len(ratings)

            # Reviewer burstiness
            max_day = 1
            max_week = 1
            if count > 1:
                # Daily max
                daily_counts = group.set_index("date").resample("1D")["rating"].count()
                max_day = int(daily_counts.max()) if len(daily_counts) > 0 else 1

                # Weekly max (rolling 7 days)
                weekly_counts = group.set_index("date").resample("7D")["rating"].count()
                max_week = int(weekly_counts.max()) if len(weekly_counts) > 0 else 1

                # Mean gap
                date_diffs = [(dates[i + 1] - dates[i]).total_seconds() / 86400.0 for i in range(count - 1)]
                mean_gap = float(np.mean(date_diffs))
                account_span = float((dates[-1] - dates[0]).total_seconds() / 86400.0)
            else:
                mean_gap = 0.0
                account_span = 0.0

            # Extreme ratings
            share_5 = sum(1 for r in ratings if r == 5) / count
            share_1 = sum(1 for r in ratings if r == 1) / count

            # Self-similarity (max cosine between own reviews)
            self_sim_max = 0.0
            if count >= 2:
                try:
                    vec = TfidfVectorizer(max_features=500, stop_words="english")
                    tfidf_mat = vec.fit_transform(texts)
                    cos_sim = cosine_similarity(tfidf_mat)
                    np.fill_diagonal(cos_sim, 0.0)
                    self_sim_max = float(np.max(cos_sim))
                except Exception:
                    self_sim_max = 0.0

            self.reviewer_profiles[rev_id] = {
                "rev_count": count,
                "rev_max_day": float(max_day),
                "rev_max_week": float(max_week),
                "rev_mean_gap": float(mean_gap),
                "account_span": float(account_span),
                "share_5star": float(share_5),
                "share_1star": float(share_1),
                "distinct_prod": float(len(set(prods))),
                "self_sim_max": float(self_sim_max),
                "single_review": 1.0 if count == 1 else 0.0,
            }

        # 2. Product profiling (Daily review counts & mean ratings)
        product_groups = df.groupby("product_id")
        for prod_id, group in product_groups:
            ratings = group["rating"].tolist()
            dates = pd.to_datetime(group["date"])

            # Daily review series
            daily_series = group.set_index("date").resample("1D")["rating"].count()
            mu_p = float(daily_series.mean()) if len(daily_series) > 0 else 1.0
            sigma_p = float(daily_series.std()) if len(daily_series) > 1 and not np.isnan(daily_series.std()) else 1.0

            self.product_profiles[prod_id] = {
                "ratings": ratings,
                "mean_rating": float(np.mean(ratings)),
                "sum_rating": float(np.sum(ratings)),
                "count_rating": len(ratings),
                "daily_series": daily_series,
                "mu_p": mu_p,
                "sigma_p": sigma_p,
            }

    def update_product_batch(self, df: pd.DataFrame):
        """Dynamically indexes product-level consensus and reviewer batch patterns for live auditing."""
        if df is None or df.empty or "product_id" not in df.columns or "rating" not in df.columns:
            return

        for prod_id, group in df.groupby("product_id"):
            ratings = group["rating"].astype(float).tolist()
            if "date" in group.columns:
                group_dates = pd.to_datetime(group["date"], errors="coerce")
                group_dates = group_dates.fillna(pd.Timestamp.now())
                daily_series = group.assign(d=group_dates).set_index("d").resample("1D")["rating"].count()
            else:
                daily_series = pd.Series([len(ratings)], index=[pd.Timestamp.now()])

            mu_p = float(daily_series.mean()) if len(daily_series) > 0 else 1.0
            sigma_p = float(daily_series.std()) if len(daily_series) > 1 and not np.isnan(daily_series.std()) else 1.0

            self.product_profiles[str(prod_id)] = {
                "ratings": ratings,
                "mean_rating": float(np.mean(ratings)),
                "sum_rating": float(np.sum(ratings)),
                "count_rating": len(ratings),
                "daily_series": daily_series,
                "mu_p": mu_p,
                "sigma_p": sigma_p,
            }

        # Index reviewer profiles within this batch
        if "reviewer_id" in df.columns:
            for rev_id, group in df.groupby("reviewer_id"):
                count = len(group)
                ratings = group["rating"].astype(float).tolist()
                texts = group["text"].tolist() if "text" in group.columns else []

                max_day = 1.0
                if "date" in group.columns and count > 1:
                    g_dates = pd.to_datetime(group["date"], errors="coerce").fillna(pd.Timestamp.now())
                    d_counts = group.assign(d=g_dates).set_index("d").resample("1D")["rating"].count()
                    max_day = float(d_counts.max()) if len(d_counts) > 0 else 1.0

                self_sim = 0.0
                if count >= 2 and texts:
                    try:
                        vec = TfidfVectorizer(max_features=200, stop_words="english")
                        tfidf_mat = vec.fit_transform(texts)
                        cos = cosine_similarity(tfidf_mat)
                        np.fill_diagonal(cos, 0.0)
                        self_sim = float(np.max(cos))
                    except Exception:
                        self_sim = 0.0

                self.reviewer_profiles[str(rev_id)] = {
                    "rev_count": float(count),
                    "rev_max_day": max_day,
                    "rev_max_week": max_day,
                    "rev_mean_gap": 0.0,
                    "account_span": 0.0,
                    "share_5star": float(sum(1 for r in ratings if r == 5) / max(count, 1)),
                    "share_1star": float(sum(1 for r in ratings if r == 1) / max(count, 1)),
                    "distinct_prod": 1.0,
                    "self_sim_max": self_sim,
                    "single_review": 1.0 if count == 1 else 0.0,
                }

    def compute_review_features(
        self, text: str, rating: int, reviewer_id: str, product_id: str, date: datetime
    ) -> Dict[str, float]:
        """Extracts the 13 behavioral features for an individual review record."""
        # 1. Rating-text mismatch features
        c_t = self.sentiment_analyzer.polarity_score(text)
        s_hat = (rating - 3.0) / 2.0  # Mapped to [-1.0, 1.0]
        mismatch = float(abs(s_hat - c_t))
        mismatch_flag = 1.0 if mismatch > self.tau_m else 0.0
        signed_mismatch = float(s_hat - c_t)

        # 2. Reviewer features (fallback for cold-start / unseen reviewer)
        rev_prof = self.reviewer_profiles.get(
            reviewer_id,
            {
                "rev_count": 1.0,
                "rev_max_day": 1.0,
                "rev_max_week": 1.0,
                "rev_mean_gap": 0.0,
                "account_span": 0.0,
                "share_5star": 1.0 if rating == 5 else 0.0,
                "share_1star": 1.0 if rating == 1 else 0.0,
                "distinct_prod": 1.0,
                "self_sim_max": 0.0,
                "single_review": 1.0,
            },
        )

        # 3. Product burst z-score zp(d)
        prod_prof = self.product_profiles.get(product_id)
        if prod_prof:
            # 7-day rolling count centered on review date
            daily_series = prod_prof["daily_series"]
            target_date = pd.to_datetime(date).floor("D")
            start_date = target_date - timedelta(days=3)
            end_date = target_date + timedelta(days=3)

            window_count = 1.0
            if len(daily_series) > 0:
                sliced = daily_series.loc[start_date:end_date]
                if len(sliced) > 0:
                    window_count = float(sliced.sum() / 7.0)

            if prod_prof["sigma_p"] > 1e-4:
                prod_burst_z = float((window_count - prod_prof["mu_p"]) / prod_prof["sigma_p"])
            else:
                prod_burst_z = 0.0

            # 4. Leave-one-out rating deviation: |s - s_bar_leave_one_out|
            n_prod = prod_prof["count_rating"]
            if n_prod > 1:
                leave_one_out_mean = (prod_prof["sum_rating"] - rating) / (n_prod - 1)
                rating_dev = float(abs(rating - leave_one_out_mean))
            else:
                rating_dev = float(abs(rating - prod_prof["mean_rating"]))
        else:
            prod_burst_z = 0.0
            rating_dev = float(abs(rating - 4.0))

        return {
            "rev_max_day": rev_prof["rev_max_day"],
            "rev_max_week": rev_prof["rev_max_week"],
            "rev_mean_gap": rev_prof["rev_mean_gap"],
            "prod_burst_z": prod_burst_z,
            "rating_dev": rating_dev,
            "mismatch": mismatch,
            "mismatch_flag": mismatch_flag,
            "signed_mismatch": signed_mismatch,
            "rev_count": rev_prof["rev_count"],
            "share_5star": rev_prof["share_5star"],
            "share_1star": rev_prof["share_1star"],
            "distinct_prod": rev_prof["distinct_prod"],
            "account_span": rev_prof["account_span"],
            "self_sim_max": rev_prof["self_sim_max"],
            "single_review": rev_prof["single_review"],
        }
