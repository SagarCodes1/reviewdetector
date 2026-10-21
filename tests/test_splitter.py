"""Unit Tests for Reviewer-Disjoint Splitting and Anti-Leakage Verification."""

import unittest
import pandas as pd
from src.data.loader import generate_benchmark_dataset
from src.data.splitter import reviewer_disjoint_split, random_review_split


class TestReviewerDisjointSplitter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generates a reproducible dataset for testing."""
        cls.df = generate_benchmark_dataset(
            n_reviews=1000,
            n_reviewers=200,
            n_products=50,
            fake_share=0.15,
            random_seed=42,
        )

    def test_strict_reviewer_disjointness(self):
        """Asserts zero reviewer intersection between train, val, and test."""
        train_df, val_df, test_df = reviewer_disjoint_split(
            self.df, reviewer_col="reviewer_id", train_size=0.70, val_size=0.10, test_size=0.20, random_state=42
        )

        u_train = set(train_df["reviewer_id"].unique())
        u_val = set(val_df["reviewer_id"].unique())
        u_test = set(test_df["reviewer_id"].unique())

        # Assert no intersection
        self.assertEqual(len(u_train.intersection(u_val)), 0, "Train and Val share reviewers!")
        self.assertEqual(len(u_train.intersection(u_test)), 0, "Train and Test share reviewers!")
        self.assertEqual(len(u_val.intersection(u_test)), 0, "Val and Test share reviewers!")

    def test_partition_proportions(self):
        """Asserts partition sizes roughly correspond to specified ratios."""
        train_df, val_df, test_df = reviewer_disjoint_split(
            self.df, reviewer_col="reviewer_id", train_size=0.70, val_size=0.10, test_size=0.20, random_state=42
        )
        total_reviewers = len(self.df["reviewer_id"].unique())
        n_train_reviewers = len(train_df["reviewer_id"].unique())
        n_test_reviewers = len(test_df["reviewer_id"].unique())

        self.assertAlmostEqual(n_train_reviewers / total_reviewers, 0.70, delta=0.05)
        self.assertAlmostEqual(n_test_reviewers / total_reviewers, 0.20, delta=0.05)

    def test_random_split_exhibits_leakage(self):
        """Demonstrates that standard random review splitting DOES leak reviewer identities."""
        train_df, val_df, test_df = random_review_split(
            self.df, train_size=0.70, val_size=0.10, test_size=0.20, random_state=42
        )
        u_train = set(train_df["reviewer_id"].unique())
        u_test = set(test_df["reviewer_id"].unique())
        overlap = u_train.intersection(u_test)
        # Because reviewers have multiple reviews, random split should leak reviewers
        self.assertGreater(len(overlap), 0, "Expected leakage in random split, but found none!")


if __name__ == "__main__":
    unittest.main()
