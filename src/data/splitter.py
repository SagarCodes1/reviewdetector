"""Reviewer-Disjoint and Leakage-Aware Dataset Splitting Module.

Ensures that reviews from any single reviewer appear strictly within one
partition (train, validation, or test) to prevent train/test identity leakage.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def reviewer_disjoint_split(
    df: pd.DataFrame,
    reviewer_col: str = "reviewer_id",
    train_size: float = 0.70,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a review DataFrame into train, validation, and test partitions

    grouped strictly by reviewer_id (reviewer-disjoint).

    Args:
        df: Input DataFrame containing reviews and reviewer_id.
        reviewer_col: Name of the column identifying the reviewer.
        train_size: Proportion of reviewers for training (default: 0.70).
        val_size: Proportion of reviewers for validation (default: 0.10).
        test_size: Proportion of reviewers for testing (default: 0.20).
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)

    Raises:
        ValueError: If partition ratios do not sum to ~1.0 or if leakage is detected.
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0 (got {total:.2f})")

    # Step 1: Split out test set (20% of reviewers)
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(gss_test.split(df, groups=df[reviewer_col]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Step 2: Split train_val into train and validation (val is val_size / (train_size + val_size))
    relative_val_size = val_size / (train_size + val_size)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=random_state
    )
    train_idx, val_idx = next(
        gss_val.split(train_val_df, groups=train_val_df[reviewer_col])
    )

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Integrity verification: zero identity overlap
    u_train = set(train_df[reviewer_col].unique())
    u_val = set(val_df[reviewer_col].unique())
    u_test = set(test_df[reviewer_col].unique())

    overlap_train_val = u_train.intersection(u_val)
    overlap_train_test = u_train.intersection(u_test)
    overlap_val_test = u_val.intersection(u_test)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Leakage detected! Overlaps: train-val={len(overlap_train_val)}, "
            f"train-test={len(overlap_train_test)}, val-test={len(overlap_val_test)}"
        )

    return train_df, val_df, test_df


def random_review_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a review DataFrame randomly at the review level (flawed baseline split).

    Used for Table 7 to evaluate and quantify data leakage.
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0 (got {total:.2f})")

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    relative_val = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val, random_state=random_state, shuffle=True
    )
    return train_df, val_df, test_df
