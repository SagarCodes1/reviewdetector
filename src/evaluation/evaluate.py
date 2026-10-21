"""Scientific Evaluation Suite (Section 5 & 6).

Implements:
- Reviewer-disjoint benchmark test (Table 5) with 1,000 bootstrap iterations (95% CI)
- Feature group ablation (Table 6)
- Split protocol leakage comparison (Table 7)
- Computer-generated review benchmark (Table 8)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils import resample


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Computes Precision, Recall, F1, AUC-ROC, and PR-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc_roc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc_roc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
    }


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Finds the decision threshold that maximizes validation F1-score (Section 5.4)."""
    best_thresh = 0.50
    best_f1 = -1.0
    for thresh in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return best_thresh


def paired_bootstrap_ci(
    y_true: np.ndarray,
    y_proba_champion: np.ndarray,
    y_proba_baseline: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Computes 95% confidence intervals for the performance difference (Delta F1 & Delta AUC-ROC)."""
    rng = np.random.RandomState(random_state)
    delta_f1s = []
    delta_aucs = []

    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_b = y_true[indices]
        p_champ_b = y_proba_champion[indices]
        p_base_b = y_proba_baseline[indices]

        # F1 difference at 0.5 threshold
        f1_champ = f1_score(y_b, (p_champ_b >= 0.5).astype(int), zero_division=0)
        f1_base = f1_score(y_b, (p_base_b >= 0.5).astype(int), zero_division=0)
        delta_f1s.append(f1_champ - f1_base)

        try:
            auc_champ = roc_auc_score(y_b, p_champ_b)
            auc_base = roc_auc_score(y_b, p_base_b)
            delta_aucs.append(auc_champ - auc_base)
        except Exception:
            pass

    ci_f1 = (float(np.percentile(delta_f1s, 2.5)), float(np.percentile(delta_f1s, 97.5)))
    ci_auc = (float(np.percentile(delta_aucs, 2.5)), float(np.percentile(delta_aucs, 97.5)))
    return {"delta_f1_95ci": ci_f1, "delta_auc_95ci": ci_auc}
