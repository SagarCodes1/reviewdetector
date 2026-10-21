"""End-to-End Training, Calibration, and Benchmark Execution Pipeline.

Executes Phases 1 through 4:
- Ingestion & Reviewer-Disjoint Splitting
- Feature Pipeline Transformation
- Model Zoo Training (S1 - S7)
- Isotonic Calibration & TreeSHAP Explainer Setup
- Evaluation Benchmark Tables (Tables 5, 6, 7, 8)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from src.data.loader import generate_benchmark_dataset, sanitize_dataset
from src.data.splitter import reviewer_disjoint_split, random_review_split
from src.features.fusion import FeaturePipeline
from src.models.baselines import (
    SentimentOnlyBaseline,
    TfidfLogisticBaseline,
    TfidfRandomForestBaseline,
    BehavioralXGBoostBaseline,
)
from src.models.fused_trees import (
    FusedRandomForestClassifier,
    FusedXGBoostClassifier,
)
from src.evaluation.calibration import IsotonicScoreCalibrator
from src.evaluation.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    paired_bootstrap_ci,
)


def run_full_experiment_pipeline(
    n_reviews: int = 3500,
    random_seed: int = 42,
    output_dir: str = "/Users/sagar0908/Projects",
):
    print("=" * 70)
    print("AI-POWERED FAKE PRODUCT REVIEW DETECTION SYSTEM: FULL PIPELINE RUN")
    print("=" * 70)

    # 1. Ingestion & Benchmark Generation
    print("\n[Phase 1] Ingesting dataset & creating reviewer-disjoint split...")
    df = generate_benchmark_dataset(
        n_reviews=n_reviews,
        n_reviewers=800,
        n_products=250,
        fake_share=0.15,
        random_seed=random_seed,
    )
    print(f"Total reviews: {len(df)} | Fake ratio: {df['label'].mean():.2%}")

    train_df, val_df, test_df = reviewer_disjoint_split(
        df, train_size=0.70, val_size=0.10, test_size=0.20, random_state=random_seed
    )
    print(f"Partitions: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # 2. Feature Extraction & Pipeline Fitting
    print("\n[Phase 2] Fitting dual-layer feature pipeline strictly on Train split...")
    pipeline = FeaturePipeline(svd_dim=64, pca_dim=32, tau_m=0.60, random_state=random_seed)
    X_train, feature_names = pipeline.fit_transform(train_df)
    X_val, _ = pipeline.transform(val_df)
    X_test, _ = pipeline.transform(test_df)

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    print(f"Fused Feature Matrix shape: Train={X_train.shape}, Test={X_test.shape}")

    # Standard feature slices for baselines & ablation
    slices = pipeline.feature_slices
    X_train_tfidf = X_train[:, slices["tfidf"]]
    X_val_tfidf = X_val[:, slices["tfidf"]]
    X_test_tfidf = X_test[:, slices["tfidf"]]

    X_train_beh = X_train[:, slices["behavioral"]]
    X_val_beh = X_val[:, slices["behavioral"]]
    X_test_beh = X_test[:, slices["behavioral"]]

    # Sentiment features [c(t), s_hat] for S1
    sentiment_idx = [feature_names.index("mismatch"), feature_names.index("signed_mismatch")]
    X_train_sent = X_train[:, sentiment_idx]
    X_val_sent = X_val[:, sentiment_idx]
    X_test_sent = X_test[:, sentiment_idx]

    # 3. Model Zoo Training (Phase 3)
    print("\n[Phase 3] Training Model Zoo (S1 through S7)...")

    # S1: Sentiment-only baseline
    s1 = SentimentOnlyBaseline(random_state=random_seed).fit(X_train_sent, y_train)

    # S2: TF-IDF + Logistic Regression
    s2 = TfidfLogisticBaseline(random_state=random_seed).fit(X_train_tfidf, y_train)

    # S3: TF-IDF + Random Forest
    s3 = TfidfRandomForestBaseline(random_state=random_seed).fit(X_train_tfidf, y_train)

    # S4: Behavioral-only XGBoost
    s4 = BehavioralXGBoostBaseline(random_state=random_seed).fit(X_train_beh, y_train)

    # S6: Fused Random Forest
    s6 = FusedRandomForestClassifier(n_estimators=250, max_depth=15, min_samples_leaf=4, random_state=random_seed).fit(X_train, y_train)

    # S7: Fused XGBoost (Champion)
    s7 = FusedXGBoostClassifier(max_depth=5, learning_rate=0.05, n_estimators=300, random_state=random_seed)
    s7.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # 4. Calibration & Explainability (Phase 4)
    print("\n[Phase 4] Fitting Isotonic Probability Calibration...")
    val_probs_s7 = s7.predict_proba(X_val)
    calibrator = IsotonicScoreCalibrator().fit(val_probs_s7, y_val)

    # 5. Scientific Benchmark Evaluation
    print("\n[Evaluation] Evaluating models on reviewer-disjoint test partition...")
    p_s1 = s1.predict_proba(X_test_sent)
    p_s2 = s2.predict_proba(X_test_tfidf)
    p_s3 = s3.predict_proba(X_test_tfidf)
    p_s4 = s4.predict_proba(X_test_beh)
    p_s6 = s6.predict_proba(X_test)
    raw_p_s7 = s7.predict_proba(X_test)
    p_s7 = calibrator.predict_proba(raw_p_s7)

    # Compute optimal thresholds on validation
    t_s1 = find_optimal_threshold(y_val, s1.predict_proba(X_val_sent))
    t_s2 = find_optimal_threshold(y_val, s2.predict_proba(X_val_tfidf))
    t_s3 = find_optimal_threshold(y_val, s3.predict_proba(X_val_tfidf))
    t_s4 = find_optimal_threshold(y_val, s4.predict_proba(X_val_beh))
    t_s6 = find_optimal_threshold(y_val, s6.predict_proba(X_val))
    t_s7 = find_optimal_threshold(y_val, calibrator.predict_proba(s7.predict_proba(X_val)))

    table5 = {
        "S1 (Sentiment-Only)": compute_metrics(y_test, p_s1, t_s1),
        "S2 (TF-IDF + LR)": compute_metrics(y_test, p_s2, t_s2),
        "S3 (TF-IDF + RF)": compute_metrics(y_test, p_s3, t_s3),
        "S4 (Behaviour Only)": compute_metrics(y_test, p_s4, t_s4),
        "S6 (Ling+Beh, RF)": compute_metrics(y_test, p_s6, t_s6),
        "S7 (Ling+Beh, XGB Champion)": compute_metrics(y_test, p_s7, t_s7),
    }

    # Bootstrap 95% Confidence Intervals
    bootstrap_results = paired_bootstrap_ci(y_test, p_s7, p_s1, n_bootstraps=1000, random_state=random_seed)

    # Table 6: Feature Group Ablation on S7
    print("\n[Evaluation] Computing Feature Group Ablation (Table 6)...")
    ablation_results = {"Full Fused Model (S7)": table5["S7 (Ling+Beh, XGB Champion)"]}
    groups = {
        "Drop TF-IDF": np.hstack([X_train[:, slices["bert"]], X_train[:, slices["stylometric"]], X_train[:, slices["behavioral"]]]),
        "Drop BERT PCA": np.hstack([X_train[:, slices["tfidf"]], X_train[:, slices["stylometric"]], X_train[:, slices["behavioral"]]]),
        "Drop Stylometry": np.hstack([X_train[:, slices["tfidf"]], X_train[:, slices["bert"]], X_train[:, slices["behavioral"]]]),
        "Drop Behavioral": np.hstack([X_train[:, slices["tfidf"]], X_train[:, slices["bert"]], X_train[:, slices["stylometric"]]]),
    }
    test_groups = {
        "Drop TF-IDF": np.hstack([X_test[:, slices["bert"]], X_test[:, slices["stylometric"]], X_test[:, slices["behavioral"]]]),
        "Drop BERT PCA": np.hstack([X_test[:, slices["tfidf"]], X_test[:, slices["stylometric"]], X_test[:, slices["behavioral"]]]),
        "Drop Stylometry": np.hstack([X_test[:, slices["tfidf"]], X_test[:, slices["bert"]], X_test[:, slices["behavioral"]]]),
        "Drop Behavioral": np.hstack([X_test[:, slices["tfidf"]], X_test[:, slices["bert"]], X_test[:, slices["stylometric"]]]),
    }
    for name, X_sub_train in groups.items():
        sub_s7 = FusedXGBoostClassifier(max_depth=5, n_estimators=200, random_state=random_seed)
        sub_s7.fit(X_sub_train, y_train)
        X_sub_test = test_groups[name]
        ablation_results[name] = compute_metrics(y_test, sub_s7.predict_proba(X_sub_test), 0.5)

    # Table 7: Leakage Comparison (Random Split vs Reviewer-Disjoint Split)
    print("\n[Evaluation] Computing Data Leakage Gap (Table 7)...")
    rnd_train_df, rnd_val_df, rnd_test_df = random_review_split(df, random_state=random_seed)
    rnd_pipeline = FeaturePipeline(svd_dim=64, pca_dim=32, tau_m=0.60, random_state=random_seed)
    X_rnd_train, _ = rnd_pipeline.fit_transform(rnd_train_df)
    X_rnd_test, _ = rnd_pipeline.transform(rnd_test_df)
    rnd_s7 = FusedXGBoostClassifier(max_depth=5, n_estimators=300, random_state=random_seed)
    rnd_s7.fit(X_rnd_train, rnd_train_df["label"].values)
    rnd_metrics = compute_metrics(rnd_test_df["label"].values, rnd_s7.predict_proba(X_rnd_test), 0.5)

    table7 = {
        "Random Review-Level Split (Leakage)": rnd_metrics,
        "Reviewer-Disjoint Split (Honest)": table5["S7 (Ling+Beh, XGB Champion)"],
        "Leakage Delta F1": rnd_metrics["f1"] - table5["S7 (Ling+Beh, XGB Champion)"]["f1"],
        "Leakage Delta AUC": rnd_metrics["auc_roc"] - table5["S7 (Ling+Beh, XGB Champion)"]["auc_roc"],
    }

    # Save Checkpoints & Results
    print("\n[Saving] Storing trained models and benchmark results...")
    os.makedirs(f"{output_dir}/artifacts/models", exist_ok=True)
    os.makedirs(f"{output_dir}/artifacts/calibrators", exist_ok=True)
    os.makedirs(f"{output_dir}/artifacts/scalers", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)

    joblib.dump(s7, f"{output_dir}/artifacts/models/fused_xgboost_model.joblib")
    joblib.dump(s6, f"{output_dir}/artifacts/models/fused_rf_model.joblib")
    joblib.dump(pipeline, f"{output_dir}/artifacts/scalers/feature_pipeline.joblib")
    calibrator.save(f"{output_dir}/artifacts/calibrators/isotonic_calibrator.joblib")

    # Save benchmark JSON
    benchmark_payload = {
        "Table5_Main_Comparison": table5,
        "Bootstrap_95CI": bootstrap_results,
        "Table6_Ablation": ablation_results,
        "Table7_Leakage_Comparison": table7,
    }
    with open(f"{output_dir}/results/benchmark_tables.json", "w") as f:
        json.dump(benchmark_payload, f, indent=2)

    # Save sample processed test set for dashboard
    test_df_enriched = test_df.copy()
    test_df_enriched["calibrated_prob"] = p_s7
    test_df_enriched["predicted_risk"] = [calibrator.assign_risk_tier(p) for p in p_s7]
    test_df_enriched.to_csv(f"{output_dir}/data/processed/test_reviews_scored.csv", index=False)

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE & ALL ARTIFACTS SERIALIZED!")
    print("=" * 70)
    print("\nTable 5: Reviewer-Disjoint Test Performance:")
    print(f"{'System':<30} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'AUC-ROC':<7}")
    print("-" * 65)
    for sys_name, m in table5.items():
        print(f"{sys_name:<30} | {m['precision']:<7.3f} | {m['recall']:<7.3f} | {m['f1']:<7.3f} | {m['auc_roc']:<7.3f}")

    print("\nTable 7: Data Leakage Gap:")
    print(f"Random Split F1: {table7['Random Review-Level Split (Leakage)']['f1']:.3f} vs Reviewer-Disjoint F1: {table7['Reviewer-Disjoint Split (Honest)']['f1']:.3f}")
    print(f"Artificial Leakage Inflation: +{table7['Leakage Delta F1']:.3f} F1 points")

    return benchmark_payload


if __name__ == "__main__":
    run_full_experiment_pipeline()
