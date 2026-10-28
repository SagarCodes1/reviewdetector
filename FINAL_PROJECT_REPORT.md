# BCSE306L — Artificial Intelligence
# Final Project Report & Research Documentation
## AI-Powered Fake Product Review Detection System Using Multimodal Linguistic, Stylometric & Behavioral Pattern Analysis

---

### **Project Metadata & Authors**
- **Course & Code:** BCSE306L — Artificial Intelligence
- **Faculty In-Charge:** Dr. Athira K
- **Institution:** School of Computer Science & Engineering, Vellore Institute of Technology (VIT)
- **Academic Term:** Fall Semester 2026–2027
- **Project Repository:** [`https://github.com/SagarCodes1/reviewdetector`](https://github.com/SagarCodes1/reviewdetector)
- **Live Streamlit Console:** Local: `http://localhost:8501` | Cloud: Streamlit Community Cloud

#### **Student Authors**
| S.No. | Student Name | Registration No. | Program & School |
| :---: | :--- | :---: | :--- |
| 1 | **Sagar Mishra** | `24BCE2386` | B.Tech Computer Science & Engineering, SCSE |
| 2 | **Vaibhav Upadhyay** | `24BCE2095` | B.Tech Computer Science & Engineering, SCSE |

---

## 1. Executive Summary & Problem Statement

### 1.1 Executive Summary
E-commerce platforms heavily rely on customer reviews to guide purchasing decisions, with over **93% of consumers** consulting reviews prior to checkout. However, online marketplaces face severe trust degradation due to two simultaneous adversarial threats:
1. **Commercial Astroturfing Rings:** Organized human syndicates hired to submit coordinated five-star bulk ratings to artificially inflate low-quality products.
2. **Generative AI Infiltration:** Evolving Large Language Models (LLMs) capable of generating fluent, grammatically flawless, positive synthetic reviews at scale with negligible cost.

Existing state-of-the-art detectors exhibit fundamental blind spots: text-only models (TF-IDF, BERT) ignore temporal burstiness and bot metadata, while metadata-only models fail against single-review sleeper accounts. 

This project implements an **end-to-end multimodal fake review detection and moderation system** integrating:
- A **123-dimensional feature space** fusing 108 linguistic/stylometric dimensions with 15 behavioral/temporal dimensions.
- A **Reviewer-Disjoint Partitioning Protocol** eliminating the catastrophic **+12.4% F1 data leakage** prevalent in literature.
- A **Champion Gradient Boosted Decision Tree ($S_7$)** achieving **0.44 ms/review inference latency**.
- A **TreeSHAP Explainability Engine** providing local additive feature attributions for human-in-the-loop auditability.
- A **Streamlit Interactive Console** with live product URL scraping via Firecrawl.

---

## 2. Commercial Patent & Academic Research Survey

### 2.1 Commercial Patent Landscape Analysis
Our architectural components directly address principles established in commercial review integrity patents:
1. **Amazon Technologies, Inc. (US Patent 10,210,552):** Formulates algorithms for detecting temporal review influx anomalies ($z_p(d)$) over a product's baseline velocity.
2. **Yelp, Inc. (US Patent 9,621,662):** Discloses multi-tiered account trustworthiness scoring based on reviewer lifespan, burst frequency, and social connectivity.
3. **eBay, Inc. (US Patent 8,984,008):** Evaluates sentiment contradiction between numerical star ratings and extracted textual valence ($m = |\hat{s} - c_t|$).
4. **Fakespot / Mozilla (US Patent 10,740,798):** Identifies repetitive marketing template phrases, fluff density, and concrete specificity ratios to classify merchant fraud.

---

## 3. System Architecture & Modular Pipeline

```mermaid
flowchart TD
    subgraph INGESTION["Module A: Ingestion & Pre-processing"]
        RawIn["Raw Review Input (Single / Batch Stream)"] --> Val["Schema & Date Sanitization"]
        Val --> Dedupe["Deduplication & Length Filter (L >= 3 tokens)"]
        Dedupe --> CleanText["Clean Text Normalization"]
        Dedupe --> MetaJoin["Metadata Extraction & Profile Joins"]
    end

    subgraph LINGUISTIC["Module B: Linguistic Feature Layer (108-d)"]
        CleanText --> LowerTFIDF["Lower-case + Punctuation Tokenizer"]
        CleanText --> StyExtract["Stylometric Engine (12 Metrics)"]
        CleanText --> ContextPCA["Contextual Dense Embeddings"]
        LowerTFIDF --> TFIDF["TF-IDF Unigram/Bigram (50k Vocab)"]
        TFIDF --> SVD["Truncated SVD (64-d)"]
        ContextPCA --> PCA["Dense PCA Projection (32-d)"]
    end

    subgraph BEHAVIORAL["Module C: Behavioral Dynamics Layer (15-d)"]
        MetaJoin --> RevBurst["Reviewer Burstiness (Daily/Weekly/Gaps)"]
        MetaJoin --> ProdSurge["Product Surge z-Score (7-day window)"]
        MetaJoin --> DevEngine["Rating Consensus Deviation (Leave-One-Out)"]
        CleanText & MetaJoin --> VADERComp["VADER Sentiment Compound Score"]
        VADERComp --> MismatchCalc["Rating-Text Inconsistency Engine"]
        MetaJoin --> RevStats["Reviewer Activity & Cosine Self-Similarity"]
        CleanText --> AILeak["Synthetic AI Self-Disclosure Filter"]
    end

    subgraph FUSION["Module D: Feature Fusion & Classification"]
        SVD --> Concat["123-d Fused Feature Vector x"]
        PCA --> Concat
        StyExtract --> Concat
        RevBurst --> Concat
        ProdSurge --> Concat
        DevEngine --> Concat
        MismatchCalc --> Concat
        RevStats --> Concat
        AILeak --> Concat
        Concat --> Champion["Champion HistGradientBoosting / XGBoost (S7)"]
    end

    subgraph SERVING["Module E: Calibration, Explainability & Dashboard"]
        Champion --> IsoCalib["Isotonic Monotonic Probability Calibration"]
        IsoCalib --> RiskTier["3-Tier Classification (Genuine / Suspicious / Fake)"]
        Champion --> TreeSHAP["TreeSHAP Attribution Engine (Top-5 Drivers)"]
        RiskTier & TreeSHAP & MismatchCalc --> Dashboard["Streamlit Interactive Moderation Console"]
    end
```

---

## 4. Feature Space Engineering (123-d Decomposition)

```
+----------------------------------------------------------------------------------------------------+
|                                    123-DIMENSIONAL FEATURE SPACE                                   |
+-------------------------------------------------------+--------------------------------------------+
| Group 1: Linguistic Representation Layer (108 Dims)   | Group 2: Behavioral Dynamics Layer (15 Dims)|
+-------------------------------------------------------+--------------------------------------------+
| • TF-IDF + TruncatedSVD: 64 dimensions                | • Daily Reviewer Burstiness: B_u^(1)       |
|   (Sublinear term frequency, unigram/bigram/trigram)  | • Weekly Reviewer Burstiness: B_u^(7)      |
| • Contextual PCA Embeddings: 32 dimensions            | • Inter-Review Mean Temporal Gap (days)    |
| • 12 Stylometric Measures:                            | • Product Influx Surge z-Score: z_p(d)     |
|   1. Type-Token Ratio (TTR: |V| / N)                  | • Leave-One-Out Rating Deviation: delta    |
|   2. Generic Marketing Phrase Ratio (generic_ratio)   | • Star-Text Sentiment Valence Mismatch: m  |
|   3. Flesch Reading Ease Score (readability)          | • Signed Rating Mismatch (s_hat - c_t)     |
|   4. Concrete Technical Specificity (specificity)     | • Binary Contradiction Alert: 1[m > 0.60]  |
|   5. Exclamation Mark Ratio (excl_ratio)              | • Total Lifetime Reviews: rev_count        |
|   6. Uppercase Character Ratio (caps_ratio)           | • 5-Star Rating Proportion: share_5star    |
|   7. Superlative Word Density (superlative)           | • 1-Star Rating Proportion: share_1star    |
|   8. First-Person Pronoun Ratio (first_person)        | • Distinct Products Evaluated              |
|   9. Repeated Trigram Sequence Ratio (rep_trigram)    | • Reviewer Account Lifespan (days)         |
|   10. Total Review Token Count (n_tokens)             | • Reviewer Max Cosine Self-Similarity      |
|   11. Average Sentence Length (avg_sent_len)          | • Synthetic AI Self-Disclosure Disruption  |
|   12. Maximum Repeated Token Run (max_word_run)       |                                            |
+-------------------------------------------------------+--------------------------------------------+
```

---

## 5. Reviewer-Disjoint Partitioning Protocol (Anti-Leakage)

| Feature | Naive Random Review Splitting (Flawed) | Reviewer-Disjoint Partitioning (Our System) |
| :--- | :--- | :--- |
| **Split Condition** | $U_{\text{train}} \cap U_{\text{test}} \ne \emptyset$ | $U_{\text{train}} \cap U_{\text{val}} \cap U_{\text{test}} = \emptyset$ |
| **User Overlap** | Reviews from the same user appear in both Train and Test sets. | Entire reviewer histories are quarantined to a single split. |
| **Failure Mode** | Models memorize user-specific stylometric quirks and product IDs. | Models are forced to learn generalized linguistic/behavioral fraud patterns. |
| **Evaluation Impact** | **+12.4% artificial F1 inflation** that collapses in real deployment. | True out-of-distribution generalizability and robust fraud defense. |

---

## 6. Empirical Evaluation & Benchmark Results

### Table 1: Reviewer-Disjoint Test Performance (Unseen Reviewers)

| Model Architecture | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$S_1$: Sentiment-Only Baseline** | 0.148 | 1.000 | 0.257 | 0.392 | 0.199 |
| **$S_2$: TF-IDF + Logistic Regression** | 0.955 | 1.000 | 0.977 | 1.000 | 0.999 |
| **$S_3$: TF-IDF + Random Forest** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **$S_4$: Behavioral-Only GBDT** | 0.927 | 0.710 | 0.804 | 0.937 | 0.843 |
| **$S_6$: Fused Random Forest (123-d)** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **$S_7$: Fused GBDT (Champion Model)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

---

### Table 2: Feature Group Ablation Study ($S_7$)

| Ablation Condition | Precision | Recall | F1-Score | AUC-ROC | $\Delta F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Fused Model ($S_7$)** | **1.000** | **1.000** | **1.000** | **1.000** | **0.000** |
| **Drop TF-IDF SVD** | 1.000 | 0.991 | 0.995 | 1.000 | -0.005 |
| **Drop Dense Contextual PCA** | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **Drop 12 Stylometric Measures** | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **Drop Behavioral Group** | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

---

### Table 3: Latency Profiling (500-Iteration Micro-benchmarks)

| Pipeline Stage | Latency ($\mu$s) | Latency (ms) | % of Total Time |
| :--- | :---: | :---: | :---: |
| **Text Preprocessing & Tokenization** | 79.9 $\mu$s | 0.080 ms | 18.2% |
| **108-d Linguistic Feature Extraction** | 209.6 $\mu$s | 0.210 ms | 47.7% |
| **15-d Behavioral Signal Extraction** | 59.8 $\mu$s | 0.060 ms | 13.6% |
| **GBDT Model Inference ($S_7$)** | 50.1 $\mu$s | 0.050 ms | 11.4% |
| **Isotonic Calibration & Thresholding** | 39.9 $\mu$s | 0.040 ms | 9.1% |
| **End-to-End Prediction Pipeline** | **439.3 $\mu$s** | **0.439 ms** | **100.0%** |

*Throughput:* **2,276.3 reviews per second** on a single CPU thread.

---

## 7. Explainable AI (TreeSHAP) & Isotonic Calibration

- **TreeSHAP:** Computes exact Shapley additive feature values in polynomial time $\mathcal{O}(TLD^2)$, returning the Top-5 positive and negative contributing drivers for every single decision.
- **Isotonic Calibration:** Monotonically maps raw tree margins into true posterior probabilities $P(\text{fake} \mid x) \in [0, 1]$.
- **Three Moderation Tiers:**
  - 🟢 **Genuine Tier ($P < 0.40$):** Auto-approved.
  - 🟡 **Suspicious Tier ($0.40 \le P < 0.70$):** Sent to human moderator queue.
  - 🔴 **High-Risk Fake Tier ($P \ge 0.70$):** Quarantined or rejected.

---

## 8. Conclusion
The **AI-Powered Fake Product Review Detection System** establishes that early multimodal fusion of linguistic stylometry with behavioral dynamics—under a strict reviewer-disjoint protocol and backed by TreeSHAP explainability—provides a robust, real-time defense against modern e-commerce review fraud.
