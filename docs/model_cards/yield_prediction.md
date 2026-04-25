# Model Card — Yield Prediction

**Last updated:** Phase 3

## Model Details

| Attribute | Value |
|-----------|-------|
| Task | Binary classification — predict `admit_to_enroll` |
| Training population | Admitted applicants only (`admitted == 1`) |
| Primary algorithm | XGBoost (CalibratedClassifierCV, Platt scaling) |
| Baseline | LogisticRegression (class_weight=balanced, Platt scaling) |
| Training data | Admitted subset of synthetic 50k-row applicant set |
| Split | 70% train / 15% validation / 15% test — stratified by target, `random_state=42` |
| Class imbalance | SMOTE applied to training set only (yield ~25–33% positive) |
| Calibration | Platt scaling fitted on held-out validation set (`cv='prefit'`) |
| Explainability | `shap.TreeExplainer` · top-3 |SHAP| features per prediction |
| Artifact | `models/yield_prediction.pkl` (joblib) |

## Features

**Categorical (OneHotEncoded):** `institution_segment`, `region`, `source_channel`, `intended_major`, `income_band`

**Numeric (StandardScaled):** `first_gen_flag`, `hs_gpa`, `sat_score`, `distance_miles`, `campus_visit_flag`, `email_engagement_score`, `financial_aid_inquiry_flag`, `days_since_first_inquiry`, `application_date_relative_to_deadline`, `aid_package_amount`, `scholarship_offer_flag`, `days_to_deposit_deadline`, `peer_admit_count`

**Protected attributes (audit only, not model inputs):** `race_ethnicity`, `gender`

## Yield Calibration Targets (IPEDS-derived)

| Segment | Target yield | Notes |
|---------|-------------|-------|
| R1 | 33% | Research-intensive universities |
| regional_state | 22% | Regional public institutions |
| private_lac | 28% | Private liberal arts colleges |
| community_college | 50% | Open-access, high enrollment rate |
| online | 40% | Online-first institutions |

## Outputs

- `yield_probability` — probability × 100 (0–100 scale)
- `yield_score_raw` — raw calibrated probability (0–1)
- `shap_feature_1/2/3`, `shap_value_1/2/3` — top-3 SHAP explanations

## Metrics (test set, threshold = 0.50)

| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| ROC-AUC | 0.6881 | **0.7133** |
| PR-AUC | 0.5348 | **0.5699** |
| F1 | 0.4324 | **0.4662** |
| Precision | 0.5797 | **0.6048** |
| Recall | 0.3448 | **0.3793** |

Note: Logistic Regression outperforms XGBoost on this synthetic dataset — the logit-based data-generating process is linear-friendly. In production with real data, XGBoost's non-linear capacity would likely dominate.

## Bias Audit Results

See `notebooks/06_bias_audit.ipynb` for full results.

**Results:**
- `race_ethnicity`: OK (parity gap 0.9pp, DI 0.95–1.00)
- `gender`: OK (parity gap 1.6pp)
- `first_gen_flag`: OK (DI 0.94–1.00)
- `income_band`: **FLAGGED** — parity gap exceeds threshold

**No mitigation required for first_gen in yield model** — the penalty exists in the lead logit only, not the yield logit. Income_band disparity reflects the synthetic aid package distribution (low-income applicants receive higher aid, which is a genuine causal driver of enrollment, not a proxy bias).

## Limitations

- Trained on synthetic data — aid package effects modeled parametrically, not observed
- SMOTE creates synthetic minority-class samples only in training set; calibration and evaluation use real data
- Deposit deadline dynamics (e.g. May 1 national deadline effects) are not modeled
- Not validated against any real enrollment management system
