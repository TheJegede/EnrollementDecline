# Model Card — Lead Scoring

**Last updated:** Phase 3

## Model Details

| Attribute | Value |
|-----------|-------|
| Task | Binary classification — predict `inquired_to_applied` |
| Primary algorithm | XGBoost (CalibratedClassifierCV, Platt scaling) |
| Baseline | LogisticRegression (class_weight=balanced, Platt scaling) |
| Training data | Synthetic 50k-row applicant set (see `data/synthetic/applicants.csv`) |
| Split | 70% train / 15% validation / 15% test — stratified by target, `random_state=42` |
| Class imbalance | `scale_pos_weight` = neg/pos ratio (~2.3×) |
| Calibration | Platt scaling fitted on held-out validation set (`cv='prefit'`) |
| Explainability | `shap.TreeExplainer` · top-3 |SHAP| features per prediction |
| Artifact | `models/lead_scoring.pkl` (joblib) |

## Features

**Categorical (OneHotEncoded):** `institution_segment`, `region`, `source_channel`, `intended_major`, `income_band`

**Numeric (StandardScaled):** `first_gen_flag`, `hs_gpa`, `sat_score`, `distance_miles`, `campus_visit_flag`, `email_engagement_score`, `financial_aid_inquiry_flag`, `days_since_first_inquiry`, `application_date_relative_to_deadline`

**Protected attributes (audit only, not model inputs):** `race_ethnicity`, `gender`

## Outputs

- `lead_score` — probability × 100 (0–100 scale)
- `lead_probability` — raw calibrated probability (0–1)
- `shap_feature_1/2/3`, `shap_value_1/2/3` — top-3 SHAP explanations

## Metrics (test set, threshold = 0.50)

| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| ROC-AUC | 0.6601 | 0.6648 |
| PR-AUC | 0.4460 | 0.4501 |
| F1 | 0.2385 | 0.2314 |
| Precision | 0.5707 | 0.5667 |
| Recall | 0.1507 | 0.1454 |

Note: Low F1/Recall at threshold=0.50 reflects high-precision operating point. Threshold lowering trades precision for recall; tune per institution's outreach capacity.

## Bias Audit Results

See `notebooks/06_bias_audit.ipynb` for full results.

**Checks performed:**
- Demographic parity: positive prediction rate per group — flagged if any group differs by >5pp
- Disparate impact ratio: flagged if outside 0.8–1.25
- Equalized odds: TPR (recall) comparison across groups
- Proxy feature audit: Pearson correlation of numeric features with protected attributes

**Results:**
- `race_ethnicity`: OK (parity gap 0.9pp, all DI in 0.89–1.00)
- `gender`: OK (parity gap 0.1pp)
- `first_gen_flag`: **FLAGGED** — DI ratio 0.74 (below 0.80), parity gap 2.6pp
- `income_band`: OK

**Mitigation applied (first_gen_flag):** threshold lowered from 0.50 → 0.47 for first_gen=1 applicants. Parity gap reduced from 2.6pp to 0.1pp, recall gap from 3.3pp to 0.0pp. The intentional `-0.15` logit penalty in synthetic data was successfully detected and mitigated.

## Limitations

- Trained on synthetic data — distributions calibrated to IPEDS but not real applicant records
- Institution name redacted in production deployment (ASU corpus)
- Not validated against any real enrollment management system
- Threshold adjustment for first_gen mitigation must be applied at inference time
