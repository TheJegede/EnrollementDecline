"""Lead scoring — Phase 3.

Target: inquired_to_applied. XGBoost primary + LogisticRegression baseline.
70/15/15 stratified split, class weighting, Platt calibration, top-3 SHAP per prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils import MODELS_DIR, RANDOM_SEED, set_seeds

TARGET = "inquired_to_applied"

CATEGORICAL_FEATURES = [
    "institution_segment",
    "region",
    "source_channel",
    "intended_major",
    "income_band",
]
NUMERIC_FEATURES = [
    "first_gen_flag",
    "hs_gpa",
    "sat_score",
    "distance_miles",
    "campus_visit_flag",
    "email_engagement_score",
    "financial_aid_inquiry_flag",
    "days_since_first_inquiry",
    "application_date_relative_to_deadline",
]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
PROTECTED_COLS = ["race_ethnicity", "gender", "first_gen_flag", "income_band"]

_XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "min_child_weight": [1, 5],
}


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def _eval_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, Any]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _top3_shap(shap_vals: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Return (n, 6) array: alternating feature_name, shap_value for top-3 |SHAP|."""
    abs_vals = np.abs(shap_vals)
    n = shap_vals.shape[0]
    result = np.empty((n, 6), dtype=object)
    for i in range(n):
        top_idx = np.argsort(abs_vals[i])[::-1][:3]
        for rank, idx in enumerate(top_idx):
            result[i, rank * 2] = feature_names[idx]
            result[i, rank * 2 + 1] = round(float(shap_vals[i, idx]), 4)
    return result


def train(df: pd.DataFrame) -> dict[str, Any]:
    """Train lead scoring models on all 50k rows. Returns artifact dict."""
    set_seeds()

    X = df[FEATURE_COLS]
    y = df[TARGET].values

    # Two-step 70/15/15 stratified split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(sss1.split(X, y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_loc, test_loc = next(sss2.split(X.iloc[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_loc]
    test_idx = temp_idx[test_loc]

    preprocessor = _build_preprocessor()
    X_train = preprocessor.fit_transform(X.iloc[train_idx])
    X_val = preprocessor.transform(X.iloc[val_idx])
    X_test = preprocessor.transform(X.iloc[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Class weighting for imbalanced lead label (~30% positive)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # XGBoost GridSearchCV (5-fold CV on training set)
    gs = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        _XGB_PARAM_GRID,
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    best_xgb = gs.best_estimator_

    # Platt calibration fitted on held-out validation set
    xgb_cal = CalibratedClassifierCV(best_xgb, method="sigmoid", cv="prefit")
    xgb_cal.fit(X_val, y_val)

    # Logistic regression baseline
    lr_base = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    lr_base.fit(X_train, y_train)
    lr_cal = CalibratedClassifierCV(lr_base, method="sigmoid", cv="prefit")
    lr_cal.fit(X_val, y_val)

    # Test set evaluation
    xgb_prob = xgb_cal.predict_proba(X_test)[:, 1]
    lr_prob = lr_cal.predict_proba(X_test)[:, 1]

    metrics = {
        "xgb": _eval_metrics(y_test, (xgb_prob >= 0.5).astype(int), xgb_prob),
        "lr": _eval_metrics(y_test, (lr_prob >= 0.5).astype(int), lr_prob),
        "best_xgb_params": gs.best_params_,
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
    }

    ohe_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    feature_names = ohe_names + NUMERIC_FEATURES

    return {
        "xgb": xgb_cal,
        "xgb_base": best_xgb,
        "lr": lr_cal,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metrics": metrics,
        "split": {"train": train_idx, "val": val_idx, "test": test_idx},
        "explainer": shap.TreeExplainer(best_xgb),
        "X_test": X_test,
        "y_test": y_test,
        "test_idx": test_idx,
    }


def predict(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame. Returns DataFrame with lead_score (0–100) and top-3 SHAP."""
    X = artifact["preprocessor"].transform(df[FEATURE_COLS])
    prob = artifact["xgb"].predict_proba(X)[:, 1]
    shap_vals = artifact["explainer"].shap_values(X)
    top3 = _top3_shap(shap_vals, artifact["feature_names"])

    out = df[["applicant_id"]].copy() if "applicant_id" in df.columns else pd.DataFrame(index=df.index)
    out["lead_score"] = (prob * 100).round(1)
    out["lead_probability"] = prob.round(4)
    for rank in range(3):
        out[f"shap_feature_{rank+1}"] = top3[:, rank * 2]
        out[f"shap_value_{rank+1}"] = top3[:, rank * 2 + 1]
    return out


def explain(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    """Return full SHAP value matrix (n_samples, n_features) for df."""
    X = artifact["preprocessor"].transform(df[FEATURE_COLS])
    return artifact["explainer"].shap_values(X)


def save(artifact: dict, path: Path | None = None) -> Path:
    out = path or MODELS_DIR / "lead_scoring.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    return out


def load(path: Path | None = None) -> dict:
    return joblib.load(path or MODELS_DIR / "lead_scoring.pkl")
