# src/ensemble_models.py
"""
Ensemble utilities for blending and stacking (Project Notebook 07).

Key design choice (repro / avoid feature mismatch):
- 02_preprocess.py outputs X_*_base with encoded numeric/binary columns (e.g., 34 cols).
- For Notebook/Script 07 we treat those columns as the *single source of truth*.
- Therefore we do NOT one-hot encode again inside stacking.

This file provides:
- safe feature alignment for loaded models (feature_names_in_)
- blending methods (avg / weighted / NNLS / rank-avg)
- stacking (OOF) using sklearn's StackingClassifier without extra preprocessing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import StackingClassifier


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def coerce_bool_to_int(X: pd.DataFrame) -> pd.DataFrame:
    """Convert bool / pandas BooleanDtype columns to int8 to avoid sklearn SimpleImputer errors."""
    X = X.copy()
    bool_cols = list(X.select_dtypes(include=["bool"]).columns)
    # pandas nullable boolean
    bool_cols += [c for c in X.columns if str(X[c].dtype) == "boolean"]
    bool_cols = sorted(set(bool_cols))
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")
    return X


def ensure_1d_y(y) -> np.ndarray:
    """Return y as 1D numpy array."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y DataFrame must have 1 column, got {y.shape[1]}")
        y = y.iloc[:, 0]
    if isinstance(y, pd.Series):
        return y.to_numpy().reshape(-1)
    return np.asarray(y).reshape(-1)


def get_expected_feature_names(model: BaseEstimator) -> Optional[List[str]]:
    """
    Best-effort to read expected feature names from fitted models.
    Works for sklearn estimators and sklearn Pipelines (pipeline.feature_names_in_).
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        # Try pipeline last step
        try:
            if hasattr(model, "named_steps"):
                last = list(model.named_steps.values())[-1]
                names = getattr(last, "feature_names_in_", None)
        except Exception:
            names = None
    if names is None:
        return None
    return [str(x) for x in list(names)]


def align_X_to_expected(
    X: pd.DataFrame,
    expected_cols: List[str],
    *,
    fill_value: float = 0.0,
    max_missing_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Align X to expected columns, filling missing with fill_value.
    If too many expected columns are missing, raise.
    """
    expected_cols = list(expected_cols)
    missing = [c for c in expected_cols if c not in X.columns]
    if missing:
        missing_ratio = len(missing) / max(len(expected_cols), 1)
        if missing_ratio > max_missing_ratio:
            preview = missing[:20]
            raise ValueError(
                f"Too many missing columns for model input: {len(missing)}/{len(expected_cols)} "
                f"(missing_ratio={missing_ratio:.2%}). Example missing: {preview}"
            )
    X_aligned = X.reindex(columns=expected_cols, fill_value=fill_value)
    return X_aligned


def predict_proba_binary(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probability."""
    # Some pickled LogisticRegression objects (from newer sklearn) may miss the
    # multi_class attribute when loaded on older versions. Patch it to default.
    def _patch_logit_multiclass(est):
        if est.__class__.__name__ == "LogisticRegression" and not hasattr(est, "multi_class"):
            est.multi_class = "auto"

    _patch_logit_multiclass(model)
    if hasattr(model, "named_steps"):
        for est in model.named_steps.values():
            _patch_logit_multiclass(est)
    elif hasattr(model, "steps"):
        for _, est in model.steps:
            _patch_logit_multiclass(est)

    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    # fallback
    return proba.reshape(-1)


@dataclass
class EvalResult:
    roc_auc: float
    pr_auc: float
    f1_fixed: Optional[float]
    recall_fixed: Optional[float]
    best_threshold: float
    f1_best: float
    recall_best: float


def evaluate_probs(y_true, y_prob, fixed_threshold: Optional[float] = None) -> EvalResult:
    y_true = ensure_1d_y(y_true)
    y_prob = np.asarray(y_prob).reshape(-1)

    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))

    f1_fixed = recall_fixed = None
    if fixed_threshold is not None:
        y_hat = (y_prob >= float(fixed_threshold)).astype(int)
        f1_fixed = float(f1_score(y_true, y_hat))
        recall_fixed = float((y_hat[y_true == 1].sum() / max((y_true == 1).sum(), 1)))

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has len = len(precision)-1
    f1s = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_i = int(np.argmax(f1s))
    best_th = float(thresholds[best_i])
    best_f1 = float(f1s[best_i])
    best_rec = float(recall[best_i])

    return EvalResult(
        roc_auc=roc,
        pr_auc=pr,
        f1_fixed=f1_fixed,
        recall_fixed=recall_fixed,
        best_threshold=best_th,
        f1_best=best_f1,
        recall_best=best_rec,
    )


# ---------------------------------------------------------------------
# Prediction helpers used by scripts
# ---------------------------------------------------------------------

def get_predictions(
    model: BaseEstimator,
    X: pd.DataFrame,
    features_fallback: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Predict probabilities with feature alignment.

    Priority:
    1) model.feature_names_in_ (if exists)  -> align X to it
    2) features_fallback (from script)      -> align X to it
    3) use X as-is
    """
    X = coerce_bool_to_int(X)

    expected = get_expected_feature_names(model)
    if expected is None and features_fallback is not None:
        expected = list(features_fallback)

    if expected is not None:
        X = align_X_to_expected(X, expected, fill_value=0.0, max_missing_ratio=0.20)

    return predict_proba_binary(model, X)


# ---------------------------------------------------------------------
# Blending methods
# ---------------------------------------------------------------------

def blend_simple_average(preds: pd.DataFrame) -> np.ndarray:
    return preds.mean(axis=1).to_numpy()


def blend_weighted(preds: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    w = np.array([weights.get(c, 0.0) for c in preds.columns], dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return (preds.to_numpy() * w.reshape(1, -1)).sum(axis=1)


def fit_nnls_weights(preds_val: pd.DataFrame, y_val) -> Dict[str, float]:
    """
    Non-negative least squares style weights.
    Uses sklearn LinearRegression(positive=True) to avoid scipy dependency.
    """
    y_val = ensure_1d_y(y_val).astype(float)
    X = preds_val.to_numpy()
    lr = LinearRegression(positive=True)
    lr.fit(X, y_val)
    coef = np.maximum(lr.coef_, 0.0)
    if coef.sum() <= 0:
        coef = np.ones_like(coef)
    coef = coef / coef.sum()
    return {c: float(w) for c, w in zip(preds_val.columns, coef)}


def blend_rank_average(preds: pd.DataFrame) -> np.ndarray:
    ranks = preds.rank(axis=0, method="average")
    return ranks.mean(axis=1).to_numpy()


# ---------------------------------------------------------------------
# Stacking (OOF)
# ---------------------------------------------------------------------

def train_stacking_oof(
    base_estimators: Dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train,
    *,
    features: Optional[List[str]] = None,
    cv_folds: int = 5,
    passthrough: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
) -> StackingClassifier:
    """
    Train sklearn StackingClassifier with OOF predictions.
    IMPORTANT: We do NOT add extra preprocessing here. X_train is expected
    to be the encoded X_base from 02_preprocess.py.
    """
    X_train = coerce_bool_to_int(X_train)
    y_train = ensure_1d_y(y_train)

    if features is not None:
        X_train = X_train[features]

    estimators = [(name, est) for name, est in base_estimators.items()]

    final_est = LogisticRegression(max_iter=2000, random_state=random_state)

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=cv_folds,
        stack_method="predict_proba",
        passthrough=passthrough,
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------
# Diversity (diagnostics)
# ---------------------------------------------------------------------

def diversity_correlation(preds: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of base predictions (higher = less diverse)."""
    return preds.corr()
