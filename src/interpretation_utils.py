# src/interpretation_utils.py
"""
Interpretation utilities for model explainability (SHAP) and error analysis.

Used by scripts/08_interpretation.py for:
- SHAP analysis (TreeExplainer, sample management)
- Plotting (ROC comparison, confusion matrix, SHAP visualizations)
- Feature importance aggregation

Design principles:
- Stable sampling (save/load indices for reproducibility)
- Safe SHAP handling (detect tree models, handle pipelines)
- Production-ready plots (no plt.show(), save with utils)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# SHAP - conditional import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# =========================================================================
# SHAP Analysis
# =========================================================================

def sample_for_shap(
    X: pd.DataFrame,
    n: int,
    seed: int,
    cache_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Sample n rows from X for SHAP analysis.
    If cache_path exists, load indices from there (reproducibility).
    Otherwise sample and save indices.
    
    Returns: (X_sampled, indices)
    """
    if cache_path and cache_path.exists():
        with open(cache_path, "r") as f:
            indices = np.array(json.load(f)["shap_sample_indices"], dtype=int)
        if len(indices) > len(X):
            indices = indices[:len(X)]
        return X.iloc[indices].copy(), indices
    
    # Sample
    np.random.seed(seed)
    n_actual = min(n, len(X))
    indices = np.random.choice(len(X), size=n_actual, replace=False)
    indices = np.sort(indices)
    
    # Save
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"shap_sample_indices": indices.tolist()}, f, indent=2)
    
    return X.iloc[indices].copy(), indices


def is_tree_model(model) -> bool:
    """Check if model is tree-based (for TreeExplainer)."""
    model_class = model.__class__.__name__
    tree_types = [
        "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
        "RandomForestClassifier", "GradientBoostingClassifier",
        "ExtraTreesClassifier", "DecisionTreeClassifier",
    ]
    return any(t in model_class for t in tree_types)


def is_pipeline(model) -> bool:
    return hasattr(model, "named_steps") or hasattr(model, "steps")


def get_final_estimator(model):
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Get appropriate SHAP explainer for model.
    Handles sklearn Pipeline (extract final estimator).
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap not installed. Install with: pip install shap")
    import xgboost as xgb

    estimator = get_final_estimator(model)
    # Unwrap sklearn XGB* to raw Booster to avoid TreeExplainer parse issues
    if isinstance(estimator, (xgb.XGBClassifier, xgb.XGBRegressor)):
        try:
            estimator = estimator.get_booster()
        except Exception:
            pass
    
    # Choose explainer
    def _ensure_binary_n_classes(est):
        # Some deserialized xgboost/sklearn models may miss n_classes_ attribute
        if est is not None and not hasattr(est, "n_classes_"):
            try:
                est.n_classes_ = 2
            except Exception:
                pass

    def predict_fn(X):
        _ensure_binary_n_classes(model)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    if is_tree_model(estimator):
        try:
            print("    [SHAP] Using TreeExplainer (booster)")
            return shap.TreeExplainer(estimator, X_background, model_output="probability")
        except Exception:
            print("    [SHAP] TreeExplainer failed, skipping SHAP")
            raise
    else:
        # KernelExplainer is slower but general
        print("    [SHAP] Using KernelExplainer (non-tree model)")
        return shap.KernelExplainer(predict_fn, X_background)


def compute_shap_values(
    model,
    X_sample: pd.DataFrame,
    X_background: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Compute SHAP values for X_sample.
    
    Returns: shap_values (N, F) array
    """
    if X_background is None:
        X_background = X_sample
    
    explainer = get_shap_explainer(model, X_background)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-output (binary classification returns list of 2 arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1
    
    return np.asarray(shap_values)


# =========================================================================
# SHAP Plotting
# =========================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    save_path: Path,
    max_display: int = 20,
) -> None:
    """
    SHAP beeswarm summary plot.
    
    Args:
        shap_values: (N, F) SHAP values
        X_sample: (N, F) feature values
        save_path: where to save figure
        max_display: top N features to show
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    save_path: Path,
    max_display: int = 20,
) -> None:
    """
    SHAP bar plot (mean absolute SHAP).
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_name: str,
    save_path: Path,
    interaction_index: str = "auto",
) -> None:
    """
    SHAP dependence plot for a single feature.
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_sample,
        interaction_index=interaction_index,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_top_features_by_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> List[str]:
    """
    Return top K features by mean absolute SHAP value.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    return [feature_names[i] for i in indices]


# =========================================================================
# Model Comparison Plots
# =========================================================================

def plot_roc_comparison(
    results: Dict[str, Dict],
    y_true: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve Comparison",
) -> None:
    """
    Plot ROC curves for multiple models on same axes.
    
    results: {model_name: {"proba": array, "auc": float}}
    """
    plt.figure(figsize=(10, 8))
    
    for name, info in sorted(results.items(), key=lambda x: x[1].get("auc", 0), reverse=True):
        proba = info["proba"]
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = info.get("auc", auc(fpr, tpr))
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_comparison(
    results: Dict[str, Dict],
    y_true: np.ndarray,
    save_path: Path,
    title: str = "Precision-Recall Curve Comparison",
) -> None:
    """
    Plot PR curves for multiple models.
    
    results: {model_name: {"proba": array, "pr_auc": float}}
    """
    plt.figure(figsize=(10, 8))
    
    baseline_precision = y_true.mean()
    
    for name, info in sorted(results.items(), key=lambda x: x[1].get("pr_auc", 0), reverse=True):
        proba = info["proba"]
        precision, recall, _ = precision_recall_curve(y_true, proba)
        pr_auc = info.get("pr_auc", average_precision_score(y_true, proba))
        plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.4f})", linewidth=2)
    
    plt.axhline(baseline_precision, color="k", linestyle="--", linewidth=1, label=f"Baseline ({baseline_precision:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
) -> None:
    """
    Plot confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ["No Churn", "Churn"]
    
    plt.figure(figsize=(8, 6))
    if SEABORN_AVAILABLE:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
    else:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(label="Count")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================================
# Feature Importance Aggregation (for one-hot features)
# =========================================================================

def aggregate_onehot_importance(
    feature_importance: pd.Series,
    prefix_sep: str = "_",
) -> pd.Series:
    """
    Aggregate one-hot encoded feature importance by prefix.
    
    Example:
        Occupation_Professional: 0.10
        Occupation_Student: 0.05
        -> Occupation: 0.15
    
    Args:
        feature_importance: Series with feature names as index
        prefix_sep: separator for one-hot prefix
    
    Returns:
        Series with aggregated importance (sorted descending)
    """
    agg = {}
    for feat, imp in feature_importance.items():
        if prefix_sep in feat:
            prefix = feat.split(prefix_sep)[0]
            agg[prefix] = agg.get(prefix, 0.0) + imp
        else:
            agg[feat] = agg.get(feat, 0.0) + imp
    
    return pd.Series(agg).sort_values(ascending=False)
