"""
Tree-based Models Utility Module
=================================


 Decision TreeRandom ForestXGBoostLightGBM

Functions:
- load_processed_splits():  02_preprocess 
- select_tree_features():  feature_config 
- fit_dt_baseline():  Decision Tree baseline
- fit_rf_baseline():  Random Forest baseline  
- fit_xgb_baseline():  XGBoost baseline
- fit_xgb_optuna_or_load():  XGBoost (Optuna tuning)  cache
- fit_lgbm_small():  LightGBM (fixed hyperparameters)
- evaluate_binary_classifier(): 
- tune_threshold_with_constraint(): Threshold tuning with recall constraint
- plot_roc_pr_with_point(): ROC + PR curves with threshold point
- plot_all_models_roc_pr(): Multiple models ROC + PR comparison
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import warnings

# Ensure project root on path so src modules (optuna_utils) resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt

# Optional imports with fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Constants
# ============================================================================

RANDOM_SEED = 42

# Expected input files from 02_preprocess
INPUT_FILES = {
    "X_train": "X_train_base.csv",
    "X_val": "X_val_base.csv",
    "X_test": "X_test_base.csv",
    "y_train": "y_train.csv",
    "y_val": "y_val.csv",
    "y_test": "y_test.csv",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_processed_splits(data_dir: Path, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load processed splits from 02_preprocess output.
    
    Args:
        data_dir: Path to data/processed directory
        verbose: Whether to print loading info
        
    Returns:
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    if verbose:
        print("\\n" + "="*80)
        print("LOADING PROCESSED DATA")
        print("="*80)
    
    # Check all files exist first
    missing = []
    for key, filename in INPUT_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            missing.append(str(filepath))
    
    if missing:
        raise FileNotFoundError(
            f"Missing required files:\\n" + "\\n".join(f"  - {p}" for p in missing) +
            "\\n\\nPlease run scripts/02_preprocess.py first."
        )
    
    # Load all splits
    data = {}
    for key, filename in INPUT_FILES.items():
        filepath = data_dir / filename
        df = pd.read_csv(filepath)
        # Targets stored as single-column CSVs; flatten to Series for safety
        data[key] = df.iloc[:, 0] if key.startswith("y_") else df
        if verbose:
            print(f"  Loaded {key:10s}: {df.shape}")
    
    return data


def select_tree_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_config_path: Optional[Path] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    Select tree features using the registry (feature_config).
    The config uses raw feature names; X_base columns are one-hot expanded.
    We therefore expand raw names by prefix matching (e.g., Occupation -> Occupation_*).
    """
    cols = list(X_train.columns)

    # Fallback: if no config, keep all columns (but avoid silent strength-by-bug in final runs)
    if not feature_config_path or not Path(feature_config_path).exists():
        if verbose:
            print("[WARN]  Feature config not found, using all available features")
            print(f"[OK] Final feature count: {len(cols)}")
        return X_train, X_val, X_test

    cfg = pd.read_csv(feature_config_path)

    # Basic validation
    required = {"feature", "decision_type_short", "keep_tree"}
    missing = required - set(cfg.columns)
    if missing:
        raise ValueError(f"feature_config missing columns: {sorted(missing)}")

    keep_mask = cfg["decision_type_short"].astype(str).str.lower().eq("keep") & cfg["keep_tree"].astype(bool)
    raw_feats = cfg.loc[keep_mask, "feature"].astype(str).tolist()

    selected = set()
    for f in raw_feats:
        if f in cols:
            selected.add(f)
            continue

        # One-hot expansion: f -> all columns that start with f + "_"
        prefix = f"{f}_"
        matches = [c for c in cols if c.startswith(prefix)]
        selected.update(matches)

    # Preserve original column order
    selected_cols = [c for c in cols if c in selected]

    if verbose:
        print("\n================================================================================")
        print("FEATURE SELECTION FOR TREE MODELS (registry-driven)")
        print("================================================================================")
        print(f"[OK] Raw keep_tree features: {len(raw_feats)}")
        print(f"[OK] Selected encoded columns: {len(selected_cols)} (from {len(cols)} available)")

        # Optional debug: show a few raw feats that selected nothing
        no_hit = []
        for f in raw_feats:
            if (f not in cols) and not any(c.startswith(f"{f}_") for c in cols):
                no_hit.append(f)
        if no_hit:
            print(f"[WARN]  Raw features with no match in X_base (first 10): {no_hit[:10]}")

    # Safety: if selection somehow becomes empty, do not proceed silently
    if len(selected_cols) == 0:
        raise RuntimeError("Tree feature selection produced 0 columns. Check encoding/column naming.")

    X_train_sel = X_train[selected_cols].copy()
    X_val_sel = X_val[selected_cols].copy()
    X_test_sel = X_test[selected_cols].copy()

    return X_train_sel, X_val_sel, X_test_sel, selected_cols




# ============================================================================
# Model Training Functions
# ============================================================================

def fit_dt_baseline(X_train, y_train, max_depth: int = 10, verbose: bool = True):
    """Train Decision Tree baseline with fixed hyperparameters."""
    if verbose:
        print("\\n" + "="*80)
        print("TRAINING: DECISION TREE BASELINE")
        print("="*80)
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    
    model.fit(X_train, y_train.values.ravel())
    
    if verbose:
        print(f"[OK] Decision Tree trained (max_depth={max_depth})")
    
    return model


def fit_rf_baseline(X_train, y_train, n_estimators: int = 100, verbose: bool = True):
    """Train Random Forest baseline with fixed hyperparameters."""
    if verbose:
        print("\\n" + "="*80)
        print("TRAINING: RANDOM FOREST BASELINE")
        print("="*80)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    
    model.fit(X_train, y_train.values.ravel())
    
    if verbose:
        print(f"[OK] Random Forest trained (n_estimators={n_estimators})")
    
    return model


def fit_xgb_baseline(X_train, y_train, verbose: bool = True):
    """Train XGBoost baseline with fixed hyperparameters."""
    if not HAS_XGB:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    y_series = y_train.squeeze()
    
    if verbose:
        print("\\n" + "="*80)
        print("TRAINING: XGBOOST BASELINE")
        print("="*80)
    
    # Calculate scale_pos_weight for class imbalance
    n_pos = (y_series == 1).sum()
    n_neg = (y_series == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    
    model.fit(X_train, y_series.values.ravel())
    
    if verbose:
        print(f"[OK] XGBoost baseline trained (scale_pos_weight={scale_pos_weight:.2f})")
    
    return model


def fit_xgb_optuna_or_load(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    checkpoint_dir: Path,
    n_trials: int = 50,
    force_retrain: bool = False,
    verbose: bool = True,
):
    """
    Train XGBoost with Optuna tuning, or load from cache.
    
    This function integrates with optuna_utils.run_optuna_optimization.
    If model exists in checkpoint_dir and force_retrain=False, load it.
    Otherwise, run Optuna optimization.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits
        checkpoint_dir: Directory for caching models/studies
        n_trials: Number of Optuna trials
        force_retrain: Force retraining even if cache exists
        verbose: Print progress
        
    Returns:
        (model, info_dict)
    """
    if not HAS_XGB:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    y_train_series = y_train.squeeze()
    y_val_series = y_val.squeeze()
    y_test_series = y_test.squeeze()
    
    # Import optuna_utils
    try:
        import optuna_utils as OU
    except ImportError:
        raise ImportError(
            "optuna_utils.py not found. Ensure it\\'s in the project root or src/."
        )
    
    if verbose:
        print("\\n" + "="*80)
        print("TRAINING: XGBOOST WITH OPTUNA TUNING")
        print("="*80)
    
    # Calculate scale_pos_weight
    n_pos = (y_train_series == 1).sum()
    n_neg = (y_train_series == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Base estimator for Optuna
    base_estimator = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    
    # Run Optuna optimization (with cache support)
    model, info = OU.run_optuna_optimization(
        model_name="04_xgb_optuna",
        model_type="xgb",
        X_train=X_train,
        y_train=y_train_series,
        X_val=X_val,
        y_val=y_val_series,
        X_test=X_test,
        y_test=y_test_series,
        preprocessor="passthrough",  # No preprocessing, data already processed
        tree_features=list(X_train.columns),  # Already filtered
        base_estimator=base_estimator,
        n_trials=n_trials,
        cv_folds=5,
        checkpoint_dir=str(checkpoint_dir),
        force_retrain=force_retrain,
    )
    
    return model, info


def fit_lgbm_small(X_train, y_train, verbose: bool = True):
    """
    Train LightGBM with fixed hyperparameters (small tuned version).
    
    These hyperparameters are from notebook experimentation.
    """
    if not HAS_LGBM:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    if verbose:
        print("\\n" + "="*80)
        print("TRAINING: LIGHTGBM SMALL TUNED")
        print("="*80)
    
    y_series = y_train.squeeze()
    # Calculate scale_pos_weight
    n_pos = (y_series == 1).sum()
    n_neg = (y_series == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(X_train, y_series.values.ravel())
    
    if verbose:
        print(f"[OK] LightGBM trained (n_estimators=500, scale_pos_weight={scale_pos_weight:.2f})")
    
    return model


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_binary_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    split_name: str = "Test",
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Comprehensive evaluation of binary classifier.
    
    Args:
        model: Trained classifier with predict_proba method
        X: Features
        y: True labels
        threshold: Classification threshold
        split_name: Name of data split (for display)
        
    Returns:
        (metrics_dict, y_proba)
    """
    # Predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    brier = brier_score_loss(y, y_proba)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "split": split_name,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    
    return metrics, y_proba


def tune_threshold_with_constraint(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float = 0.70,
    threshold_range: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal threshold on validation set with recall constraint.
    
    Strategy: Maximize F1-score subject to recall >= min_recall.
    
    Args:
        model: Trained classifier
        X_val: Validation features
        y_val: Validation labels
        min_recall: Minimum recall constraint
        threshold_range: Array of thresholds to sweep
        verbose: Print results
        
    Returns:
        (optimal_threshold, threshold_metrics_df)
    """
    if verbose:
        print("\\n" + "="*80)
        print("THRESHOLD TUNING ON VALIDATION SET")
        print("="*80)
    
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.6, 0.01)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluate metrics at different thresholds
    results = []
    for threshold in threshold_range:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        
        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    
    threshold_metrics = pd.DataFrame(results)
    
    # Find optimal threshold: max F1 with recall constraint
    valid_thresholds = threshold_metrics[
        threshold_metrics["recall"] >= min_recall
    ]
    
    if len(valid_thresholds) == 0:
        if verbose:
            print(f"[WARN]  No threshold satisfies min_recall >= {min_recall:.2f}")
            print(f"   Using threshold with maximum recall")
        optimal_idx = threshold_metrics["recall"].idxmax()
    else:
        optimal_idx = valid_thresholds["f1"].idxmax()
    
    optimal_threshold = threshold_metrics.loc[optimal_idx, "threshold"]
    optimal_metrics = threshold_metrics.loc[optimal_idx]
    
    if verbose:
        print(f"\n[OK] Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Precision: {optimal_metrics['precision']:.4f}")
        print(f"  Recall:    {optimal_metrics['recall']:.4f}")
        print(f"  F1-score:  {optimal_metrics['f1']:.4f}")
    
    return optimal_threshold, threshold_metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_roc_pr_with_point(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str = "Model",
    split_name: str = "Test",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ROC and PR curves with threshold point marked.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Classification threshold to mark
        model_name: Model name for title
        split_name: Data split name
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Calculate point at threshold
    y_pred_thresh = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()
    fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_point = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    axes[0].plot(fpr_point, tpr_point, "ro", markersize=10, 
                label=f"Threshold={threshold:.3f}")
    
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title(f"ROC Curve - {split_name} Set", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    # Point at threshold
    precision_point, recall_point, _, _ = precision_recall_fscore_support(
        y_true, y_pred_thresh, average="binary", zero_division=0
    )
    
    axes[1].plot(recall_curve, precision_curve, linewidth=2, 
                label=f"{model_name} (AP={pr_auc:.4f})")
    
    baseline = y_true.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", linewidth=1,
                   label=f"Baseline (Prevalence={baseline:.4f})")
    axes[1].plot(recall_point, precision_point, "ro", markersize=10,
                label=f"Threshold={threshold:.3f}")
    
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title(f"Precision-Recall Curve - {split_name} Set", 
                     fontsize=12, fontweight="bold")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")
    
    return fig


def plot_all_models_roc_pr(
    models_dict: Dict[str, Tuple[Any, np.ndarray, np.ndarray]],
    split_name: str = "Test",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ROC and PR curves for multiple models on same axes.
    
    Args:
        models_dict: Dict of {model_name: (model, X, y)}
        split_name: Data split name
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))
    
    for (model_name, (model, X, y)), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X)[:, 1]
        
        # ROC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{model_name} ({roc_auc:.4f})",
                    color=color)
        
        # PR
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        axes[1].plot(recall, precision, linewidth=2, label=f"{model_name} ({pr_auc:.4f})",
                    color=color)
    
    # ROC baseline
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title(f"ROC Curves - {split_name} Set", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # PR baseline
    if len(models_dict) > 0:
        first_y = list(models_dict.values())[0][2]
        baseline = first_y.mean()
        axes[1].axhline(baseline, color="k", linestyle="--", linewidth=1, alpha=0.5,
                       label=f"Baseline ({baseline:.3f})")
    
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title(f"Precision-Recall Curves - {split_name} Set", 
                     fontsize=12, fontweight="bold")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")
    
    return fig
