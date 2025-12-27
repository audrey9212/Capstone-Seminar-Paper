#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate the Logistic Regression baseline model, producing the metrics
and visualizations needed for the paper.

Workflow:
1. Load processed data produced by 02_preprocess.py
2. Select GLM-suitable features according to the feature config
3. Train Logistic Regression (with hyperparameter search)
4. Tune the classification threshold on the validation set
5. Evaluate performance and generate paper-ready artifacts

Outputs:
- models/logit_baseline.pkl: trained model
- artifacts/figures/: ROC, PR, Calibration curves
- artifacts/tables/: metrics summary, best hyperparameters
- artifacts/meta/manifest_03_logit.json: output manifest

Usage:
    python scripts/03_logit.py
"""

import os
import sys
from pathlib import Path
import warnings

# Project setup - must be done before importing utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["CAPSTONE_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib

import utils as U

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Input paths (from 02_preprocess.py output)
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_CONFIG = PROJECT_ROOT / "data" / "feature_config_clean.csv"

# Expected input files
INPUT_FILES = {
    "X_train": "X_train_base.csv",
    "X_val": "X_val_base.csv",
    "X_test": "X_test_base.csv",
    "y_train": "y_train.csv",
    "y_val": "y_val.csv",
    "y_test": "y_test.csv",
}

# Output paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

MODEL_FILE = MODEL_DIR / "logit_baseline.pkl"

# Hyperparameter search grid
PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],  # Required for l1 penalty
    "class_weight": ["balanced"],
    "random_state": [RANDOM_SEED],
    "max_iter": [1000],
}

# Threshold tuning config
THRESHOLD_RANGE = np.arange(0.1, 0.6, 0.01)
MIN_RECALL_CONSTRAINT = 0.70  # Minimum recall constraint for churn prediction


# ============================================================================
# Helper Functions
# ============================================================================

def check_input_files():
    """Verify all required input files exist before proceeding."""
    print("\n" + "="*80)
    print("CHECKING INPUT FILES")
    print("="*80)
    
    missing = []
    for key, filename in INPUT_FILES.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            missing.append(str(filepath))
        else:
            print(f"[OK] Found: {filepath}")
    
    if missing:
        print("\n[ERROR] ERROR: Missing required input files:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease run scripts/02_preprocess.py first.")
        sys.exit(1)
    
    # Check feature config
    if not FEATURE_CONFIG.exists():
        print(f"\n[WARN]  WARNING: Feature config not found: {FEATURE_CONFIG}")
        print("   Will use all features except target/ID columns.")
    else:
        print(f"[OK] Found: {FEATURE_CONFIG}")
    
    print("\n[OK] All required files present.\n")


def load_data():
    """Load processed data from 02_preprocess.py output."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data = {}
    for key, filename in INPUT_FILES.items():
        filepath = DATA_DIR / filename
        df = pd.read_csv(filepath)
        # Targets read as Series
        data[key] = df.iloc[:, 0] if key.startswith("y_") else df
        print(f"Loaded {key:10s}: shape {df.shape}")
    
    return data


def select_features(X_train, X_val, X_test):
    """
    Select features suitable for GLM based on feature config.
    
    If feature_config_clean.csv exists, filter to keep_glm=True features.
    Otherwise, exclude common non-feature columns.
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION FOR LOGISTIC REGRESSION")
    print("="*80)
    
    # Common columns to exclude (regardless of config)
    EXCLUDE_COLS = {
        "Churn", "churn", "target", "TARGET",  # Target
        "CustomerID", "customer_id", "id", "ID",  # ID columns
    }
    
    available_features = [c for c in X_train.columns if c not in EXCLUDE_COLS]
    
    # Try to use feature config
    if FEATURE_CONFIG.exists():
        try:
            config_df = pd.read_csv(FEATURE_CONFIG)
            print(f"\n[OK] Loaded feature config: {len(config_df)} features")
            
            # normalize bool-like columns
            def to_bool(series):
                return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])
            
            # Filter to GLM-suitable features
            if "keep_glm" in config_df.columns:
                glm_mask = to_bool(config_df["keep_glm"])
                if "decision_type_short" in config_df.columns:
                    glm_mask &= config_df["decision_type_short"].astype(str).str.lower().eq("keep")
                glm_features = config_df.loc[glm_mask, "feature"].tolist()
                glm_features = [f for f in glm_features if f in available_features]
                print(f"[OK] Selected {len(glm_features)} features with keep_glm=True")
                selected_features = glm_features
            else:
                print("'keep_glm' column not found in config, using all available features")
                selected_features = available_features
                
        except Exception as e:
            print(f"Error reading feature config: {e}")
            print("   Falling back to all available features")
            selected_features = available_features
    else:
        print("Feature config not found, using all available features")
        selected_features = available_features
    
    # Ensure features exist in all splits
    missing_in_val = set(selected_features) - set(X_val.columns)
    missing_in_test = set(selected_features) - set(X_test.columns)
    
    if missing_in_val or missing_in_test:
        print(f"[WARN]  Removing features not present in all splits: {missing_in_val | missing_in_test}")
        selected_features = [f for f in selected_features 
                           if f in X_val.columns and f in X_test.columns]
    
    print(f"\n[OK] Final feature count: {len(selected_features)}")
    
    # Apply feature selection
    X_train_selected = X_train[selected_features].copy()
    X_val_selected = X_val[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    print(f"  Train: {X_train_selected.shape}")
    print(f"  Val:   {X_val_selected.shape}")
    print(f"  Test:  {X_test_selected.shape}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features


def train_with_gridsearch(X_train, y_train):
    """
    Train Logistic Regression with GridSearchCV for hyperparameter tuning.
    
    Returns:
        best_model: Fitted LogisticRegression model
        cv_results: Dictionary of cross-validation results
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*80)
    
    print("\nParameter grid:")
    for param, values in PARAM_GRID.items():
        if param != "random_state" and param != "max_iter":
            print(f"  {param:15s}: {values}")
    
    print(f"\nRunning 5-fold CV on {len(PARAM_GRID['C']) * len(PARAM_GRID['penalty'])} combinations...")
    
    # Initialize base model
    base_model = LogisticRegression()
    
    # GridSearchCV with ROC-AUC as scoring metric
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    
    # Fit
    grid_search.fit(X_train, y_train.values.ravel())
    
    # Extract results
    best_model = grid_search.best_estimator_
    cv_results = {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "cv_results_df": pd.DataFrame(grid_search.cv_results_),
    }
    
    print(f"\n[OK] GridSearchCV completed")
    print(f"  Best CV ROC-AUC: {cv_results['best_cv_score']:.4f}")
    print(f"  Best parameters:")
    for param, value in cv_results["best_params"].items():
        if param not in ["random_state", "max_iter", "solver"]:
            print(f"    {param:15s}: {value}")
    
    return best_model, cv_results


def find_optimal_threshold(model, X_val, y_val):
    """
    Find optimal classification threshold on validation set.
    
    Strategy: Maximize F1-score subject to minimum recall constraint.
    
    Returns:
        optimal_threshold: Selected threshold
        threshold_metrics: DataFrame of metrics at different thresholds
    """
    print("\n" + "="*80)
    print("THRESHOLD TUNING ON VALIDATION SET")
    print("="*80)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluate metrics at different thresholds
    results = []
    for threshold in THRESHOLD_RANGE:
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
        threshold_metrics["recall"] >= MIN_RECALL_CONSTRAINT
    ]
    
    if len(valid_thresholds) == 0:
        print(f"[WARN]  No threshold satisfies min_recall >= {MIN_RECALL_CONSTRAINT:.2f}")
        print(f"   Using threshold with maximum recall")
        optimal_idx = threshold_metrics["recall"].idxmax()
    else:
        optimal_idx = valid_thresholds["f1"].idxmax()
    
    optimal_threshold = threshold_metrics.loc[optimal_idx, "threshold"]
    optimal_metrics = threshold_metrics.loc[optimal_idx]
    
    print(f"\n[OK] Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  Recall:    {optimal_metrics['recall']:.4f}")
    print(f"  F1-score:  {optimal_metrics['f1']:.4f}")
    
    return optimal_threshold, threshold_metrics


def evaluate_model(model, X, y, split_name, threshold=0.5):
    """
    Comprehensive model evaluation on a given dataset split.
    
    Returns:
        metrics: Dictionary of evaluation metrics
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
    
    return metrics


def plot_roc_curves(model, X_val, y_val, X_test, y_test):
    """Generate ROC curves for validation and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, X, y, split_name in zip(
        axes, [X_val, X_test], [y_val, y_test], ["Validation", "Test"]
    ):
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        ax.plot(fpr, tpr, linewidth=2, label=f"Logistic Regression (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC Curve - {split_name} Set", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pr_curves(model, X_val, y_val, X_test, y_test):
    """Generate Precision-Recall curves for validation and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, X, y, split_name in zip(
        axes, [X_val, X_test], [y_val, y_test], ["Validation", "Test"]
    ):
        y_proba = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        
        # Baseline (prevalence)
        baseline = y.mean()
        
        ax.plot(recall, precision, linewidth=2, label=f"Logistic Regression (AP={pr_auc:.4f})")
        ax.axhline(baseline, color="k", linestyle="--", linewidth=1, 
                   label=f"Baseline (Prevalence={baseline:.4f})")
        
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"Precision-Recall Curve - {split_name} Set", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_calibration(model, X_test, y_test):
    """Generate calibration curve (reliability diagram)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10, strategy="uniform"
    )
    
    # Plot
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2, 
            label="Logistic Regression", markersize=8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
    
    # Brier score
    brier = brier_score_loss(y_test, y_proba)
    ax.text(0.05, 0.95, f"Brier Score: {brier:.4f}", 
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Calibration Curve (Reliability Diagram) - Test Set", 
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrices(model, X_test, y_test, threshold_default=0.5, threshold_tuned=None):
    """Generate confusion matrices for default and tuned thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    thresholds = [threshold_default, threshold_tuned]
    titles = [f"Default Threshold ({threshold_default:.2f})", 
              f"Tuned Threshold ({threshold_tuned:.3f})"]
    
    for ax, threshold, title in zip(axes, thresholds, titles):
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count", fontsize=10)
        
        # Labels
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["No Churn", "Churn"], fontsize=10)
        ax.set_yticklabels(["No Churn", "Churn"], fontsize=10)
        
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], "d"),
                       ha="center", va="center", fontsize=14,
                       color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig


def plot_threshold_sweep(threshold_metrics):
    """Visualize precision, recall, F1 across different thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(threshold_metrics["threshold"], threshold_metrics["precision"], 
            "o-", label="Precision", linewidth=2, markersize=4)
    ax.plot(threshold_metrics["threshold"], threshold_metrics["recall"], 
            "s-", label="Recall", linewidth=2, markersize=4)
    ax.plot(threshold_metrics["threshold"], threshold_metrics["f1"], 
            "^-", label="F1-Score", linewidth=2, markersize=4)
    
    # Mark recall constraint
    ax.axhline(MIN_RECALL_CONSTRAINT, color="red", linestyle="--", linewidth=1,
               label=f"Min Recall Constraint ({MIN_RECALL_CONSTRAINT:.2f})")
    
    ax.set_xlabel("Classification Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Threshold Sweep: Precision, Recall, F1-Score", 
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_artifacts(model, cv_results, metrics_df, threshold_metrics, 
                   optimal_threshold, figures, selected_features):
    """
    Save all artifacts: model, tables, figures, and generate manifest.
    """
    print("\n" + "="*80)
    print("SAVING ARTIFACTS")
    print("="*80)
    
    # 1. Save model
    joblib.dump(model, MODEL_FILE)
    print(f"\n[OK] Model saved: {MODEL_FILE}")
    
    # 2. Save tables
    print("\n[TABLES] Saving tables...")
    
    # Metrics summary
    U.save_df(metrics_df, "logit_metrics_summary", folder=U.DIRS.tables)
    
    # Best hyperparameters
    best_params_df = pd.DataFrame([cv_results["best_params"]])
    best_params_df["best_cv_roc_auc"] = cv_results["best_cv_score"]
    U.save_df(best_params_df, "logit_best_params", folder=U.DIRS.tables)
    
    # Full GridSearchCV results (for appendix)
    cv_results_summary = cv_results["cv_results_df"][[
        "params", "mean_test_score", "std_test_score", "rank_test_score"
    ]].sort_values("rank_test_score")
    U.save_df(cv_results_summary, "logit_gridsearch_full", folder=U.DIRS.tables)
    
    # Threshold sweep results (for appendix)
    U.save_df(threshold_metrics, "logit_threshold_sweep", folder=U.DIRS.tables)
    
    # 3. Save figures
    print("\n[FIGURES] Saving figures...")
    
    for fig_name, fig in figures.items():
        U.save_fig(fig, fig_name)
        plt.close(fig)
    
    # 4. Save meta (optional but handy for reproducibility/debugging)
    meta_payload = {
        "selected_features": selected_features,
        "n_features": len(selected_features),
        "optimal_threshold": float(optimal_threshold),
        "best_params": cv_results["best_params"],
        "best_cv_roc_auc": float(cv_results["best_cv_score"]),
    }
    U.save_json(meta_payload, "logit_run_meta", folder=U.DIRS.meta)
    
    # 5. Generate manifest
    print("\n[MANIFEST] Generating manifest...")
    
    core_figures = [
        "logit_roc_curves.png",
        "logit_pr_curves.png",
        "logit_calibration.png",
    ]
    
    core_tables = [
        "logit_metrics_summary.csv",
        "logit_best_params.csv",
    ]
    
    U.write_manifest(
        run_name="03_logit",
        core_figures=core_figures,
        core_tables=core_tables,
        reset=True,  # This will reset the run log
    )
    
    print("\n[OK] All artifacts saved successfully")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION BASELINE TRAINING")
    print("="*80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Reset run log at the start
    U.reset_run_log()
    
    # 0. Check inputs
    check_input_files()
    
    # 1. Load data
    data = load_data()
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    
    # 2. Feature selection
    X_train, X_val, X_test, selected_features = select_features(
        X_train, X_val, X_test
    )
    
    # 3. Train with hyperparameter tuning
    model, cv_results = train_with_gridsearch(X_train, y_train)
    
    # 4. Find optimal threshold on validation set
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        model, X_val, y_val
    )
    
    # 5. Evaluate on all splits
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Evaluate with optimal threshold
    metrics_list = []
    for X, y, split_name in [
        (X_train, y_train, "Train"),
        (X_val, y_val, "Validation"),
        (X_test, y_test, "Test"),
    ]:
        metrics = evaluate_model(model, X, y, split_name, threshold=optimal_threshold)
        metrics_list.append(metrics)
        
        print(f"\n{split_name} Set (threshold={optimal_threshold:.3f}):")
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
        print(f"  Brier:      {metrics['brier_score']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # 6. Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    figures = {}
    
    print("  - ROC curves...")
    figures["logit_roc_curves"] = plot_roc_curves(model, X_val, y_val, X_test, y_test)
    
    print("  - PR curves...")
    figures["logit_pr_curves"] = plot_pr_curves(model, X_val, y_val, X_test, y_test)
    
    print("  - Calibration curve...")
    figures["logit_calibration"] = plot_calibration(model, X_test, y_test)
    
    print("  - Confusion matrices...")
    figures["logit_confusion_matrices"] = plot_confusion_matrices(
        model, X_test, y_test, 
        threshold_default=0.5, 
        threshold_tuned=optimal_threshold
    )
    
    print("  - Threshold sweep...")
    figures["logit_threshold_sweep"] = plot_threshold_sweep(threshold_metrics)
    
    # 7. Save everything
    save_artifacts(
        model=model,
        cv_results=cv_results,
        metrics_df=metrics_df,
        threshold_metrics=threshold_metrics,
        optimal_threshold=optimal_threshold,
        figures=figures,
        selected_features=selected_features,
    )
    
    # 8. Print summary
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION BASELINE - COMPLETED")
    print("="*80)
    print(f"\n[OK] Model saved to: {MODEL_FILE}")
    print(f"[OK] Artifacts saved to: {PROJECT_ROOT / 'artifacts'}")
    print(f"[OK] Manifest: artifacts/meta/manifest_03_logit.json")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
