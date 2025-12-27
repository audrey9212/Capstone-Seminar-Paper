#!/usr/bin/env python3
"""
08 - Model Interpretation & Error Analysis (with Semi-supervised Appendix)

Purpose:
- Consolidate results from 04 (Trees/GBM), 05 (Neural Networks), 07 (Ensemble)
- Generate unified leaderboard across all model families
- Perform SHAP explainability analysis on champion model
- Include Appendix: Semi-supervised Learning with Real Holdout (from 06)

Design Principles:
- NO retraining: only load pre-trained models from 04/05/07
- Feature alignment: use 02's X_*_base.csv as single source of truth
- Stable sampling: fixed SHAP sample indices for reproducibility
- Integrate 06 SSL results as appendix (not main leaderboard)

Dependencies:
    02_preprocess.py -> X_*_base.csv (encoded features)
    03_logit.py -> logit_baseline.pkl
    04_trees_gbm.py -> XGBoost/LightGBM/RandomForest models
    05_nn.py -> neural network models (optional)
    06_autoencoder.py -> SSL/AE results (appendix)
    07_stacking.py -> ensemble models (optional)

Run:
    python scripts/08_interpretation.py
    python scripts/08_interpretation.py --fast
    python scripts/08_interpretation.py --force --shap_n 3000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["CAPSTONE_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

import utils as U
import src.ensemble_models as EM
import src.interpretation_utils as IU

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> Dict:
    """Load train/val/test splits from 02_preprocess outputs."""
    processed = U.DIRS.processed
    
    def _read_df(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return pd.read_csv(csv_path)
    
    def _read_y(csv_path: Path, npy_path: Path) -> np.ndarray:
        if npy_path.exists():
            return np.load(npy_path).reshape(-1)
        y = pd.read_csv(csv_path)
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        return np.asarray(y).reshape(-1)
    
    X_train = _read_df(processed / "X_train_base.csv", processed / "X_train_base.parquet")
    X_val = _read_df(processed / "X_val_base.csv", processed / "X_val_base.parquet")
    X_test = _read_df(processed / "X_test_base.csv", processed / "X_test_base.parquet")
    
    y_train = _read_y(processed / "y_train.csv", processed / "y_train.npy")
    y_val = _read_y(processed / "y_val.csv", processed / "y_val.npy")
    y_test = _read_y(processed / "y_test.csv", processed / "y_test.npy")
    
    # Normalize dtypes
    X_train = EM.coerce_bool_to_int(X_train.reset_index(drop=True))
    X_val = EM.coerce_bool_to_int(X_val.reset_index(drop=True))
    X_test = EM.coerce_bool_to_int(X_test.reset_index(drop=True))
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "features": list(X_train.columns),
    }


# =============================================================================
# Model Loading
# =============================================================================

def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_all_models() -> Dict[str, Dict]:
    """Load models from 03/04/05/07."""
    models_dir = U.DIRS.models
    meta_dir = U.DIRS.meta
    
    inventory = {}
    
    # 03 - Logistic Regression
    logit_candidates = [
        models_dir / "logit_baseline.pkl",
        models_dir / "03_logit_baseline.pkl",
    ]
    logit_path = _first_existing(logit_candidates)
    if logit_path:
        inventory["Logistic"] = {
            "model": U.load_pickle(logit_path),
            "path": str(logit_path),
            "source": "03_logit",
        }
    
    # 04 - XGBoost
    xgb_candidates = [
        models_dir / "04_xgb_optuna.pkl",
        models_dir / "XGBoost_Optuna.pkl",
        models_dir / "04_xgb_baseline.pkl",
    ]
    xgb_path = _first_existing(xgb_candidates)
    if xgb_path:
        inventory["XGBoost"] = {
            "model": U.load_pickle(xgb_path),
            "path": str(xgb_path),
            "source": "04_trees_gbm",
        }
    
    # 04 - LightGBM
    lgbm_candidates = [
        models_dir / "04_lgbm_small.pkl",
        models_dir / "LightGBM_small_tuned_v1.pkl",
    ]
    lgbm_path = _first_existing(lgbm_candidates)
    if lgbm_path:
        inventory["LightGBM"] = {
            "model": U.load_pickle(lgbm_path),
            "path": str(lgbm_path),
            "source": "04_trees_gbm",
        }
    
    # 04 - RandomForest
    rf_candidates = [
        models_dir / "04_rf_baseline.pkl",
        models_dir / "RandomForest_baseline_v1.pkl",
    ]
    rf_path = _first_existing(rf_candidates)
    if rf_path:
        inventory["RandomForest"] = {
            "model": U.load_pickle(rf_path),
            "path": str(rf_path),
            "source": "04_trees_gbm",
        }
    
    # 05 - Neural Networks
    nn_candidates = [
        models_dir / "05_nn_best.pkl",
        models_dir / "05_nn_dense.pkl",
    ]
    nn_path = _first_existing(nn_candidates)
    if nn_path:
        inventory["NeuralNet"] = {
            "model": U.load_pickle(nn_path),
            "path": str(nn_path),
            "source": "05_nn",
        }
    
    # 07 - Stacking
    stacking_candidates = [
        models_dir / "07_stacking_oof.pkl",
        models_dir / "07_stacking_stacking_oof.pkl",
    ]
    stacking_path = _first_existing(stacking_candidates)
    if stacking_path:
        inventory["Stacking_OOF"] = {
            "model": U.load_pickle(stacking_path),
            "path": str(stacking_path),
            "source": "07_stacking",
        }
    
    if not inventory:
        raise FileNotFoundError("No trained models found. Run 03/04 first.")
    
    return inventory


# =============================================================================
# Threshold Policy
# =============================================================================

def load_threshold_policy() -> Tuple[float, str]:
    """Load fixed threshold from 04 or 07."""
    candidates = [
        (U.DIRS.meta / "04_best_threshold_rule.json", "04_trees_gbm"),
        (U.DIRS.meta / "06_threshold_policy.json", "06_autoencoder"),
        (U.DIRS.meta / "07_blend_weights.json", "07_stacking"),
    ]
    
    for path, source in candidates:
        if path.exists():
            obj = U.load_json(path)
            if isinstance(obj, dict):
                if "best_threshold" in obj:
                    return float(obj["best_threshold"]), source
                if "fixed_threshold" in obj:
                    return float(obj["fixed_threshold"]), source
    
    return 0.5, "default_fallback"


# =============================================================================
# Prediction Generation & Caching
# =============================================================================

def generate_predictions(
    models: Dict[str, Dict],
    data: Dict,
    cache_path: Path,
    force: bool = False,
) -> Dict[str, Dict]:
    """Generate val/test predictions for all models."""
    features = data["features"]
    
    # Check cache
    if cache_path.exists() and not force:
        print(f"  Loading cached predictions from {cache_path.name}")
        cached = np.load(cache_path)
        predictions = {}
        for name in models.keys():
            val_key = f"{name}_val"
            test_key = f"{name}_test"
            if val_key in cached and test_key in cached:
                predictions[name] = {
                    "val": cached[val_key],
                    "test": cached[test_key],
                }
        if len(predictions) == len(models):
            print(f"  [OK] All {len(models)} models loaded from cache")
            return predictions
    
    # Generate predictions
    print(f"  Generating predictions for {len(models)} models...")
    predictions = {}
    
    for name, info in models.items():
        model = info["model"]
        print(f"    - {name}...", end=" ")
        try:
            p_val = EM.get_predictions(model, data["X_val"][features], features_fallback=features)
            p_test = EM.get_predictions(model, data["X_test"][features], features_fallback=features)
            predictions[name] = {"val": p_val, "test": p_test}
            print("[OK]")
        except Exception as e:
            print(f" ({e})")
    
    # Save cache
    cache_data = {}
    for name, preds in predictions.items():
        cache_data[f"{name}_val"] = preds["val"]
        cache_data[f"{name}_test"] = preds["test"]
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache_data)
    print(f"  [OK] Predictions cached to {cache_path.name}")
    
    return predictions


def load_nn_cached_predictions() -> Dict[str, Dict]:
    """Load cached NN predictions from artifacts/preds."""
    preds_dir = U.DIRS.artifacts / "preds"
    if not preds_dir.exists():
        return {}
    
    nn_preds: Dict[str, Dict] = {}
    for path in sorted(preds_dir.glob("05_nn_*_test_proba.npy")):
        name = path.stem.replace("_test_proba", "")
        test_proba = np.load(path)
        
        val_path = preds_dir / f"{name}_val_proba.npy"
        val_proba = np.load(val_path) if val_path.exists() else None
        
        metrics_path = U.DIRS.meta / f"{name}_metrics.json"
        metrics_json = json.load(open(metrics_path)) if metrics_path.exists() else {}
        
        nn_preds[name] = {
            "val": val_proba,
            "test": test_proba,
            "precomputed_metrics": {
                "val_roc_auc": metrics_json.get("val_auc"),
                "val_pr_auc": metrics_json.get("val_pr_auc"),
                "test_roc_auc": metrics_json.get("test_auc"),
                "test_pr_auc": metrics_json.get("test_pr_auc"),
            },
            "source": "05_nn_cached",
        }
    
    return nn_preds


def load_ssl_cached_predictions() -> Dict[str, Dict]:
    """Load cached SSL predictions from 06_autoencoder."""
    preds_dir = U.DIRS.artifacts / "preds"
    if not preds_dir.exists():
        return {}
    
    ssl_preds: Dict[str, Dict] = {}
    for prefix in ["06_teacher", "06_student"]:
        test_path = preds_dir / f"{prefix}_test_proba.npy"
        val_path = preds_dir / f"{prefix}_val_proba.npy"
        metrics_path = U.DIRS.meta / f"{prefix}_metrics.json"
        
        p_test = np.load(test_path) if test_path.exists() else None
        p_val = np.load(val_path) if val_path.exists() else None
        metrics_json = json.load(open(metrics_path)) if metrics_path.exists() else {}
        
        if p_test is None and not metrics_json:
            continue
        
        ssl_preds[prefix] = {
            "val": p_val,
            "test": p_test,
            "precomputed_metrics": {
                "val_roc_auc": metrics_json.get("val_auc"),
                "val_pr_auc": metrics_json.get("val_pr_auc"),
                "test_roc_auc": metrics_json.get("test_auc"),
                "test_pr_auc": metrics_json.get("test_pr_auc"),
            },
            "source": "06_autoencoder_cached",
        }
    
    return ssl_preds


# =============================================================================
# Leaderboard Construction
# =============================================================================

def build_leaderboard(
    predictions: Dict[str, Dict],
    y_val: np.ndarray,
    y_test: np.ndarray,
    fixed_threshold: float,
) -> pd.DataFrame:
    """Build unified leaderboard with all metrics."""
    rows = []
    
    for name, preds in predictions.items():
        p_val = preds.get("val")
        p_test = preds.get("test")
        metrics_override = preds.get("precomputed_metrics", {})
        
        # Validation metrics
        if p_val is not None:
            val_roc = roc_auc_score(y_val, p_val)
            val_pr = average_precision_score(y_val, p_val)
        else:
            val_roc = metrics_override.get("val_roc_auc", np.nan)
            val_pr = metrics_override.get("val_pr_auc", np.nan)
        
        # Test metrics
        if p_test is not None:
            test_roc = roc_auc_score(y_test, p_test)
            test_pr = average_precision_score(y_test, p_test)
            test_brier = brier_score_loss(y_test, p_test)
            y_pred = (p_test >= fixed_threshold).astype(int)
            test_f1 = f1_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
        else:
            test_roc = metrics_override.get("test_roc_auc", np.nan)
            test_pr = metrics_override.get("test_pr_auc", np.nan)
            test_brier = np.nan
            test_f1 = np.nan
            test_recall = np.nan
            test_precision = np.nan
        
        rows.append({
            "model": name,
            "val_roc_auc": val_roc,
            "val_pr_auc": val_pr,
            "test_roc_auc": test_roc,
            "test_pr_auc": test_pr,
            "test_brier": test_brier,
            "test_f1": test_f1,
            "test_recall": test_recall,
            "test_precision": test_precision,
            "threshold": fixed_threshold,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(["val_roc_auc", "val_pr_auc"], ascending=False).reset_index(drop=True)
    return df


def select_champion(leaderboard: pd.DataFrame) -> str:
    """Select champion model (highest val_roc_auc)."""
    return leaderboard.iloc[0]["model"]


# =============================================================================
# SHAP Analysis
# =============================================================================

def run_shap_analysis(
    champion_name: str,
    model,
    data: Dict,
    shap_n: int,
    seed: int,
    output_dir: Path,
    fast: bool = False,
) -> Dict:
    """Run SHAP analysis on champion model."""
    if not IU.SHAP_AVAILABLE:
        print("  [WARN] SHAP not installed, skipping")
        return {}
    
    base_estimator = IU.get_final_estimator(model)
    if not IU.is_tree_model(base_estimator):
        print(f"  [WARN] {champion_name} is not tree-based; skipping SHAP")
        return {}
    if IU.is_pipeline(model):
        print(f"  [WARN] {champion_name} is a pipeline; skipping SHAP")
        return {}
    
    print(f"\n  Running SHAP analysis on {champion_name}...")
    
    idx_path = U.DIRS.meta / "08_shap_sample_idx.json"
    shap_values_path = U.DIRS.meta / "08_shap_values.npy"
    shap_sample_path = U.DIRS.meta / "08_shap_X_sample.parquet"
    
    X_sample, sample_idx = IU.sample_for_shap(
        data["X_test"], n=shap_n, seed=seed, cache_path=idx_path,
    )
    print(f"    Sampled {len(X_sample)} rows")
    
    if shap_values_path.exists() and shap_sample_path.exists():
        print("    Loading cached SHAP values...")
        shap_values = np.load(shap_values_path)
        X_sample = pd.read_parquet(shap_sample_path)
    else:
        print("    Computing SHAP values...")
        shap_values = IU.compute_shap_values(
            model, X_sample,
            X_background=data["X_train"].sample(min(100, len(data["X_train"])), random_state=seed),
        )
        print(f"    [OK] SHAP values: {shap_values.shape}")
        np.save(shap_values_path, shap_values)
        X_sample.to_parquet(shap_sample_path, index=False)
    
    top_features = IU.get_top_features_by_shap(
        shap_values, feature_names=list(X_sample.columns), top_k=20,
    )
    
    print("    Generating SHAP plots...")
    
    IU.plot_shap_summary(shap_values, X_sample, 
                         save_path=output_dir / "08_shap_summary.png", max_display=20)
    print(f"      [OK] 08_shap_summary.png")
    
    IU.plot_shap_importance(shap_values, X_sample,
                            save_path=output_dir / "08_shap_importance.png", max_display=20)
    print(f"      [OK] 08_shap_importance.png")
    
    n_dependence = 1 if fast else 2
    for i, feat in enumerate(top_features[:n_dependence]):
        IU.plot_shap_dependence(
            shap_values, X_sample, feature_name=feat,
            save_path=output_dir / f"08_shap_dependence_{i+1}_{feat}.png",
        )
        print(f"      [OK] 08_shap_dependence_{i+1}_{feat}.png")
    
    return {
        "shap_values": shap_values,
        "X_sample": X_sample,
        "sample_indices": sample_idx,
        "top_features": top_features,
    }


# =============================================================================
# Semi-supervised Learning Appendix (from 06)
# =============================================================================

def generate_ssl_appendix(output_dir: Path) -> Dict:
    """
    Generate Semi-supervised Learning appendix section.
    Integrates results from 06_autoencoder.py (real holdout experiment).
    
    Returns: dict with SSL analysis summary
    """
    print("\n" + "=" * 60)
    print("Appendix: Semi-supervised Learning with Real Holdout")
    print("=" * 60)
    
    meta_dir = U.DIRS.meta
    figures_dir = U.DIRS.figures
    tables_dir = U.DIRS.tables
    preds_dir = U.DIRS.artifacts / "preds"
    
    ssl_summary = {"enabled": False}
    
    # 1. Load pseudo-label stats
    stats_path = meta_dir / "06_pseudolabel_stats.json"
    if not stats_path.exists():
        print("  [WARN] No 06_pseudolabel_stats.json found. Run 06_autoencoder.py first.")
        return ssl_summary
    
    with open(stats_path, "r") as f:
        pseudo_stats = json.load(f)
    
    print(f"\n  Pseudo-label Statistics:")
    print(f"    Total unlabeled samples: {pseudo_stats['total_unlabeled']}")
    print(f"    High-confidence labels: {pseudo_stats['high_conf_total']} ({pseudo_stats['high_conf_ratio']*100:.1f}%)")
    print(f"    Thresholds: high={pseudo_stats['threshold_high']}, low={pseudo_stats['threshold_low']}")
    print(f"    Mode: {'Real holdout' if pseudo_stats.get('use_real_holdout') else 'Simulated'}")
    
    if pseudo_stats.get('pseudo_accuracy') is not None:
        print(f"    Pseudo-label accuracy: {pseudo_stats['pseudo_accuracy']:.2%}")
    
    ssl_summary.update({
        "enabled": True,
        "pseudo_stats": pseudo_stats,
    })
    
    # 2. Load teacher/student results
    results_path = tables_dir / "06_pseudolabel_results.csv"
    if results_path.exists():
        ssl_results = pd.read_csv(results_path)
        print(f"\n  Teacher-Student Results:")
        print(ssl_results.to_string(index=False))
        ssl_summary["ssl_results"] = ssl_results.to_dict(orient="records")
    
    # 3. Load ablation results
    ablation_path = tables_dir / "06_autoencoder_ablation.csv"
    if ablation_path.exists():
        ablation_df = pd.read_csv(ablation_path)
        print(f"\n  AE Ablation Study (Latent Feature Augmentation):")
        cols_to_show = ["feature_set", "test_roc_auc", "test_pr_auc"] if "feature_set" in ablation_df.columns else ablation_df.columns[:5]
        print(ablation_df[cols_to_show].to_string(index=False))
        ssl_summary["ablation_results"] = ablation_df.to_dict(orient="records")
    
    # 4. Copy confidence distribution plot to 08 outputs
    conf_plot_src = figures_dir / "06_pseudo_confidence_dist.png"
    conf_plot_dst = output_dir / "08_ssl_confidence_dist.png"
    if conf_plot_src.exists():
        shutil.copy(conf_plot_src, conf_plot_dst)
        print(f"\n  [OK] Copied confidence distribution plot to 08_ssl_confidence_dist.png")
        ssl_summary["confidence_plot"] = str(conf_plot_dst)
    
    # 5. Generate SSL summary figure
    if ssl_summary.get("ssl_results") and len(ssl_summary["ssl_results"]) >= 1:
        _plot_ssl_summary(
            ssl_summary["ssl_results"],
            pseudo_stats,
            save_path=output_dir / "08_ssl_summary.png"
        )
        print(f"  [OK] Generated 08_ssl_summary.png")
    
    # 6. Key findings
    print(f"\n  Key Findings:")
    if pseudo_stats['high_conf_ratio'] < 0.01:
        print(f"    - Coverage too low ({pseudo_stats['high_conf_ratio']*100:.1f}%) for meaningful SSL improvement")
        print(f"    - Teacher model produced conservative probability estimates on real holdout")
        print(f"    - This gap between simulated vs real unlabeled data is expected in production")
    else:
        print(f"    - {pseudo_stats['high_conf_total']} high-confidence pseudo-labels generated")
        if pseudo_stats.get('pseudo_accuracy'):
            print(f"    - Pseudo-label accuracy: {pseudo_stats['pseudo_accuracy']:.2%}")
    
    return ssl_summary


def _plot_ssl_summary(
    ssl_results: List[Dict],
    pseudo_stats: Dict,
    save_path: Path
) -> None:
    """Generate SSL summary visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: Teacher vs Student
    ax1 = axes[0]
    models = [r.get("model", "Unknown") for r in ssl_results]
    aucs = [r.get("test_auc", 0) for r in ssl_results]
    
    colors = ['steelblue', 'forestgreen'][:len(models)]
    bars = ax1.bar(models, aucs, color=colors, edgecolor='black')
    ax1.set_ylabel('Test ROC-AUC', fontsize=12)
    ax1.set_title('Semi-supervised Learning: Teacher vs Student', fontsize=14)
    ax1.set_ylim([0.5, max(aucs) * 1.1] if aucs else [0.5, 1.0])
    for bar, auc in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{auc:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart: Pseudo-label coverage
    ax2 = axes[1]
    n_total = pseudo_stats['total_unlabeled']
    n_high = pseudo_stats['high_conf_total']
    n_uncertain = n_total - n_high
    
    sizes = [n_high, n_uncertain]
    labels = [f'High-conf\n{n_high} ({n_high/n_total*100:.1f}%)',
              f'Uncertain\n{n_uncertain} ({n_uncertain/n_total*100:.1f}%)']
    colors = ['#2ecc71', '#95a5a6']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax2.set_title(f'Pseudo-label Coverage (n={n_total})', fontsize=14)
    
    mode_text = "Real Holdout" if pseudo_stats.get('use_real_holdout') else "Simulated"
    fig.suptitle(f'Semi-supervised Learning Summary ({mode_text})', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Plotting
# =============================================================================

def generate_plots(
    predictions: Dict[str, Dict],
    y_test: np.ndarray,
    champion_name: str,
    champion_preds: np.ndarray,
    fixed_threshold: float,
    output_dir: Path,
) -> None:
    """Generate all comparison plots."""
    print("\n  Generating comparison plots...")
    
    # Filter predictions with valid test proba
    valid_predictions = {k: v for k, v in predictions.items() if v.get("test") is not None}
    
    # ROC comparison
    roc_data = {}
    for name, preds in valid_predictions.items():
        p_test = preds["test"]
        auc_val = roc_auc_score(y_test, p_test)
        roc_data[name] = {"proba": p_test, "auc": auc_val}
    
    IU.plot_roc_comparison(
        roc_data, y_test,
        save_path=output_dir / "08_model_comparison_roc.png",
        title="ROC Curve Comparison - Test Set",
    )
    print(f"    [OK] 08_model_comparison_roc.png")
    
    # PR comparison
    pr_data = {}
    for name, preds in valid_predictions.items():
        p_test = preds["test"]
        pr_auc_val = average_precision_score(y_test, p_test)
        pr_data[name] = {"proba": p_test, "pr_auc": pr_auc_val}
    
    IU.plot_pr_comparison(
        pr_data, y_test,
        save_path=output_dir / "08_model_comparison_pr.png",
        title="Precision-Recall Curve Comparison - Test Set",
    )
    print(f"    [OK] 08_model_comparison_pr.png")
    
    # Confusion matrix (champion)
    if champion_preds is not None:
        y_pred = (champion_preds >= fixed_threshold).astype(int)
        IU.plot_confusion_matrix(
            y_test, y_pred,
            save_path=output_dir / "08_confusion_matrix.png",
            title=f"Confusion Matrix - {champion_name} (threshold={fixed_threshold:.3f})",
        )
        print(f"    [OK] 08_confusion_matrix.png")


# =============================================================================
# Output Management
# =============================================================================

def save_outputs(
    leaderboard: pd.DataFrame,
    models: Dict[str, Dict],
    predictions: Dict[str, Dict],
    champion_name: str,
    fixed_threshold: float,
    threshold_source: str,
    shap_results: Dict,
    ssl_summary: Dict,
    seed: int,
) -> None:
    """Save all outputs and manifest."""
    U.ensure_dirs()
    
    U.save_df(leaderboard, U.DIRS.tables / "08_model_leaderboard.csv")
    
    manifest = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "seed": seed,
        "threshold": {"value": fixed_threshold, "source": threshold_source},
        "models": {
            name: {"path": info["path"], "source": info["source"]}
            for name, info in models.items()
        },
        "champion": {
            "model": champion_name,
            "test_roc_auc": float(leaderboard.iloc[0]["test_roc_auc"]),
            "test_pr_auc": float(leaderboard.iloc[0]["test_pr_auc"]),
        },
        "shap": {
            "enabled": bool(shap_results),
            "sample_size": len(shap_results.get("sample_indices", [])) if shap_results else 0,
            "top_features": shap_results.get("top_features", [])[:10] if shap_results else [],
        },
        "ssl_appendix": {
            "enabled": ssl_summary.get("enabled", False),
            "pseudo_stats": ssl_summary.get("pseudo_stats"),
        },
        "outputs": {
            "core_tables": ["08_model_leaderboard.csv"],
            "core_figures": [
                "08_model_comparison_roc.png",
                "08_shap_summary.png",
                "08_confusion_matrix.png",
            ],
            "appendix_figures": [
                "08_model_comparison_pr.png",
                "08_shap_importance.png",
                "08_ssl_summary.png",
                "08_ssl_confidence_dist.png",
            ],
        },
    }
    
    U.save_json(manifest, U.DIRS.meta / "manifest_08_interpretation.json")
    print(f"\n  [OK] Manifest saved")


# =============================================================================
# Main Pipeline
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="08 - Model Interpretation & Error Analysis")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shap_n", type=int, default=2000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_ssl", action="store_true", help="Skip SSL appendix")
    return ap.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("08 - Model Interpretation & Error Analysis Pipeline")
    print("=" * 70)
    print(f"Config: seed={args.seed}, shap_n={args.shap_n}, fast={args.fast}")
    
    # Load data
    print("\n[1/8] Loading data...")
    data = load_data()
    print(f"  [OK] Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
    
    # Load models
    print("\n[2/8] Loading trained models...")
    models = load_all_models()
    print(f"  [OK] Loaded {len(models)} models")
    for name, info in models.items():
        print(f"    - {name:20s} from {info['source']}")
    
    # Load threshold
    print("\n[3/8] Loading threshold policy...")
    fixed_threshold, threshold_source = load_threshold_policy()
    print(f"  [OK] Fixed threshold: {fixed_threshold:.4f} (source: {threshold_source})")
    
    # Generate predictions
    print("\n[4/8] Generating predictions...")
    cache_path = U.DIRS.meta / "08_cached_proba.npz"
    predictions = generate_predictions(models, data, cache_path, force=args.force)
    
    # Add cached NN predictions
    nn_cached = load_nn_cached_predictions()
    if nn_cached:
        predictions.update(nn_cached)
        for name in nn_cached.keys():
            models[name] = {"model": None, "path": "artifacts/preds", "source": "05_nn_cached"}
        print(f"  [OK] Added {len(nn_cached)} cached NN models")
    
    # Add cached SSL predictions
    ssl_cached = load_ssl_cached_predictions()
    if ssl_cached:
        predictions.update(ssl_cached)
        for name in ssl_cached.keys():
            models[name] = {"model": None, "path": "artifacts/preds", "source": "06_ssl_cached"}
        print(f"  [OK] Added {len(ssl_cached)} cached SSL models")
    
    # Build leaderboard
    print("\n[5/8] Building unified leaderboard...")
    leaderboard = build_leaderboard(predictions, data["y_val"], data["y_test"], fixed_threshold)
    print("\n  Leaderboard (sorted by Val ROC-AUC):")
    cols = ["model", "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc", "test_f1"]
    print(leaderboard.head(10)[cols].to_string(index=False))
    
    champion_name = select_champion(leaderboard)
    print(f"\n  Champion Model: {champion_name}")
    
    # SHAP analysis
    print("\n[6/8] SHAP Explainability Analysis")
    champion_model = models.get(champion_name, {}).get("model")
    if champion_model is None:
        print(f"  [WARN] Champion {champion_name} has no model object; skipping SHAP")
        shap_results = {}
    else:
        try:
            shap_results = run_shap_analysis(
                champion_name, champion_model, data,
                shap_n=args.shap_n, seed=args.seed,
                output_dir=U.DIRS.figures, fast=args.fast,
            )
        except Exception as e:
            print(f"  [WARN] SHAP failed ({e}); skipping")
            shap_results = {}
    
    # Generate plots
    print("\n[7/8] Generating comparison plots...")
    champion_preds = predictions.get(champion_name, {}).get("test")
    generate_plots(
        predictions, data["y_test"],
        champion_name, champion_preds, fixed_threshold,
        output_dir=U.DIRS.figures,
    )
    
    # SSL Appendix
    ssl_summary = {}
    if not args.skip_ssl:
        ssl_summary = generate_ssl_appendix(output_dir=U.DIRS.figures)
    
    # Save outputs
    print("\n[8/8] Saving outputs...")
    save_outputs(
        leaderboard, models, predictions,
        champion_name, fixed_threshold, threshold_source,
        shap_results, ssl_summary, args.seed,
    )
    
    print("\n" + "=" * 70)
    print("08 - Interpretation Pipeline Completed Successfully!")
    print("=" * 70)
    print("\nCore Outputs:")
    print("  - artifacts/tables/08_model_leaderboard.csv")
    print("  - artifacts/figures/08_model_comparison_roc.png")
    print("  - artifacts/figures/08_shap_summary.png")
    print("  - artifacts/figures/08_confusion_matrix.png")
    if ssl_summary.get("enabled"):
        print("\nSSL Appendix:")
        print("  - artifacts/figures/08_ssl_summary.png")
        print("  - artifacts/figures/08_ssl_confidence_dist.png")
    print("\nManifest:")
    print("  - artifacts/meta/manifest_08_interpretation.json")

    val_pred = predictions.get("05_nn_wide_deep_focal", {}).get("val")
    if val_pred is None:
        print("val_pred is None (no val predictions found for 05_nn_wide_deep_focal)")
        print("predictions keys:", list(predictions.keys()))
        print("05_nn_wide_deep_focal keys:",
              list(predictions.get("05_nn_wide_deep_focal", {}).keys()))
    else:
        y_val = data["y_val"]
        if isinstance(y_val, pd.Series):
            if y_val.dtype == object:
                y_val = y_val.map({"Yes": 1, "No": 0}).astype(int)
            mask = y_val.notna()
            y_val = y_val[mask].to_numpy()
        else:
            y_val = np.asarray(y_val)
            mask = ~pd.isna(y_val)
            y_val = y_val[mask]
        print("y_val counts:", np.unique(y_val, return_counts=True))
        print("val_pred NaN:", np.isnan(val_pred).any())
        print("val_pred min/max:", np.min(val_pred), np.max(val_pred))
        print("leaderboard val_pr_auc:",
              leaderboard.loc[leaderboard["model"] == "05_nn_wide_deep_focal", "val_pr_auc"])


if __name__ == "__main__":
    main()
