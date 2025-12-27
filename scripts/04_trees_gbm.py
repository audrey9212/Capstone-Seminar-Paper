"""
Tree-based Models Training Script (04)
======================================

Train and evaluate multiple tree models (Decision Tree, Random Forest, XGBoost, LightGBM),
and generate comparison tables, ROC/PR curves, and threshold tuning results.

This script runs the following steps:
1. Load processed data from 02_preprocess
2. Select tree-compatible features from feature_config
3. Train multiple baseline + tuned models
4. Tune thresholds on the validation set (recall constraint)
5. Evaluate all models and produce tables and figures

Outputs:
- models/: trained model files
- artifacts/figures/: ROC+PR curves (single best + all models)
- artifacts/tables/: leaderboard, threshold sweep
- artifacts/meta/: best params, threshold rule, manifest

Usage:
    python scripts/04_trees_gbm.py                    # full training (with Optuna)
    python scripts/04_trees_gbm.py --fast             # fast mode (skip Optuna)
    python scripts/04_trees_gbm.py --force_retrain    # force retrain
    python scripts/04_trees_gbm.py --n_trials 100     # custom Optuna trials
"""


import os
import sys
from pathlib import Path
import argparse
import warnings
import json

# Project setup - must be done before importing utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["CAPSTONE_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Add src to path for tree_models
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import joblib

import utils as U

# Import tree_models module
try:
    import src.tree_models as TM
except ImportError:
    print("[ERROR] ERROR: tree_models.py not found in src/")
    print("   Please ensure src/tree_models.py exists.")
    sys.exit(1)

# Import optuna_utils for visualization
try:
    import src.optuna_utils as OU
except ImportError:
    OU = None
    print("[WARN]  optuna_utils.py not found. Optuna tuning will be unavailable.")

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
candidates = [
    U.DIRS.data / "feature_config_clean.csv",
    U.DIRS.root / "feature_config_clean.csv",
]
FEATURE_CONFIG = next((p for p in candidates if p.exists()), None)
feature_config_path=FEATURE_CONFIG


# Threshold tuning config
MIN_RECALL_CONSTRAINT = 0.70
THRESHOLD_RANGE = np.arange(0.1, 0.6, 0.01)


# ============================================================================
# Main Pipeline
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate tree-based models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip Optuna tuning, only train baselines + LightGBM"
    )
    
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining even if cached models exist"
    )
    
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials for XGBoost tuning"
    )
    
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_args()
    
    print("\\n" + "="*80)
    print("TREE-BASED MODELS TRAINING PIPELINE")
    print("="*80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Fast mode: {args.fast}")
    print(f"Force retrain: {args.force_retrain}")
    if not args.fast:
        print(f"Optuna trials: {args.n_trials}")
    
    # Reset run log at the start
    U.reset_run_log()
    
    # -----------------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------------
    try:
        data = TM.load_processed_splits(DATA_DIR, verbose=True)
    except FileNotFoundError as e:
        print(f"\\n[ERROR] ERROR: {e}")
        sys.exit(1)
    
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    
    # -----------------------------------------------------------------------
    # 2. Feature Selection for Trees
    # -----------------------------------------------------------------------
    X_train_tree, X_val_tree, X_test_tree, tree_features = TM.select_tree_features(
        X_train, X_val, X_test,
        feature_config_path=FEATURE_CONFIG,
        verbose=True,
    )
    
    # -----------------------------------------------------------------------
    # 3. Train Models
    # -----------------------------------------------------------------------
    models = {}
    model_infos = {}
    
    print("\\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    # 3.1 Decision Tree Baseline
    try:
        models["dt_baseline"] = TM.fit_dt_baseline(
            X_train_tree, y_train, max_depth=10, verbose=True
        )
        # Save model
        model_path = U.DIRS.models / "04_dt_baseline.pkl"
        joblib.dump(models["dt_baseline"], model_path)
        print(f"  [OK] Saved: {model_path}")
    except Exception as e:
        print(f"  [WARN]  Decision Tree training failed: {e}")
    
    # 3.2 Random Forest Baseline
    try:
        models["rf_baseline"] = TM.fit_rf_baseline(
            X_train_tree, y_train, n_estimators=100, verbose=True
        )
        # Save model
        model_path = U.DIRS.models / "04_rf_baseline.pkl"
        joblib.dump(models["rf_baseline"], model_path)
        print(f"  [OK] Saved: {model_path}")
    except Exception as e:
        print(f"  [WARN]  Random Forest training failed: {e}")
    
    # 3.3 XGBoost Baseline
    if TM.HAS_XGB:
        try:
            models["xgb_baseline"] = TM.fit_xgb_baseline(
                X_train_tree, y_train, verbose=True
            )
            # Save model
            model_path = U.DIRS.models / "04_xgb_baseline.pkl"
            joblib.dump(models["xgb_baseline"], model_path)
            print(f"  [OK] Saved: {model_path}")
        except Exception as e:
            print(f"  [WARN]  XGBoost baseline training failed: {e}")
    else:
        print("\\n[WARN]  XGBoost not installed, skipping XGBoost baseline")
    
    # 3.4 XGBoost with Optuna (optional)
    if not args.fast and TM.HAS_XGB and OU is not None:
        try:
            # Create checkpoint directory for Optuna
            optuna_checkpoint_dir = U.DIRS.tuner / "04_optuna"
            optuna_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            models["xgb_optuna"], model_infos["xgb_optuna"] = TM.fit_xgb_optuna_or_load(
                X_train=X_train_tree,
                y_train=y_train,
                X_val=X_val_tree,
                y_val=y_val,
                X_test=X_test_tree,
                y_test=y_test,
                checkpoint_dir=optuna_checkpoint_dir,
                n_trials=args.n_trials,
                force_retrain=args.force_retrain,
                verbose=True,
            )
            
            # Save best params
            if "best_params" in model_infos["xgb_optuna"]:
                best_params = {
                    k: v for k, v in model_infos["xgb_optuna"]["best_params"].items()
                    if k not in ["random_state", "n_jobs", "verbosity"]
                }
                U.save_json(
                    best_params,
                    "04_xgb_optuna_best_params",
                    folder=U.DIRS.meta,
                )
            
            # Generate Optuna visualizations
            print("\\n[TABLES] Generating Optuna visualizations...")
            optuna_fig_dir = U.DIRS.figures / "optuna"
            optuna_fig_dir.mkdir(parents=True, exist_ok=True)
            
            OU.plot_optuna_optimization(
                model_infos["xgb_optuna"],
                save_dir=str(optuna_fig_dir),
                show_inline=False, 
            )
            
            # Manually log Optuna figures to manifest
            for fp in sorted(list(optuna_fig_dir.glob("*.png")) + 
                           list(optuna_fig_dir.glob("*.html"))):
                U._log_figure(fp)
                
        except Exception as e:
            print(f"  [WARN]  XGBoost Optuna tuning failed: {e}")
            print(f"     Continuing with baseline models only...")
    

    # ==========================================================================
    # [NEW] 1. Search Space CSV (Using Explicit Relative Paths)
    # ==========================================================================
    project_root = Path(__file__).resolve().parent.parent
    
    target_tables_dir = project_root / "artifacts" / "tables"
    target_figures_dir = project_root / "artifacts" / "figures"
    
    target_tables_dir.mkdir(parents=True, exist_ok=True)
    target_figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving files to:")
    print(f"  - Tables:  {target_tables_dir}")
    print(f"  - Figures: {target_figures_dir}")

    xgb_search_space = [
        {"Parameter": "max_depth", "Type": "Integer", "Range": "3 - 10", "Step": 1},
        {"Parameter": "learning_rate", "Type": "Float (Log)", "Range": "0.01 - 0.3", "Step": "-"},
        {"Parameter": "n_estimators", "Type": "Integer", "Range": "100 - 500", "Step": 50},
        {"Parameter": "subsample", "Type": "Float", "Range": "0.6 - 1.0", "Step": "-"},
        {"Parameter": "colsample_bytree", "Type": "Float", "Range": "0.6 - 1.0", "Step": "-"},
        {"Parameter": "min_child_weight", "Type": "Integer", "Range": "1 - 10", "Step": 1},
        {"Parameter": "gamma", "Type": "Float", "Range": "0.0 - 5.0", "Step": "-"},
        {"Parameter": "reg_alpha", "Type": "Float", "Range": "0.0 - 1.0", "Step": "-"},
        {"Parameter": "reg_lambda", "Type": "Float", "Range": "0.0 - 1.0", "Step": "-"}
    ]
    search_space_df = pd.DataFrame(xgb_search_space)
    
    csv_path = target_tables_dir / "04_hyperparameter_search_space.csv"
    search_space_df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved search space CSV: {csv_path}")

    # ==========================================================================
    # [NEW] 2. Decision Tree Visualization (Depth 3) - BETTER FOR APPENDIX
    # ==========================================================================
    try:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        import joblib
        
        if "dt_baseline" in models:
            dt_model = models["dt_baseline"]
        else:
            dt_path = U.DIRS.models / "04_dt_baseline.pkl"
            print(f"  [INFO] Loading DT from: {dt_path}")
            dt_model = joblib.load(dt_path)
            
        if hasattr(dt_model, "named_steps"):
            dt_model = dt_model.named_steps["model"]
            
        plt.figure(figsize=(24, 10), dpi=300)
        
        plot_tree(
            dt_model, 
            max_depth=3, 
            feature_names=X_train.columns.tolist(),
            class_names=["No Churn", "Churn"],
            filled=True, 
            fontsize=10,
            rounded=True
        )
        
        plt.title("Baseline Decision Tree Rules (Top 3 Levels)", fontsize=16)
        
        dt_viz_path = target_figures_dir / "04_decision_tree_viz_depth3.png"
        plt.savefig(dt_viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  [OK] Saved Decision Tree visualization: {dt_viz_path}")
        
    except Exception as e:
        print(f"  [WARN] Failed to plot Decision Tree: {e}")

    # 3.5 LightGBM Small Tuned (always run, fast)
    if TM.HAS_LGBM:
        try:
            models["lgbm_small"] = TM.fit_lgbm_small(
                X_train_tree, y_train, verbose=True
            )
            # Save model
            model_path = U.DIRS.models / "04_lgbm_small.pkl"
            joblib.dump(models["lgbm_small"], model_path)
            print(f"  [OK] Saved: {model_path}")
        except Exception as e:
            print(f"  [WARN]  LightGBM training failed: {e}")
    else:
        print("\\n[WARN]  LightGBM not installed, skipping LightGBM")
    
    # -----------------------------------------------------------------------
    # 4. Evaluate All Models
    # -----------------------------------------------------------------------
    print("\\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    leaderboard_rows = []
    
    for model_name, model in models.items():
        print(f"\\nEvaluating: {model_name}")
        
        # Evaluate on val and test
        for split_name, X, y in [
            ("val", X_val_tree, y_val),
            ("test", X_test_tree, y_test),
        ]:
            metrics, _ = TM.evaluate_binary_classifier(
                model, X, y, threshold=0.5, split_name=split_name
            )
            
            # Add model name
            metrics["model"] = model_name
            leaderboard_rows.append(metrics)
            
            # Print key metrics
            print(f"  {split_name.capitalize():5s}: "
                  f"AUC={metrics['roc_auc']:.4f} "
                  f"PR-AUC={metrics['pr_auc']:.4f} "
                  f"Brier={metrics['brier_score']:.4f}")
            
            # Log to model results
            U.log_model_result(
                model_name=f"04_{model_name}_{split_name}",
                params={},  # Could add hyperparams here
                metrics=metrics,
                notes=f"{model_name} on {split_name} set"
            )
    
    # Create leaderboard
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    
    # Reorder columns for better readability
    col_order = [
        "model", "split", "roc_auc", "pr_auc", "brier_score",
        "threshold", "precision", "recall", "f1_score",
        "true_negatives", "false_positives", "false_negatives", "true_positives",
    ]
    leaderboard_df = leaderboard_df[col_order]
    
    # Save leaderboard
    U.save_df(leaderboard_df, "04_tree_leaderboard", folder=U.DIRS.tables)
    
    print("\\n" + "="*80)
    print("LEADERBOARD (sorted by Val AUC)")
    print("="*80)
    val_only = leaderboard_df[leaderboard_df["split"] == "val"].sort_values(
        "roc_auc", ascending=False
    )
    print(val_only[["model", "roc_auc", "pr_auc", "brier_score"]].to_string(index=False))
    
    # -----------------------------------------------------------------------
    # 5. Threshold Tuning on Best Model
    # -----------------------------------------------------------------------
    # Select best model based on validation AUC
    val_metrics = leaderboard_df[leaderboard_df["split"] == "val"]
    best_model_name = val_metrics.loc[val_metrics["roc_auc"].idxmax(), "model"]
    best_model = models[best_model_name]
    
    print("\\n" + "="*80)
    print(f"THRESHOLD TUNING: {best_model_name.upper()}")
    print("="*80)
    
    optimal_threshold, threshold_metrics = TM.tune_threshold_with_constraint(
        best_model,
        X_val_tree,
        y_val,
        min_recall=MIN_RECALL_CONSTRAINT,
        threshold_range=THRESHOLD_RANGE,
        verbose=True,
    )
    
    # Save threshold results
    U.save_df(threshold_metrics, "04_best_threshold_table_val", folder=U.DIRS.tables)
    
    threshold_rule = {
        "model": best_model_name,
        "rule": f"maximize F1 subject to recall >= {MIN_RECALL_CONSTRAINT:.2f}",
        "best_threshold": float(optimal_threshold),
        "min_recall_constraint": MIN_RECALL_CONSTRAINT,
        "val_precision": float(threshold_metrics.loc[
            threshold_metrics["threshold"] == optimal_threshold, "precision"
        ].values[0]),
        "val_recall": float(threshold_metrics.loc[
            threshold_metrics["threshold"] == optimal_threshold, "recall"
        ].values[0]),
        "val_f1": float(threshold_metrics.loc[
            threshold_metrics["threshold"] == optimal_threshold, "f1"
        ].values[0]),
    }
    
    U.save_json(threshold_rule, "04_best_threshold_rule", folder=U.DIRS.meta)
    
    # -----------------------------------------------------------------------
    # 6. Generate Visualizations
    # -----------------------------------------------------------------------
    print("\\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 6.1 Best model ROC+PR with threshold point (Test set)
    print("\\n[FIGURES] Best model ROC+PR curves with threshold point...")
    _, y_proba_test = TM.evaluate_binary_classifier(
        best_model, X_test_tree, y_test, threshold=optimal_threshold, split_name="test"
    )
    
    fig1 = TM.plot_roc_pr_with_point(
        y_true=y_test.values.ravel(),
        y_proba=y_proba_test,
        threshold=optimal_threshold,
        model_name=best_model_name,
        split_name="Test",
        save_path=None,
    )
    U.save_fig(fig1, "04_best_roc_pr_point_test")
    
    # 6.2 All models comparison (Test set)
    print("\\n[FIGURES] All models ROC+PR comparison...")
    models_for_plot = {
        name: (model, X_test_tree, y_test)
        for name, model in models.items()
    }
    
    fig2 = TM.plot_all_models_roc_pr(
        models_dict=models_for_plot,
        split_name="Test",
        save_path=None,
    )
    U.save_fig(fig2, "04_all_tree_roc_pr_test")
    
    # -----------------------------------------------------------------------
    # 7. Write Manifest
    # -----------------------------------------------------------------------
    print("\\n" + "="*80)
    print("WRITING MANIFEST")
    print("="*80)
    
    core_figures = ["04_best_roc_pr_point_test.png"]
    core_tables = ["04_tree_leaderboard.csv"]
    
    appendix_figures = ["04_all_tree_roc_pr_test.png"]
    appendix_tables = ["04_best_threshold_table_val.csv"]
    
    U.write_manifest(
        run_name="04_trees_gbm",
        core_figures=core_figures,
        core_tables=core_tables,
        reset=True,
    )
    
    # -----------------------------------------------------------------------
    # 8. Summary
    # -----------------------------------------------------------------------
    print("\\n" + "="*80)
    print("TREE-BASED MODELS TRAINING - COMPLETED")
    print("="*80)
    print(f"\\n[OK] Best model: {best_model_name}")
    print(f"[OK] Optimal threshold: {optimal_threshold:.4f}")
    print(f"[OK] Val Recall: {threshold_rule['val_recall']:.4f}")
    print(f"[OK] Val F1: {threshold_rule['val_f1']:.4f}")
    print(f"\\n[OK] Models saved to: {U.DIRS.models}")
    print(f"[OK] Artifacts saved to: {PROJECT_ROOT / 'artifacts'}")
    print(f"[OK] Manifest: artifacts/meta/manifest_04_trees_gbm.json")
    print("\\n" + "="*80)


if __name__ == "__main__":
    main()
