# scripts/07_stacking.py
"""
07 - Ensemble (Blending + Stacking) on encoded X_base features.

Design principles (thesis + reproducibility):
- Treat 02_preprocess.py outputs (X_*_base) as the single source of truth.
- Do NOT one-hot encode again here; base features are already encoded numeric/binary.
- Robustly align features when loading older models (feature_names_in_).

Outputs (artifacts/):
- tables/07_ensemble_leaderboard.csv
- tables/07_base_pred_val.csv, tables/07_base_pred_test.csv
- meta/07_blend_weights.json (for weighted + NNLS)
- meta/manifest_07_stacking.json

Run:
    python scripts/07_stacking.py
    python scripts/07_stacking.py --force_retrain
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Project setup - must be before importing utils / src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["CAPSTONE_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import utils as U
import src.ensemble_models as EM


# ---------------------------
# IO helpers
# ---------------------------

def _read_df(path_csv: Path, path_parquet: Path) -> pd.DataFrame:
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
    return pd.read_csv(path_csv)

def _read_y(path_csv: Path, path_npy: Path):
    if path_npy.exists():
        return np.load(path_npy).reshape(-1)
    y = pd.read_csv(path_csv)
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    return np.asarray(y).reshape(-1)


def load_data() -> Dict:
    processed = U.DIRS.processed

    X_train = _read_df(processed / "X_train_base.csv", processed / "X_train_base.parquet")
    X_val   = _read_df(processed / "X_val_base.csv", processed / "X_val_base.parquet")
    X_test  = _read_df(processed / "X_test_base.csv", processed / "X_test_base.parquet")

    y_train = _read_y(processed / "y_train.csv", processed / "y_train.npy")
    y_val   = _read_y(processed / "y_val.csv", processed / "y_val.npy")
    y_test  = _read_y(processed / "y_test.csv", processed / "y_test.npy")

    # Normalize columns / types
    X_train = EM.coerce_bool_to_int(X_train.reset_index(drop=True))
    X_val   = EM.coerce_bool_to_int(X_val.reset_index(drop=True))
    X_test  = EM.coerce_bool_to_int(X_test.reset_index(drop=True))

    y_train = y_train.reshape(-1)
    y_val   = y_val.reshape(-1)
    y_test  = y_test.reshape(-1)

    X_train_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_train_full": X_train_full, "y_train_full": y_train_full,
    }


def encoded_base_columns(data: Dict) -> List[str]:
    return list(data["X_train"].columns)


def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate model paths exist: {[str(p) for p in paths]}")


def load_base_models() -> Dict[str, Dict]:
    """
    Load the 4 base models from disk.
    Adjust the candidate paths if your repo stores them differently.
    """
    models_dir = U.DIRS.models
    meta_dir = U.DIRS.meta

    base = {
        "Logistic": {
            "paths": [
                models_dir / "logit_baseline.pkl",
                models_dir / "03_logit_baseline.pkl",
                meta_dir / "logit_baseline.pkl",
            ],
        },
        "XGBoost": {
            "paths": [
                models_dir / "04_xgb_optuna.pkl",
                models_dir / "04_xgb_baseline.pkl",
                meta_dir / "xgb_baseline_v1.pkl",
            ],
        },
        "LightGBM": {
            "paths": [
                models_dir / "04_lgbm_small.pkl",
                meta_dir / "lgbm_small_v1.pkl",
            ],
        },
        "RandomForest": {
            "paths": [
                models_dir / "04_rf_baseline.pkl",
                meta_dir / "rf_baseline_v1.pkl",
            ],
        },
    }

    out: Dict[str, Dict] = {}
    for name, spec in base.items():
        path = _first_existing([Path(p) for p in spec["paths"]])
        model = U.load_pickle(path)
        out[name] = {"model": model, "path": str(path)}
    return out


# ---------------------------
# Experiments
# ---------------------------

def generate_base_predictions(
    base_models: Dict[str, Dict],
    data: Dict,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    Predict on val/test using loaded models.
    Returns val_preds_df, test_preds_df, and per-model metrics.
    """
    y_val = data["y_val"]
    y_test = data["y_test"]

    val_preds = {}
    test_preds = {}
    metrics: Dict[str, Dict] = {}

    for name, info in base_models.items():
        model = info["model"]
        p_val = EM.get_predictions(model, data["X_val"][features], features_fallback=features)
        p_test = EM.get_predictions(model, data["X_test"][features], features_fallback=features)

        val_auc = float(roc_auc_score(y_val, p_val))
        test_auc = float(roc_auc_score(y_test, p_test))

        val_preds[name] = p_val
        test_preds[name] = p_test
        metrics[name] = {"val_auc": val_auc, "test_auc": test_auc, "model_path": info["path"]}

    return pd.DataFrame(val_preds), pd.DataFrame(test_preds), metrics


def run_experiments(
    base_models: Dict[str, Dict],
    data: Dict,
    features: List[str],
    fixed_threshold: float,
    cv_folds: int,
    seed: int,
) -> Dict[str, Dict]:
    """
    Run:
    - blends: avg / weighted(AUC) / NNLS / rank-avg
    - stacking: OOF
    """
    results: Dict[str, Dict] = {}

    y_val = data["y_val"]
    y_test = data["y_test"]

    # Base prediction matrices
    val_preds_df, test_preds_df, base_metrics = generate_base_predictions(base_models, data, features)

    # 1) Simple average
    p_val = EM.blend_simple_average(val_preds_df)
    p_test = EM.blend_simple_average(test_preds_df)
    results["Blend_SimpleAvg"] = {
        "val": EM.evaluate_probs(y_val, p_val, fixed_threshold=fixed_threshold).__dict__,
        "test": EM.evaluate_probs(y_test, p_test, fixed_threshold=fixed_threshold).__dict__,
    }

    # 2) Weighted by validation AUC (heuristic)
    weights_auc = {k: max(v["val_auc"] - 0.5, 0.0) + 1e-6 for k, v in base_metrics.items()}
    p_val = EM.blend_weighted(val_preds_df, weights_auc)
    p_test = EM.blend_weighted(test_preds_df, weights_auc)
    results["Blend_WeightedAUC"] = {
        "weights": weights_auc,
        "val": EM.evaluate_probs(y_val, p_val, fixed_threshold=fixed_threshold).__dict__,
        "test": EM.evaluate_probs(y_test, p_test, fixed_threshold=fixed_threshold).__dict__,
    }

    # 3) NNLS weights (fit on val preds)
    weights_nnls = EM.fit_nnls_weights(val_preds_df, y_val)
    p_val = EM.blend_weighted(val_preds_df, weights_nnls)
    p_test = EM.blend_weighted(test_preds_df, weights_nnls)
    results["Blend_NNLS"] = {
        "weights": weights_nnls,
        "val": EM.evaluate_probs(y_val, p_val, fixed_threshold=fixed_threshold).__dict__,
        "test": EM.evaluate_probs(y_test, p_test, fixed_threshold=fixed_threshold).__dict__,
    }

    # 4) Rank average
    p_val = EM.blend_rank_average(val_preds_df)
    p_test = EM.blend_rank_average(test_preds_df)
    results["Blend_RankAvg"] = {
        "val": EM.evaluate_probs(y_val, p_val, fixed_threshold=fixed_threshold).__dict__,
        "test": EM.evaluate_probs(y_test, p_test, fixed_threshold=fixed_threshold).__dict__,
    }

    # 5) Stacking (OOF)
    base_estimators = {name: info["model"] for name, info in base_models.items()}
    stacking = EM.train_stacking_oof(
        base_estimators=base_estimators,
        X_train=data["X_train"][features],
        y_train=data["y_train"],
        features=features,
        cv_folds=cv_folds,
        passthrough=False,
        random_state=seed,
        n_jobs=-1,
    )
    p_val = EM.get_predictions(stacking, data["X_val"][features], features_fallback=features)
    p_test = EM.get_predictions(stacking, data["X_test"][features], features_fallback=features)
    results["Stacking_OOF"] = {
        "val": EM.evaluate_probs(y_val, p_val, fixed_threshold=fixed_threshold).__dict__,
        "test": EM.evaluate_probs(y_test, p_test, fixed_threshold=fixed_threshold).__dict__,
    }

    # Also return base preds & base metrics for saving
    results["_base_val_preds_df"] = val_preds_df
    results["_base_test_preds_df"] = test_preds_df
    results["_base_metrics"] = base_metrics
    results["_base_model_paths"] = {k: v["path"] for k, v in base_models.items()}

    return results


# ---------------------------
# Saving
# ---------------------------

def save_outputs(results: Dict[str, Dict], fixed_threshold: float, features: List[str]) -> None:
    U.ensure_dirs()

    val_preds_df: pd.DataFrame = results.pop("_base_val_preds_df")
    test_preds_df: pd.DataFrame = results.pop("_base_test_preds_df")
    base_metrics: Dict[str, Dict] = results.pop("_base_metrics")
    base_model_paths: Dict[str, str] = results.pop("_base_model_paths")

    # Base preds (for appendix / diagnostics)
    U.save_df(val_preds_df, U.DIRS.tables / "07_base_pred_val.csv")
    U.save_df(test_preds_df, U.DIRS.tables / "07_base_pred_test.csv")

    # Leaderboard (test ROC-AUC + PR-AUC)
    rows = []
    for name, info in results.items():
        rows.append({
            "model": name,
            "val_roc_auc": info["val"]["roc_auc"],
            "val_pr_auc": info["val"]["pr_auc"],
            "test_roc_auc": info["test"]["roc_auc"],
            "test_pr_auc": info["test"]["pr_auc"],
            "fixed_threshold": fixed_threshold,
            "val_best_threshold": info["val"]["best_threshold"],
            "test_best_threshold": info["test"]["best_threshold"],
        })
    leaderboard = pd.DataFrame(rows).sort_values(["test_roc_auc", "test_pr_auc"], ascending=False)
    U.save_df(leaderboard, U.DIRS.tables / "07_ensemble_leaderboard.csv")

    # Blend weights
    weights_out = {}
    for k, v in results.items():
        if "weights" in v:
            weights_out[k] = v["weights"]
    if weights_out:
        U.save_json(weights_out, U.DIRS.meta / "07_blend_weights.json")

    # Diversity (correlation)  appendix-friendly
    corr = EM.diversity_correlation(val_preds_df)
    U.save_df(corr, U.DIRS.tables / "07_diversity_corr_val.csv")

    # Manifest
    manifest = {
        "fixed_threshold": fixed_threshold,
        "features_n": len(features),
        "features": features,
        "base_models": base_model_paths,
        "base_model_metrics": base_metrics,
        "results": results,
        "tables": {
            "leaderboard": "artifacts/tables/07_ensemble_leaderboard.csv",
            "base_pred_val": "artifacts/tables/07_base_pred_val.csv",
            "base_pred_test": "artifacts/tables/07_base_pred_test.csv",
            "diversity_corr_val": "artifacts/tables/07_diversity_corr_val.csv",
        },
        "meta": {
            "blend_weights": "artifacts/meta/07_blend_weights.json" if weights_out else None,
        },
    }
    U.save_json(manifest, U.DIRS.meta / "manifest_07_stacking.json")


# ---------------------------
# Main
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="(reserved) speed up experiments")
    ap.add_argument("--force_retrain", action="store_true", help="retrain stacking model (base models still loaded)")
    return ap.parse_args()


def load_fixed_threshold() -> float:
    # Prefer 04 best threshold if present, else default 0.40
    cand = [
        U.DIRS.meta / "04_best_threshold_rule.json",
        U.DIRS.meta / "06_threshold_policy.json",
    ]
    for p in cand:
        if p.exists():
            obj = U.load_json(p)
            if isinstance(obj, dict) and "best_threshold" in obj:
                return float(obj["best_threshold"])
            if isinstance(obj, dict) and "fixed_threshold" in obj:
                return float(obj["fixed_threshold"])
    return 0.40


def main():
    args = parse_args()

    print("=" * 60)
    print("07 Ensemble Stacking Pipeline")
    print("=" * 60)
    print(f"Config: seed={args.seed}, cv_folds={args.cv_folds}, fast={args.fast}")

    print("\nLoading data...")
    data = load_data()
    print(f"  Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
    print(f"  Train+Val: {data['X_train_full'].shape}")

    features = encoded_base_columns(data)
    print(f"\nEncoded base columns: {len(features)}")

    fixed_threshold = load_fixed_threshold()
    print(f"\nLoaded fixed threshold: {fixed_threshold:.4f}")

    print("\nLoading base models...")
    base_models = load_base_models()
    for k, v in base_models.items():
        print(f"  [OK] {k} loaded from {v['path']}")

    print("\nRunning ensemble experiments...")
    results = run_experiments(
        base_models=base_models,
        data=data,
        features=features,
        fixed_threshold=fixed_threshold,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )

    print("\nSaving outputs...")
    save_outputs(results, fixed_threshold, features)

    print("\n07 - Stacking Completed Successfully!")


if __name__ == "__main__":
    main()
