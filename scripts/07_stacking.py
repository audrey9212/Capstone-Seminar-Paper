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
from sklearn.model_selection import KFold, cross_val_predict

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


def load_nn_predictions() -> Dict[str, np.ndarray]:
    """
    Load pre-computed Wide & Deep (focal) val/test probabilities saved by 05_nn.py.
    Returns dict with 'val' and 'test' arrays, or empty dict if files not found.
    """
    preds_dir = U.DIRS.artifacts / "preds"
    val_path  = preds_dir / "05_nn_wide_deep_focal_val_proba.npy"
    test_path = preds_dir / "05_nn_wide_deep_focal_test_proba.npy"
    if val_path.exists() and test_path.exists():
        return {
            "val":  np.load(val_path).reshape(-1),
            "test": np.load(test_path).reshape(-1),
        }
    print("  [WARN] Wide & Deep predictions not found; skipping NN base model.")
    return {}


def load_deep_data() -> Dict | None:
    """
    Load deep-feature splits (X_*_deep) and feature groups for Wide & Deep OOF.
    Returns None if the deep CSV files are not present.
    """
    data_dir = U.DIRS.processed
    paths = [data_dir / "X_train_deep.csv", data_dir / "X_val_deep.csv", data_dir / "X_test_deep.csv"]
    if not all(p.exists() for p in paths):
        print("  [WARN] X_*_deep.csv files not found; Wide & Deep OOF will be skipped.")
        return None

    X_train = pd.read_csv(paths[0])
    X_val   = pd.read_csv(paths[1])
    X_test  = pd.read_csv(paths[2])

    # Load feature groups
    pp_path = data_dir / "preprocess_params.json"
    fg_path = U.DIRS.meta / "feature_groups.json"
    feature_groups: Dict = {}
    cat_mappings: Dict = {}

    if fg_path.exists():
        feature_groups = U.load_json(fg_path)
        feature_groups = {k: [c for c in v if c in X_train.columns] for k, v in feature_groups.items()}
    elif pp_path.exists():
        pp = U.load_json(pp_path)
        if "feature_groups" in pp:
            feature_groups = pp["feature_groups"]
            feature_groups = {k: [c for c in v if c in X_train.columns] for k, v in feature_groups.items()}
        else:
            # Infer from categorical_mappings / numeric_scaler keys
            cat_mappings = pp.get("categorical_mappings", {})
            cat_cols_inferred  = sorted(set(cat_mappings.keys()) & set(X_train.columns))
            cont_cols_inferred = sorted(set(pp.get("numeric_scaler", {}).keys()) & set(X_train.columns))
            bin_cols_inferred  = sorted(
                c for c in X_train.columns
                if c not in cat_cols_inferred and c not in cont_cols_inferred
                and X_train[c].nunique() == 2
            )
            feature_groups = {
                "continuous": cont_cols_inferred,
                "categorical": cat_cols_inferred,
                "binary": bin_cols_inferred,
            }
        cat_mappings = pp.get("categorical_mappings", {})

    # Build categorical cardinalities
    cat_cols = feature_groups.get("categorical", [])
    cat_cards: Dict[str, int] = {}
    for col in cat_cols:
        if col in cat_mappings:
            cat_cards[col] = max(cat_mappings[col].values()) + 1 if cat_mappings[col] else 1
        else:
            cat_cards[col] = int(X_train[col].max() + 1)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "feature_groups": feature_groups,
        "cat_cards": cat_cards,
    }


def _build_nn_loader(X_df: pd.DataFrame, y_arr: np.ndarray, feature_groups: Dict,
                     cat_cards: Dict, batch_size: int, shuffle: bool, seed: int = 42):
    """Create a DataLoader and return (loader, input_dims)."""
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import nn_models as NM

    dataset = NM.ChurnDataset(
        X_df, y_arr,
        continuous_cols=feature_groups.get("continuous", []),
        categorical_cols=feature_groups.get("categorical", []),
        binary_cols=feature_groups.get("binary", []),
        categorical_cardinalities=cat_cards,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=0,
    )
    return loader, dataset.get_input_dims()


def _train_one_fold_nn(X_train_f, y_train_f, X_val_f, y_val_f,
                       feature_groups, cat_cards, input_dims, device,
                       epochs: int, batch_size: int, patience: int, seed: int):
    """Train a single Wide & Deep fold; return (model, oof_val_proba)."""
    import inspect
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import nn_models as NM

    train_loader, _ = _build_nn_loader(X_train_f, y_train_f, feature_groups, cat_cards, batch_size, True,  seed)
    val_loader,   _ = _build_nn_loader(X_val_f,   y_val_f,   feature_groups, cat_cards, batch_size, False, seed)

    model = NM.create_wide_deep_model(
        input_dims, embedding_dim=8, deep_hidden_dims=[128, 64], dropout=0.3, alpha=0.5,
    ).to(device)

    criterion = NM.FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_kwargs = dict(mode="max", factor=0.5, patience=5, min_lr=1e-6)
    sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__)
    if "verbose" in sig.parameters:
        scheduler_kwargs["verbose"] = False
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    _, best_state, _, _ = NM.train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=epochs, device=device, patience=patience, verbose=False,
    )
    model.load_state_dict(best_state)

    _, val_proba, _ = NM.evaluate_model(model, val_loader, device)
    return model, val_proba


def _predict_nn(model, X_df: pd.DataFrame, y_dummy: np.ndarray,
                feature_groups: Dict, cat_cards: Dict, batch_size: int, device) -> np.ndarray:
    """Get probability predictions from a Wide & Deep model."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import nn_models as NM
    loader, _ = _build_nn_loader(X_df, y_dummy, feature_groups, cat_cards, batch_size, False)
    _, proba, _ = NM.evaluate_model(model, loader, device)
    return proba


def train_wide_deep_oof(
    X_train_deep: pd.DataFrame,
    y_train: np.ndarray,
    X_val_deep: pd.DataFrame,
    X_test_deep: pd.DataFrame,
    feature_groups: Dict,
    cat_cards: Dict,
    cv_folds: int = 5,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    5-fold OOF training of Wide & Deep (focal) model.
    Returns (oof_train_preds, val_preds, test_preds).
    val/test predictions are averaged across the 5 fold models.
    """
    try:
        import torch
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import nn_models as NM
    except ImportError as exc:
        raise RuntimeError(f"PyTorch / nn_models required for Wide & Deep OOF: {exc}")

    NM.set_seed(seed)
    device = NM.DEVICE

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_train))
    y_dummy_val  = np.zeros(len(X_val_deep),  dtype=np.float32)
    y_dummy_test = np.zeros(len(X_test_deep), dtype=np.float32)

    # Derive input_dims from first fold's training slice
    first_train_idx = next(iter(kf.split(X_train_deep)))[0]
    _, input_dims = _build_nn_loader(
        X_train_deep.iloc[first_train_idx], y_train[first_train_idx],
        feature_groups, cat_cards, batch_size=batch_size, shuffle=False, seed=seed,
    )

    fold_val_preds  = []
    fold_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_deep)):
        print(f"    Fold {fold + 1}/{cv_folds}  (train={len(train_idx)}, oof={len(val_idx)}) ...", flush=True)

        X_fold_tr = X_train_deep.iloc[train_idx].reset_index(drop=True)
        X_fold_va = X_train_deep.iloc[val_idx].reset_index(drop=True)
        y_fold_tr = y_train[train_idx].astype(np.float32)
        y_fold_va = y_train[val_idx].astype(np.float32)

        fold_model, oof_fold_proba = _train_one_fold_nn(
            X_fold_tr, y_fold_tr, X_fold_va, y_fold_va,
            feature_groups, cat_cards, input_dims, device,
            epochs=epochs, batch_size=batch_size, patience=patience, seed=seed + fold,
        )

        oof_preds[val_idx] = oof_fold_proba
        fold_val_preds.append( _predict_nn(fold_model, X_val_deep,  y_dummy_val,  feature_groups, cat_cards, batch_size, device))
        fold_test_preds.append(_predict_nn(fold_model, X_test_deep, y_dummy_test, feature_groups, cat_cards, batch_size, device))

        del fold_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_preds  = np.mean(fold_val_preds,  axis=0)
    test_preds = np.mean(fold_test_preds, axis=0)
    return oof_preds, val_preds, test_preds


def load_base_models() -> Dict[str, Dict]:
    """
    Load the 4 sklearn base models from disk.
    Wide & Deep is handled separately via load_nn_predictions().
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
    nn_preds: Dict[str, np.ndarray] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    Predict on val/test using loaded sklearn models, then optionally merge
    pre-computed Wide & Deep predictions (nn_preds).
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

    # Merge pre-computed Wide & Deep predictions (blending only; not used in OOF)
    if nn_preds:
        p_val  = nn_preds["val"]
        p_test = nn_preds["test"]
        val_auc  = float(roc_auc_score(y_val,  p_val))
        test_auc = float(roc_auc_score(y_test, p_test))
        val_preds["WideDeep"]  = p_val
        test_preds["WideDeep"] = p_test
        metrics["WideDeep"] = {"val_auc": val_auc, "test_auc": test_auc, "model_path": "precomputed_npy"}

    return pd.DataFrame(val_preds), pd.DataFrame(test_preds), metrics


def run_experiments(
    base_models: Dict[str, Dict],
    data: Dict,
    features: List[str],
    fixed_threshold: float,
    cv_folds: int,
    seed: int,
    nn_preds: Dict[str, np.ndarray] | None = None,
    deep_data: Dict | None = None,
) -> Dict[str, Dict]:
    """
    Run:
    - blends: avg / weighted(AUC) / NNLS / rank-avg  (includes WideDeep if available)
    - stacking: Stacking_OOF  (4 sklearn base models via StackingClassifier)
    - stacking: Stacking_OOF_5  (4 sklearn + Wide & Deep, manual OOF; only if deep_data provided)
    """
    results: Dict[str, Dict] = {}

    y_val = data["y_val"]
    y_test = data["y_test"]

    # Base prediction matrices (includes WideDeep for blending)
    val_preds_df, test_preds_df, base_metrics = generate_base_predictions(
        base_models, data, features, nn_preds=nn_preds
    )

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

    # 6) Stacking OOF with Wide & Deep as 5th base learner (proper OOF for NN)
    if deep_data is not None:
        print("\n  [Stacking_OOF_5] Computing sklearn OOF predictions via cross_val_predict ...")
        base_estimators_dict = {name: info["model"] for name, info in base_models.items()}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        sklearn_oof: Dict[str, np.ndarray] = {}
        for name, model in base_estimators_dict.items():
            print(f"    {name} (cross_val_predict) ...")
            sklearn_oof[name] = cross_val_predict(
                model, data["X_train"][features], data["y_train"],
                cv=kf, method="predict_proba", n_jobs=1,
            )[:, 1]

        # Shut down loky worker pool before starting PyTorch to avoid OpenMP deadlock
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except Exception:
            pass
        import torch as _torch
        _torch.set_num_threads(1)

        print("\n  [Stacking_OOF_5] Training Wide & Deep 5-fold OOF ...")
        wd_oof, wd_val, wd_test = train_wide_deep_oof(
            deep_data["X_train"], data["y_train"],
            deep_data["X_val"], deep_data["X_test"],
            deep_data["feature_groups"], deep_data["cat_cards"],
            cv_folds=cv_folds, seed=seed, epochs=50, batch_size=256, patience=10,
        )

        # Build meta-feature matrices
        sklearn_names = list(sklearn_oof.keys())
        oof_matrix  = np.column_stack([sklearn_oof[n] for n in sklearn_names] + [wd_oof])
        val_meta    = np.column_stack([val_preds_df[n].values for n in sklearn_names] + [wd_val])
        test_meta   = np.column_stack([test_preds_df[n].values for n in sklearn_names] + [wd_test])

        from sklearn.linear_model import LogisticRegression as _MetaLR
        meta_lr = _MetaLR(max_iter=2000, random_state=seed)
        meta_lr.fit(oof_matrix, data["y_train"])

        p_val  = meta_lr.predict_proba(val_meta)[:, 1]
        p_test = meta_lr.predict_proba(test_meta)[:, 1]

        results["Stacking_OOF_5"] = {
            "val":  EM.evaluate_probs(y_val,  p_val,  fixed_threshold=fixed_threshold).__dict__,
            "test": EM.evaluate_probs(y_test, p_test, fixed_threshold=fixed_threshold).__dict__,
        }

        # Persist WideDeep OOF val/test preds for downstream use
        results["_wd_oof"]  = wd_oof
        results["_wd_val"]  = wd_val
        results["_wd_test"] = wd_test

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
    # Pop WideDeep OOF arrays if present (not JSON-serialisable as-is)
    wd_oof  = results.pop("_wd_oof",  None)
    wd_val  = results.pop("_wd_val",  None)
    wd_test = results.pop("_wd_test", None)
    if wd_oof is not None:
        preds_dir = U.DIRS.artifacts / "preds"
        preds_dir.mkdir(exist_ok=True, parents=True)
        np.save(preds_dir / "07_wd_oof_train_proba.npy",  wd_oof)
        np.save(preds_dir / "07_wd_oof_val_proba.npy",    wd_val)
        np.save(preds_dir / "07_wd_oof_test_proba.npy",   wd_test)

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

    print("\nLoading Wide & Deep predictions (blending)...")
    nn_preds = load_nn_predictions()
    if nn_preds:
        print(f"  [OK] WideDeep val/test proba loaded (val_auc will be computed)")
    else:
        print("  [SKIP] WideDeep predictions not available")

    print("\nLoading Wide & Deep deep features (OOF stacking)...")
    deep_data = load_deep_data()
    if deep_data is not None:
        print(f"  [OK] Deep features loaded: X_train={deep_data['X_train'].shape}, "
              f"feature_groups keys={list(deep_data['feature_groups'].keys())}")
    else:
        print("  [SKIP] Deep features not available; Stacking_OOF_5 will be skipped")

    print("\nRunning ensemble experiments...")
    results = run_experiments(
        base_models=base_models,
        data=data,
        features=features,
        fixed_threshold=fixed_threshold,
        cv_folds=args.cv_folds,
        seed=args.seed,
        nn_preds=nn_preds or None,
        deep_data=deep_data,
    )

    print("\nSaving outputs...")
    save_outputs(results, fixed_threshold, features)

    print("\n07 - Stacking Completed Successfully!")


if __name__ == "__main__":
    main()
