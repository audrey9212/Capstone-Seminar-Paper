#!/usr/bin/env python3
"""
02_preprocess.py: Data Splitting & Preprocessing Pipeline (Dual-Output Version)

Purpose:
  1. Fix train/val/test split indices (cached in split_indices.json)
  2. Fit preprocessing parameters on training set only (prevent data leakage)
  3. Generate THREE preprocessing variants:
     - base: for supervised models (03/04/05/07/08) - one-hot encoded
     - deep: for embedding models (05 NN) - integer encoded
     - ae_base: for autoencoder (06) - numeric only, no one-hot

Outputs to data/processed/:
  [Supervised - for 03/04/05/07/08]
    X_train_base.csv, X_val_base.csv, X_test_base.csv
    X_train_deep.csv, X_val_deep.csv, X_test_deep.csv
    y_train.csv, y_val.csv, y_test.csv
    X_train_processed.npy, y_train.npy, etc.

  [AE - for 06 only]
    X_train_ae_base.csv, X_val_ae_base.csv, X_test_ae_base.csv
    X_holdout_ae_base.csv (from cell2cellholdout.csv)
    holdout_customer_ids.csv, holdout_metadata.json
"""
import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["CAPSTONE_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import utils as U

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROCESSED_DIR = U.DIRS.processed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Columns never included in model features
ALWAYS_EXCLUDE = {"Churn", "Churn01", "CustomerID"}
# Leakage risk columns
LEAKAGE_COLS = {"MadeCallToRetentionTeam", "RetentionCalls", 
                "RetentionOffersAccepted", "RespondsToMailOffers"}


# =============================================================================
# 1. Load data
# =============================================================================
def load_raw_data() -> pd.DataFrame:
    """Load raw Cell2Cell training data."""
    raw_file = U.raw_path("cell2celltrain.csv")
    print(f"Loading raw data from {raw_file}...")
    df = pd.read_csv(raw_file, low_memory=False)
    print(f"  Shape: {df.shape}")
    return df


def load_holdout_data() -> pd.DataFrame:
    """Load raw Cell2Cell holdout data (unlabeled)."""
    raw_file = U.raw_path("cell2cellholdout.csv")
    print(f"Loading holdout data from {raw_file}...")
    df = pd.read_csv(raw_file, low_memory=False)
    print(f"  Shape: {df.shape}")
    return df


def load_feature_config() -> Optional[pd.DataFrame]:
    """Load feature metadata (type, zero_as_missing, etc.)"""
    config_paths = [
        U.DIRS.data / "feature_config_clean.csv",
        U.DIRS.root / "feature_config_clean.csv",
    ]
    for path in config_paths:
        if path.exists():
            print(f"Loading feature config from {path}...")
            return pd.read_csv(path)
    print("[WARN]  No feature_config_clean.csv found. Using auto-detection.")
    return None


def select_feature_columns(config: pd.DataFrame, variant: str = "base") -> List[str]:
    """
    Select features based on feature_config_clean.csv.
    variant: "base" (GLM+Tree), "deep" (NN embedding), "ae" (numeric only)
    """
    if config is None:
        raise ValueError("feature_config_clean.csv required for leakage-safe selection.")
    
    cfg = config.copy()

    def to_bool(series):
        return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])
    
    keep_mask = cfg["decision_type_short"].str.lower().eq("keep")
    
    if variant == "base":
        keep_mask &= to_bool(cfg["keep_glm"]) | to_bool(cfg["keep_tree"])
    elif variant == "deep":
        keep_mask &= to_bool(cfg["keep_nn"])
    elif variant == "ae":
        # AE uses only numeric features (no categorical)
        numeric_types = [
            "Numeric (continuous)", "Numeric (count)",
            "Numeric (ratio)", "Numeric (momentum)"
        ]
        keep_mask &= cfg["type"].isin(numeric_types)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    cols = cfg.loc[keep_mask, "feature"].tolist()
    # Remove leakage and always-excluded columns
    cols = [c for c in cols if c not in (ALWAYS_EXCLUDE | LEAKAGE_COLS)]
    return cols


# =============================================================================
# 2. Create target & fix dtypes
# =============================================================================
def create_churn01(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Churn (Yes/No string) to Churn01 (0/1 binary)."""
    df = df.copy()
    churn_col = df["Churn"].astype(str).str.lower().str.strip()
    df["Churn01"] = churn_col.map({
        "yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0
    }).fillna(-1).astype(int)
    df = df[df["Churn01"] != -1]
    print(f"Churn01 distribution: {df['Churn01'].value_counts().to_dict()}")
    return df


def fix_dtype_from_config(df: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to correct dtypes based on feature config."""
    if config is None:
        return df
    df = df.copy()
    for _, row in config.iterrows():
        col = row.get("feature_name") or row.get("Feature") or row.get("feature")
        feat_type = row.get("type")
        if col not in df.columns:
            continue
        if feat_type in ["Numeric", "Ordinal"] or "Numeric" in str(feat_type):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_zero_as_missing(df: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    """For features where zero_as_missing=True, convert 0  NaN."""
    if config is None:
        return df
    df = df.copy()
    for _, row in config.iterrows():
        col = row.get("feature_name") or row.get("Feature") or row.get("feature")
        is_zero_missing = row.get("zero_as_missing") in [True, 1, "True", "TRUE"]
        if col not in df.columns or not is_zero_missing:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] == 0, col] = np.nan
    return df


# =============================================================================
# 3. Fixed data splitting with stratification
# =============================================================================
def create_or_load_split_indices(
    df: pd.DataFrame,
    temp_size: float = 0.30,
    val_frac_of_temp: float = 0.50
) -> tuple:
    """Generate or load fixed train/val/test split indices (70/15/15)."""
    split_file = U.processed_path("split_indices.json")

    if split_file.exists():
        print("Loading split indices from cache...")
        with open(split_file, "r") as f:
            indices_data = json.load(f)
        train_idx = np.array(indices_data["train"])
        val_idx = np.array(indices_data["val"])
        test_idx = np.array(indices_data["test"])
    else:
        print("Generating new split indices...")
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=temp_size, random_state=RANDOM_SEED
        )
        train_idx, temp_idx = next(splitter1.split(df, df["Churn01"]))

        df_temp = df.iloc[temp_idx]
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac_of_temp, random_state=RANDOM_SEED
        )
        val_in_temp, test_in_temp = next(splitter2.split(df_temp, df_temp["Churn01"]))
        val_idx = temp_idx[val_in_temp]
        test_idx = temp_idx[test_in_temp]

        indices_data = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
            "random_seed": RANDOM_SEED,
            "split": "70/15/15",
        }
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(indices_data, f, indent=2)
        print(f"Saved split indices to {split_file}")

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


# =============================================================================
# 4. Fit preprocessing parameters on training set only
# =============================================================================
def fit_preprocess_params(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    config: pd.DataFrame
) -> Dict[str, Any]:
    """Fit preprocessing params on training set to prevent data leakage."""
    print("\nFitting preprocessing parameters on training set...")
    params = {"random_seed": RANDOM_SEED}
    df_train = df.iloc[train_idx]
    
    # Numeric columns
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ALWAYS_EXCLUDE]
    
    params["numeric_medians"] = {}
    params["numeric_scaler"] = {}
    
    for col in numeric_cols:
        median_val = df_train[col].median()
        params["numeric_medians"][col] = float(median_val) if not pd.isna(median_val) else 0.0
        
        mean_val = df_train[col].mean()
        std_val = df_train[col].std()
        if pd.isna(mean_val):
            mean_val = 0.0
        if pd.isna(std_val) or std_val == 0:
            std_val = 1.0
        params["numeric_scaler"][col] = {
            "mean": float(mean_val),
            "std": float(std_val),
        }
    
    # Categorical columns
    cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in {"Churn"}]
    
    params["categorical_mappings"] = {}
    for col in cat_cols:
        categories = sorted(df_train[col].dropna().astype(str).unique())
        code_dict = {cat: i + 1 for i, cat in enumerate(categories)}
        params["categorical_mappings"][col] = code_dict
    
    print(f"Fitted {len(numeric_cols)} numeric + {len(cat_cols)} categorical features")
    return params


def save_preprocess_params(params: Dict) -> None:
    """Save preprocessing parameters to JSON."""
    params_file = U.processed_path("preprocess_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2, default=str)
    print(f"Saved preprocessing params to {params_file}")


# =============================================================================
# 5. Build X_base: for supervised models (RF, XGB, Logistic)
# =============================================================================
def build_x_base(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    params: Dict
) -> tuple:
    """Build X_base: numeric standardized + categorical one-hot encoded."""
    print("\nBuilding X_base (numeric standardized + one-hot categorical)...")
    
    base_features = params.get("feature_cols_base", [])
    df_processed = df.copy()
    
    # Impute numeric
    numeric_cols = [c for c in params["numeric_medians"].keys() 
                    if not base_features or c in base_features]
    for col in numeric_cols:
        if col in df_processed.columns:
            median_val = params["numeric_medians"][col]
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # Standardize numeric
    for col in numeric_cols:
        if col in params["numeric_scaler"] and col in df_processed.columns:
            mean_val = params["numeric_scaler"][col]["mean"]
            std_val = params["numeric_scaler"][col]["std"]
            if std_val > 0:
                df_processed[col] = (df_processed[col] - mean_val) / std_val
    
    # One-hot encode categorical
    cat_cols = [c for c in params["categorical_mappings"].keys() 
                if not base_features or c in base_features]
    
    available_numeric = [c for c in numeric_cols if c in df_processed.columns]
    available_cat = [c for c in cat_cols if c in df_processed.columns]
    
    df_train_base = df_processed.iloc[train_idx][available_numeric + available_cat]
    df_val_base = df_processed.iloc[val_idx][available_numeric + available_cat]
    df_test_base = df_processed.iloc[test_idx][available_numeric + available_cat]
    
    # One-hot on train, then align val/test
    df_train_oh = pd.get_dummies(df_train_base, columns=available_cat, drop_first=True)
    df_val_oh = pd.get_dummies(df_val_base, columns=available_cat, drop_first=True)
    df_test_oh = pd.get_dummies(df_test_base, columns=available_cat, drop_first=True)
    
    # Align columns
    all_cols = df_train_oh.columns
    df_val_oh = df_val_oh.reindex(columns=all_cols, fill_value=0)
    df_test_oh = df_test_oh.reindex(columns=all_cols, fill_value=0)
    
    # Save
    U.save_df(df_train_oh, "X_train_base.csv", folder=PROCESSED_DIR, log_table=False)
    U.save_df(df_val_oh, "X_val_base.csv", folder=PROCESSED_DIR, log_table=False)
    U.save_df(df_test_oh, "X_test_base.csv", folder=PROCESSED_DIR, log_table=False)
    
    print(f"  X_train_base: {df_train_oh.shape}")
    print(f"  X_val_base: {df_val_oh.shape}")
    print(f"  X_test_base: {df_test_oh.shape}")
    
    return df_train_oh, df_val_oh, df_test_oh


# =============================================================================
# 6. Build X_deep: for embedding layers (PyTorch)
# =============================================================================
def build_x_deep(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    params: Dict
) -> tuple:
    """Build X_deep: numeric standardized + categorical integer encoded."""
    print("\nBuilding X_deep (numeric standardized + integer categorical)...")
    
    deep_features = params.get("feature_cols_deep", [])
    df_processed = df.copy()
    
    # Impute numeric
    numeric_cols = [c for c in params["numeric_medians"].keys() 
                    if not deep_features or c in deep_features]
    for col in numeric_cols:
        if col in df_processed.columns:
            median_val = params["numeric_medians"][col]
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # Standardize numeric
    for col in numeric_cols:
        if col in params["numeric_scaler"] and col in df_processed.columns:
            mean_val = params["numeric_scaler"][col]["mean"]
            std_val = params["numeric_scaler"][col]["std"]
            if std_val > 0:
                df_processed[col] = (df_processed[col] - mean_val) / std_val
    
    # Integer encode categorical
    cat_cols = [c for c in params["categorical_mappings"].keys() 
                if not deep_features or c in deep_features]
    for col in cat_cols:
        if col in df_processed.columns:
            code_dict = params["categorical_mappings"][col]
            df_processed[col] = df_processed[col].astype(str).map(code_dict).fillna(0).astype(int)
    
    # Extract splits
    available_numeric = [c for c in numeric_cols if c in df_processed.columns]
    available_cat = [c for c in cat_cols if c in df_processed.columns]
    feature_cols = available_numeric + available_cat
    
    df_train_deep = df_processed.iloc[train_idx][feature_cols]
    df_val_deep = df_processed.iloc[val_idx][feature_cols]
    df_test_deep = df_processed.iloc[test_idx][feature_cols]
    
    # Save
    U.save_df(df_train_deep, "X_train_deep.csv", folder=PROCESSED_DIR, log_table=False)
    U.save_df(df_val_deep, "X_val_deep.csv", folder=PROCESSED_DIR, log_table=False)
    U.save_df(df_test_deep, "X_test_deep.csv", folder=PROCESSED_DIR, log_table=False)
    
    print(f"  X_train_deep: {df_train_deep.shape}")
    print(f"  X_val_deep: {df_val_deep.shape}")
    print(f"  X_test_deep: {df_test_deep.shape}")
    
    return df_train_deep, df_val_deep, df_test_deep


# =============================================================================
# 7. Build X_ae_base: for autoencoder (06) - NUMERIC ONLY
# =============================================================================
def build_x_ae_base(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    params: Dict,
    ae_feature_cols: List[str]
) -> tuple:
    """
    Build X_ae_base: numeric-only features for autoencoder.
    Separate schema to avoid overwriting X_*_base.csv.
    """
    print("\nBuilding X_ae_base (numeric-only for autoencoder)...")
    
    df_processed = df.copy()
    
    # Filter to available AE features
    available_cols = [c for c in ae_feature_cols if c in df_processed.columns]
    
    # Impute + standardize numeric
    for col in available_cols:
        if col in params["numeric_medians"]:
            median_val = params["numeric_medians"][col]
            df_processed[col] = df_processed[col].fillna(median_val)
        if col in params["numeric_scaler"]:
            mean_val = params["numeric_scaler"][col]["mean"]
            std_val = params["numeric_scaler"][col]["std"]
            if std_val > 0:
                df_processed[col] = (df_processed[col] - mean_val) / std_val
    
    # Extract splits
    df_train_ae = df_processed.iloc[train_idx][available_cols].reset_index(drop=True)
    df_val_ae = df_processed.iloc[val_idx][available_cols].reset_index(drop=True)
    df_test_ae = df_processed.iloc[test_idx][available_cols].reset_index(drop=True)
    
    # Save with distinct filenames
    U.save_df(df_train_ae, "X_train_ae_base.csv", folder=PROCESSED_DIR, log_table=False)
    U.save_df(df_val_ae, "X_val_ae_base.csv", folder=PROCESSED_DIR, log_table=False)
    U.save_df(df_test_ae, "X_test_ae_base.csv", folder=PROCESSED_DIR, log_table=False)
    
    print(f"  X_train_ae_base: {df_train_ae.shape}")
    print(f"  X_val_ae_base: {df_val_ae.shape}")
    print(f"  X_test_ae_base: {df_test_ae.shape}")
    
    return df_train_ae, df_val_ae, df_test_ae


# =============================================================================
# 8. Build X_holdout_ae: for real unlabeled data (06)
# =============================================================================
def build_x_holdout_ae(
    df_holdout: pd.DataFrame,
    params: Dict,
    ae_feature_cols: List[str],
    config: pd.DataFrame
) -> pd.DataFrame:
    """
    Process holdout data using TRAIN-fitted params (no leakage).
    Returns: Processed holdout DataFrame aligned with AE schema.
    """
    print("\nBuilding X_holdout_ae (real unlabeled data for autoencoder)...")
    
    # Apply same preprocessing as train
    df_processed = df_holdout.copy()
    df_processed = fix_dtype_from_config(df_processed, config)
    df_processed = apply_zero_as_missing(df_processed, config)
    
    # Filter to available AE features
    available_cols = [c for c in ae_feature_cols if c in df_processed.columns]
    missing_cols = [c for c in ae_feature_cols if c not in df_processed.columns]
    
    if missing_cols:
        print(f"  [WARN]  Missing {len(missing_cols)} cols in holdout: {missing_cols[:5]}...")
    
    # Impute + standardize using TRAIN params
    for col in available_cols:
        if col in params["numeric_medians"]:
            median_val = params["numeric_medians"][col]
            df_processed[col] = df_processed[col].fillna(median_val)
        if col in params["numeric_scaler"]:
            mean_val = params["numeric_scaler"][col]["mean"]
            std_val = params["numeric_scaler"][col]["std"]
            if std_val > 0:
                df_processed[col] = (df_processed[col] - mean_val) / std_val
    
    # Extract and add zeros for missing columns
    df_holdout_ae = df_processed[available_cols].copy()
    for col in missing_cols:
        df_holdout_ae[col] = 0.0
    
    # Reorder to match ae_feature_cols
    df_holdout_ae = df_holdout_ae[ae_feature_cols].reset_index(drop=True)
    
    # Save
    U.save_df(df_holdout_ae, "X_holdout_ae_base.csv", folder=PROCESSED_DIR, log_table=False)
    
    # Save CustomerIDs for traceability
    if "CustomerID" in df_holdout.columns:
        customer_ids = df_holdout["CustomerID"].reset_index(drop=True)
        customer_ids.to_csv(PROCESSED_DIR / "holdout_customer_ids.csv", index=False)
        print(f"  Saved holdout_customer_ids.csv ({len(customer_ids)} rows)")
    
    # Save metadata
    holdout_meta = {
        "n_samples": len(df_holdout_ae),
        "n_features": len(ae_feature_cols),
        "features": ae_feature_cols,
        "missing_cols_filled_zero": missing_cols,
        "source": "cell2cellholdout.csv",
        "preprocessing": "train-fitted params (no leakage)",
    }
    with open(PROCESSED_DIR / "holdout_metadata.json", "w") as f:
        json.dump(holdout_meta, f, indent=2)
    
    print(f"  X_holdout_ae_base: {df_holdout_ae.shape}")
    return df_holdout_ae


# =============================================================================
# 9. Save numpy arrays for neural networks
# =============================================================================
def save_numpy_arrays(
    X_train_deep: pd.DataFrame,
    X_val_deep: pd.DataFrame,
    X_test_deep: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> None:
    """Save processed data as NumPy arrays for PyTorch/TensorFlow."""
    print("\nSaving numpy arrays for neural networks...")
    
    np.save(PROCESSED_DIR / "X_train_processed.npy", X_train_deep.values.astype(np.float32))
    np.save(PROCESSED_DIR / "X_val_processed.npy", X_val_deep.values.astype(np.float32))
    np.save(PROCESSED_DIR / "X_test_processed.npy", X_test_deep.values.astype(np.float32))
    np.save(PROCESSED_DIR / "y_train.npy", y_train.values.astype(np.int32))
    np.save(PROCESSED_DIR / "y_val.npy", y_val.values.astype(np.int32))
    np.save(PROCESSED_DIR / "y_test.npy", y_test.values.astype(np.int32))
    
    print(f"  Saved numpy arrays: X={X_train_deep.shape}, y={len(y_train)}")


# =============================================================================
# 10. Summary
# =============================================================================
def print_summary() -> None:
    """Print preprocessing completion summary."""
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE (Dual-Output Version)")
    print("=" * 80)
    print("""
Outputs saved to data/processed/:

  [Supervised - for 03/04/05/07/08]
  [OK] X_train_base.csv, X_val_base.csv, X_test_base.csv (one-hot encoded)
  [OK] X_train_deep.csv, X_val_deep.csv, X_test_deep.csv (integer encoded)
  [OK] X_train_processed.npy, y_train.npy, etc. (numpy arrays)
  [OK] y_train.csv, y_val.csv, y_test.csv

  [AE - for 06 only]
  [OK] X_train_ae_base.csv, X_val_ae_base.csv, X_test_ae_base.csv (numeric only)
  [OK] X_holdout_ae_base.csv (real unlabeled data from cell2cellholdout.csv)
  [OK] holdout_customer_ids.csv, holdout_metadata.json

  [Config/Indices]
  [OK] split_indices.json (train/val/test indices)
  [OK] preprocess_params.json (fit parameters)
""")
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    U.reset_run_log()
    print("Starting preprocessing pipeline (Dual-Output Version)...\n")
    
    # 1. Load data
    df = load_raw_data()
    config = load_feature_config()
    
    # 2. Create target & prepare
    df = create_churn01(df)
    
    # 2a. Select features for each variant
    base_cols = select_feature_columns(config, variant="base")
    deep_cols = select_feature_columns(config, variant="deep")
    ae_cols = select_feature_columns(config, variant="ae")
    
    feature_cols_all = sorted(set(base_cols) | set(deep_cols) | set(ae_cols))
    keep_cols = feature_cols_all + ["Churn01"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    
    print(f"[Features] base={len(base_cols)}, deep={len(deep_cols)}, ae={len(ae_cols)}")
    
    # 2b. Type fixes
    df = fix_dtype_from_config(df, config)
    df = apply_zero_as_missing(df, config)
    
    # 3. Create or load split indices
    train_idx, val_idx, test_idx = create_or_load_split_indices(df)
    
    # 4. Fit preprocessing parameters on train only
    params = fit_preprocess_params(df, train_idx, config)
    params["feature_cols_base"] = base_cols
    params["feature_cols_deep"] = deep_cols
    params["feature_cols_ae"] = ae_cols
    
    # Sanity check: no leakage columns
    bad_leakage = (LEAKAGE_COLS & set(params["numeric_medians"].keys()))
    assert not bad_leakage, f"Leakage columns in params: {bad_leakage}"
    
    save_preprocess_params(params)
    
    # 5. Extract target variables
    y_train = df.iloc[train_idx]["Churn01"].reset_index(drop=True)
    y_val = df.iloc[val_idx]["Churn01"].reset_index(drop=True)
    y_test = df.iloc[test_idx]["Churn01"].reset_index(drop=True)
    
    # 6. Build X_base (for 03/04/05/07/08)
    X_train_base, X_val_base, X_test_base = build_x_base(
        df, train_idx, val_idx, test_idx, params
    )
    
    # 7. Build X_deep (for 05 NN)
    X_train_deep, X_val_deep, X_test_deep = build_x_deep(
        df, train_idx, val_idx, test_idx, params
    )
    
    # 8. Build X_ae_base (for 06 only) - SEPARATE SCHEMA
    X_train_ae, X_val_ae, X_test_ae = build_x_ae_base(
        df, train_idx, val_idx, test_idx, params, ae_cols
    )
    
    # 9. Process holdout data (for 06 real unlabeled)
    try:
        df_holdout = load_holdout_data()
        X_holdout_ae = build_x_holdout_ae(df_holdout, params, ae_cols, config)
    except FileNotFoundError:
        print("[WARN]  cell2cellholdout.csv not found. Skipping holdout processing.")
    
    # 10. Save numpy arrays
    save_numpy_arrays(X_train_deep, X_val_deep, X_test_deep, y_train, y_val, y_test)
    
    # 11. Save target variables as CSV
    U.save_df(y_train.to_frame(name="Churn01"), "y_train", folder=PROCESSED_DIR, log_table=False)
    U.save_df(y_val.to_frame(name="Churn01"), "y_val", folder=PROCESSED_DIR, log_table=False)
    U.save_df(y_test.to_frame(name="Churn01"), "y_test", folder=PROCESSED_DIR, log_table=False)
    
    # 12. Write manifest
    processed_files = sorted([p.name for p in PROCESSED_DIR.glob("*")])
    U.save_json(
        {"run": "02_preprocess", "processed_files": processed_files},
        "processed_files_02_preprocess.json"
    )
    
    U.write_manifest(
        run_name="02_preprocess",
        core_figures=[],
        core_tables=[],
    )
    
    print_summary()
    print("\n All preprocessing outputs saved to data/processed/")
