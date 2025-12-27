"""
PyTorch Neural Network Training Script (05)
==========================================

 PyTorch training curvesROC/PR curves

 script 
1.  02_preprocess  X_*_deep.csv (integer encoded + standardized)
2.  feature_groups  continuous/categorical/binary 
3.  7  cache 
4.  validation set  threshold tuning
5. 


- models/:  PyTorch 
- artifacts/figures/: Training curves, ROC+PR curves
- artifacts/tables/: Leaderboard
- artifacts/meta/: Config, metrics, history for each experiment
- artifacts/preds/: Test predictions (for 08)


    python scripts/05_nn.py                    # 7
    python scripts/05_nn.py --fast             # 3
    python scripts/05_nn.py --force_retrain    # 
    python scripts/05_nn.py --epochs 50        #  epochs
"""

import os
import sys
from pathlib import Path
import argparse
import warnings
import json
import hashlib

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["CAPSTONE_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib

import utils as U

# Import nn_models
try:
    import nn_models as NM
except ImportError:
    print("ERROR: nn_models.py not found in src/")
    print("   Please ensure src/nn_models.py exists.")
    sys.exit(1)

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_CONFIG = PROJECT_ROOT / "data" / "feature_config_clean.csv"
PREPROCESS_PARAMS = DATA_DIR / "preprocess_params.json"

# Experiments to run
ALL_EXPERIMENTS = [
    "dense_baseline",
    "embedding_baseline",
    "embedding_strong_weight",
    "embedding_focal_loss",
    "wide_and_deep",
    "wide_and_deep_deeper",
    "wide_deep_focal",
]

FAST_EXPERIMENTS = [
    "dense_baseline",
    "embedding_baseline",
    "wide_and_deep",
]

# Threshold tuning
MIN_RECALL_CONSTRAINT = 0.70


# ============================================================================
# Helper Functions
# ============================================================================

def compute_feature_signature(X: pd.DataFrame, preprocess_params: dict) -> str:
    """
    Compute a signature for feature set to detect cache invalidation.
    
    Includes:
    - Column names and order
    - Categorical cardinalities
    """
    signature_parts = [
        "cols:" + ",".join(X.columns.tolist()),
    ]
    
    # Add categorical mappings if available
    if "categorical_mappings" in preprocess_params:
        cat_maps = preprocess_params["categorical_mappings"]
        for col, mapping in sorted(cat_maps.items()):
            if col in X.columns:
                signature_parts.append(f"{col}:{(max(mapping.values()) + 1) if mapping else 0}")
    
    signature = "|".join(signature_parts)
    return hashlib.md5(signature.encode()).hexdigest()[:16]


def load_feature_groups(data_dir: Path, X: pd.DataFrame) -> dict[str, list[str]]:
    """
    Load feature groups (continuous/categorical/binary) from config.
    
    Priority:
    1. artifacts/meta/feature_groups.json (if exists)
    2. preprocess_params.json
    3. Infer from feature_config_clean.csv
    """
    # Try feature_groups.json
    feature_groups_candidates = [
        PROJECT_ROOT / "artifacts" / "meta" / "feature_groups.json",
        PROJECT_ROOT / "artifacts" / "feature_groups.json",
    ]
    for fg_path in feature_groups_candidates:
        if fg_path.exists():
            with open(fg_path, "r") as f:
                groups = json.load(f)
                print(f"[OK] Loaded feature_groups from: {fg_path}")
                return groups
    
    # Try preprocess_params.json
    preprocess_params_candidates = [
        data_dir / "preprocess_params.json",
        PROJECT_ROOT / "data" / "processed" / "preprocess_params.json",
    ]
    for pp_path in preprocess_params_candidates:
        if not pp_path.exists():
            continue
        with open(pp_path, "r") as f:
            params = json.load(f)
        # Preferred: explicit feature_groups
        if "feature_groups" in params:
            groups = params["feature_groups"]
            groups = {k: [c for c in v if c in X.columns] for k, v in groups.items()}
            print(f"[OK] Loaded feature_groups from: {pp_path}")
            return groups
        # Otherwise infer from categorical_mappings / numeric_scaler
        cat_cols = set(params.get("categorical_mappings", {}).keys()) & set(X.columns)
        cont_cols = set(params.get("numeric_scaler", {}).keys()) & set(X.columns)
        bin_cols = {c for c in X.columns if c not in cat_cols and c not in cont_cols and X[c].nunique() == 2}
        groups = {
            "continuous": sorted(cont_cols),
            "categorical": sorted(cat_cols),
            "binary": sorted(bin_cols),
        }
        print(f"[OK] Inferred feature_groups from preprocess_params: cont={len(cont_cols)}, cat={len(cat_cols)}, bin={len(bin_cols)}")
        return groups
    
    # Infer from feature_config_clean.csv
    config_candidates = [
        PROJECT_ROOT / "data" / "feature_config_clean.csv",
        PROJECT_ROOT / "feature_config_clean.csv",
        data_dir / "feature_config_clean.csv",
    ]
    for config_path in config_candidates:
        if not config_path.exists():
            continue
        print(f"[WARN]  Inferring feature_groups from: {config_path}")
        config = pd.read_csv(config_path)
        
        # Filter to columns in X
        config = config[config["feature"].isin(X.columns)]
        
        # Infer based on type column (if available)
        continuous = []
        categorical = []
        binary = []
        
        for _, row in config.iterrows():
            feat = row["feature"]
            if "type" in config.columns:
                ftype = str(row["type"]).lower()
                if "continuous" in ftype or "numeric" in ftype:
                    continuous.append(feat)
                elif "binary" in ftype or "bool" in ftype:
                    binary.append(feat)
                elif "categorical" in ftype:
                    categorical.append(feat)
                else:
                    # Default: treat as continuous
                    continuous.append(feat)
            else:
                # No type info, use heuristics
                if X[feat].nunique() == 2:
                    binary.append(feat)
                elif X[feat].nunique() < 20:
                    categorical.append(feat)
                else:
                    continuous.append(feat)
        
        groups = {
            "continuous": continuous,
            "categorical": categorical,
            "binary": binary,
        }
        
        print(f"  Continuous: {len(continuous)}")
        print(f"  Categorical: {len(categorical)}")
        print(f"  Binary: {len(binary)}")
        
        return groups
    
    # Fallback: treat all as continuous
    print("[WARN]  No feature config found, treating all features as continuous")
    return {
        "continuous": X.columns.tolist(),
        "categorical": [],
        "binary": [],
    }


def load_data_and_groups():
    """Load data splits and feature groups."""
    print("\\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load splits
    X_train = pd.read_csv(DATA_DIR / "X_train_deep.csv")
    X_val = pd.read_csv(DATA_DIR / "X_val_deep.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_deep.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel().astype(np.float32)
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").values.ravel().astype(np.float32)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel().astype(np.float32)
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # Load feature groups
    feature_groups = load_feature_groups(DATA_DIR, X_train)
    
    # Load preprocess params for signature
    preprocess_params = {}
    if PREPROCESS_PARAMS.exists():
        with open(PREPROCESS_PARAMS, "r") as f:
            preprocess_params = json.load(f)
    
    # Compute feature signature
    feature_signature = compute_feature_signature(X_train, preprocess_params)
    print(f"\\n[OK] Feature signature: {feature_signature}")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_groups": feature_groups,
        "feature_signature": feature_signature,
        "preprocess_params": preprocess_params,
    }


def create_datasets_and_loaders(data, batch_size: int, seed: int):
    """Create PyTorch datasets and dataloaders with deterministic shuffle."""
    # Extract feature groups
    continuous_cols = data["feature_groups"]["continuous"]
    categorical_cols = data["feature_groups"]["categorical"]
    binary_cols = data["feature_groups"]["binary"]
    
    # Get categorical cardinalities from preprocess_params
    categorical_cardinalities = {}
    if "categorical_mappings" in data["preprocess_params"]:
        cat_maps = data["preprocess_params"]["categorical_mappings"]
        for col in categorical_cols:
            if col in cat_maps:
                categorical_cardinalities[col] = (max(cat_maps[col].values()) + 1) if cat_maps[col] else 1
            else:
                # Infer from data
                categorical_cardinalities[col] = int(data["X_train"][col].max() + 1)
    else:
        # Infer all from data
        for col in categorical_cols:
            categorical_cardinalities[col] = int(data["X_train"][col].max() + 1)
    
    # Create datasets
    train_dataset = NM.ChurnDataset(
        data["X_train"], data["y_train"],
        continuous_cols, categorical_cols, binary_cols,
        categorical_cardinalities,
    )
    
    val_dataset = NM.ChurnDataset(
        data["X_val"], data["y_val"],
        continuous_cols, categorical_cols, binary_cols,
        categorical_cardinalities,
    )
    
    test_dataset = NM.ChurnDataset(
        data["X_test"], data["y_test"],
        continuous_cols, categorical_cols, binary_cols,
        categorical_cardinalities,
    )
    
    # Deterministic generator for shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    # Create loaders (num_workers=0 for stability)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def check_cache_validity(exp_name: str, feature_signature: str, config: dict) -> bool:
    """
    Check if cached model is still valid.
    
    Returns True if cache is valid, False otherwise.
    """
    model_path = U.DIRS.models / f"05_nn_{exp_name}.pth"
    config_path = U.DIRS.meta / f"05_nn_{exp_name}_config.json"
    
    if not (model_path.exists() and config_path.exists()):
        return False
    
    # Load cached config
    with open(config_path, "r") as f:
        cached_config = json.load(f)
    
    # Check feature signature
    if cached_config.get("feature_signature") != feature_signature:
        print(f"  Feature signature mismatch, cache invalid")
        return False
    
    # Check config hash (hyperparameters)
    current_config_hash = hashlib.md5(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:16]
    
    if cached_config.get("config_hash") != current_config_hash:
        print(f"  [WARN]  Config changed, cache invalid")
        return False
    
    return True


# ============================================================================
# Experiment Definitions
# ============================================================================

def get_experiment_config(exp_name: str, args, pos_weight: float) -> dict:
    """Get configuration for a specific experiment."""
    base_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": 15,
        "pos_weight": pos_weight,
    }
    
    if exp_name == "dense_baseline":
        return {
            **base_config,
            "model_type": "dense",
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "loss_type": "bce_weighted",
        }
    
    elif exp_name == "embedding_baseline":
        return {
            **base_config,
            "model_type": "embedding",
            "embedding_dim": 8,
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "loss_type": "bce_weighted",
        }
    
    elif exp_name == "embedding_strong_weight":
        return {
            **base_config,
            "model_type": "embedding",
            "embedding_dim": 8,
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "loss_type": "bce_weighted",
            "pos_weight": pos_weight * 1.5,  # Stronger weight
        }
    
    elif exp_name == "embedding_focal_loss":
        return {
            **base_config,
            "model_type": "embedding",
            "embedding_dim": 8,
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "loss_type": "focal",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        }
    
    elif exp_name == "wide_and_deep":
        return {
            **base_config,
            "model_type": "wide_deep",
            "embedding_dim": 8,
            "deep_hidden_dims": [128, 64],
            "dropout": 0.3,
            "loss_type": "bce_weighted",
            "wide_deep_alpha": 0.5,
        }
    
    elif exp_name == "wide_and_deep_deeper":
        return {
            **base_config,
            "model_type": "wide_deep",
            "embedding_dim": 8,
            "deep_hidden_dims": [256, 128, 64],  # Deeper
            "dropout": 0.3,
            "loss_type": "bce_weighted",
            "wide_deep_alpha": 0.5,
        }
    
    elif exp_name == "wide_deep_focal":
        return {
            **base_config,
            "model_type": "wide_deep",
            "embedding_dim": 8,
            "deep_hidden_dims": [128, 64],
            "dropout": 0.3,
            "loss_type": "focal",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "wide_deep_alpha": 0.5,
        }
    
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")


def run_experiment(
    exp_name: str,
    config: dict,
    datasets,
    input_dims: dict,
    device,
    force_retrain: bool,
    feature_signature: str,
    verbose: bool = True,
):
    """
    Run a single experiment: train or load from cache.
    
    Returns:
        model: Trained model
        history: Training history
        best_epoch: Best epoch number
        train_time: Training time in seconds
    """
    if verbose:
        print("\\n" + "="*80)
        print(f"EXPERIMENT: {exp_name.upper()}")
        print("="*80)
    
    model_path = U.DIRS.models / f"05_nn_{exp_name}.pth"
    history_path = U.DIRS.meta / f"05_nn_{exp_name}_history.pkl"
    config_path = U.DIRS.meta / f"05_nn_{exp_name}_config.json"
    
    # Check cache validity
    cache_valid = check_cache_validity(exp_name, feature_signature, config)
    
    if cache_valid and not force_retrain:
        if verbose:
            print(f" Loading cached model from: {model_path}")
        
        # Create model architecture
        if config["model_type"] == "dense":
            model = NM.create_dense_model(
                input_dims,
                hidden_dims=config["hidden_dims"],
                dropout=config["dropout"],
            )
        elif config["model_type"] == "embedding":
            model = NM.create_embedding_model(
                input_dims,
                embedding_dim=config["embedding_dim"],
                hidden_dims=config["hidden_dims"],
                dropout=config["dropout"],
            )
        elif config["model_type"] == "wide_deep":
            model = NM.create_wide_deep_model(
                input_dims,
                embedding_dim=config["embedding_dim"],
                deep_hidden_dims=config["deep_hidden_dims"],
                dropout=config["dropout"],
                alpha=config["wide_deep_alpha"],
            )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        # Load history
        with open(history_path, "rb") as f:
            history = joblib.load(f)
        
        # Load config for best_epoch and train_time
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        best_epoch = saved_config.get("best_epoch", 0)
        train_time = saved_config.get("train_time", 0)
        
        if verbose:
            print(f"[OK] Model loaded (best epoch: {best_epoch+1})")
        
        return model, history, best_epoch, train_time
    
    # Train new model
    if verbose:
        print(f" Training new model...")
        print(f"  Config: {config}")
    
    # Create model
    if config["model_type"] == "dense":
        model = NM.create_dense_model(
            input_dims,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        )
    elif config["model_type"] == "embedding":
        model = NM.create_embedding_model(
            input_dims,
            embedding_dim=config["embedding_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        )
    elif config["model_type"] == "wide_deep":
        model = NM.create_wide_deep_model(
            input_dims,
            embedding_dim=config["embedding_dim"],
            deep_hidden_dims=config["deep_hidden_dims"],
            dropout=config["dropout"],
            alpha=config["wide_deep_alpha"],
        )
    
    model = model.to(device)
    
    if verbose:
        n_params = NM.count_parameters(model)
        print(f"  Model parameters: {n_params:,}")
    
    # Create loss function
    criterion = NM.create_loss_function(
        loss_type=config["loss_type"],
        pos_weight=config.get("pos_weight"),
        alpha=config.get("focal_alpha", 0.25),
        gamma=config.get("focal_gamma", 2.0),
    )
    
    # Create optimizer and scheduler (compatible with all torch versions)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    import inspect
    scheduler_kwargs = dict(mode="max", factor=0.5, patience=5, min_lr=1e-6)
    sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__)
    if "verbose" in sig.parameters:
        scheduler_kwargs["verbose"] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    
    # Train
    history, best_model_state, best_epoch, train_time = NM.train_model(
        model,
        datasets["train_loader"],
        datasets["val_loader"],
        criterion,
        optimizer,
        scheduler,
        epochs=config["epochs"],
        device=device,
        patience=config["patience"],
        verbose=verbose,
    )
    
    # After training, load best weights back into model
    model.load_state_dict(best_model_state)
    model = model.to(device)
    
    # Save model
    torch.save(best_model_state, model_path)
    
    # Save history
    with open(history_path, "wb") as f:
        joblib.dump(history, f)
    
    # Save config with metadata
    config_with_meta = {
        **config,
        "feature_signature": feature_signature,
        "config_hash": hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16],
        "best_epoch": int(best_epoch),
        "train_time": float(train_time),
        "n_params": NM.count_parameters(model),
    }
    U.save_json(config_with_meta, f"05_nn_{exp_name}_config", folder=U.DIRS.meta)
    
    if verbose:
        print(f"\\n[OK] Model saved: {model_path}")
    
    return model, history, best_epoch, train_time


# ============================================================================
# Main Pipeline
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PyTorch neural networks for churn prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: only 3 experiments with reduced epochs"
    )
    
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining even if cached models exist"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_args()
    
    print("\\n" + "="*80)
    print("PYTORCH NEURAL NETWORK TRAINING PIPELINE")
    print("="*80)
    print(f"Device: {NM.DEVICE}")
    print(f"Random seed: {args.seed}")
    print(f"Fast mode: {args.fast}")
    print(f"Force retrain: {args.force_retrain}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Set seed
    NM.set_seed(args.seed)
    
    # Reset run log
    U.reset_run_log()
    
    # Adjust epochs for fast mode
    if args.fast:
        args.epochs = min(args.epochs, 30)
        print(f"\\n Fast mode: reducing epochs to {args.epochs}")
    
    # Load data
    data = load_data_and_groups()
    
    # Calculate pos_weight for class imbalance
    n_pos = data["y_train"].sum()
    n_neg = len(data["y_train"]) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\\n[OK] Class imbalance: pos_weight = {pos_weight:.2f}")
    
    # Create datasets and loaders
    datasets = create_datasets_and_loaders(data, args.batch_size, seed=args.seed)
    input_dims = datasets["train_dataset"].get_input_dims()
    
    print(f"\\n[OK] Dataset created:")
    print(f"  Continuous features: {input_dims['n_continuous']}")
    print(f"  Categorical features: {input_dims['n_categorical']}")
    print(f"  Binary features: {input_dims['n_binary']}")
    
    # Select experiments to run
    experiments = FAST_EXPERIMENTS if args.fast else ALL_EXPERIMENTS
    
    print(f"\\n[OK] Running {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp}")
    
    # Run experiments
    results = []
    
    for exp_name in experiments:
        config = get_experiment_config(exp_name, args, pos_weight)
        
        # Run experiment
        model, history, best_epoch, train_time = run_experiment(
            exp_name=exp_name,
            config=config,
            datasets=datasets,
            input_dims=input_dims,
            device=NM.DEVICE,
            force_retrain=args.force_retrain,
            feature_signature=data["feature_signature"],
            verbose=True,
        )
        
        # Evaluate on all splits
        print(f"\\n[TABLES] Evaluating {exp_name}...")
        
        train_y, train_proba, train_metrics = NM.evaluate_model(
            model, datasets["train_loader"], NM.DEVICE
        )
        val_y, val_proba, val_metrics = NM.evaluate_model(
            model, datasets["val_loader"], NM.DEVICE
        )
        test_y, test_proba, test_metrics = NM.evaluate_model(
            model, datasets["test_loader"], NM.DEVICE
        )
        
        # Threshold tuning on validation set
        best_threshold, threshold_metrics = NM.tune_threshold_with_constraint(
            val_y, val_proba, min_recall=MIN_RECALL_CONSTRAINT
        )
        
        # Evaluate test with best threshold
        test_pred = (test_proba >= best_threshold).astype(int)
        from sklearn.metrics import precision_recall_fscore_support
        test_precision_thresh, test_recall_thresh, test_f1_thresh, _ = precision_recall_fscore_support(
            test_y, test_pred, average="binary", zero_division=0
        )
        
        print(f"  Train AUC: {train_metrics['auc']:.4f}")
        print(f"  Val AUC:   {val_metrics['auc']:.4f}")
        print(f"  Test AUC:  {test_metrics['auc']:.4f}")
        print(f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"  Best threshold: {best_threshold:.4f}")
        print(f"  Test F1 @ threshold: {test_f1_thresh:.4f}")
        
        # Save metrics
        metrics_dict = {
            "experiment": exp_name,
            "train_auc": float(train_metrics["auc"]),
            "val_auc": float(val_metrics["auc"]),
            "test_auc": float(test_metrics["auc"]),
            "test_pr_auc": float(test_metrics["pr_auc"]),
            "test_brier": float(test_metrics.get("brier", np.nan)),
            "best_threshold_val": float(best_threshold),
            "test_precision_at_best_threshold": float(test_precision_thresh),
            "test_recall_at_best_threshold": float(test_recall_thresh),
            "test_f1_at_best_threshold": float(test_f1_thresh),
            "best_epoch": int(best_epoch),
            "train_time_sec": float(train_time),
            "loss_type": config["loss_type"],
            "pos_weight": float(config.get("pos_weight", 0)),
            "focal_alpha": float(config.get("focal_alpha", 0)),
            "focal_gamma": float(config.get("focal_gamma", 0)),
        }
        
        U.save_json(metrics_dict, f"05_nn_{exp_name}_metrics", folder=U.DIRS.meta)
        
        # Add to results
        results.append(metrics_dict)
        
        # Save training curves
        fig_curves = NM.plot_training_curves(history)
        U.save_fig(fig_curves, f"05_nn_{exp_name}_curves")
        
        # Save predictions (for 08)
        preds_dir = U.DIRS.artifacts / "preds"
        preds_dir.mkdir(exist_ok=True, parents=True)
        np.save(preds_dir / f"05_nn_{exp_name}_val_proba.npy", val_proba)
        np.save(preds_dir / f"05_nn_{exp_name}_test_proba.npy", test_proba)
        np.save(preds_dir / f"05_nn_{exp_name}_test_preds.npy", test_pred)
        
        # Log to model results
        U.log_model_result(
            model_name=f"05_nn_{exp_name}",
            params=config,
            metrics=metrics_dict,
            notes=f"PyTorch NN - {exp_name}"
        )
    
    # Create leaderboard
    leaderboard_df = pd.DataFrame(results)
    
    # Sort by val_auc
    leaderboard_df = leaderboard_df.sort_values("val_auc", ascending=False)
    
    # Save leaderboard
    U.save_df(leaderboard_df, "05_nn_leaderboard", folder=U.DIRS.tables)
    
    print("\\n" + "="*80)
    print("LEADERBOARD (sorted by Val AUC)")
    print("="*80)
    print(leaderboard_df[[
        "experiment", "val_auc", "test_auc", "test_pr_auc", 
        "test_f1_at_best_threshold", "train_time_sec"
    ]].to_string(index=False))
    
    # Select best experiment
    best_exp = leaderboard_df.iloc[0]
    best_exp_name = best_exp["experiment"]
    
    # Tie-breaker: if top 2 are within 0.001, prefer wide_and_deep
    if len(leaderboard_df) >= 2:
        top2_diff = leaderboard_df.iloc[0]["val_auc"] - leaderboard_df.iloc[1]["val_auc"]
        if top2_diff < 0.001 and "wide_and_deep" in leaderboard_df.iloc[1]["experiment"]:
            best_exp = leaderboard_df.iloc[1]
            best_exp_name = best_exp["experiment"]
            print(f"\\n[OK] Tie-breaker: selecting {best_exp_name} (within 0.001 AUC)")
    
    print(f"\\n[OK] Best experiment: {best_exp_name}")
    print(f"  Val AUC: {best_exp['val_auc']:.4f}")
    print(f"  Test AUC: {best_exp['test_auc']:.4f}")
    
    # Generate best model visualizations
    print("\\n" + "="*80)
    print(f"GENERATING VISUALIZATIONS FOR BEST MODEL: {best_exp_name.upper()}")
    print("="*80)
    
    # Load best model
    best_model_path = U.DIRS.models / f"05_nn_{best_exp_name}.pth"
    best_config = get_experiment_config(best_exp_name, args, pos_weight)
    
    if best_config["model_type"] == "dense":
        best_model = NM.create_dense_model(
            input_dims,
            hidden_dims=best_config["hidden_dims"],
            dropout=best_config["dropout"],
        )
    elif best_config["model_type"] == "embedding":
        best_model = NM.create_embedding_model(
            input_dims,
            embedding_dim=best_config["embedding_dim"],
            hidden_dims=best_config["hidden_dims"],
            dropout=best_config["dropout"],
        )
    elif best_config["model_type"] == "wide_deep":
        best_model = NM.create_wide_deep_model(
            input_dims,
            embedding_dim=best_config["embedding_dim"],
            deep_hidden_dims=best_config["deep_hidden_dims"],
            dropout=best_config["dropout"],
            alpha=best_config["wide_deep_alpha"],
        )
    
    best_model.load_state_dict(torch.load(best_model_path, map_location=NM.DEVICE))
    best_model = best_model.to(NM.DEVICE)
    
    # Get test predictions
    test_y, test_proba, _ = NM.evaluate_model(best_model, datasets["test_loader"], NM.DEVICE)
    best_threshold = best_exp["best_threshold_val"]
    
    # ROC + PR with point
    fig_roc_pr = NM.plot_roc_pr_with_point(
        test_y, test_proba, best_threshold, model_name=best_exp_name
    )
    U.save_fig(fig_roc_pr, f"05_nn_{best_exp_name}_roc_pr_test")
    
    # Write manifest
    print("\\n" + "="*80)
    print("WRITING MANIFEST")
    print("="*80)
    
    core_tables = ["05_nn_leaderboard.csv"]
    core_figures = [
        f"05_nn_{best_exp_name}_curves.png",
        f"05_nn_{best_exp_name}_roc_pr_test.png",
    ]
    
    U.write_manifest(
        run_name="05_nn",
        core_tables=core_tables,
        core_figures=core_figures,
        reset=True,
    )
    
    # Summary
    print("\\n" + "="*80)
    print("PYTORCH NEURAL NETWORK TRAINING - COMPLETED")
    print("="*80)
    print(f"\\n[OK] Best experiment: {best_exp_name}")
    print(f"[OK] Best val AUC: {best_exp['val_auc']:.4f}")
    print(f"[OK] Best test AUC: {best_exp['test_auc']:.4f}")
    print(f"[OK] Best threshold: {best_threshold:.4f}")
    print(f"\\n[OK] Models saved to: {U.DIRS.models}")
    print(f"[OK] Artifacts saved to: {PROJECT_ROOT / 'artifacts'}")
    print(f"[OK] Manifest: artifacts/meta/manifest_05_nn.json")
    print("\\n" + "="*80)


if __name__ == "__main__":
    main()
