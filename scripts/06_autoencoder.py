#!/usr/bin/env python3
"""
06 - Autoencoder for Unsupervised Representation Learning (Real Holdout Version)

Key Features:
- Stage 1 (Unsupervised): Train DAE on labeled + unlabeled, extract latent Z
- Stage 2 (Semi-supervised): Teacher-Student pseudo-labeling
- Real Holdout Mode: Uses cell2cellholdout.csv (~20k truly unlabeled samples)
- Simulated Mode: Falls back to train split simulation for comparison

Data Source:
- Reads X_*_ae_base.csv (numeric-only) from 02_preprocess.py
- Does NOT overwrite X_*_base.csv (supervised schema is separate)

Outputs (artifacts/):
- models/06_ae_preprocessor.pkl
- models/06_dae_best.pth
- meta/06_dae_history.json
- meta/06_ae_config.json
- figures/06_dae_loss_curve.png
- tables/06_autoencoder_ablation.csv
- tables/06_threshold_table_val.csv
- figures/06_ablation_comparison.png
- preds/06_unlabeled_proba.npy
- figures/06_pseudo_confidence_dist.png
- meta/06_pseudolabel_stats.json
- preds/06_teacher_val_proba.npy, preds/06_teacher_test_proba.npy
- meta/06_teacher_metrics.json
- tables/06_pseudolabel_results.csv
- preds/06_student_val_proba.npy, preds/06_student_test_proba.npy
- meta/06_student_metrics.json
- meta/manifest_06_autoencoder.json
- figures/06_latent_tsne.png (if --with_tsne)
- figures/06_recon_error_by_label.png (if --with_recon_plot)

Usage:
    python scripts/06_autoencoder.py                    # Real holdout (default)
    python scripts/06_autoencoder.py --simulate_unlabeled  # Simulate from train
    python scripts/06_autoencoder.py --fast --skip_pseudolabel
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import utils as U
from src.autoencoder_models import (
    Autoencoder, DenoisingAutoencoder,
    train_autoencoder, extract_latent, compute_reconstruction_error,
    load_fixed_threshold, run_ablation_study,
    generate_pseudo_labels, train_student_with_pseudolabels,
    plot_loss_curve, plot_tsne, plot_reconstruction_error, plot_ablation_comparison
)

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class Config:
    RANDOM_SEED = 42
    
    # Autoencoder Architecture
    LATENT_DIM = 32
    HIDDEN_DIMS = [128, 64]
    
    # Training
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 10
    
    # Denoising
    NOISE_RATE = 0.2
    USE_DENOISING = True
    
    # Unlabeled Data Simulation (only used if --simulate_unlabeled)
    UNLABELED_RATIO = 0.3
    
    # Pseudo-labeling
    PSEUDO_THRESHOLD_HIGH = 0.95
    PSEUDO_THRESHOLD_LOW = 0.05
    PSEUDO_WEIGHT = 0.3
    MIN_PSEUDO_SAMPLES = 20
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_feature_signature(feature_list: list, input_dim: int) -> str:
    import hashlib
    sig_str = f"{tuple(sorted(feature_list))}_{input_dim}"
    return hashlib.md5(sig_str.encode()).hexdigest()[:16]


def save_or_load_unlabeled_indices(
    data_dir: Path,
    n_total: int,
    unlabeled_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Save/load unlabeled split indices (for simulated mode)."""
    indices_path = data_dir / "processed" / "06_unlabeled_indices.json"
    
    if indices_path.exists():
        print(f"Loading unlabeled split from {indices_path}")
        with open(indices_path, 'r') as f:
            saved = json.load(f)
        unlabeled_idx = np.array(saved['unlabeled_idx'])
        labeled_idx = np.array(saved['labeled_idx'])
        if len(unlabeled_idx) + len(labeled_idx) == n_total:
            return unlabeled_idx, labeled_idx
    
    print(f"Generating unlabeled split (ratio={unlabeled_ratio})")
    n_unlabeled = int(n_total * unlabeled_ratio)
    np.random.seed(seed)
    indices = np.random.permutation(n_total)
    unlabeled_idx = indices[:n_unlabeled]
    labeled_idx = indices[n_unlabeled:]
    
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(indices_path, 'w') as f:
        json.dump({
            'unlabeled_idx': unlabeled_idx.tolist(),
            'labeled_idx': labeled_idx.tolist(),
            'unlabeled_ratio': unlabeled_ratio,
            'seed': seed,
        }, f, indent=2)
    
    return unlabeled_idx, labeled_idx


def load_xgb_params(meta_dir: Path) -> Tuple[Dict, Optional[Path]]:
    """Load XGBoost params from 04 artifacts if available."""
    candidates = [
        meta_dir / "04_xgb_optuna_best_params.json",
        meta_dir / "XGBoost_Optuna_tuned_threshold_info.json",
        meta_dir / "XGBoost_baseline_v1_info.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "best_params" in data:
                    return data["best_params"], p
                if "params" in data:
                    return data["params"], p
                if "n_estimators" in data:
                    return data, p
        except Exception:
            pass
    
    return {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }, None


def plot_pseudo_confidence_distribution(
    proba: np.ndarray,
    high_thresh: float,
    low_thresh: float,
    save_path: Path,
    title: str = "Pseudo-label Confidence Distribution"
) -> None:
    """Visualize pseudo-label probability distribution with histogram + pie."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(proba, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(high_thresh, color='red', linestyle='--', linewidth=2, 
                label=f'High threshold ({high_thresh})')
    ax1.axvline(low_thresh, color='green', linestyle='--', linewidth=2, 
                label=f'Low threshold ({low_thresh})')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Probability Distribution on Unlabeled Data', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Pie chart
    ax2 = axes[1]
    n_high = (proba >= high_thresh).sum()
    n_low = (proba <= low_thresh).sum()
    n_uncertain = len(proba) - n_high - n_low
    
    sizes = [n_high, n_low, n_uncertain]
    labels = [
        f'High conf ({high_thresh})\n{n_high} ({n_high/len(proba)*100:.1f}%)',
        f'Low conf ({low_thresh})\n{n_low} ({n_low/len(proba)*100:.1f}%)',
        f'Uncertain\n{n_uncertain} ({n_uncertain/len(proba)*100:.1f}%)'
    ]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0) if n_high + n_low > 0 else (0, 0, 0)
    
    ax2.pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='', startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Pseudo-label Coverage Breakdown', fontsize=14)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence distribution plot: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='06 - Autoencoder (Real Holdout Version)')
    
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts')
    
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED)
    parser.add_argument('--latent_dim', type=int, default=Config.LATENT_DIM)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    
    # Mode flags
    parser.add_argument('--simulate_unlabeled', action='store_true',
                        help='Simulate unlabeled from train (backward compat)')
    parser.add_argument('--unlabeled_ratio', type=float, default=Config.UNLABELED_RATIO,
                        help='Ratio for simulated unlabeled (only with --simulate_unlabeled)')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--skip_pseudolabel', action='store_true')
    parser.add_argument('--with_tsne', action='store_true')
    parser.add_argument('--with_recon_plot', action='store_true')
    parser.add_argument('--fast', action='store_true')
    
    args = parser.parse_args()
    
    if args.fast:
        args.with_tsne = False
        args.with_recon_plot = False
        Config.MIN_PSEUDO_SAMPLES = 200
    
    Config.RANDOM_SEED = args.seed
    Config.LATENT_DIM = args.latent_dim
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    Config.UNLABELED_RATIO = args.unlabeled_ratio
    
    set_seed(Config.RANDOM_SEED)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    artifacts_dir = Path(args.artifacts_dir)
    
    meta_dir = artifacts_dir / "meta"
    figures_dir = artifacts_dir / "figures"
    tables_dir = artifacts_dir / "tables"
    preds_dir = artifacts_dir / "preds"
    
    for d in [models_dir, meta_dir, figures_dir, tables_dir, preds_dir, data_dir / "processed"]:
        d.mkdir(parents=True, exist_ok=True)
    
    U.reset_run_log()
    
    print("=" * 60)
    print("06 - Autoencoder (Real Holdout Version)")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Mode: {'SIMULATED unlabeled' if args.simulate_unlabeled else 'REAL holdout'}")
    
    # ========================================================================
    # Load Data (AE schema)
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Loading Data (AE Schema)...")
    print("=" * 60)
    
    processed_path = data_dir / "processed"
    
    # Try AE-specific files first, fall back to base if not found
    ae_train_path = processed_path / "X_train_ae_base.csv"
    if ae_train_path.exists():
        print("  Using AE-specific schema (X_*_ae_base.csv)")
        X_train = pd.read_csv(processed_path / "X_train_ae_base.csv")
        X_val = pd.read_csv(processed_path / "X_val_ae_base.csv")
        X_test = pd.read_csv(processed_path / "X_test_ae_base.csv")
        use_ae_schema = True
    else:
        print("  [WARN]  AE schema not found, falling back to X_*_base.csv")
        X_train = pd.read_csv(processed_path / "X_train_base.csv")
        X_val = pd.read_csv(processed_path / "X_val_base.csv")
        X_test = pd.read_csv(processed_path / "X_test_base.csv")
        use_ae_schema = False
    
    # Drop unnamed index columns
    for df in [X_train, X_val, X_test]:
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
    
    y_train = pd.read_csv(processed_path / "y_train.csv")["Churn01"]
    y_val = pd.read_csv(processed_path / "y_val.csv")["Churn01"]
    y_test = pd.read_csv(processed_path / "y_test.csv")["Churn01"]
    
    all_features = list(X_train.columns)
    print(f"Train: {X_train.shape}, Churn rate: {y_train.mean():.2%}")
    print(f"Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Total features: {len(all_features)}")
    
    # ========================================================================
    # Load Unlabeled Data
    # ========================================================================
    
    print("\n" + "=" * 60)
    if args.simulate_unlabeled:
        print("Creating SIMULATED Unlabeled Data (from train split)...")
    else:
        print("Loading REAL Holdout Data (from cell2cellholdout.csv)...")
    print("=" * 60)
    
    use_real_holdout = False
    y_unlabeled_hidden = None
    
    if args.simulate_unlabeled:
        # Simulated mode: split train into labeled/unlabeled
        unlabeled_idx, labeled_idx = save_or_load_unlabeled_indices(
            data_dir, len(X_train), Config.UNLABELED_RATIO, Config.RANDOM_SEED
        )
        X_unlabeled = X_train.iloc[unlabeled_idx].reset_index(drop=True)
        y_unlabeled_hidden = y_train.iloc[unlabeled_idx].reset_index(drop=True)
        X_train_labeled = X_train.iloc[labeled_idx].reset_index(drop=True)
        y_train_labeled = y_train.iloc[labeled_idx].reset_index(drop=True)
        
        print(f"Labeled Train: {X_train_labeled.shape}")
        print(f"Simulated Unlabeled: {X_unlabeled.shape} (labels hidden)")
    else:
        # Real holdout mode
        holdout_path = processed_path / "X_holdout_ae_base.csv"
        if holdout_path.exists():
            X_holdout = pd.read_csv(holdout_path)
            # Drop unnamed columns
            unnamed_cols = [c for c in X_holdout.columns if c.startswith("Unnamed")]
            if unnamed_cols:
                X_holdout.drop(columns=unnamed_cols, inplace=True)
            
            # Align columns with train schema
            missing_cols = set(all_features) - set(X_holdout.columns)
            extra_cols = set(X_holdout.columns) - set(all_features)
            
            if missing_cols:
                print(f"  Adding {len(missing_cols)} missing cols as zeros")
                for col in missing_cols:
                    X_holdout[col] = 0.0
            if extra_cols:
                print(f"  Dropping {len(extra_cols)} extra cols")
                X_holdout = X_holdout.drop(columns=list(extra_cols))
            
            X_holdout = X_holdout[all_features]  # Ensure same column order
            
            # Use ALL train as labeled, holdout as unlabeled
            X_train_labeled = X_train.copy()
            y_train_labeled = y_train.copy()
            X_unlabeled = X_holdout
            y_unlabeled_hidden = None  # No ground truth for real holdout
            use_real_holdout = True
            
            print(f"Labeled Train: {X_train_labeled.shape} (all train data)")
            print(f"Real Holdout Unlabeled: {X_unlabeled.shape}")
        else:
            print("  [WARN]  X_holdout_ae_base.csv not found, falling back to simulated mode")
            args.simulate_unlabeled = True
            unlabeled_idx, labeled_idx = save_or_load_unlabeled_indices(
                data_dir, len(X_train), Config.UNLABELED_RATIO, Config.RANDOM_SEED
            )
            X_unlabeled = X_train.iloc[unlabeled_idx].reset_index(drop=True)
            y_unlabeled_hidden = y_train.iloc[unlabeled_idx].reset_index(drop=True)
            X_train_labeled = X_train.iloc[labeled_idx].reset_index(drop=True)
            y_train_labeled = y_train.iloc[labeled_idx].reset_index(drop=True)
    
    # ========================================================================
    # AE Preprocessing
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("AE Preprocessing (Scaling)")
    print("=" * 60)
    
    # Since data is already preprocessed by 02, we just need simple scaling
    ae_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    print("Fitting preprocessor on labeled train only...")
    ae_preprocessor.fit(X_train_labeled)
    
    X_ae_train_labeled = ae_preprocessor.transform(X_train_labeled).astype(np.float32)
    X_ae_unlabeled = ae_preprocessor.transform(X_unlabeled).astype(np.float32)
    X_ae_val = ae_preprocessor.transform(X_val).astype(np.float32)
    X_ae_test = ae_preprocessor.transform(X_test).astype(np.float32)
    
    # Combine for unsupervised AE training
    X_ae_train_all = np.vstack([X_ae_train_labeled, X_ae_unlabeled])
    INPUT_DIM = X_ae_train_labeled.shape[1]
    
    print(f"\nPreprocessed Data:")
    print(f"  Labeled Train: {X_ae_train_labeled.shape}")
    print(f"  Unlabeled: {X_ae_unlabeled.shape}")
    print(f"  AE Train (all): {X_ae_train_all.shape}")
    print(f"  Input Dim: {INPUT_DIM}")
    
    feature_signature = compute_feature_signature(all_features, INPUT_DIM)
    print(f"Feature Signature: {feature_signature}")
    
    # Save preprocessor
    preprocessor_path = models_dir / "06_ae_preprocessor.pkl"
    joblib.dump(ae_preprocessor, preprocessor_path)
    
    # ========================================================================
    # Train Autoencoder
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Training Autoencoder...")
    print("=" * 60)
    
    model_name = "DAE" if Config.USE_DENOISING else "AE"
    checkpoint_path = models_dir / "06_dae_best.pth"
    history_path = meta_dir / "06_dae_history.json"
    config_path = meta_dir / "06_ae_config.json"
    
    # Check cache
    cache_valid = False
    if checkpoint_path.exists() and config_path.exists() and not args.force_retrain:
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        if (saved_config.get('feature_signature') == feature_signature and
            saved_config.get('input_dim') == INPUT_DIM and
            saved_config.get('latent_dim') == Config.LATENT_DIM):
            cache_valid = True
            print("Cache valid. Loading existing model...")
    
    # Build model
    if Config.USE_DENOISING:
        model = DenoisingAutoencoder(
            input_dim=INPUT_DIM,
            hidden_dims=Config.HIDDEN_DIMS,
            latent_dim=Config.LATENT_DIM,
            noise_rate=Config.NOISE_RATE,
        )
    else:
        model = Autoencoder(
            input_dim=INPUT_DIM,
            hidden_dims=Config.HIDDEN_DIMS,
            latent_dim=Config.LATENT_DIM,
        )
    
    print(f"Model: {model_name}")
    print(f"Architecture: {INPUT_DIM} -> {Config.HIDDEN_DIMS} -> {Config.LATENT_DIM}")
    
    if cache_valid:
        state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(Config.DEVICE)
        history = json.load(open(history_path)) if history_path.exists() else {}
    else:
        X_train_tensor = torch.FloatTensor(X_ae_train_all)
        X_val_tensor = torch.FloatTensor(X_ae_val)
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
        
        history = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=Config.EPOCHS,
            lr=Config.LEARNING_RATE,
            patience=Config.EARLY_STOPPING_PATIENCE,
            device=Config.DEVICE,
            checkpoint_path=checkpoint_path,
        )
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save config
    ae_config = {
        'model_type': model_name,
        'input_dim': INPUT_DIM,
        'latent_dim': Config.LATENT_DIM,
        'hidden_dims': Config.HIDDEN_DIMS,
        'best_val_loss': history.get('best_val_loss'),
        'train_size': len(X_ae_train_all),
        'labeled_size': len(X_ae_train_labeled),
        'unlabeled_size': len(X_ae_unlabeled),
        'feature_signature': feature_signature,
        'use_real_holdout': use_real_holdout,
        'seed': Config.RANDOM_SEED,
    }
    with open(config_path, 'w') as f:
        json.dump(ae_config, f, indent=2)
    
    plot_loss_curve(history, figures_dir / "06_dae_loss_curve.png", model_name)
    
    # ========================================================================
    # Extract Latent Features
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Extracting Latent Features...")
    print("=" * 60)
    
    Z_train_labeled = extract_latent(model, X_ae_train_labeled, Config.DEVICE)
    Z_unlabeled = extract_latent(model, X_ae_unlabeled, Config.DEVICE)
    Z_val = extract_latent(model, X_ae_val, Config.DEVICE)
    Z_test = extract_latent(model, X_ae_test, Config.DEVICE)
    
    print(f"Z_train_labeled: {Z_train_labeled.shape}")
    print(f"Z_unlabeled: {Z_unlabeled.shape}")
    print(f"Z_val: {Z_val.shape}")
    print(f"Z_test: {Z_test.shape}")
    
    # Save latent features
    np.save(data_dir / "processed" / "06_Z_train_labeled.npy", Z_train_labeled)
    np.save(data_dir / "processed" / "06_Z_unlabeled.npy", Z_unlabeled)
    np.save(data_dir / "processed" / "06_Z_val.npy", Z_val)
    np.save(data_dir / "processed" / "06_Z_test.npy", Z_test)
    
    # ========================================================================
    # Optional Visualizations
    # ========================================================================
    
    if args.with_tsne:
        print("\nGenerating t-SNE...")
        n_sample = min(3000, len(Z_test))
        sample_idx = np.random.choice(len(Z_test), n_sample, replace=False)
        plot_tsne(Z_test[sample_idx], y_test.iloc[sample_idx], 
                  figures_dir / "06_latent_tsne.png", model_name, n_sample, Config.RANDOM_SEED)
    
    if args.with_recon_plot:
        print("\nGenerating reconstruction error plot...")
        recon_error_test = compute_reconstruction_error(model, X_ae_test, Config.DEVICE)
        plot_reconstruction_error(recon_error_test, y_test, 
                                  figures_dir / "06_recon_error_by_label.png")
    
    # ========================================================================
    # Downstream Ablation Study
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Downstream Ablation Study")
    print("=" * 60)
    
    fixed_threshold = load_fixed_threshold(meta_dir)
    
    feature_sets = {
        'Baseline (X only)': {
            'train': X_ae_train_labeled,
            'val': X_ae_val,
            'test': X_ae_test,
        },
        'Latent-only (Z only)': {
            'train': Z_train_labeled,
            'val': Z_val,
            'test': Z_test,
        },
        'Augmented (X + Z)': {
            'train': np.hstack([X_ae_train_labeled, Z_train_labeled]),
            'val': np.hstack([X_ae_val, Z_val]),
            'test': np.hstack([X_ae_test, Z_test]),
        },
    }
    
    xgb_params, xgb_source = load_xgb_params(meta_dir)
    print(f"XGBoost params from: {xgb_source.name if xgb_source else 'default'}")
    
    ablation_df, threshold_table_val = run_ablation_study(
        feature_sets=feature_sets,
        y_train=y_train_labeled,
        y_val=y_val,
        y_test=y_test,
        xgb_params=xgb_params,
        fixed_threshold=fixed_threshold,
        tuning_rule="recall_0.7",
        seed=Config.RANDOM_SEED,
    )
    
    ablation_df.to_csv(tables_dir / "06_autoencoder_ablation.csv", index=False)
    threshold_table_val.to_csv(tables_dir / "06_threshold_table_val.csv", index=False)
    
    plot_ablation_comparison(ablation_df, figures_dir / "06_ablation_comparison.png")
    
    # ========================================================================
    # Semi-supervised Pseudo-labeling
    # ========================================================================
    
    if not args.skip_pseudolabel:
        print("\n" + "=" * 60)
        print("Semi-supervised Pseudo-labeling")
        print("=" * 60)
        
        try:
            from xgboost import XGBClassifier
            
            # Train Teacher (augmented features)
            print("\nTraining Teacher model...")
            X_teacher_train = np.hstack([X_ae_train_labeled, Z_train_labeled])
            X_teacher_val = np.hstack([X_ae_val, Z_val])
            X_teacher_unlabeled = np.hstack([X_ae_unlabeled, Z_unlabeled])
            X_teacher_test = np.hstack([X_ae_test, Z_test])
            
            teacher_clf = XGBClassifier(
                random_state=Config.RANDOM_SEED,
                objective='binary:logistic',
                eval_metric='logloss',
                n_jobs=-1,
                **xgb_params,
            )
            
            teacher_clf.fit(
                X_teacher_train, y_train_labeled,
                eval_set=[(X_teacher_val, y_val)],
                verbose=False,
            )
            
            teacher_val_proba = teacher_clf.predict_proba(X_teacher_val)[:, 1]
            teacher_test_proba = teacher_clf.predict_proba(X_teacher_test)[:, 1]
            teacher_val_auc = roc_auc_score(y_val, teacher_val_proba)
            teacher_val_prauc = average_precision_score(y_val, teacher_val_proba)
            teacher_test_auc = roc_auc_score(y_test, teacher_test_proba)
            teacher_test_prauc = average_precision_score(y_test, teacher_test_proba)
            
            print(f"Teacher Val AUC: {teacher_val_auc:.4f}, PR-AUC: {teacher_val_prauc:.4f}")
            print(f"Teacher Test AUC: {teacher_test_auc:.4f}, PR-AUC: {teacher_test_prauc:.4f}")
            
            # Generate pseudo-labels
            print("\nGenerating pseudo-labels...")
            unlabeled_proba, pseudo_labels, high_conf_mask = generate_pseudo_labels(
                teacher_clf, X_teacher_unlabeled,
                Config.PSEUDO_THRESHOLD_HIGH, Config.PSEUDO_THRESHOLD_LOW
            )
            
            # Save unlabeled probabilities for 08 analysis
            np.save(preds_dir / "06_unlabeled_proba.npy", unlabeled_proba)
            
            n_pseudo = high_conf_mask.sum()
            print(f"High-confidence pseudo-labels: {n_pseudo} ({n_pseudo/len(unlabeled_proba)*100:.1f}%)")
            
            # Verify quality (only if we have hidden labels)
            pseudo_accuracy = None
            if y_unlabeled_hidden is not None and n_pseudo > 0:
                pseudo_accuracy = (pseudo_labels[high_conf_mask] == y_unlabeled_hidden.values[high_conf_mask]).mean()
                print(f"Pseudo-label accuracy (on hidden labels): {pseudo_accuracy:.2%}")
            
            # Plot confidence distribution
            plot_pseudo_confidence_distribution(
                unlabeled_proba,
                Config.PSEUDO_THRESHOLD_HIGH,
                Config.PSEUDO_THRESHOLD_LOW,
                figures_dir / "06_pseudo_confidence_dist.png",
                title=f"Pseudo-label Confidence ({'Real Holdout' if use_real_holdout else 'Simulated'})"
            )
            
            # Save pseudo-label stats
            pseudo_stats = {
                'total_unlabeled': len(unlabeled_proba),
                'high_conf_total': int(n_pseudo),
                'high_conf_ratio': float(n_pseudo / len(unlabeled_proba)),
                'pseudo_accuracy': float(pseudo_accuracy) if pseudo_accuracy else None,
                'threshold_high': Config.PSEUDO_THRESHOLD_HIGH,
                'threshold_low': Config.PSEUDO_THRESHOLD_LOW,
                'use_real_holdout': use_real_holdout,
                'proba_mean': float(unlabeled_proba.mean()),
                'proba_std': float(unlabeled_proba.std()),
            }
            with open(meta_dir / "06_pseudolabel_stats.json", 'w') as f:
                json.dump(pseudo_stats, f, indent=2)
            
            # Save teacher predictions
            np.save(preds_dir / "06_teacher_val_proba.npy", teacher_val_proba)
            np.save(preds_dir / "06_teacher_test_proba.npy", teacher_test_proba)
            U.save_json({
                "model": "06_teacher",
                "val_auc": float(teacher_val_auc),
                "val_pr_auc": float(teacher_val_prauc),
                "test_auc": float(teacher_test_auc),
                "test_pr_auc": float(teacher_test_prauc),
            }, "06_teacher_metrics", folder=meta_dir)
            
            # Train Student (if sufficient samples)
            if n_pseudo >= Config.MIN_PSEUDO_SAMPLES:
                student_clf = train_student_with_pseudolabels(
                    X_labeled=X_teacher_train,
                    y_labeled=y_train_labeled,
                    X_unlabeled=X_teacher_unlabeled,
                    pseudo_labels=pseudo_labels,
                    high_conf_mask=high_conf_mask,
                    X_val=X_teacher_val,
                    y_val=y_val,
                    xgb_params=xgb_params,
                    pseudo_weight=Config.PSEUDO_WEIGHT,
                    seed=Config.RANDOM_SEED,
                )
                
                student_val_proba = student_clf.predict_proba(X_teacher_val)[:, 1]
                student_test_proba = student_clf.predict_proba(X_teacher_test)[:, 1]
                student_val_auc = roc_auc_score(y_val, student_val_proba)
                student_val_prauc = average_precision_score(y_val, student_val_proba)
                student_test_auc = roc_auc_score(y_test, student_test_proba)
                student_test_prauc = average_precision_score(y_test, student_test_proba)
                
                print(f"Student Val AUC: {student_val_auc:.4f}")
                print(f"Student Test AUC: {student_test_auc:.4f}")
                
                ssl_improvement = (student_test_auc - teacher_test_auc) / teacher_test_auc * 100
                print(f"SSL Improvement: {ssl_improvement:+.2f}%")
                
                ssl_results = pd.DataFrame([
                    {'model': 'Teacher (labeled only)', 'test_auc': teacher_test_auc, 'test_prauc': teacher_test_prauc},
                    {'model': 'Student (+ pseudo-labels)', 'test_auc': student_test_auc, 'test_prauc': student_test_prauc},
                ])
                ssl_results.to_csv(tables_dir / "06_pseudolabel_results.csv", index=False)
                
                np.save(preds_dir / "06_student_val_proba.npy", student_val_proba)
                np.save(preds_dir / "06_student_test_proba.npy", student_test_proba)
                U.save_json({
                    "model": "06_student",
                    "val_auc": float(student_val_auc),
                    "val_pr_auc": float(student_val_prauc),
                    "test_auc": float(student_test_auc),
                    "test_pr_auc": float(student_test_prauc),
                }, "06_student_metrics", folder=meta_dir)
            else:
                print(f"\nSkipping student training: insufficient pseudo-labels ({n_pseudo} < {Config.MIN_PSEUDO_SAMPLES})")
                ssl_results = pd.DataFrame([
                    {'model': 'Teacher (labeled only)', 'test_auc': teacher_test_auc, 'test_prauc': teacher_test_prauc},
                ])
                ssl_results.to_csv(tables_dir / "06_pseudolabel_results.csv", index=False)
        
        except ImportError:
            print("\nXGBoost not available. Skipping pseudo-labeling.")
    
    # ========================================================================
    # Write Manifest
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Writing Manifest...")
    print("=" * 60)
    
    manifest = {
        'script': '06_autoencoder.py',
        'timestamp': pd.Timestamp.now().isoformat(),
        'config': ae_config,
        'use_real_holdout': use_real_holdout,
        
        'core_tables': ["06_autoencoder_ablation.csv"],
        'core_figures': ["06_ablation_comparison.png", "06_dae_loss_curve.png"],
        'appendix_tables': ["06_threshold_table_val.csv", "06_pseudolabel_results.csv"],
        'appendix_figures': ["06_latent_tsne.png", "06_recon_error_by_label.png", 
                            "06_pseudo_confidence_dist.png"],
    }
    
    with open(meta_dir / "manifest_06_autoencoder.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "=" * 60)
    print("06 - Autoencoder Completed Successfully!")
    print("=" * 60)
    print(f"\nMode: {'REAL holdout' if use_real_holdout else 'SIMULATED unlabeled'}")
    print(f"Unlabeled samples: {len(X_unlabeled)}")


if __name__ == "__main__":
    main()
