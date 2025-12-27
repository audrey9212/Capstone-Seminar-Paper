"""
Autoencoder Models for Unsupervised Representation Learning (06)

Contains:
- Autoencoder / Denoising Autoencoder classes
- Training loop with early stopping
- Latent feature extraction
- Downstream ablation study (fixed + tuned threshold)
- Semi-supervised pseudo-labeling (Teacher-Student)
- Evaluation and plotting utilities
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, recall_score, precision_score, roc_curve
)
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None  # optional dependency


# ============================================================================
# Autoencoder Architecture
# ============================================================================

class Autoencoder(nn.Module):
    """Autoencoder with configurable architecture.
    
    Architecture:
        Encoder: input_dim -> hidden_dims -> latent_dim
        Decoder: latent_dim -> hidden_dims (reversed) -> input_dim
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on latent layer
                encoder_layers.append(nn.ReLU())
                if dropout_rate > 0:
                    encoder_layers.append(nn.Dropout(dropout_rate))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build Decoder (reverse)
        decoder_layers = []
        dims_dec = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(dims_dec) - 1):
            decoder_layers.append(nn.Linear(dims_dec[i], dims_dec[i+1]))
            if i < len(dims_dec) - 2:
                decoder_layers.append(nn.ReLU())
                if dropout_rate > 0:
                    decoder_layers.append(nn.Dropout(dropout_rate))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class DenoisingAutoencoder(Autoencoder):
    """Denoising Autoencoder - adds dropout noise during training."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        noise_rate: float = 0.2,
        dropout_rate: float = 0.0,
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.noise_rate = noise_rate
        self.noise_layer = nn.Dropout(noise_rate)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add noise only during training
        if self.training:
            x_noisy = self.noise_layer(x)
        else:
            x_noisy = x
        
        z = self.encode(x_noisy)
        x_recon = self.decode(z)
        return x_recon, z


# ============================================================================
# Training
# ============================================================================

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device: str,
    checkpoint_path: Path,
) -> Dict:
    """Train autoencoder with early stopping."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining {model.__class__.__name__}...")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            
            optimizer.zero_grad()
            X_recon, _ = model(X_batch)
            loss = criterion(X_recon, X_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                X_recon, _ = model(X_batch)
                loss = criterion(X_recon, X_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    print(f"\nTraining completed. Best val_loss: {best_val_loss:.6f}")
    
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = len(history['val_loss']) - patience_counter
    
    return history


# ============================================================================
# Latent Extraction
# ============================================================================

def extract_latent(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Extract latent representation Z from autoencoder."""
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        _, Z = model(X_tensor)
    
    return Z.cpu().numpy()


def compute_reconstruction_error(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Compute per-sample reconstruction error (MSE)."""
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        X_recon, _ = model(X_tensor)
    
    # Per-sample MSE
    mse = ((X_tensor - X_recon) ** 2).mean(dim=1).cpu().numpy()
    return mse


# ============================================================================
# Threshold Utilities (Fixed + Tuned)
# ============================================================================

def load_fixed_threshold(meta_dir: Path) -> float:
    """Load best_threshold from 04's artifacts (for fixed-threshold evaluation)."""
    threshold_rule_path = meta_dir / "04_best_threshold_rule.json"
    
    if threshold_rule_path.exists():
        with open(threshold_rule_path, 'r') as f:
            rule = json.load(f)
        threshold = rule.get('best_threshold', 0.5)
        print(f"Loaded fixed threshold from 04: {threshold:.4f}")
        return threshold
    else:
        print(f"Warning: 04 threshold rule not found. Using default 0.5")
        return 0.5


def tune_threshold_on_val(y_val: pd.Series, proba_val: np.ndarray, rule: str = "recall_0.7") -> Tuple[float, pd.DataFrame]:
    """
    Tune threshold on validation set with specified rule.
    
    Args:
        y_val: True labels
        proba_val: Predicted probabilities
        rule: Tuning rule (default: "recall_0.7" = recall >= 0.7, maximize F1)
    
    Returns:
        best_threshold, threshold_table_val
    """
    thresholds = np.linspace(0, 1, 101)
    results = []
    
    for th in thresholds:
        preds = (proba_val >= th).astype(int)
        
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        results.append({
            'threshold': th,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    
    df = pd.DataFrame(results)
    
    # Apply tuning rule
    if rule == "recall_0.7":
        # Recall >= 0.7, maximize F1
        valid = df[df['recall'] >= 0.7]
        if len(valid) == 0:
            # Fallback: max F1 without constraint
            best_idx = df['f1'].idxmax()
        else:
            best_idx = valid['f1'].idxmax()
    else:
        # Default: max F1
        best_idx = df['f1'].idxmax()
    
    best_threshold = df.loc[best_idx, 'threshold']
    
    return best_threshold, df


# ============================================================================
# Ablation Study (Fixed + Tuned Threshold)
# ============================================================================

def run_ablation_study(
    feature_sets: Dict[str, Dict[str, np.ndarray]],
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    xgb_params: Dict,
    fixed_threshold: float,
    tuning_rule: str = "recall_0.7",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ablation study with both fixed and tuned threshold evaluation.
    
    Returns:
        ablation_df: Main results (both fixed & tuned columns)
        threshold_table_val: Threshold sweep results on validation set
    """
    # Defensive: ensure labels are 1D (XGBoost expects 1D arrays)
    y_train = np.asarray(y_train).reshape(-1)
    y_val = np.asarray(y_val).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    try:
        from xgboost import XGBClassifier
        # Ensure 1D labels for XGBoost (avoid (n,1) DataFrame issues)
        y_train = np.asarray(y_train).reshape(-1)
        y_val = np.asarray(y_val).reshape(-1)
        y_test = np.asarray(y_test).reshape(-1)
    except ImportError:
        print("XGBoost not available. Skipping ablation.")
        return pd.DataFrame(), pd.DataFrame()
    
    ablation_results = []
    threshold_tables = []
    
    for name, data in feature_sets.items():
        print(f"\nTraining: {name}...")
        
        clf = XGBClassifier(
            random_state=seed,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            **xgb_params,
        )
        
        clf.fit(
            data['train'], y_train,
            eval_set=[(data['val'], y_val)],
            verbose=False,
        )
        
        # Predict probabilities
        val_proba = clf.predict_proba(data['val'])[:, 1]
        test_proba = clf.predict_proba(data['test'])[:, 1]
        
        # Threshold-free metrics
        val_auc = roc_auc_score(y_val, val_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        test_prauc = average_precision_score(y_test, test_proba)
        test_brier = brier_score_loss(y_test, test_proba)
        
        # Fixed-threshold evaluation
        test_preds_fixed = (test_proba >= fixed_threshold).astype(int)
        test_precision_fixed = precision_score(y_test, test_preds_fixed, zero_division=0)
        test_recall_fixed = recall_score(y_test, test_preds_fixed, zero_division=0)
        test_f1_fixed = f1_score(y_test, test_preds_fixed, zero_division=0)
        
        # Tuned-threshold evaluation
        tuned_threshold, threshold_table = tune_threshold_on_val(y_val, val_proba, tuning_rule)
        threshold_table['version'] = name
        threshold_tables.append(threshold_table)
        
        test_preds_tuned = (test_proba >= tuned_threshold).astype(int)
        test_precision_tuned = precision_score(y_test, test_preds_tuned, zero_division=0)
        test_recall_tuned = recall_score(y_test, test_preds_tuned, zero_division=0)
        test_f1_tuned = f1_score(y_test, test_preds_tuned, zero_division=0)
        
        result = {
            'version': name,
            'split': 'test',
            'n_features': data['train'].shape[1],
            
            # Threshold-free
            'roc_auc': test_auc,
            'pr_auc': test_prauc,
            'brier_score': test_brier,
            
            # Fixed threshold
            'threshold_fixed': fixed_threshold,
            'precision_fixed': test_precision_fixed,
            'recall_fixed': test_recall_fixed,
            'f1_fixed': test_f1_fixed,
            
            # Tuned threshold
            'threshold_tuned': tuned_threshold,
            'precision_tuned': test_precision_tuned,
            'recall_tuned': test_recall_tuned,
            'f1_tuned': test_f1_tuned,
        }
        ablation_results.append(result)
        
        print(f"  ROC-AUC: {test_auc:.4f}, PR-AUC: {test_prauc:.4f}")
        print(f"  Fixed (th={fixed_threshold:.2f}): F1={test_f1_fixed:.4f}, Recall={test_recall_fixed:.4f}")
        print(f"  Tuned (th={tuned_threshold:.2f}): F1={test_f1_tuned:.4f}, Recall={test_recall_tuned:.4f}")
    
    ablation_df = pd.DataFrame(ablation_results)
    threshold_table_val = pd.concat(threshold_tables, ignore_index=True)
    
    return ablation_df, threshold_table_val


# ============================================================================
# Semi-supervised Pseudo-labeling
# ============================================================================

def generate_pseudo_labels(
    teacher_clf,
    X_unlabeled: np.ndarray,
    threshold_high: float = 0.95,
    threshold_low: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate pseudo-labels for unlabeled data using teacher model.
    
    Returns:
        unlabeled_proba: Predicted probabilities
        pseudo_labels: Binary labels (0/1)
        high_conf_mask: Boolean mask for high-confidence samples
    """
    unlabeled_proba = teacher_clf.predict_proba(X_unlabeled)[:, 1]
    
    high_conf_pos = unlabeled_proba >= threshold_high
    high_conf_neg = unlabeled_proba <= threshold_low
    high_conf_mask = high_conf_pos | high_conf_neg
    
    pseudo_labels = (unlabeled_proba >= 0.5).astype(int)
    
    return unlabeled_proba, pseudo_labels, high_conf_mask


def train_student_with_pseudolabels(
    X_labeled: np.ndarray,
    y_labeled: pd.Series,
    X_unlabeled: np.ndarray,
    pseudo_labels: np.ndarray,
    high_conf_mask: np.ndarray,
    X_val: np.ndarray,
    y_val: pd.Series,
    xgb_params: Dict,
    pseudo_weight: float = 0.3,
    seed: int = 42,
):
    """Train student model with pseudo-labeled data."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost not available. Skipping student training.")
        return None
    
    # Combine labeled + high-confidence pseudo-labeled
    X_pseudo = X_unlabeled[high_conf_mask]
    y_pseudo = pseudo_labels[high_conf_mask]
    
    X_student_train = np.vstack([X_labeled, X_pseudo])
    y_student_train = np.concatenate([y_labeled.values, y_pseudo])
    
    # Sample weights
    w_labeled = np.ones(len(y_labeled))
    w_pseudo = np.ones(len(y_pseudo)) * pseudo_weight
    sample_weights = np.concatenate([w_labeled, w_pseudo])
    
    print(f"\nTraining Student with {len(y_pseudo)} pseudo-labeled samples (weight={pseudo_weight})")
    
    student_clf = XGBClassifier(
        random_state=seed,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        **xgb_params,
    )
    
    student_clf.fit(
        X_student_train, y_student_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    return student_clf


# ============================================================================
# Plotting
# ============================================================================

def plot_loss_curve(history: Dict, save_path: Path, model_name: str):
    """Plot training loss curve."""
    if not history.get('train_loss'):
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(history['train_loss'], label='Train Loss', color='blue')
    ax.plot(history['val_loss'], label='Val Loss', color='orange')
    
    if 'best_epoch' in history:
        ax.axvline(x=history['best_epoch'], color='red', linestyle='--',
                   label=f"Best Epoch ({history['best_epoch']})")
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{model_name} Training Loss Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne(
    Z_sample: np.ndarray,
    y_sample: pd.Series,
    save_path: Path,
    model_name: str,
    n_sample: int,
    seed: int = 42,
):
    """Plot t-SNE visualization of latent space."""
    print(f"\nRunning t-SNE ({n_sample} samples)...")
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed)
    Z_tsne = tsne.fit_transform(Z_sample)
    
    print("t-SNE completed.")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db' if y == 0 else '#e74c3c' for y in y_sample]
    ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=colors, alpha=0.5, s=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Non-Churn'),
        Patch(facecolor='#e74c3c', label='Churn'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'{model_name} Latent Space (n={n_sample})\nColored by Churn Label (post-hoc)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction_error(
    recon_error: np.ndarray,
    y_true: pd.Series,
    save_path: Path,
):
    """Plot reconstruction error distribution by label."""
    churn_mask = y_true.values == 1
    error_churn = recon_error[churn_mask]
    error_non_churn = recon_error[~churn_mask]
    
    print(f"\nReconstruction Error Statistics:")
    print(f"  Churn: mean={error_churn.mean():.6f}, std={error_churn.std():.6f}")
    print(f"  Non-Churn: mean={error_non_churn.mean():.6f}, std={error_non_churn.std():.6f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(error_non_churn, bins=50, alpha=0.6, label='Non-Churn', color='#3498db')
    axes[0].hist(error_churn, bins=50, alpha=0.6, label='Churn', color='#e74c3c')
    axes[0].axvline(error_non_churn.mean(), color='#3498db', linestyle='--', linewidth=2)
    axes[0].axvline(error_churn.mean(), color='#e74c3c', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Reconstruction Error (MSE)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Reconstruction Error Distribution')
    axes[0].legend()
    
    # Box plot
    box_data = [
        pd.DataFrame({'Error': error_non_churn, 'Label': 'Non-Churn'}),
        pd.DataFrame({'Error': error_churn, 'Label': 'Churn'}),
    ]
    box_df = pd.concat(box_data)
    if sns is not None:
        sns.boxplot(data=box_df, x='Label', y='Error', ax=axes[1], palette=['#3498db', '#e74c3c'])
    else:
        # Fallback: simple matplotlib boxplot
        grouped = [box_df[box_df['Label'] == lbl]['Error'] for lbl in ['Non-Churn', 'Churn']]
        axes[1].boxplot(grouped, labels=['Non-Churn', 'Churn'])
    axes[1].set_title('Reconstruction Error by Churn Status')
    axes[1].set_ylabel('Reconstruction Error (MSE)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(ablation_df: pd.DataFrame, save_path: Path):
    """Plot ablation study results (fixed threshold)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ablation_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ablation_df['roc_auc'], width, label='ROC-AUC', color='#3498db')
    bars2 = ax.bar(x + width/2, ablation_df['pr_auc'], width, label='PR-AUC', color='#e74c3c')
    
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Effect of Latent Features', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('(')[0].strip() for s in ablation_df['version']], rotation=15, ha='right')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_ylim(0, max(ablation_df['roc_auc'].max(), ablation_df['pr_auc'].max()) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
