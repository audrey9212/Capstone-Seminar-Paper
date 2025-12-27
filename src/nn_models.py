"""
PyTorch Neural Network Models Module
====================================

Dataset/


- ChurnDataset: PyTorch Dataset for churn prediction
- Model Architectures: Dense, Embedding, Wide & Deep
- Loss Functions: Weighted BCE, Focal Loss
- Training: train_epoch, validate_epoch, train_model (with early stopping)
- Evaluation: evaluate_model, tune_threshold_with_constraint
- Visualization: plot_training_curves, plot_roc_pr_with_point


    from nn_models import ChurnDataset, create_dense_model, train_model
    
    dataset = ChurnDataset(X, y, continuous_cols, categorical_cols, binary_cols)
    model = create_dense_model(dataset.get_input_dims(), hidden_dims=[128, 64])
    history, metrics = train_model(model, train_loader, val_loader, ...)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import time
import copy
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Dataset
# ============================================================================

class ChurnDataset(Dataset):
    """
    PyTorch Dataset for churn prediction with mixed feature types.
    
    Handles three types of features:
    - Continuous: Standardized numerical features
    - Categorical: Integer-encoded categorical features (for embedding)
    - Binary: Binary features (treated as continuous)
    
    Args:
        X: DataFrame with all features
        y: Target labels (1D array-like)
        continuous_cols: List of continuous feature names
        categorical_cols: List of categorical feature names
        binary_cols: List of binary feature names
        categorical_cardinalities: Dict mapping categorical col -> cardinality
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
        binary_cols: List[str],
        categorical_cardinalities: Optional[Dict[str, int]] = None,
    ):
        self.X = X.copy()
        self.y = torch.FloatTensor(y.astype(np.float32).reshape(-1))
        
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        
        # Prepare continuous features
        if continuous_cols:
            self.X_continuous = torch.FloatTensor(
                X[continuous_cols].values.astype(np.float32)
            )
        else:
            self.X_continuous = torch.FloatTensor(np.zeros((len(X), 0)))
        
        # Prepare categorical features
        if categorical_cols:
            self.X_categorical = torch.LongTensor(
                X[categorical_cols].values.astype(np.int64)
            )
            # Store cardinalities
            if categorical_cardinalities is None:
                self.categorical_cardinalities = {
                    col: int(X[col].max() + 1) for col in categorical_cols
                }
            else:
                self.categorical_cardinalities = categorical_cardinalities
        else:
            self.X_categorical = torch.LongTensor(np.zeros((len(X), 0), dtype=np.int64))
            self.categorical_cardinalities = {}
        
        # Prepare binary features
        if binary_cols:
            self.X_binary = torch.FloatTensor(
                X[binary_cols].values.astype(np.float32)
            )
        else:
            self.X_binary = torch.FloatTensor(np.zeros((len(X), 0)))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            "continuous": self.X_continuous[idx],
            "categorical": self.X_categorical[idx],
            "binary": self.X_binary[idx],
            "target": self.y[idx],
        }
    
    def get_input_dims(self) -> Dict[str, Any]:
        """Return input dimensions for model construction."""
        return {
            "n_continuous": len(self.continuous_cols),
            "n_categorical": len(self.categorical_cols),
            "n_binary": len(self.binary_cols),
            "categorical_cardinalities": self.categorical_cardinalities,
        }


# ============================================================================
# Model Architectures
# ============================================================================

class DenseModel(nn.Module):
    """
    Fully-connected dense neural network for tabular data.
    
    Concatenates all features and passes through dense layers.
    """
    
    def __init__(
        self,
        n_continuous: int,
        n_binary: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        input_dim = n_continuous + n_binary
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, batch):
        x = torch.cat([batch["continuous"], batch["binary"]], dim=1)
        return self.network(x).squeeze(-1)


class EmbeddingModel(nn.Module):
    """
    Neural network with embeddings for categorical features.
    
    Uses embedding layers for categorical features, then concatenates
    with continuous/binary features and passes through dense layers.
    """
    
    def __init__(
        self,
        n_continuous: int,
        n_binary: int,
        categorical_cardinalities: Dict[str, int],
        embedding_dim: int = 8,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinalities.values()
        ])
        
        # Calculate input dimension
        n_embedding_features = len(categorical_cardinalities) * embedding_dim
        input_dim = n_continuous + n_binary + n_embedding_features
        
        # Dense layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, batch):
        # Embed categorical features
        if batch["categorical"].shape[1] > 0:
            embeddings = [
                emb(batch["categorical"][:, i])
                for i, emb in enumerate(self.embeddings)
            ]
            cat_embedded = torch.cat(embeddings, dim=1)
        else:
            bs = batch["continuous"].shape[0]
            cat_embedded = batch["continuous"].new_zeros((bs, 0))
        
        # Concatenate all features
        x = torch.cat([batch["continuous"], batch["binary"], cat_embedded], dim=1)
        
        return self.network(x).squeeze(-1)


class WideAndDeepModel(nn.Module):
    """
    Wide & Deep model architecture.
    
    - Wide: Linear combination of all features
    - Deep: Embedding + dense network
    - Output: Weighted sum of wide and deep components
    """
    
    def __init__(
        self,
        n_continuous: int,
        n_binary: int,
        categorical_cardinalities: Dict[str, int],
        embedding_dim: int = 8,
        deep_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
        alpha: float = 0.5,  # Weight for deep component (1-alpha for wide)
    ):
        super().__init__()
        
        self.alpha = alpha
        
        # Wide component (linear)
        n_categorical = len(categorical_cardinalities)
        wide_input_dim = n_continuous + n_binary + n_categorical
        self.wide = nn.Linear(wide_input_dim, 1)
        
        # Deep component (embeddings + MLP)
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinalities.values()
        ])
        
        n_embedding_features = len(categorical_cardinalities) * embedding_dim
        deep_input_dim = n_continuous + n_binary + n_embedding_features
        
        deep_layers = []
        prev_dim = deep_input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_layers)
    
    def forward(self, batch):
        # Wide component: use categorical as-is (integer values)
        wide_input = torch.cat([
            batch["continuous"],
            batch["binary"],
            batch["categorical"].float()
        ], dim=1)
        wide_out = self.wide(wide_input)
        
        # Deep component: embed categorical
        if batch["categorical"].shape[1] > 0:
            embeddings = [
                emb(batch["categorical"][:, i])
                for i, emb in enumerate(self.embeddings)
            ]
            cat_embedded = torch.cat(embeddings, dim=1)
        else:
            bs = batch["continuous"].shape[0]
            cat_embedded = batch["continuous"].new_zeros((bs, 0))
        
        deep_input = torch.cat([batch["continuous"], batch["binary"], cat_embedded], dim=1)
        deep_out = self.deep(deep_input)
        
        # Combined output
        output = (1 - self.alpha) * wide_out + self.alpha * deep_out
        
        return output.squeeze(-1)


# ============================================================================
# Model Creation Helpers
# ============================================================================

def create_dense_model(input_dims: Dict, hidden_dims: List[int] = [128, 64], dropout: float = 0.3):
    """Create a dense baseline model."""
    return DenseModel(
        n_continuous=input_dims["n_continuous"],
        n_binary=input_dims["n_binary"],
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


def create_embedding_model(
    input_dims: Dict,
    embedding_dim: int = 8,
    hidden_dims: List[int] = [128, 64],
    dropout: float = 0.3,
):
    """Create an embedding-based model."""
    return EmbeddingModel(
        n_continuous=input_dims["n_continuous"],
        n_binary=input_dims["n_binary"],
        categorical_cardinalities=input_dims["categorical_cardinalities"],
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


def create_wide_deep_model(
    input_dims: Dict,
    embedding_dim: int = 8,
    deep_hidden_dims: List[int] = [128, 64],
    dropout: float = 0.3,
    alpha: float = 0.5,
):
    """Create a Wide & Deep model."""
    return WideAndDeepModel(
        n_continuous=input_dims["n_continuous"],
        n_binary=input_dims["n_binary"],
        categorical_cardinalities=input_dims["categorical_cardinalities"],
        embedding_dim=embedding_dim,
        deep_hidden_dims=deep_hidden_dims,
        dropout=dropout,
        alpha=alpha,
    )


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


def create_loss_function(loss_type: str, pos_weight: Optional[float] = None, alpha: float = 0.25, gamma: float = 2.0):
    """
    Create loss function based on type.
    
    Args:
        loss_type: "bce_weighted" or "focal"
        pos_weight: Weight for positive class (for bce_weighted)
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
    """
    if loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "bce_weighted":
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in loader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch)
        loss = criterion(logits, batch["target"])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * len(batch["target"])
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(batch["target"].cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_targets, all_preds)
    
    return avg_loss, auc


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch)
            loss = criterion(logits, batch["target"])
            
            total_loss += loss.item() * len(batch["target"])
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(batch["target"].cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_targets, all_preds)
    
    return avg_loss, auc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    device,
    patience: int = 15,
    verbose: bool = True,
):
    """
    Train model with early stopping.
    
    Returns:
        history: Dict with loss/auc/lr per epoch
        best_model_state: State dict of best model
        best_epoch: Epoch number of best model
    """
    history = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
        "lr": [],
    }
    
    best_val_auc = 0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_auc)  # ReduceLROnPlateau mode
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)
        
        # Check improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            if verbose:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
                      f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} | "
                      f"LR: {current_lr:.2e} | [OK] NEW BEST")
        else:
            patience_counter += 1
            
            if verbose:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
                      f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} | "
                      f"LR: {current_lr:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\\nEarly stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - start_time
    
    if verbose:
        print(f"\\nTraining completed in {train_time:.1f}s")
        print(f"Best epoch: {best_epoch+1} | Best val AUC: {best_val_auc:.4f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return history, best_model_state, best_epoch, train_time


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, loader, device):
    """
    Evaluate model and return predictions + metrics.
    
    Returns:
        y_true: True labels
        y_proba: Predicted probabilities
        metrics: Dict of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(batch["target"].cpu().numpy())
    
    y_true = np.array(all_targets)
    y_proba = np.array(all_preds)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    
    metrics = {
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }
    
    return y_true, y_proba, metrics


def tune_threshold_with_constraint(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.70,
    threshold_range: Optional[np.ndarray] = None,
) -> Tuple[float, Dict]:
    """
    Find optimal threshold with recall constraint.
    
    Strategy: Maximize F1 subject to recall >= min_recall.
    
    Returns:
        optimal_threshold: Selected threshold
        threshold_metrics: Dict with precision/recall/f1 at optimal threshold
    """
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.6, 0.01)
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}
    
    for threshold in threshold_range:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        
        if recall >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    
    return best_threshold, best_metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_curves(history: Dict, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot training curves (loss and AUC)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # AUC
    axes[1].plot(epochs, history["train_auc"], "o-", label="Train AUC", linewidth=2)
    axes[1].plot(epochs, history["val_auc"], "s-", label="Val AUC", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("AUC", fontsize=11)
    axes[1].set_title("Training AUC", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_roc_pr_with_point(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ROC and PR curves with operating point marked."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    # Point at threshold
    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    # FPR/TPR at threshold
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_point = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    axes[0].plot(fpr_point, tpr_point, "ro", markersize=10, 
                label=f"Threshold={threshold:.3f}")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("ROC Curve - Test Set", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()
    
    axes[1].plot(recall_curve, precision_curve, linewidth=2, 
                label=f"{model_name} (AP={pr_auc:.4f})")
    axes[1].axhline(baseline, color="k", linestyle="--", linewidth=1,
                   label=f"Baseline ({baseline:.3f})")
    axes[1].plot(recall, precision, "ro", markersize=10,
                label=f"Threshold={threshold:.3f}")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curve - Test Set", fontsize=12, fontweight="bold")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Strict deterministic (may raise on some ops; best-effort)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
