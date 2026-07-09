# -*- coding: utf-8 -*-

"""
BggDeepLearning PyTorch training utilities for tabular deep learning

File:
python/bggdeep/models/deep_learning/training.py

Purpose:
1. Training loop with progress logging
2. Validation loop
3. Early stopping based on validation loss
4. Model checkpointing (save best model)
5. Full train/evaluate pipeline for clinical risk prediction

Clinical task:
poor_outcome prediction from structured clinical variables
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from bggdeep.models.deep_learning.mlp import TabularMLP
from bggdeep.models.deep_learning.tabular_dataset import ClinicalTabularDataset


@dataclass
class TrainingConfig:
    """
    Configuration for PyTorch training.

    Parameters
    ----------
    batch_size:
        Batch size for DataLoader.

    learning_rate:
        Initial learning rate for Adam optimizer.

    weight_decay:
        L2 regularization coefficient.

    max_epochs:
        Maximum number of training epochs.

    early_stopping_patience:
        Number of epochs with no validation loss improvement before stopping.

    early_stopping_min_delta:
        Minimum absolute change in validation loss to count as improvement.

    grad_clip_norm:
        Maximum gradient norm for gradient clipping. None disables clipping.

    device:
        Device string. "cpu" or "cuda". Defaults to "cpu" for maximum compatibility.

    random_seed:
        Random seed for reproducibility.
    """

    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 200
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cpu"
    random_seed: int = 42


class EarlyStopping:
    """
    Early stopping handler.

    Monitors validation loss and stops training when no improvement
    is observed for `patience` consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check whether to stop training.

        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False


class TabularTrainer:
    """
    Trainer for PyTorch tabular MLP models.

    Handles:
    - Training loop with BCE loss and pos_weight for class imbalance
    - Validation loop
    - Early stopping
    - Best model checkpointing
    - Epoch-level metric logging
    """

    def __init__(
        self,
        model: TabularMLP,
        train_config: TrainingConfig,
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.device = torch.device(train_config.device)

        self.model.to(self.device)

        self.optimizer: optim.Optimizer | None = None
        self.criterion: nn.Module | None = None
        self.early_stopping: EarlyStopping | None = None

        # Training history
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

        self.best_model_state: Dict | None = None
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")

    def _setup_training(self, pos_weight: torch.Tensor) -> None:
        """
        Set up optimizer, loss, and early stopping before training.
        """
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )

        # BCEWithLogitsLoss is more numerically stable than BCELoss + sigmoid.
        # However, our model already applies sigmoid in forward(),
        # so we use BCELoss here.
        self.criterion = nn.BCELoss()

        self.pos_weight = pos_weight.to(self.device)

        self.early_stopping = EarlyStopping(
            patience=self.train_config.early_stopping_patience,
            min_delta=self.train_config.early_stopping_min_delta,
        )

        # Reset history
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

        self.best_model_state = None
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def _compute_auc(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute ROC AUC score from numpy arrays.
        """
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(y_true)) < 2:
                return 0.5  # Cannot compute AUC with single class

            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return 0.5

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train one epoch. Return average training loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(batch_x).squeeze()
            loss = self.criterion(predictions, batch_y)

            loss.backward()

            # Gradient clipping
            if self.train_config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a DataLoader.

        Returns
        -------
        Tuple of (avg_loss, auc, all_labels, all_predictions).
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        all_labels: List[np.ndarray] = []
        all_predictions: List[np.ndarray] = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x).squeeze()

            if self.criterion is not None:
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()

            n_batches += 1

            all_labels.append(batch_y.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_predictions)

        auc = self._compute_auc(y_true, y_pred)

        return avg_loss, auc, y_true, y_pred

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pos_weight: torch.Tensor,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run full training loop with early stopping.

        Parameters
        ----------
        train_loader:
            Training DataLoader.

        val_loader:
            Validation DataLoader.

        pos_weight:
            Positive class weight tensor for the loss function.

        verbose:
            Whether to print epoch-level progress.

        Returns
        -------
        Dict[str, List[float]]
            Training history dictionary.
        """
        self._setup_training(pos_weight)

        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples: {len(val_loader.dataset)}")
            print(f"Max epochs: {self.train_config.max_epochs}")
            print(f"Early stopping patience: {self.train_config.early_stopping_patience}")
            print(f"Learning rate: {self.train_config.learning_rate}")
            print(f"Batch size: {self.train_config.batch_size}")
            print(f"Positive class weight: {pos_weight.item():.2f}")
            print("-" * 60)

        for epoch in range(1, self.train_config.max_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_auc, _, _ = self.evaluate_loader(val_loader)

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            if verbose:
                marker = ""
                if epoch == self.best_epoch:
                    marker = " *"
                print(
                    f"Epoch {epoch:4d}/{self.train_config.max_epochs} | "
                    f"train_loss: {train_loss:.6f} | "
                    f"val_loss: {val_loss:.6f} | "
                    f"val_auc: {val_auc:.4f}{marker}"
                )

            # Early stopping
            if self.early_stopping(val_loss, epoch):
                if verbose:
                    print(
                        f"\nEarly stopping triggered at epoch {epoch}. "
                        f"Best epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.6f})"
                    )
                break

        # Restore best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"Restored best model from epoch {self.best_epoch}")

        if verbose:
            print("=" * 60)

        return self.history

    def predict_dataframe(
        self,
        loader: DataLoader,
    ) -> pd.DataFrame:
        """
        Generate predictions as a DataFrame.

        Returns DataFrame with columns:
        - predicted_probability
        - predicted_label (threshold 0.5)
        """
        _, _, y_true, y_pred = self.evaluate_loader(loader)

        return pd.DataFrame({
            "true_label": y_true.astype(int),
            "predicted_probability": y_pred,
            "predicted_label": (y_pred >= 0.5).astype(int),
        })

    def save_model(self, file_path: Path) -> None:
        """
        Save model state dict and config to file.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.best_model_state
            if self.best_model_state is not None
            else self.model.state_dict(),
            "model_config": self.model.config,
            "train_config": self.train_config,
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, file_path)

    @staticmethod
    def load_model(file_path: Path, device: str = "cpu") -> TabularMLP:
        """
        Load model from checkpoint file.
        """
        checkpoint = torch.load(
            file_path,
            map_location=torch.device(device),
            weights_only=False,
        )

        model = TabularMLP(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(torch.device(device))
        model.eval()

        return model


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true:
        True binary labels.

    y_pred_prob:
        Predicted probabilities.

    threshold:
        Classification threshold.

    Returns
    -------
    Dict with auc, accuracy, sensitivity, specificity, precision, recall, f1.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred_label = (y_pred_prob >= threshold).astype(int)

    # AUC
    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, y_pred_prob))

    accuracy = float(accuracy_score(y_true, y_pred_label))
    precision = float(precision_score(y_true, y_pred_label, zero_division=0))
    recall = float(recall_score(y_true, y_pred_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_label, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred_label,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
    }
