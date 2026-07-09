# -*- coding: utf-8 -*-

"""
BggDeepLearning PyTorch Dataset and DataLoader for tabular clinical data

File:
python/bggdeep/models/deep_learning/tabular_dataset.py

Purpose:
1. Wrap processed tabular data as PyTorch Dataset
2. Provide DataLoader factory for train/val/test splits
3. Support imbalanced class weighting for Binary Cross Entropy loss
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class ClinicalTabularDataset(Dataset):
    """
    PyTorch Dataset for clinical tabular data.

    Each sample is (feature_tensor, label_tensor).

    Parameters
    ----------
    features:
        DataFrame or numpy array of shape (n_samples, n_features).
        Already preprocessed (imputed, scaled, one-hot encoded).

    labels:
        Series or numpy array of shape (n_samples,) with binary labels (0 or 1).
    """

    def __init__(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: pd.Series | pd.DataFrame | np.ndarray,
    ) -> None:
        # Convert to float32 numpy arrays for PyTorch
        if isinstance(features, pd.DataFrame):
            self.x = features.to_numpy(dtype=np.float32)
        else:
            self.x = np.asarray(features, dtype=np.float32)

        if isinstance(labels, (pd.Series, pd.DataFrame)):
            self.y = labels.to_numpy(dtype=np.float32).ravel()
        else:
            self.y = np.asarray(labels, dtype=np.float32).ravel()

        if len(self.x) != len(self.y):
            raise ValueError(
                f"Features and labels must have same length. "
                f"Got {len(self.x)} vs {len(self.y)}."
            )

        self.n_features = self.x.shape[1]
        self.n_samples = len(self.x)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)
        return x, y

    @property
    def positive_rate(self) -> float:
        """
        Return the proportion of positive samples.
        """
        return float(self.y.mean())

    @property
    def class_counts(self) -> Dict[str, int]:
        """
        Return counts of negative (0) and positive (1) samples.
        """
        neg = int((self.y == 0).sum())
        pos = int((self.y == 1).sum())
        return {"negative": neg, "positive": pos}

    def compute_pos_weight(self) -> torch.Tensor:
        """
        Compute positive class weight for BCELoss / BCEWithLogitsLoss.

        pos_weight = n_negative / n_positive

        This is the standard weight for imbalanced binary classification.
        """
        counts = self.class_counts
        if counts["positive"] == 0:
            return torch.tensor(1.0)

        weight = counts["negative"] / counts["positive"]
        return torch.tensor(weight, dtype=torch.float32)

    def summary(self) -> str:
        """
        Return a human-readable summary string.
        """
        counts = self.class_counts
        lines = [
            f"ClinicalTabularDataset summary:",
            f"  Samples: {self.n_samples}",
            f"  Features: {self.n_features}",
            f"  Negative (0): {counts['negative']}",
            f"  Positive (1): {counts['positive']}",
            f"  Positive rate: {self.positive_rate:.4f}",
        ]
        return "\n".join(lines)


def create_dataloaders(
    train_dataset: ClinicalTabularDataset,
    val_dataset: ClinicalTabularDataset,
    test_dataset: ClinicalTabularDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Parameters
    ----------
    train_dataset:
        Training dataset.

    val_dataset:
        Validation dataset.

    test_dataset:
        Test dataset.

    batch_size:
        Batch size for all loaders.

    num_workers:
        Number of worker processes. Use 0 on Windows to avoid issues.

    use_weighted_sampler:
        Whether to use WeightedRandomSampler for the training set
        to handle class imbalance.

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary with keys "train", "val", "test".
    """
    loaders: Dict[str, DataLoader] = {}

    # Training loader with optional weighted sampling
    if use_weighted_sampler:
        # Compute sample weights: positive samples get higher weight
        counts = train_dataset.class_counts
        class_weights = {
            0: 1.0 / counts["negative"] if counts["negative"] > 0 else 0,
            1: 1.0 / counts["positive"] if counts["positive"] > 0 else 0,
        }
        sample_weights = [
            class_weights[int(label)] for label in train_dataset.y
        ]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(train_dataset),
            replacement=True,
        )

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )

    # Validation and test loaders — no shuffling or weighted sampling
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    loaders["test"] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return loaders


def load_datasets_from_processed(
    processed_data_dir: Path,
) -> Tuple[ClinicalTabularDataset, ClinicalTabularDataset, ClinicalTabularDataset]:
    """
    Load train/val/test datasets from the preprocessed data directory.

    This is the standard entry point used by training scripts.

    Parameters
    ----------
    processed_data_dir:
        Path to data/processed/ containing tabular_*_features.csv and
        tabular_*_labels.csv.

    Returns
    -------
    Tuple of (train_dataset, val_dataset, test_dataset).
    """
    required_files = [
        "tabular_train_features.csv",
        "tabular_train_labels.csv",
        "tabular_val_features.csv",
        "tabular_val_labels.csv",
        "tabular_test_features.csv",
        "tabular_test_labels.csv",
    ]

    for filename in required_files:
        file_path = processed_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required processed data file was not found: {file_path}\n"
                f"Please run: python scripts\\python\\run_tabular_preprocessing.py"
            )

    x_train = pd.read_csv(processed_data_dir / "tabular_train_features.csv")
    y_train = pd.read_csv(processed_data_dir / "tabular_train_labels.csv")
    x_val = pd.read_csv(processed_data_dir / "tabular_val_features.csv")
    y_val = pd.read_csv(processed_data_dir / "tabular_val_labels.csv")
    x_test = pd.read_csv(processed_data_dir / "tabular_test_features.csv")
    y_test = pd.read_csv(processed_data_dir / "tabular_test_labels.csv")

    # Ensure same feature order
    common_columns = list(x_train.columns)
    x_val = x_val[common_columns]
    x_test = x_test[common_columns]

    train_dataset = ClinicalTabularDataset(x_train, y_train)
    val_dataset = ClinicalTabularDataset(x_val, y_val)
    test_dataset = ClinicalTabularDataset(x_test, y_test)

    return train_dataset, val_dataset, test_dataset


# Quick self-test
if __name__ == "__main__":
    import sys

    # Create dummy data
    np.random.seed(42)
    x_dummy = np.random.randn(100, 26).astype(np.float32)
    y_dummy = (np.random.rand(100) < 0.1).astype(np.float32)

    dataset = ClinicalTabularDataset(x_dummy, y_dummy)
    print(dataset.summary())
    print(f"pos_weight: {dataset.compute_pos_weight().item():.4f}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"\nBatch x shape: {batch_x.shape}")
    print(f"Batch y shape: {batch_y.shape}")

    print("\nTabularDataset module self-test: PASSED")
