# -*- coding: utf-8 -*-

"""
BggDeepLearning PyTorch MLP training script

File:
scripts/python/train_deep_mlp.py

Purpose:
1. Load preprocessed tabular data as PyTorch Datasets
2. Build and train a TabularMLP model with Early Stopping
3. Evaluate on train/val/test splits
4. Save model checkpoint, metrics, predictions, and history
5. Output results compatible with the existing model leaderboard

Usage:
python scripts\\python\\train_deep_mlp.py
python scripts\\python\\train_deep_mlp.py --hidden-dims 256 128 64 --dropout 0.4
python scripts\\python\\train_deep_mlp.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import torch


def find_project_root() -> Path:
    """
    Find project root by locating configs/app.yaml.
    """
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / "configs" / "app.yaml").exists():
            return parent

    raise FileNotFoundError(
        "Project root was not found. configs/app.yaml is missing."
    )


def setup_python_path(project_root: Path) -> None:
    """
    Add project python folder to sys.path.
    """
    python_dir = project_root / "python"

    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train PyTorch MLP for BggDeepLearning tabular clinical data."
    )

    # Model architecture
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64, 32],
        help="Hidden layer dimensions. Default: 128 64 32",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate. Default: 0.3",
    )

    parser.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Disable batch normalization.",
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "leaky_relu"],
        help="Activation function. Default: relu",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size. Default: 32",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate. Default: 0.001",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 weight decay. Default: 1e-5",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum training epochs. Default: 200",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Early stopping patience. Default: 30",
    )

    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Default: 1.0",
    )

    # Reproducibility and device
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training. Default: cpu",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_metrics_df(
    model_name: str,
    model_display_name: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> pd.DataFrame:
    """
    Build a unified metrics DataFrame compatible with the leaderboard.
    """
    rows = []

    for split_name, metrics in [
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics),
    ]:
        row = {
            "model_name": model_name,
            "model_display_name": model_display_name,
            "model_group": "deep_learning",
            "split": split_name,
            **metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(
    model_path: Path,
    metrics_df: pd.DataFrame,
    history: Dict[str, list],
    train_predictions: pd.DataFrame,
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    model_summary_text: str,
    output_model_dir: Path,
    output_table_dir: Path,
    output_prediction_dir: Path,
    output_report_dir: Path,
) -> Dict[str, Path]:
    """
    Save all outputs from deep learning training.
    """
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = output_model_dir / "deep_mlp_baseline.pt"

    # Metrics CSV
    metrics_file = output_table_dir / "deep_mlp_baseline_metrics.csv"

    # Training history CSV
    history_file = output_table_dir / "deep_mlp_training_history.csv"

    # Predictions CSV
    train_pred_file = output_prediction_dir / "deep_mlp_train_predictions.csv"
    val_pred_file = output_prediction_dir / "deep_mlp_val_predictions.csv"
    test_pred_file = output_prediction_dir / "deep_mlp_test_predictions.csv"

    # Report
    report_file = output_report_dir / "deep_mlp_baseline_report.txt"

    # Copy model file from trainer's path
    import shutil
    shutil.copy2(model_path, model_file)

    # Save DataFrames
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")

    pd.DataFrame(history).to_csv(
        history_file,
        index=False,
        encoding="utf-8-sig",
    )

    train_predictions.to_csv(train_pred_file, index=False, encoding="utf-8-sig")
    val_predictions.to_csv(val_pred_file, index=False, encoding="utf-8-sig")
    test_predictions.to_csv(test_pred_file, index=False, encoding="utf-8-sig")

    # Build report text
    report_text = build_report(
        metrics_df=metrics_df,
        history=history,
        model_summary_text=model_summary_text,
    )
    report_file.write_text(report_text, encoding="utf-8")

    return {
        "model_file": model_file,
        "metrics_file": metrics_file,
        "history_file": history_file,
        "train_pred_file": train_pred_file,
        "val_pred_file": val_pred_file,
        "test_pred_file": test_pred_file,
        "report_file": report_file,
    }


def build_report(
    metrics_df: pd.DataFrame,
    history: Dict[str, list],
    model_summary_text: str,
) -> str:
    """
    Build plain text report for deep MLP training.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Deep MLP Baseline Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Model Architecture")
    lines.append("-" * 70)
    lines.append(model_summary_text)
    lines.append("")
    lines.append("2. Classification Metrics")
    lines.append("-" * 70)
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("3. Training Summary")
    lines.append("-" * 70)

    if history and history.get("epoch"):
        epochs = history["epoch"]
        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])
        val_aucs = history.get("val_auc", [])

        best_val_epoch = val_losses.index(min(val_losses)) + 1 if val_losses else 0

        lines.append(f"Total epochs trained: {len(epochs)}")
        lines.append(f"Best epoch (val_loss): {best_val_epoch}")
        lines.append(f"Best train_loss: {min(train_losses):.6f}" if train_losses else "")
        lines.append(f"Best val_loss: {min(val_losses):.6f}" if val_losses else "")
        lines.append(f"Best val_auc: {max(val_aucs):.4f}" if val_aucs else "")
        lines.append(f"Final train_loss: {train_losses[-1]:.6f}" if train_losses else "")
        lines.append(f"Final val_loss: {val_losses[-1]:.6f}" if val_losses else "")
        lines.append(f"Final val_auc: {val_aucs[-1]:.4f}" if val_aucs else "")

    lines.append("")
    lines.append("4. Notes")
    lines.append("-" * 70)
    lines.append("This is a PyTorch MLP baseline for tabular clinical data.")
    lines.append("The model was trained with BCE loss and Adam optimizer.")
    lines.append("Early stopping was used to prevent overfitting.")
    lines.append("Class imbalance was handled via positive class weighting.")
    lines.append("Current model is trained on simulated data and is for ")
    lines.append("engineering demonstration only — NOT for real clinical use.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Project setup
    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.models.deep_learning.mlp import MLPConfig, TabularMLP
    from bggdeep.models.deep_learning.tabular_dataset import (
        ClinicalTabularDataset,
        create_dataloaders,
        load_datasets_from_processed,
    )
    from bggdeep.models.deep_learning.training import (
        TrainingConfig,
        TabularTrainer,
        compute_classification_metrics,
    )

    # Set random seed
    set_seed(args.seed)

    # Load config and set up paths
    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)
    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Deep MLP training started")

    # Load data
    train_dataset, val_dataset, test_dataset = load_datasets_from_processed(
        project_paths.processed_data_dir,
    )

    logger.info("Train dataset: %s", train_dataset.summary())
    logger.info("Val dataset: %s", val_dataset.summary())
    logger.info("Test dataset: %s", test_dataset.summary())

    input_dim = train_dataset.n_features
    pos_weight = train_dataset.compute_pos_weight()

    logger.info("Input dimension: %s", input_dim)
    logger.info("Positive class weight: %.2f", pos_weight.item())

    # Build model
    mlp_config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_batch_norm=not args.no_batch_norm,
        activation=args.activation,
        random_seed=args.seed,
    )

    model = TabularMLP(mlp_config)
    model_summary_text = model.summary()

    logger.info("Model created with %s parameters", model.get_model_size())

    # Build training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        random_seed=args.seed,
    )

    # Create DataLoaders
    loaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # 0 for Windows safety
        use_weighted_sampler=True,
    )

    # Train
    trainer = TabularTrainer(model=model, train_config=train_config)

    history = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        pos_weight=pos_weight,
        verbose=True,
    )

    logger.info("Training completed. Best epoch: %s", trainer.best_epoch)

    # Evaluate on all splits
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("-" * 60)

    all_metrics = {}

    for split_name, loader in [
        ("train", loaders["train"]),
        ("validation", loaders["val"]),
        ("test", loaders["test"]),
    ]:
        _, _, y_true, y_pred = trainer.evaluate_loader(loader)
        metrics = compute_classification_metrics(y_true, y_pred)
        all_metrics[split_name] = metrics

        print(f"\n{split_name.upper()}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    # Build metrics DataFrame
    metrics_df = build_metrics_df(
        model_name="deep_mlp_baseline",
        model_display_name="Deep MLP (PyTorch)",
        train_metrics=all_metrics["train"],
        val_metrics=all_metrics["validation"],
        test_metrics=all_metrics["test"],
    )

    # Generate predictions
    train_predictions = trainer.predict_dataframe(loaders["train"])
    val_predictions = trainer.predict_dataframe(loaders["val"])
    test_predictions = trainer.predict_dataframe(loaders["test"])

    # Save model checkpoint (temporary path)
    temp_model_path = project_paths.model_dir / "_temp_deep_mlp.pt"
    trainer.save_model(temp_model_path)

    # Save all outputs
    output_paths = save_outputs(
        model_path=temp_model_path,
        metrics_df=metrics_df,
        history=history,
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        model_summary_text=model_summary_text,
        output_model_dir=project_paths.model_dir,
        output_table_dir=project_paths.table_dir,
        output_prediction_dir=project_paths.prediction_dir,
        output_report_dir=project_paths.report_dir,
    )

    # Clean up temp file
    temp_model_path.unlink(missing_ok=True)

    logger.info("Deep MLP training finished")
    logger.info("Model file: %s", output_paths["model_file"])
    logger.info("Metrics file: %s", output_paths["metrics_file"])

    print("\n" + "=" * 70)
    print("BggDeepLearning Deep MLP training succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Input dim: {input_dim}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Parameters: {model.get_model_size():,}")
    print(f"Best epoch: {trainer.best_epoch}")
    print("-" * 70)
    print("Metrics summary:")
    print(metrics_df.to_string(index=False))
    print("-" * 70)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
