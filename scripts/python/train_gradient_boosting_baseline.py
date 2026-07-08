"""
BggDeepLearning Gradient Boosting baseline training script

File:
scripts/python/train_gradient_boosting_baseline.py

Usage:
python scripts\\python\\train_gradient_boosting_baseline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_project_root() -> Path:
    """
    Find project root by locating configs/app.yaml.
    """
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / "configs" / "app.yaml").exists():
            return parent

    raise FileNotFoundError("Project root was not found. configs/app.yaml is missing.")


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
        description="Train Gradient Boosting baseline for BggDeepLearning."
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting stages. Default: 200",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate. Default: 0.05",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth of individual regression estimators. Default: 3",
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples per leaf. Default: 2",
    )

    parser.add_argument(
        "--subsample",
        type=float,
        default=0.9,
        help="Subsample ratio. Default: 0.9",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold. Default: 0.5",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--no-balanced-sample-weight",
        action="store_true",
        help="Disable balanced sample weight.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.models.traditional.gradient_boosting import (
        GradientBoostingBaselineConfig,
        GradientBoostingBaselineTrainer,
        build_confusion_matrix_table,
        build_three_model_comparison_table,
        load_processed_tabular_data,
        save_gradient_boosting_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Gradient Boosting baseline training started")

    data = load_processed_tabular_data(project_paths.processed_data_dir)

    x_train = data["x_train"]
    x_val = data["x_val"]
    x_test = data["x_test"]

    y_train = data["y_train"]["poor_outcome"]
    y_val = data["y_val"]["poor_outcome"]
    y_test = data["y_test"]["poor_outcome"]

    model_config = GradientBoostingBaselineConfig(
        random_state=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        threshold=args.threshold,
        use_balanced_sample_weight=not args.no_balanced_sample_weight,
    )

    trainer = GradientBoostingBaselineTrainer(model_config)

    logger.info("Training rows: %s", len(x_train))
    logger.info("Validation rows: %s", len(x_val))
    logger.info("Test rows: %s", len(x_test))
    logger.info("Number of features: %s", x_train.shape[1])
    logger.info("Number of estimators: %s", model_config.n_estimators)
    logger.info("Learning rate: %s", model_config.learning_rate)
    logger.info("Max depth: %s", model_config.max_depth)
    logger.info("Balanced sample weight: %s", model_config.use_balanced_sample_weight)

    trainer.train(x_train, y_train)

    train_metrics, train_predictions = trainer.evaluate_split(
        x_train,
        y_train,
        split_name="train",
    )

    val_metrics, val_predictions = trainer.evaluate_split(
        x_val,
        y_val,
        split_name="validation",
    )

    test_metrics, test_predictions = trainer.evaluate_split(
        x_test,
        y_test,
        split_name="test",
    )

    metrics_df = pd.DataFrame(
        [
            train_metrics,
            val_metrics,
            test_metrics,
        ]
    )

    confusion_df = build_confusion_matrix_table(metrics_df)

    feature_importance_df = trainer.build_feature_importance_table(
        feature_names=x_train.columns.tolist()
    )

    model_comparison_df = build_three_model_comparison_table(
        gradient_boosting_metrics=metrics_df,
        table_dir=project_paths.table_dir,
    )

    output_paths = save_gradient_boosting_outputs(
        trainer=trainer,
        metrics_df=metrics_df,
        confusion_df=confusion_df,
        feature_importance_df=feature_importance_df,
        model_comparison_df=model_comparison_df,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        output_model_dir=project_paths.model_dir,
        output_table_dir=project_paths.table_dir,
        output_prediction_dir=project_paths.prediction_dir,
        output_report_dir=project_paths.report_dir,
    )

    logger.info("Gradient Boosting baseline training finished")
    logger.info("Metrics file: %s", output_paths["metrics_file"])
    logger.info("Model file: %s", output_paths["model_file"])
    logger.info("Model comparison file: %s", output_paths["model_comparison_file"])

    print("=" * 70)
    print("BggDeepLearning Gradient Boosting baseline training succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Train shape: {x_train.shape}")
    print(f"Validation shape: {x_val.shape}")
    print(f"Test shape: {x_test.shape}")
    print("-" * 70)
    print("Gradient Boosting metrics:")
    print(metrics_df.to_string(index=False))
    print("-" * 70)
    print("Top 20 feature importances:")
    print(feature_importance_df.head(20).to_string(index=False))
    print("-" * 70)
    print("Model comparison:")
    print(model_comparison_df.to_string(index=False))
    print("-" * 70)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()