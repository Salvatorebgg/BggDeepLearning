"""
BggDeepLearning Gradient Boosting baseline plotting script

File:
scripts/python/plot_gradient_boosting_baseline.py

Usage:
python scripts\\python\\plot_gradient_boosting_baseline.py
"""

from __future__ import annotations

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


def build_gradient_boosting_plot_report(
    gb_plot_summary: pd.DataFrame,
    gb_top_features: pd.DataFrame,
    roc_comparison_summary: pd.DataFrame,
    generated_files: dict[str, Path],
) -> str:
    """
    Build plain text report for Gradient Boosting plots and model comparison ROC.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Gradient Boosting Plot Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Gradient Boosting Plot Summary")
    lines.append("-" * 70)
    lines.append(gb_plot_summary.to_string(index=False))
    lines.append("")
    lines.append("2. Top 20 Gradient Boosting Feature Importances")
    lines.append("-" * 70)
    lines.append(gb_top_features.to_string(index=False))
    lines.append("")
    lines.append("3. Three-Model ROC Comparison")
    lines.append("-" * 70)
    lines.append(roc_comparison_summary.to_string(index=False))
    lines.append("")
    lines.append("4. Generated Files")
    lines.append("-" * 70)

    for name, path in generated_files.items():
        lines.append(f"{name}: {path}")

    lines.append("")
    lines.append("5. Notes")
    lines.append("-" * 70)
    lines.append("Gradient Boosting is a sequential tree ensemble model.")
    lines.append("Feature importance is model-specific and not causal evidence.")
    lines.append("Validation ROC comparison should be used for model selection.")
    lines.append("Test ROC comparison should be treated as held-out performance evaluation.")
    lines.append("Calibration and DCA will be added in later steps.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.classification_plots import (
        plot_binary_classification_evaluation,
        plot_feature_importance,
        plot_multi_model_roc_comparison,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Gradient Boosting plotting started")

    gb_val_prediction_file = (
        project_paths.prediction_dir / "gradient_boosting_val_predictions.csv"
    )
    gb_test_prediction_file = (
        project_paths.prediction_dir / "gradient_boosting_test_predictions.csv"
    )
    gb_feature_importance_file = (
        project_paths.table_dir / "gradient_boosting_feature_importance.csv"
    )

    gb_plot_results = []

    gb_plot_results.append(
        plot_binary_classification_evaluation(
            prediction_file=gb_val_prediction_file,
            figure_dir=project_paths.figure_dir,
            model_prefix="gradient_boosting",
            split_name="validation",
        )
    )

    gb_plot_results.append(
        plot_binary_classification_evaluation(
            prediction_file=gb_test_prediction_file,
            figure_dir=project_paths.figure_dir,
            model_prefix="gradient_boosting",
            split_name="test",
        )
    )

    gb_feature_importance_figure = (
        project_paths.figure_dir / "gradient_boosting_feature_importance_top20.png"
    )

    gb_top_features = plot_feature_importance(
        feature_importance_file=gb_feature_importance_file,
        output_path=gb_feature_importance_figure,
        title="Gradient Boosting Top 20 Feature Importances",
        top_n=20,
    )

    gb_plot_summary = pd.DataFrame(gb_plot_results)

    gb_plot_summary_file = (
        project_paths.table_dir / "gradient_boosting_plot_summary.csv"
    )
    gb_plot_summary.to_csv(
        gb_plot_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    gb_top_feature_table_file = (
        project_paths.table_dir / "gradient_boosting_top20_feature_importance.csv"
    )
    gb_top_features.to_csv(
        gb_top_feature_table_file,
        index=False,
        encoding="utf-8-sig",
    )

    validation_prediction_files = {
        "Logistic Regression": project_paths.prediction_dir / "logistic_val_predictions.csv",
        "Random Forest": project_paths.prediction_dir / "random_forest_val_predictions.csv",
        "Gradient Boosting": project_paths.prediction_dir / "gradient_boosting_val_predictions.csv",
    }

    test_prediction_files = {
        "Logistic Regression": project_paths.prediction_dir / "logistic_test_predictions.csv",
        "Random Forest": project_paths.prediction_dir / "random_forest_test_predictions.csv",
        "Gradient Boosting": project_paths.prediction_dir / "gradient_boosting_test_predictions.csv",
    }

    validation_roc_comparison_figure = (
        project_paths.figure_dir / "model_comparison_validation_roc_curve.png"
    )

    test_roc_comparison_figure = (
        project_paths.figure_dir / "model_comparison_test_roc_curve.png"
    )

    validation_roc_summary = plot_multi_model_roc_comparison(
        prediction_files=validation_prediction_files,
        output_path=validation_roc_comparison_figure,
        title="Validation ROC Comparison",
        split_name="validation",
    )

    test_roc_summary = plot_multi_model_roc_comparison(
        prediction_files=test_prediction_files,
        output_path=test_roc_comparison_figure,
        title="Test ROC Comparison",
        split_name="test",
    )

    roc_comparison_summary = pd.concat(
        [
            validation_roc_summary,
            test_roc_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    roc_comparison_summary_file = (
        project_paths.table_dir / "model_comparison_roc_summary.csv"
    )
    roc_comparison_summary.to_csv(
        roc_comparison_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    generated_files = {
        "gradient_boosting_validation_roc": project_paths.figure_dir / "gradient_boosting_validation_roc_curve.png",
        "gradient_boosting_validation_pr": project_paths.figure_dir / "gradient_boosting_validation_pr_curve.png",
        "gradient_boosting_validation_confusion": project_paths.figure_dir / "gradient_boosting_validation_confusion_matrix.png",
        "gradient_boosting_test_roc": project_paths.figure_dir / "gradient_boosting_test_roc_curve.png",
        "gradient_boosting_test_pr": project_paths.figure_dir / "gradient_boosting_test_pr_curve.png",
        "gradient_boosting_test_confusion": project_paths.figure_dir / "gradient_boosting_test_confusion_matrix.png",
        "gradient_boosting_feature_importance": gb_feature_importance_figure,
        "validation_roc_comparison": validation_roc_comparison_figure,
        "test_roc_comparison": test_roc_comparison_figure,
        "gradient_boosting_plot_summary": gb_plot_summary_file,
        "gradient_boosting_top20_feature_table": gb_top_feature_table_file,
        "model_comparison_roc_summary": roc_comparison_summary_file,
    }

    report_text = build_gradient_boosting_plot_report(
        gb_plot_summary=gb_plot_summary,
        gb_top_features=gb_top_features,
        roc_comparison_summary=roc_comparison_summary,
        generated_files=generated_files,
    )

    report_file = project_paths.report_dir / "gradient_boosting_plot_report.txt"
    report_file.write_text(report_text, encoding="utf-8")

    logger.info("Gradient Boosting plotting finished")
    logger.info("Gradient Boosting plot report: %s", report_file)
    logger.info("ROC comparison summary: %s", roc_comparison_summary_file)

    print("=" * 70)
    print("BggDeepLearning Gradient Boosting plotting succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print("-" * 70)
    print("Gradient Boosting plot summary:")
    print(
        gb_plot_summary[
            ["split", "roc_auc", "average_precision", "tn", "fp", "fn", "tp"]
        ].to_string(index=False)
    )
    print("-" * 70)
    print("Top 20 Gradient Boosting feature importances:")
    print(gb_top_features.to_string(index=False))
    print("-" * 70)
    print("Three-model ROC comparison:")
    print(roc_comparison_summary.to_string(index=False))
    print("-" * 70)
    print("Generated files:")
    for name, path in generated_files.items():
        print(f"{name}: {path}")
    print(f"report_file: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()