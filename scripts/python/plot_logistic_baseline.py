"""
BggDeepLearning logistic baseline plotting script

File:
scripts/python/plot_logistic_baseline.py

Usage:
python scripts\\python\\plot_logistic_baseline.py
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


def main() -> None:
    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.classification_plots import (
        build_plot_report,
        plot_binary_classification_evaluation,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Logistic baseline plotting started")

    val_prediction_file = project_paths.prediction_dir / "logistic_val_predictions.csv"
    test_prediction_file = project_paths.prediction_dir / "logistic_test_predictions.csv"

    plot_results = []

    plot_results.append(
        plot_binary_classification_evaluation(
            prediction_file=val_prediction_file,
            figure_dir=project_paths.figure_dir,
            model_prefix="logistic",
            split_name="validation",
        )
    )

    plot_results.append(
        plot_binary_classification_evaluation(
            prediction_file=test_prediction_file,
            figure_dir=project_paths.figure_dir,
            model_prefix="logistic",
            split_name="test",
        )
    )

    plot_summary = pd.DataFrame(plot_results)
    plot_summary_file = project_paths.table_dir / "logistic_plot_summary.csv"
    plot_summary.to_csv(plot_summary_file, index=False, encoding="utf-8-sig")

    report_text = build_plot_report(plot_results)
    report_file = project_paths.report_dir / "logistic_baseline_plot_report.txt"
    report_file.write_text(report_text, encoding="utf-8")

    logger.info("Logistic baseline plotting finished")
    logger.info("Plot summary file: %s", plot_summary_file)
    logger.info("Plot report file: %s", report_file)

    print("=" * 70)
    print("BggDeepLearning logistic baseline plotting succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print("-" * 70)
    print("Plot summary:")
    print(plot_summary[["split", "roc_auc", "average_precision", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print("-" * 70)
    print("Generated figure files:")

    for result in plot_results:
        print(result["roc_file"])
        print(result["pr_file"])
        print(result["confusion_file"])

    print("-" * 70)
    print(f"Plot summary table: {plot_summary_file}")
    print(f"Plot report: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()