"""
BggDeepLearning model calibration evaluation script

File:
scripts/python/evaluate_model_calibration.py

Usage:
python scripts\\python\\evaluate_model_calibration.py
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
        description="Evaluate model calibration for BggDeepLearning."
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of calibration bins. Default: 10",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="uniform",
        choices=["uniform", "quantile"],
        help="Calibration binning strategy. Default: uniform",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.calibration import (
        CalibrationConfig,
        CalibrationEvaluator,
        build_available_prediction_specs,
        plot_multi_model_calibration_curve,
        plot_single_calibration_curve,
        save_calibration_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Model calibration evaluation started")

    calibration_config = CalibrationConfig(
        n_bins=args.n_bins,
        strategy=args.strategy,
    )

    evaluator = CalibrationEvaluator(calibration_config)

    specs = build_available_prediction_specs(project_paths.prediction_dir)

    metric_rows = []
    curve_tables = []
    figure_paths = {}

    for spec in specs:
        logger.info(
            "Evaluating calibration: %s | %s",
            spec.model_display_name,
            spec.split,
        )

        metrics, curve_points = evaluator.evaluate_one_model(spec)

        metric_rows.append(metrics)
        curve_tables.append(curve_points)

        single_figure_file = (
            project_paths.figure_dir
            / f"{spec.model_key}_{spec.split}_calibration_curve.png"
        )

        plot_single_calibration_curve(
            curve_points=curve_points,
            output_path=single_figure_file,
            title=f"{spec.model_display_name} {spec.split} Calibration Curve",
        )

        figure_paths[
            f"{spec.model_key}_{spec.split}_calibration_curve"
        ] = single_figure_file

    metrics_df = pd.DataFrame(metric_rows)
    curve_points_df = pd.concat(
        curve_tables,
        ignore_index=True,
        sort=False,
    )

    available_splits = sorted(curve_points_df["split"].unique().tolist())

    for split in available_splits:
        comparison_figure_file = (
            project_paths.figure_dir
            / f"model_comparison_{split}_calibration_curve.png"
        )

        plot_multi_model_calibration_curve(
            all_curve_points=curve_points_df,
            split=split,
            output_path=comparison_figure_file,
            title=f"{split.title()} Calibration Comparison",
        )

        figure_paths[
            f"model_comparison_{split}_calibration_curve"
        ] = comparison_figure_file

    output_paths = save_calibration_outputs(
        metrics_df=metrics_df,
        curve_points_df=curve_points_df,
        table_dir=project_paths.table_dir,
        report_dir=project_paths.report_dir,
        output_paths=figure_paths,
    )

    metrics_sorted = metrics_df.sort_values(
        by=["split", "brier_score"],
        ascending=[True, True],
    ).reset_index(drop=True)

    logger.info("Model calibration evaluation finished")
    logger.info("Calibration report: %s", output_paths["calibration_report"])

    print("=" * 70)
    print("BggDeepLearning model calibration evaluation succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Calibration bins: {calibration_config.n_bins}")
    print(f"Calibration strategy: {calibration_config.strategy}")
    print("-" * 70)
    print("Calibration metrics sorted by split and Brier score:")
    print(
        metrics_sorted[
            [
                "model_display_name",
                "split",
                "n_samples",
                "positive_rate",
                "brier_score",
                "n_bins_returned",
            ]
        ].to_string(index=False)
    )
    print("-" * 70)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()