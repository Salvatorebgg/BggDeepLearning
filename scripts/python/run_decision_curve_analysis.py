"""
BggDeepLearning Decision Curve Analysis script

File:
scripts/python/run_decision_curve_analysis.py

Usage:
python scripts\\python\\run_decision_curve_analysis.py
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
        description="Run Decision Curve Analysis for BggDeepLearning models."
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.01,
        help="Minimum threshold probability. Default: 0.01",
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
        help="Maximum threshold probability. Default: 0.99",
    )

    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=99,
        help="Number of threshold points. Default: 99",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.dca import (
        DCAConfig,
        DecisionCurveAnalyzer,
        build_available_dca_specs,
        build_reference_dca_tables,
        build_selected_threshold_summary,
        plot_dca_curve,
        save_dca_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Decision Curve Analysis started")

    dca_config = DCAConfig(
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        n_thresholds=args.n_thresholds,
    )

    analyzer = DecisionCurveAnalyzer(dca_config)

    specs = build_available_dca_specs(project_paths.prediction_dir)

    dca_tables = []

    for spec in specs:
        logger.info(
            "Calculating DCA: %s | %s",
            spec.model_display_name,
            spec.split,
        )

        dca_table = analyzer.analyze_one_prediction_file(spec)
        dca_tables.append(dca_table)

    reference_tables = build_reference_dca_tables(
        analyzer=analyzer,
        specs=specs,
    )

    dca_points = pd.concat(
        dca_tables + reference_tables,
        ignore_index=True,
        sort=False,
    )

    output_figure_paths = {}

    available_splits = sorted(dca_points["split"].unique().tolist())

    for split in available_splits:
        figure_file = project_paths.figure_dir / f"dca_{split}_curve.png"

        plot_dca_curve(
            dca_points=dca_points,
            split=split,
            output_path=figure_file,
            title=f"{split.title()} Decision Curve Analysis",
        )

        output_figure_paths[f"dca_{split}_curve"] = figure_file

    selected_threshold_summary = build_selected_threshold_summary(
        dca_points=dca_points,
        selected_thresholds=dca_config.selected_thresholds,
    )

    output_paths = save_dca_outputs(
        dca_points=dca_points,
        selected_threshold_summary=selected_threshold_summary,
        table_dir=project_paths.table_dir,
        report_dir=project_paths.report_dir,
        output_paths=output_figure_paths,
    )

    logger.info("Decision Curve Analysis finished")
    logger.info("DCA report: %s", output_paths["dca_report"])

    print("=" * 70)
    print("BggDeepLearning Decision Curve Analysis succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Threshold range: {dca_config.threshold_min} to {dca_config.threshold_max}")
    print(f"Number of thresholds: {dca_config.n_thresholds}")
    print("-" * 70)
    print("Available splits:")
    print(", ".join(available_splits))
    print("-" * 70)
    print("Selected threshold summary preview:")
    preview_columns = [
        "split",
        "selected_threshold",
        "rank_at_threshold",
        "model_display_name",
        "strategy",
        "net_benefit",
        "standardized_net_benefit",
    ]
    preview_columns = [
        column for column in preview_columns
        if column in selected_threshold_summary.columns
    ]
    print(
        selected_threshold_summary[preview_columns]
        .head(50)
        .to_string(index=False)
    )
    print("-" * 70)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()