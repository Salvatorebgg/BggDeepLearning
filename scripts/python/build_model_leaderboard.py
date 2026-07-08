"""
BggDeepLearning model leaderboard script

File:
scripts/python/build_model_leaderboard.py

Usage:
python scripts\\python\\build_model_leaderboard.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
        description="Build unified model leaderboard for BggDeepLearning."
    )

    parser.add_argument(
        "--primary-metric",
        type=str,
        default="auc",
        help="Primary metric for ranking. Default: auc",
    )

    parser.add_argument(
        "--selection-split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Split used to select current best model. Default: validation",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.model_ranking import (
        ModelLeaderboardBuilder,
        ModelRankingConfig,
        save_model_leaderboard_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Model leaderboard building started")

    ranking_config = ModelRankingConfig(
        primary_metric=args.primary_metric,
        selection_split=args.selection_split,
    )

    builder = ModelLeaderboardBuilder(ranking_config)

    all_metrics = builder.load_available_metrics(project_paths.table_dir)
    leaderboard = builder.build_leaderboard(all_metrics)
    split_tables = builder.split_leaderboards(leaderboard)
    best_model_summary = builder.select_best_model(leaderboard)

    output_paths = save_model_leaderboard_outputs(
        leaderboard=leaderboard,
        split_tables=split_tables,
        best_model_summary=best_model_summary,
        table_dir=project_paths.table_dir,
        report_dir=project_paths.report_dir,
    )

    logger.info("Model leaderboard building finished")
    logger.info("Leaderboard report: %s", output_paths["report"])

    validation_table = split_tables.get("validation")
    test_table = split_tables.get("test")

    print("=" * 70)
    print("BggDeepLearning model leaderboard building succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Primary metric: {ranking_config.primary_metric}")
    print(f"Selection split: {ranking_config.selection_split}")
    print("-" * 70)
    print("Current best model:")
    print(
        best_model_summary[
            [
                "model_display_name",
                "model_name",
                "split",
                "auc",
                "accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "balanced_summary_score",
            ]
        ].to_string(index=False)
    )
    print("-" * 70)
    print("Validation leaderboard:")
    print(
        validation_table[
            [
                "rank_within_split",
                "model_display_name",
                "auc",
                "accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "balanced_summary_score",
            ]
        ].to_string(index=False)
    )
    print("-" * 70)
    print("Test leaderboard:")
    print(
        test_table[
            [
                "rank_within_split",
                "model_display_name",
                "auc",
                "accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "balanced_summary_score",
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