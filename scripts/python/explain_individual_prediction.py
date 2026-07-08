"""
BggDeepLearning individual prediction explanation script

File:
scripts/python/explain_individual_prediction.py

Usage examples:
python scripts\\python\\explain_individual_prediction.py
python scripts\\python\\explain_individual_prediction.py --model gradient_boosting --split test --sample-index 0
python scripts\\python\\explain_individual_prediction.py --model random_forest --split test --sample-index 10
python scripts\\python\\explain_individual_prediction.py --model gradient_boosting --split test --patient-id P0001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_project_root() -> Path:
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / "configs" / "app.yaml").exists():
            return parent

    raise FileNotFoundError("Project root was not found. configs/app.yaml is missing.")


def setup_python_path(project_root: Path) -> None:
    python_dir = project_root / "python"

    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain one individual prediction using SHAP."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gradient_boosting",
        choices=["random_forest", "gradient_boosting"],
        help="Model to explain. Default: gradient_boosting",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Data split to explain. Default: test",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample row index in selected split. Default: 0",
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional patient ID. If provided, it overrides sample-index.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of features displayed in waterfall plot. Default: 20",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.individual_explain import (
        IndividualExplanationConfig,
        IndividualPredictionExplainer,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Individual prediction explanation started")

    explanation_config = IndividualExplanationConfig(
        model_key=args.model,
        split=args.split,
        sample_index=args.sample_index,
        patient_id=args.patient_id,
        top_n=args.top_n,
    )

    explainer = IndividualPredictionExplainer(explanation_config)

    result = explainer.explain(
        processed_data_dir=project_paths.processed_data_dir,
        model_dir=project_paths.model_dir,
        table_dir=project_paths.table_dir,
        figure_dir=project_paths.figure_dir,
        report_dir=project_paths.report_dir,
    )

    logger.info("Individual prediction explanation finished")
    logger.info("Report file: %s", result["report_file"])

    print("=" * 70)
    print("BggDeepLearning individual prediction explanation succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Model: {result['model_display_name']}")
    print(f"Split: {result['split']}")
    print(f"Sample index: {result['sample_index']}")
    print(f"Predicted probability: {result['probability']:.6f}")
    print(f"Predicted label: {result['predicted_label']}")
    print("-" * 70)
    print("Output files:")
    print(f"Explanation table: {result['table_file']}")
    print(f"Waterfall figure: {result['waterfall_file']}")
    print(f"Report file: {result['report_file']}")
    print("=" * 70)


if __name__ == "__main__":
    main()