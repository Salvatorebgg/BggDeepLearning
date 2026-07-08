"""
BggDeepLearning batch high-risk patient explanation script

File:
scripts/python/batch_explain_high_risk_patients.py

Usage examples:
python scripts\\python\\batch_explain_high_risk_patients.py
python scripts\\python\\batch_explain_high_risk_patients.py --model gradient_boosting --split test --top-n-patients 5
python scripts\\python\\batch_explain_high_risk_patients.py --model random_forest --split test --top-n-patients 10
python scripts\\python\\batch_explain_high_risk_patients.py --model gradient_boosting --split test --top-n-patients 20 --risk-threshold 0.30
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
        description="Batch explain high-risk patients using individual SHAP waterfall plots."
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
        choices=["validation", "test"],
        help="Data split to analyze. Default: test",
    )

    parser.add_argument(
        "--top-n-patients",
        type=int,
        default=5,
        help="Number of highest-risk patients to explain. Default: 5",
    )

    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=None,
        help="Optional minimum predicted risk threshold. Example: 0.30",
    )

    parser.add_argument(
        "--top-n-features",
        type=int,
        default=20,
        help="Number of SHAP features shown in each waterfall plot. Default: 20",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.batch_individual_explain import (
        BatchHighRiskExplanationConfig,
        BatchHighRiskExplainer,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Batch high-risk patient explanation started")

    batch_config = BatchHighRiskExplanationConfig(
        model_key=args.model,
        split=args.split,
        top_n_patients=args.top_n_patients,
        risk_threshold=args.risk_threshold,
        top_n_features=args.top_n_features,
    )

    explainer = BatchHighRiskExplainer(batch_config)

    result = explainer.run(
        prediction_dir=project_paths.prediction_dir,
        processed_data_dir=project_paths.processed_data_dir,
        model_dir=project_paths.model_dir,
        table_dir=project_paths.table_dir,
        figure_dir=project_paths.figure_dir,
        report_dir=project_paths.report_dir,
    )

    logger.info("Batch high-risk patient explanation finished")
    logger.info("Batch report: %s", result["batch_report_file"])

    print("=" * 70)
    print("BggDeepLearning batch high-risk patient explanation succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Model: {batch_config.model_key}")
    print(f"Split: {batch_config.split}")
    print(f"Top N patients: {batch_config.top_n_patients}")
    print(f"Risk threshold: {batch_config.risk_threshold}")
    print(f"Candidates selected: {result['n_candidates']}")
    print(f"Individual outputs generated: {result['n_outputs']}")
    print("-" * 70)
    print("Output files:")
    print(f"Candidate table: {result['candidate_file']}")
    print(f"Output index table: {result['output_index_file']}")
    print(f"Batch report: {result['batch_report_file']}")
    print("=" * 70)


if __name__ == "__main__":
    main()