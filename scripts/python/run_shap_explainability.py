"""
BggDeepLearning SHAP explainability script

File:
scripts/python/run_shap_explainability.py

Usage:
python scripts\\python\\run_shap_explainability.py
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
        description="Run SHAP explainability for BggDeepLearning tree models."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of test samples used for SHAP. Default: 200",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features displayed in SHAP plots. Default: 30",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.evaluation.shap_explain import (
        SHAPExplainabilityConfig,
        SHAPExplainabilityRunner,
        build_available_shap_model_specs,
        save_shap_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("SHAP explainability started")

    shap_config = SHAPExplainabilityConfig(
        max_samples=args.max_samples,
        random_state=args.seed,
        top_n=args.top_n,
    )

    runner = SHAPExplainabilityRunner(shap_config)

    x_test = runner.load_test_features(project_paths.processed_data_dir)
    x_sample = runner.sample_features(x_test)

    specs = build_available_shap_model_specs(project_paths.model_dir)

    results = []

    for spec in specs:
        logger.info("Running SHAP for model: %s", spec.model_display_name)

        result = runner.explain_one_model(
            spec=spec,
            x_sample=x_sample,
            table_dir=project_paths.table_dir,
            figure_dir=project_paths.figure_dir,
        )

        results.append(result)

    output_paths = save_shap_outputs(
        results=results,
        table_dir=project_paths.table_dir,
        report_dir=project_paths.report_dir,
    )

    logger.info("SHAP explainability finished")
    logger.info("SHAP report: %s", output_paths["report_file"])

    print("=" * 70)
    print("BggDeepLearning SHAP explainability succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Samples used: {len(x_sample)}")
    print(f"Features: {x_sample.shape[1]}")
    print("-" * 70)

    for result in results:
        print(f"Model: {result['model_display_name']}")
        print(f"Importance table: {result['importance_file']}")
        print(f"Bar figure: {result['bar_file']}")
        print(f"Beeswarm figure: {result['beeswarm_file']}")
        print("Top 10 features:")
        print(
            result["importance_df"][
                [
                    "rank",
                    "feature_name",
                    "mean_abs_shap",
                ]
            ].head(10).to_string(index=False)
        )
        print("-" * 70)

    print("Combined outputs:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()