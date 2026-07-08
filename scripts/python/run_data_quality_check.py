"""
BggDeepLearning data quality check script

File:
scripts/python/run_data_quality_check.py

Usage:
python scripts\\python\\run_data_quality_check.py

Optional:
python scripts\\python\\run_data_quality_check.py --input data\\simulated\\clinical_simulated_data.csv
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
        description="Run data quality check for BggDeepLearning clinical data."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/simulated/clinical_simulated_data.csv",
        help="Input CSV file path.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="poor_outcome",
        help="Target outcome column.",
    )

    parser.add_argument(
        "--high-missing-threshold",
        type=float,
        default=0.15,
        help="High missing rate warning threshold.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.data.quality_check import (
        ClinicalDataQualityChecker,
        DataQualityConfig,
        save_quality_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    input_path = Path(args.input)

    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input data file was not found: {input_path}\n"
            "Please run: python scripts\\python\\generate_simulated_clinical_data.py"
        )

    logger.info("Data quality check started")
    logger.info("Input file: %s", input_path)

    df = pd.read_csv(input_path)

    quality_config = DataQualityConfig(
        target_column=args.target,
        high_missing_threshold=args.high_missing_threshold,
    )

    checker = ClinicalDataQualityChecker(quality_config)
    results = checker.run_all_checks(df)

    output_paths = save_quality_outputs(
        results=results,
        table_dir=project_paths.table_dir,
        report_dir=project_paths.report_dir,
    )

    logger.info("Data quality check finished")
    logger.info("Data quality report saved: %s", output_paths["text_report"])

    alerts = results["alerts"]
    warning_count = int((alerts["alert_level"] == "WARNING").sum())
    error_count = int((alerts["alert_level"] == "ERROR").sum())

    print("=" * 70)
    print("BggDeepLearning data quality check succeeded")
    print("-" * 70)
    print(f"Input file: {input_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Warnings: {warning_count}")
    print(f"Errors: {error_count}")
    print("-" * 70)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("=" * 70)

    if error_count > 0:
        print("Data quality check found ERROR-level alerts. Please review the report.")


if __name__ == "__main__":
    main()