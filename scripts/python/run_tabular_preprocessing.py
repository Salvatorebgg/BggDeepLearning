"""
BggDeepLearning tabular preprocessing script

File:
scripts/python/run_tabular_preprocessing.py

Usage:
python scripts\\python\\run_tabular_preprocessing.py

Optional:
python scripts\\python\\run_tabular_preprocessing.py --input data\\simulated\\clinical_simulated_data.csv
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
        description="Run tabular preprocessing for BggDeepLearning."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/simulated/clinical_simulated_data.csv",
        help="Input clinical CSV file.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="poor_outcome",
        help="Target outcome column.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test set proportion.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Validation set proportion.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--include-treatment",
        action="store_true",
        help="Include treatment columns as features. Default: excluded.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.data.preprocessing import (
        ClinicalTabularPreprocessor,
        TabularPreprocessingConfig,
        save_preprocessing_outputs,
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

    logger.info("Tabular preprocessing started")
    logger.info("Input file: %s", input_path)

    df = pd.read_csv(input_path)

    preprocessing_config = TabularPreprocessingConfig(
        target_column=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        exclude_treatment_columns=not args.include_treatment,
    )

    preprocessor = ClinicalTabularPreprocessor(preprocessing_config)
    results = preprocessor.run(df)

    output_paths = save_preprocessing_outputs(
        results=results,
        processed_data_dir=project_paths.processed_data_dir,
        table_dir=project_paths.table_dir,
        report_dir=project_paths.report_dir,
        model_dir=project_paths.model_dir,
        config=preprocessing_config,
    )

    train_features = results["train_features"]
    val_features = results["val_features"]
    test_features = results["test_features"]
    split_summary = results["split_summary"]

    logger.info("Tabular preprocessing finished")
    logger.info("Train features shape: %s", train_features.shape)
    logger.info("Validation features shape: %s", val_features.shape)
    logger.info("Test features shape: %s", test_features.shape)
    logger.info("Preprocessing report: %s", output_paths["report_txt"])

    print("=" * 70)
    print("BggDeepLearning tabular preprocessing succeeded")
    print("-" * 70)
    print(f"Input file: {input_path}")
    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")
    print("-" * 70)
    print("Split summary:")
    print(split_summary.to_string(index=False))
    print("-" * 70)
    print(f"Train features shape: {train_features.shape}")
    print(f"Validation features shape: {val_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print("-" * 70)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()