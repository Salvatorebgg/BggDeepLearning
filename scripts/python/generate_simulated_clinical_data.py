"""
BggDeepLearning simulated clinical data generation script

File:
scripts/python/generate_simulated_clinical_data.py

Usage:
python scripts\\python\\generate_simulated_clinical_data.py

Optional:
python scripts\\python\\generate_simulated_clinical_data.py --n 1000 --seed 2026
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
        description="Generate simulated clinical tabular data for BggDeepLearning."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of simulated patients. Default: 500",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.03,
        help="Missing value rate. Default: 0.03",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.data.simulation import (
        SimulationConfig,
        ClinicalDataSimulator,
        build_data_dictionary,
        summarize_simulated_data,
        save_simulation_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Simulated clinical data generation started")
    logger.info("Number of patients: %s", args.n)
    logger.info("Random seed: %s", args.seed)
    logger.info("Missing rate: %s", args.missing_rate)

    simulation_config = SimulationConfig(
        n_patients=args.n,
        seed=args.seed,
        missing_rate=args.missing_rate,
    )

    simulator = ClinicalDataSimulator(simulation_config)

    df = simulator.generate()
    data_dictionary = build_data_dictionary()
    summary = summarize_simulated_data(df)

    data_file, dictionary_file, summary_file = save_simulation_outputs(
        df=df,
        data_dictionary=data_dictionary,
        summary=summary,
        simulated_data_dir=project_paths.simulated_data_dir,
        table_dir=project_paths.table_dir,
    )

    logger.info("Simulated clinical data saved: %s", data_file)
    logger.info("Data dictionary saved: %s", dictionary_file)
    logger.info("Summary table saved: %s", summary_file)
    logger.info("Simulated clinical data generation finished")

    print("=" * 70)
    print("BggDeepLearning simulated clinical data generation succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Outcome rate: {df['poor_outcome'].mean():.4f}")
    print("-" * 70)
    print(f"Data file: {data_file}")
    print(f"Data dictionary: {dictionary_file}")
    print(f"Summary table: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()