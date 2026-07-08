"""
BggDeepLearning comprehensive model evaluation report script

File:
scripts/python/generate_comprehensive_model_report.py

Usage:
python scripts\\python\\generate_comprehensive_model_report.py
"""

from __future__ import annotations

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


def main() -> None:
    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.reporting.model_report import (
        ComprehensiveModelReportBuilder,
        ComprehensiveReportConfig,
        save_comprehensive_report_outputs,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Comprehensive model report generation started")

    report_config = ComprehensiveReportConfig()
    builder = ComprehensiveModelReportBuilder(report_config)

    report_inputs = builder.load_report_inputs(
        table_dir=project_paths.table_dir,
    )

    file_index = builder.build_file_index(
        project_root=project_root,
        table_dir=project_paths.table_dir,
        figure_dir=project_paths.figure_dir,
        report_dir=project_paths.report_dir,
        model_dir=project_paths.model_dir,
        prediction_dir=project_paths.prediction_dir,
    )

    markdown_report = builder.build_markdown_report(
        project_root=project_root,
        inputs=report_inputs,
        file_index=file_index,
    )

    text_report = builder.build_text_report(
        project_root=project_root,
        inputs=report_inputs,
        file_index=file_index,
    )

    output_paths = save_comprehensive_report_outputs(
        markdown_report=markdown_report,
        text_report=text_report,
        file_index=file_index,
        report_dir=project_paths.report_dir,
        table_dir=project_paths.table_dir,
    )

    logger.info("Comprehensive model report generation finished")
    logger.info("Markdown report: %s", output_paths["markdown_report"])
    logger.info("Text report: %s", output_paths["text_report"])

    existing_files = file_index[file_index["exists"] == True]
    missing_files = file_index[file_index["exists"] == False]

    print("=" * 70)
    print("BggDeepLearning comprehensive model report generation succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Existing indexed files: {len(existing_files)}")
    print(f"Missing indexed files: {len(missing_files)}")
    print("-" * 70)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()