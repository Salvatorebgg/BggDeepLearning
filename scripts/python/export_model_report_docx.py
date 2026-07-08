"""
BggDeepLearning Word report export script

File:
scripts/python/export_model_report_docx.py

Usage:
python scripts\\python\\export_model_report_docx.py
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
    from bggdeep.reporting.word_report import (
        WordModelReportBuilder,
        WordReportConfig,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("Word report export started")

    report_config = WordReportConfig()
    builder = WordModelReportBuilder(report_config)

    comprehensive_docx = (
        project_paths.report_dir / "comprehensive_model_evaluation_report.docx"
    )

    clinical_results_docx = (
        project_paths.report_dir / "clinical_model_results_report.docx"
    )

    builder.build_comprehensive_docx(
        project_root=project_root,
        table_dir=project_paths.table_dir,
        figure_dir=project_paths.figure_dir,
        output_path=comprehensive_docx,
    )

    builder.build_clinical_results_docx(
        project_root=project_root,
        table_dir=project_paths.table_dir,
        figure_dir=project_paths.figure_dir,
        output_path=clinical_results_docx,
    )

    logger.info("Word report export finished")
    logger.info("Comprehensive DOCX: %s", comprehensive_docx)
    logger.info("Clinical results DOCX: %s", clinical_results_docx)

    print("=" * 70)
    print("BggDeepLearning Word report export succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"Comprehensive DOCX: {comprehensive_docx}")
    print(f"Clinical results DOCX: {clinical_results_docx}")
    print("=" * 70)


if __name__ == "__main__":
    main()