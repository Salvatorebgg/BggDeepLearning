"""
BggDeepLearning SCI-style Word report with SHAP export script

File:
scripts/python/export_sci_word_report_with_shap.py

Usage:
python scripts\\python\\export_sci_word_report_with_shap.py
"""

from __future__ import annotations

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


def main() -> None:
    project_root = find_project_root()
    setup_python_path(project_root)

    from bggdeep.core.config import load_yaml_config
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories
    from bggdeep.reporting.shap_word_report import (
        SHAPWordReportConfig,
        SHAPWordReportIntegrator,
    )

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    logger.info("SCI-style Word report with SHAP export started")

    base_docx = project_paths.report_dir / "clinical_model_results_report_sci.docx"
    output_docx = project_paths.report_dir / "clinical_model_results_report_sci_shap.docx"

    integrator = SHAPWordReportIntegrator(
        SHAPWordReportConfig(
            target_name="poor_outcome",
            top_n_table=15,
            top_n_text=5,
        )
    )

    output_paths = integrator.append_shap_section(
        base_docx=base_docx,
        output_docx=output_docx,
        table_dir=project_paths.table_dir,
        figure_dir=project_paths.figure_dir,
        report_dir=project_paths.report_dir,
    )

    logger.info("SCI-style Word report with SHAP export finished")
    logger.info("SHAP DOCX: %s", output_paths["shap_docx"])
    logger.info("SHAP interpretation text: %s", output_paths["interpretation_text"])

    print("=" * 70)
    print("BggDeepLearning SCI-style Word report with SHAP export succeeded")
    print("-" * 70)
    print(f"Project root: {project_root}")
    print(f"SHAP DOCX: {output_paths['shap_docx']}")
    print(f"SHAP interpretation text: {output_paths['interpretation_text']}")
    print("=" * 70)


if __name__ == "__main__":
    main()