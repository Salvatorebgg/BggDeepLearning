"""
BggDeepLearning Streamlit dashboard artifact check

File:
scripts/python/check_streamlit_dashboard_artifacts.py

Usage:
python scripts\\python\\check_streamlit_dashboard_artifacts.py
"""

from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / "configs" / "app.yaml").exists():
            return parent

    raise FileNotFoundError("Project root was not found. configs/app.yaml is missing.")


def check_file(project_root: Path, relative_path: str) -> bool:
    path = project_root / relative_path

    if path.exists():
        print(f"[OK] {relative_path}")
        return True

    print(f"[MISSING] {relative_path}")
    return False


def main() -> None:
    project_root = find_project_root()

    required_files = [
        "apps/streamlit/clinical_risk_demo.py",
        "outputs/models/tabular_preprocessor.joblib",
        "outputs/models/logistic_regression_baseline.joblib",
        "outputs/models/random_forest_baseline.joblib",
        "outputs/models/gradient_boosting_baseline.joblib",
        "outputs/tables/model_best_model_summary.csv",
        "outputs/tables/model_leaderboard_validation.csv",
        "outputs/tables/model_leaderboard_test.csv",
        "outputs/tables/model_comparison_roc_summary.csv",
        "outputs/tables/model_calibration_metrics.csv",
        "outputs/tables/dca_selected_threshold_summary.csv",
        "outputs/tables/shap_model_global_importance_all.csv",
        "outputs/tables/comprehensive_model_report_file_index.csv",
        "outputs/figures/model_comparison_validation_roc_curve.png",
        "outputs/figures/model_comparison_test_roc_curve.png",
        "outputs/figures/model_comparison_validation_calibration_curve.png",
        "outputs/figures/model_comparison_test_calibration_curve.png",
        "outputs/figures/dca_validation_curve.png",
        "outputs/figures/dca_test_curve.png",
        "outputs/figures/shap_random_forest_bar.png",
        "outputs/figures/shap_gradient_boosting_bar.png",
        "outputs/figures/shap_random_forest_beeswarm.png",
        "outputs/figures/shap_gradient_boosting_beeswarm.png",
        "outputs/reports/comprehensive_model_evaluation_report.txt",
        "outputs/reports/dca_report.txt",
        "outputs/reports/shap_explainability_report.txt",
    ]

    print("=" * 70)
    print("BggDeepLearning Streamlit dashboard artifact check")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print("-" * 70)

    ok_count = 0
    missing_count = 0

    for relative_path in required_files:
        ok = check_file(project_root, relative_path)

        if ok:
            ok_count += 1
        else:
            missing_count += 1

    print("-" * 70)
    print(f"OK files: {ok_count}")
    print(f"Missing files: {missing_count}")

    if missing_count == 0:
        print("Dashboard artifact check passed.")
    else:
        print("Some dashboard files are missing.")
        print("Please run previous model evaluation scripts to regenerate missing outputs.")

    print("=" * 70)


if __name__ == "__main__":
    main()