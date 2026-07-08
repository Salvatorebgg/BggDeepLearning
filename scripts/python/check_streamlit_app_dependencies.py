"""
BggDeepLearning Streamlit dependency check

File:
scripts/python/check_streamlit_app_dependencies.py

Usage:
python scripts\\python\\check_streamlit_app_dependencies.py
"""

from __future__ import annotations

import importlib
from pathlib import Path


def check_package(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        print(f"[OK] {package_name}")
        return True
    except Exception as exc:
        print(f"[MISSING] {package_name}: {exc}")
        return False


def check_file(path: str) -> bool:
    file_path = Path(path)

    if file_path.exists():
        print(f"[OK] {path}")
        return True

    print(f"[MISSING] {path}")
    return False


def main() -> None:
    print("=" * 70)
    print("BggDeepLearning Streamlit app dependency check")
    print("=" * 70)

    packages_ok = True
    for package in ["streamlit", "pandas", "numpy", "joblib", "sklearn"]:
        packages_ok = check_package(package) and packages_ok

    print("-" * 70)

    files_ok = True
    required_files = [
        "configs/app.yaml",
        "outputs/models/tabular_preprocessor.joblib",
        "outputs/models/logistic_regression_baseline.joblib",
        "outputs/models/random_forest_baseline.joblib",
        "outputs/models/gradient_boosting_baseline.joblib",
        "apps/streamlit/clinical_risk_demo.py",
        "python/bggdeep/inference/clinical_risk_predictor.py",
    ]

    for file_path in required_files:
        files_ok = check_file(file_path) and files_ok

    print("-" * 70)

    if packages_ok and files_ok:
        print("Streamlit app dependency check passed.")
    else:
        print("Some dependencies or files are missing.")
        print("Please install missing packages or run previous training steps.")

    print("=" * 70)


if __name__ == "__main__":
    main()