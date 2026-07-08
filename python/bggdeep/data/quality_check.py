"""
BggDeepLearning data quality check module

File:
python/bggdeep/data/quality_check.py

Purpose:
1. Check basic dataset shape
2. Check missing values
3. Check variable types
4. Check numeric variable distribution
5. Check categorical variable distribution
6. Check outcome distribution
7. Generate data quality alerts
8. Save quality check outputs

This module is designed for clinical tabular data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class DataQualityConfig:
    """
    Configuration for data quality check.

    target_column:
        Outcome variable name.

    high_missing_threshold:
        If missing rate is higher than this threshold, a warning will be generated.

    rare_category_threshold:
        If a category proportion is lower than this threshold, it may be considered rare.

    """

    target_column: str = "poor_outcome"
    high_missing_threshold: float = 0.15
    rare_category_threshold: float = 0.01


class ClinicalDataQualityChecker:
    """
    Data quality checker for clinical tabular data.

    Main checks:
    1. Missing values
    2. Numeric summaries
    3. Categorical summaries
    4. Outcome distribution
    5. Clinical range alerts
    """

    def __init__(self, config: Optional[DataQualityConfig] = None) -> None:
        self.config = config or DataQualityConfig()

    def run_all_checks(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Run all quality checks.

        Returns:
            A dictionary of result tables.
        """
        missing_summary = self.build_missing_summary(df)
        variable_type_summary = self.build_variable_type_summary(df)
        numeric_summary = self.build_numeric_summary(df)
        categorical_summary = self.build_categorical_summary(df)
        outcome_summary = self.build_outcome_summary(df)
        alerts = self.build_quality_alerts(
            df=df,
            missing_summary=missing_summary,
            outcome_summary=outcome_summary,
        )

        return {
            "missing_summary": missing_summary,
            "variable_type_summary": variable_type_summary,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "outcome_summary": outcome_summary,
            "alerts": alerts,
        }

    def build_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build missing value summary for each column.
        """
        n_rows = len(df)

        rows = []

        for column in df.columns:
            missing_count = int(df[column].isna().sum())
            missing_rate = missing_count / n_rows if n_rows > 0 else 0.0

            rows.append(
                {
                    "variable_name": column,
                    "missing_count": missing_count,
                    "missing_rate": round(missing_rate, 4),
                    "non_missing_count": int(n_rows - missing_count),
                }
            )

        result = pd.DataFrame(rows)
        result = result.sort_values(
            by=["missing_rate", "missing_count"],
            ascending=[False, False],
        ).reset_index(drop=True)

        return result

    def build_variable_type_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build variable type summary.
        """
        rows = []

        for column in df.columns:
            series = df[column]
            dtype_text = str(series.dtype)
            unique_count = int(series.nunique(dropna=True))

            if pd.api.types.is_numeric_dtype(series):
                inferred_type = "numeric"
            elif pd.api.types.is_bool_dtype(series):
                inferred_type = "binary"
            else:
                inferred_type = "categorical_or_text"

            if unique_count == 2 and pd.api.types.is_numeric_dtype(series):
                inferred_type = "binary_numeric"

            rows.append(
                {
                    "variable_name": column,
                    "pandas_dtype": dtype_text,
                    "inferred_type": inferred_type,
                    "unique_count": unique_count,
                }
            )

        return pd.DataFrame(rows)

    def build_numeric_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build summary for numeric columns.
        """
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        rows = []

        for column in numeric_columns:
            series = df[column].dropna()

            if series.empty:
                rows.append(
                    {
                        "variable_name": column,
                        "count": 0,
                        "mean": None,
                        "std": None,
                        "min": None,
                        "q1": None,
                        "median": None,
                        "q3": None,
                        "max": None,
                    }
                )
                continue

            rows.append(
                {
                    "variable_name": column,
                    "count": int(series.shape[0]),
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                    "min": round(float(series.min()), 4),
                    "q1": round(float(series.quantile(0.25)), 4),
                    "median": round(float(series.median()), 4),
                    "q3": round(float(series.quantile(0.75)), 4),
                    "max": round(float(series.max()), 4),
                }
            )

        return pd.DataFrame(rows)

    def build_categorical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build summary for categorical columns.

        Object columns and low-cardinality non-float columns are summarized.
        """
        rows = []

        for column in df.columns:
            series = df[column]

            is_object = pd.api.types.is_object_dtype(series)
            unique_count = int(series.nunique(dropna=True))
            is_low_cardinality = unique_count <= 20

            if not is_object and not is_low_cardinality:
                continue

            counts = series.value_counts(dropna=False)

            for category, count in counts.items():
                category_text = "MISSING" if pd.isna(category) else str(category)
                proportion = count / len(df) if len(df) > 0 else 0.0

                rows.append(
                    {
                        "variable_name": column,
                        "category": category_text,
                        "count": int(count),
                        "proportion": round(float(proportion), 4),
                    }
                )

        return pd.DataFrame(rows)

    def build_outcome_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build outcome distribution summary.
        """
        target = self.config.target_column

        if target not in df.columns:
            return pd.DataFrame(
                [
                    {
                        "target_column": target,
                        "status": "missing_target_column",
                        "class_label": None,
                        "count": None,
                        "proportion": None,
                    }
                ]
            )

        counts = df[target].value_counts(dropna=False).sort_index()
        rows = []

        for label, count in counts.items():
            label_text = "MISSING" if pd.isna(label) else str(label)
            proportion = count / len(df) if len(df) > 0 else 0.0

            rows.append(
                {
                    "target_column": target,
                    "status": "ok",
                    "class_label": label_text,
                    "count": int(count),
                    "proportion": round(float(proportion), 4),
                }
            )

        return pd.DataFrame(rows)

    def build_quality_alerts(
        self,
        df: pd.DataFrame,
        missing_summary: pd.DataFrame,
        outcome_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build data quality alerts.
        """
        alerts: List[dict[str, str]] = []

        alerts.extend(self._check_empty_dataset(df))
        alerts.extend(self._check_high_missing(missing_summary))
        alerts.extend(self._check_target_column(df))
        alerts.extend(self._check_outcome_distribution(outcome_summary))
        alerts.extend(self._check_clinical_ranges(df))
        alerts.extend(self._check_possible_leakage_columns(df))

        if not alerts:
            alerts.append(
                {
                    "alert_level": "INFO",
                    "variable_name": "ALL",
                    "message": "No major data quality issue was detected.",
                }
            )

        return pd.DataFrame(alerts)

    def _check_empty_dataset(self, df: pd.DataFrame) -> List[dict[str, str]]:
        """
        Check whether dataset is empty.
        """
        alerts = []

        if df.empty:
            alerts.append(
                {
                    "alert_level": "ERROR",
                    "variable_name": "DATASET",
                    "message": "The dataset is empty.",
                }
            )

        return alerts

    def _check_high_missing(self, missing_summary: pd.DataFrame) -> List[dict[str, str]]:
        """
        Check variables with high missing rate.
        """
        alerts = []

        threshold = self.config.high_missing_threshold

        high_missing = missing_summary[
            missing_summary["missing_rate"] > threshold
        ]

        for _, row in high_missing.iterrows():
            alerts.append(
                {
                    "alert_level": "WARNING",
                    "variable_name": str(row["variable_name"]),
                    "message": (
                        f"High missing rate detected: "
                        f"{row['missing_rate']:.4f}. "
                        f"Threshold: {threshold:.2f}."
                    ),
                }
            )

        return alerts

    def _check_target_column(self, df: pd.DataFrame) -> List[dict[str, str]]:
        """
        Check whether target column exists.
        """
        target = self.config.target_column
        alerts = []

        if target not in df.columns:
            alerts.append(
                {
                    "alert_level": "ERROR",
                    "variable_name": target,
                    "message": f"Target column was not found: {target}",
                }
            )

        return alerts

    def _check_outcome_distribution(self, outcome_summary: pd.DataFrame) -> List[dict[str, str]]:
        """
        Check outcome distribution.
        """
        alerts = []

        if outcome_summary.empty:
            return alerts

        if "status" in outcome_summary.columns:
            if (outcome_summary["status"] == "missing_target_column").any():
                return alerts

        valid_rows = outcome_summary[
            outcome_summary["class_label"].astype(str) != "MISSING"
        ]

        if len(valid_rows) < 2:
            alerts.append(
                {
                    "alert_level": "ERROR",
                    "variable_name": self.config.target_column,
                    "message": "Outcome variable has fewer than two classes.",
                }
            )
            return alerts

        min_proportion = valid_rows["proportion"].astype(float).min()

        if min_proportion < 0.05:
            alerts.append(
                {
                    "alert_level": "WARNING",
                    "variable_name": self.config.target_column,
                    "message": (
                        f"Outcome class imbalance may be severe. "
                        f"Minimum class proportion: {min_proportion:.4f}."
                    ),
                }
            )

        return alerts

    def _check_clinical_ranges(self, df: pd.DataFrame) -> List[dict[str, str]]:
        """
        Check simple clinical ranges.

        These rules are broad sanity checks, not clinical diagnosis rules.
        """
        alerts = []

        rules = [
            ("age", 0, 120),
            ("bmi", 10, 80),
            ("heart_rate", 20, 250),
            ("systolic_bp", 40, 260),
            ("diastolic_bp", 20, 180),
            ("respiratory_rate", 4, 80),
            ("oxygen_saturation", 30, 100),
            ("temperature_c", 25, 45),
            ("hemoglobin", 20, 250),
            ("white_blood_cell", 0.1, 100),
            ("platelet", 1, 1000),
            ("creatinine", 5, 1500),
            ("lactate", 0, 30),
            ("c_reactive_protein", 0, 500),
            ("shock_index", 0, 5),
        ]

        for column, lower, upper in rules:
            if column not in df.columns:
                continue

            series = df[column].dropna()

            if series.empty:
                continue

            invalid_count = int(((series < lower) | (series > upper)).sum())

            if invalid_count > 0:
                alerts.append(
                    {
                        "alert_level": "WARNING",
                        "variable_name": column,
                        "message": (
                            f"{invalid_count} values are outside broad "
                            f"clinical sanity range [{lower}, {upper}]."
                        ),
                    }
                )

        return alerts

    def _check_possible_leakage_columns(self, df: pd.DataFrame) -> List[dict[str, str]]:
        """
        Check columns that may cause leakage in later modeling.
        """
        alerts = []

        suspicious_keywords = [
            "probability_demo",
            "prediction",
            "predicted",
            "outcome_probability",
        ]

        for column in df.columns:
            column_lower = column.lower()

            for keyword in suspicious_keywords:
                if keyword in column_lower:
                    alerts.append(
                        {
                            "alert_level": "WARNING",
                            "variable_name": column,
                            "message": (
                                "This variable may cause data leakage in modeling. "
                                "Do not use it as an input feature."
                            ),
                        }
                    )
                    break

        return alerts


def save_quality_outputs(
    results: dict[str, pd.DataFrame],
    table_dir: Path,
    report_dir: Path,
) -> dict[str, Path]:
    """
    Save quality check result tables and text report.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "missing_summary": table_dir / "data_quality_missing_summary.csv",
        "variable_type_summary": table_dir / "data_quality_variable_type_summary.csv",
        "numeric_summary": table_dir / "data_quality_numeric_summary.csv",
        "categorical_summary": table_dir / "data_quality_categorical_summary.csv",
        "outcome_summary": table_dir / "data_quality_outcome_summary.csv",
        "alerts": table_dir / "data_quality_alerts.csv",
    }

    for key, path in output_paths.items():
        results[key].to_csv(path, index=False, encoding="utf-8-sig")

    report_file = report_dir / "data_quality_report.txt"
    report_text = build_text_report(results)
    report_file.write_text(report_text, encoding="utf-8")

    output_paths["text_report"] = report_file

    return output_paths


def build_text_report(results: dict[str, pd.DataFrame]) -> str:
    """
    Build a plain text quality report.
    """
    missing_summary = results["missing_summary"]
    variable_type_summary = results["variable_type_summary"]
    numeric_summary = results["numeric_summary"]
    outcome_summary = results["outcome_summary"]
    alerts = results["alerts"]

    top_missing = missing_summary.head(10)

    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Data Quality Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Dataset Overview")
    lines.append("-" * 70)
    lines.append(f"Number of variables: {len(variable_type_summary)}")
    lines.append(f"Number of numeric variables: {(variable_type_summary['inferred_type'].str.contains('numeric')).sum()}")
    lines.append("")
    lines.append("2. Outcome Summary")
    lines.append("-" * 70)
    lines.append(outcome_summary.to_string(index=False))
    lines.append("")
    lines.append("3. Top Missing Variables")
    lines.append("-" * 70)
    lines.append(top_missing.to_string(index=False))
    lines.append("")
    lines.append("4. Numeric Summary Preview")
    lines.append("-" * 70)
    lines.append(numeric_summary.head(10).to_string(index=False))
    lines.append("")
    lines.append("5. Alerts")
    lines.append("-" * 70)
    lines.append(alerts.to_string(index=False))
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)