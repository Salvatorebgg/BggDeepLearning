"""
BggDeepLearning comprehensive model evaluation report module

File:
python/bggdeep/reporting/model_report.py

Purpose:
1. Collect model leaderboard results
2. Collect model comparison metrics
3. Collect calibration results
4. Collect Decision Curve Analysis results
5. Build Markdown and text reports
6. Build output file index

This report is designed as a bridge between model development
and later manuscript-style reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class ComprehensiveReportConfig:
    """
    Configuration for comprehensive model report.
    """

    report_title: str = "BggDeepLearning Comprehensive Model Evaluation Report"
    project_name: str = "BggDeepLearning"
    target_name: str = "poor_outcome"
    selection_split: str = "validation"
    primary_metric: str = "auc"


class ComprehensiveModelReportBuilder:
    """
    Build comprehensive model evaluation report.
    """

    def __init__(self, config: ComprehensiveReportConfig) -> None:
        self.config = config

    @staticmethod
    def read_csv_if_exists(path: Path) -> pd.DataFrame:
        """
        Read CSV if it exists. Otherwise return empty DataFrame.
        """
        if path.exists():
            return pd.read_csv(path)

        return pd.DataFrame()

    @staticmethod
    def format_table(
        df: pd.DataFrame,
        columns: List[str] | None = None,
        max_rows: int = 20,
    ) -> str:
        """
        Format DataFrame as Markdown table.

        If DataFrame is empty, return a placeholder message.
        """
        if df.empty:
            return "_No available data._"

        display_df = df.copy()

        if columns is not None:
            available_columns = [
                column for column in columns
                if column in display_df.columns
            ]

            if available_columns:
                display_df = display_df[available_columns]

        display_df = display_df.head(max_rows)

        return display_df.to_markdown(index=False)

    @staticmethod
    def format_text_table(
        df: pd.DataFrame,
        columns: List[str] | None = None,
        max_rows: int = 20,
    ) -> str:
        """
        Format DataFrame as plain text table.
        """
        if df.empty:
            return "No available data."

        display_df = df.copy()

        if columns is not None:
            available_columns = [
                column for column in columns
                if column in display_df.columns
            ]

            if available_columns:
                display_df = display_df[available_columns]

        display_df = display_df.head(max_rows)

        return display_df.to_string(index=False)

    def load_report_inputs(
        self,
        table_dir: Path,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all available result tables.
        """
        inputs = {
            "best_model_summary": self.read_csv_if_exists(
                table_dir / "model_best_model_summary.csv"
            ),
            "leaderboard_validation": self.read_csv_if_exists(
                table_dir / "model_leaderboard_validation.csv"
            ),
            "leaderboard_test": self.read_csv_if_exists(
                table_dir / "model_leaderboard_test.csv"
            ),
            "leaderboard_all": self.read_csv_if_exists(
                table_dir / "model_leaderboard_all.csv"
            ),
            "model_comparison_metrics": self.read_csv_if_exists(
                table_dir / "model_comparison_metrics.csv"
            ),
            "model_comparison_roc_summary": self.read_csv_if_exists(
                table_dir / "model_comparison_roc_summary.csv"
            ),
            "calibration_metrics": self.read_csv_if_exists(
                table_dir / "model_calibration_metrics.csv"
            ),
            "dca_selected_threshold_summary": self.read_csv_if_exists(
                table_dir / "dca_selected_threshold_summary.csv"
            ),
            "logistic_metrics": self.read_csv_if_exists(
                table_dir / "logistic_baseline_metrics.csv"
            ),
            "random_forest_metrics": self.read_csv_if_exists(
                table_dir / "random_forest_baseline_metrics.csv"
            ),
            "gradient_boosting_metrics": self.read_csv_if_exists(
                table_dir / "gradient_boosting_baseline_metrics.csv"
            ),
        }

        return inputs

    def build_file_index(
        self,
        project_root: Path,
        table_dir: Path,
        figure_dir: Path,
        report_dir: Path,
        model_dir: Path,
        prediction_dir: Path,
    ) -> pd.DataFrame:
        """
        Build key output file index.
        """
        candidate_files = [
            ("table", "Model leaderboard all", table_dir / "model_leaderboard_all.csv"),
            ("table", "Validation leaderboard", table_dir / "model_leaderboard_validation.csv"),
            ("table", "Test leaderboard", table_dir / "model_leaderboard_test.csv"),
            ("table", "Best model summary", table_dir / "model_best_model_summary.csv"),
            ("table", "Model comparison metrics", table_dir / "model_comparison_metrics.csv"),
            ("table", "Model ROC comparison summary", table_dir / "model_comparison_roc_summary.csv"),
            ("table", "Calibration metrics", table_dir / "model_calibration_metrics.csv"),
            ("table", "Calibration curve points", table_dir / "model_calibration_curve_points.csv"),
            ("table", "DCA net benefit points", table_dir / "dca_net_benefit_points.csv"),
            ("table", "DCA selected threshold summary", table_dir / "dca_selected_threshold_summary.csv"),

            ("figure", "Validation ROC comparison", figure_dir / "model_comparison_validation_roc_curve.png"),
            ("figure", "Test ROC comparison", figure_dir / "model_comparison_test_roc_curve.png"),
            ("figure", "Validation calibration comparison", figure_dir / "model_comparison_validation_calibration_curve.png"),
            ("figure", "Test calibration comparison", figure_dir / "model_comparison_test_calibration_curve.png"),
            ("figure", "Validation DCA curve", figure_dir / "dca_validation_curve.png"),
            ("figure", "Test DCA curve", figure_dir / "dca_test_curve.png"),

            ("model", "Logistic Regression model", model_dir / "logistic_regression_baseline.joblib"),
            ("model", "Random Forest model", model_dir / "random_forest_baseline.joblib"),
            ("model", "Gradient Boosting model", model_dir / "gradient_boosting_baseline.joblib"),

            ("prediction", "Logistic validation predictions", prediction_dir / "logistic_val_predictions.csv"),
            ("prediction", "Logistic test predictions", prediction_dir / "logistic_test_predictions.csv"),
            ("prediction", "Random Forest validation predictions", prediction_dir / "random_forest_val_predictions.csv"),
            ("prediction", "Random Forest test predictions", prediction_dir / "random_forest_test_predictions.csv"),
            ("prediction", "Gradient Boosting validation predictions", prediction_dir / "gradient_boosting_val_predictions.csv"),
            ("prediction", "Gradient Boosting test predictions", prediction_dir / "gradient_boosting_test_predictions.csv"),

            ("report", "Model leaderboard report", report_dir / "model_leaderboard_report.txt"),
            ("report", "Calibration report", report_dir / "model_calibration_report.txt"),
            ("report", "DCA report", report_dir / "dca_report.txt"),
        ]

        rows = []

        for file_type, description, path in candidate_files:
            exists = path.exists()

            try:
                relative_path = path.relative_to(project_root)
            except ValueError:
                relative_path = path

            rows.append(
                {
                    "file_type": file_type,
                    "description": description,
                    "exists": exists,
                    "relative_path": str(relative_path).replace("\\", "/"),
                    "absolute_path": str(path),
                }
            )

        return pd.DataFrame(rows)

    def build_markdown_report(
        self,
        project_root: Path,
        inputs: Dict[str, pd.DataFrame],
        file_index: pd.DataFrame,
    ) -> str:
        """
        Build Markdown comprehensive report.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        best_model_columns = [
            "model_display_name",
            "model_name",
            "split",
            "auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "balanced_summary_score",
            "selection_rule",
        ]

        leaderboard_columns = [
            "rank_within_split",
            "model_display_name",
            "split",
            "auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "balanced_summary_score",
        ]

        calibration_columns = [
            "model_display_name",
            "split",
            "n_samples",
            "positive_rate",
            "brier_score",
            "n_bins_returned",
            "strategy",
        ]

        dca_columns = [
            "split",
            "selected_threshold",
            "rank_at_threshold",
            "model_display_name",
            "strategy",
            "net_benefit",
            "standardized_net_benefit",
            "tp",
            "fp",
            "tn",
            "fn",
        ]

        roc_summary_columns = [
            "split",
            "model_name",
            "roc_auc",
        ]

        existing_figures = file_index[
            (file_index["file_type"] == "figure")
            & (file_index["exists"] == True)
        ].copy()

        lines: List[str] = []

        lines.append(f"# {self.config.report_title}")
        lines.append("")
        lines.append(f"**Generated time:** {now}")
        lines.append("")
        lines.append(f"**Project root:** `{project_root}`")
        lines.append("")
        lines.append(f"**Target outcome:** `{self.config.target_name}`")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## 1. Executive Summary")
        lines.append("")
        lines.append(
            "This report summarizes the current model development results, including "
            "discrimination, calibration, decision curve analysis, and unified model ranking."
        )
        lines.append("")
        lines.append(
            "Model selection is based on validation-set performance. "
            "The test set should be treated as held-out final evaluation."
        )
        lines.append("")

        lines.append("### Current Best Model")
        lines.append("")
        lines.append(
            self.format_table(
                inputs["best_model_summary"],
                columns=best_model_columns,
                max_rows=5,
            )
        )
        lines.append("")

        lines.append("## 2. Validation Leaderboard")
        lines.append("")
        lines.append(
            self.format_table(
                inputs["leaderboard_validation"],
                columns=leaderboard_columns,
                max_rows=20,
            )
        )
        lines.append("")

        lines.append("## 3. Test Leaderboard")
        lines.append("")
        lines.append(
            self.format_table(
                inputs["leaderboard_test"],
                columns=leaderboard_columns,
                max_rows=20,
            )
        )
        lines.append("")

        lines.append("## 4. ROC Comparison Summary")
        lines.append("")
        lines.append(
            self.format_table(
                inputs["model_comparison_roc_summary"],
                columns=roc_summary_columns,
                max_rows=20,
            )
        )
        lines.append("")

        lines.append("## 5. Calibration Summary")
        lines.append("")
        lines.append(
            "Brier score measures the squared difference between predicted probabilities "
            "and true labels. Lower Brier score is better."
        )
        lines.append("")
        calibration_df = inputs["calibration_metrics"]

        if not calibration_df.empty and "brier_score" in calibration_df.columns:
            calibration_df = calibration_df.sort_values(
                by=["split", "brier_score"],
                ascending=[True, True],
            )

        lines.append(
            self.format_table(
                calibration_df,
                columns=calibration_columns,
                max_rows=30,
            )
        )
        lines.append("")

        lines.append("## 6. Decision Curve Analysis Summary")
        lines.append("")
        lines.append(
            "Decision Curve Analysis evaluates clinical net benefit across threshold probabilities. "
            "A model is useful at a given threshold if it has higher net benefit than Treat All and Treat None."
        )
        lines.append("")
        lines.append(
            self.format_table(
                inputs["dca_selected_threshold_summary"],
                columns=dca_columns,
                max_rows=80,
            )
        )
        lines.append("")

        lines.append("## 7. Key Figures")
        lines.append("")

        if existing_figures.empty:
            lines.append("_No figure files were found._")
        else:
            for _, row in existing_figures.iterrows():
                description = row["description"]
                relative_path = row["relative_path"]

                lines.append(f"### {description}")
                lines.append("")
                lines.append(f"![{description}](../{relative_path.split('outputs/', 1)[-1]})")
                lines.append("")
                lines.append(f"File: `{relative_path}`")
                lines.append("")

        lines.append("## 8. Key Output File Index")
        lines.append("")
        lines.append(
            self.format_table(
                file_index,
                columns=[
                    "file_type",
                    "description",
                    "exists",
                    "relative_path",
                ],
                max_rows=100,
            )
        )
        lines.append("")

        lines.append("## 9. Interpretation Notes")
        lines.append("")
        lines.append("- **AUC** evaluates discrimination, not calibration.")
        lines.append("- **Calibration** evaluates whether predicted probabilities match observed event rates.")
        lines.append("- **Brier score** is lower when probability predictions are more accurate.")
        lines.append("- **DCA** evaluates clinical net benefit at different risk thresholds.")
        lines.append("- **Validation set** should guide model selection.")
        lines.append("- **Test set** should remain a held-out final evaluation set.")
        lines.append("- Current results are based on simulated data and should not be interpreted as validated clinical evidence.")
        lines.append("")

        lines.append("## 10. Recommended Next Steps")
        lines.append("")
        lines.append("1. Add SHAP-based model interpretability.")
        lines.append("2. Generate manuscript-style result tables.")
        lines.append("3. Export this Markdown report to Word or PDF.")
        lines.append("4. Add external validation workflow when real or external data are available.")
        lines.append("")

        return "\n".join(lines)

    def build_text_report(
        self,
        project_root: Path,
        inputs: Dict[str, pd.DataFrame],
        file_index: pd.DataFrame,
    ) -> str:
        """
        Build plain text comprehensive report.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        best_model_columns = [
            "model_display_name",
            "model_name",
            "split",
            "auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "balanced_summary_score",
            "selection_rule",
        ]

        leaderboard_columns = [
            "rank_within_split",
            "model_display_name",
            "split",
            "auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "balanced_summary_score",
        ]

        calibration_columns = [
            "model_display_name",
            "split",
            "n_samples",
            "positive_rate",
            "brier_score",
            "n_bins_returned",
            "strategy",
        ]

        dca_columns = [
            "split",
            "selected_threshold",
            "rank_at_threshold",
            "model_display_name",
            "strategy",
            "net_benefit",
            "standardized_net_benefit",
            "tp",
            "fp",
            "tn",
            "fn",
        ]

        roc_summary_columns = [
            "split",
            "model_name",
            "roc_auc",
        ]

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(self.config.report_title)
        lines.append("=" * 80)
        lines.append(f"Generated time: {now}")
        lines.append(f"Project root: {project_root}")
        lines.append(f"Target outcome: {self.config.target_name}")
        lines.append("")

        lines.append("1. CURRENT BEST MODEL")
        lines.append("-" * 80)
        lines.append(
            self.format_text_table(
                inputs["best_model_summary"],
                columns=best_model_columns,
                max_rows=5,
            )
        )
        lines.append("")

        lines.append("2. VALIDATION LEADERBOARD")
        lines.append("-" * 80)
        lines.append(
            self.format_text_table(
                inputs["leaderboard_validation"],
                columns=leaderboard_columns,
                max_rows=20,
            )
        )
        lines.append("")

        lines.append("3. TEST LEADERBOARD")
        lines.append("-" * 80)
        lines.append(
            self.format_text_table(
                inputs["leaderboard_test"],
                columns=leaderboard_columns,
                max_rows=20,
            )
        )
        lines.append("")

        lines.append("4. ROC COMPARISON SUMMARY")
        lines.append("-" * 80)
        lines.append(
            self.format_text_table(
                inputs["model_comparison_roc_summary"],
                columns=roc_summary_columns,
                max_rows=20,
            )
        )
        lines.append("")

        lines.append("5. CALIBRATION SUMMARY")
        lines.append("-" * 80)

        calibration_df = inputs["calibration_metrics"]

        if not calibration_df.empty and "brier_score" in calibration_df.columns:
            calibration_df = calibration_df.sort_values(
                by=["split", "brier_score"],
                ascending=[True, True],
            )

        lines.append(
            self.format_text_table(
                calibration_df,
                columns=calibration_columns,
                max_rows=30,
            )
        )
        lines.append("")

        lines.append("6. DECISION CURVE ANALYSIS SUMMARY")
        lines.append("-" * 80)
        lines.append(
            self.format_text_table(
                inputs["dca_selected_threshold_summary"],
                columns=dca_columns,
                max_rows=80,
            )
        )
        lines.append("")

        lines.append("7. KEY OUTPUT FILE INDEX")
        lines.append("-" * 80)
        lines.append(
            self.format_text_table(
                file_index,
                columns=[
                    "file_type",
                    "description",
                    "exists",
                    "relative_path",
                ],
                max_rows=100,
            )
        )
        lines.append("")

        lines.append("8. INTERPRETATION NOTES")
        lines.append("-" * 80)
        lines.append("AUC evaluates discrimination, not calibration.")
        lines.append("Calibration evaluates whether predicted probabilities match observed event rates.")
        lines.append("DCA evaluates clinical net benefit at different risk thresholds.")
        lines.append("Validation set should guide model selection.")
        lines.append("Test set should remain a held-out final evaluation set.")
        lines.append("Current results are based on simulated data.")
        lines.append("=" * 80)

        return "\n".join(lines)


def save_comprehensive_report_outputs(
    markdown_report: str,
    text_report: str,
    file_index: pd.DataFrame,
    report_dir: Path,
    table_dir: Path,
) -> Dict[str, Path]:
    """
    Save comprehensive report outputs.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    markdown_file = report_dir / "comprehensive_model_evaluation_report.md"
    text_file = report_dir / "comprehensive_model_evaluation_report.txt"
    file_index_file = table_dir / "comprehensive_model_report_file_index.csv"

    markdown_file.write_text(markdown_report, encoding="utf-8")
    text_file.write_text(text_report, encoding="utf-8")
    file_index.to_csv(file_index_file, index=False, encoding="utf-8-sig")

    return {
        "markdown_report": markdown_file,
        "text_report": text_file,
        "file_index": file_index_file,
    }