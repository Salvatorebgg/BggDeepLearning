"""
BggDeepLearning model ranking and leaderboard module

File:
python/bggdeep/evaluation/model_ranking.py

Purpose:
1. Load metrics from available baseline models
2. Build unified model leaderboard
3. Rank models on validation and test sets
4. Select current best model based on validation performance
5. Save leaderboard tables and report

Important:
For model selection, validation performance should be used.
Test performance should be treated as held-out final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class ModelRankingConfig:
    """
    Configuration for model ranking.

    primary_metric:
        Main metric used for ranking models.

    secondary_metrics:
        Used to break ties or provide more balanced ranking.

    selection_split:
        The split used to select the current best model.
        Usually should be validation, not test.
    """

    primary_metric: str = "auc"
    secondary_metrics: List[str] = None
    selection_split: str = "validation"

    def __post_init__(self) -> None:
        if self.secondary_metrics is None:
            self.secondary_metrics = [
                "sensitivity",
                "specificity",
                "f1",
            ]


class ModelLeaderboardBuilder:
    """
    Build model leaderboard from saved metric CSV files.
    """

    def __init__(self, config: ModelRankingConfig) -> None:
        self.config = config

    def load_available_metrics(self, table_dir: Path) -> pd.DataFrame:
        """
        Load available model metric files.

        Missing model files will be skipped.
        """
        candidate_files = [
            {
                "model_group": "traditional_ml",
                "model_display_name": "Logistic Regression",
                "file_name": "logistic_baseline_metrics.csv",
            },
            {
                "model_group": "traditional_ml",
                "model_display_name": "Random Forest",
                "file_name": "random_forest_baseline_metrics.csv",
            },
            {
                "model_group": "traditional_ml",
                "model_display_name": "Gradient Boosting",
                "file_name": "gradient_boosting_baseline_metrics.csv",
            },
            {
                "model_group": "deep_learning",
                "model_display_name": "Deep MLP (PyTorch)",
                "file_name": "deep_mlp_baseline_metrics.csv",
            },
        ]

        metric_tables = []

        for item in candidate_files:
            metric_file = table_dir / item["file_name"]

            if not metric_file.exists():
                continue

            metrics = pd.read_csv(metric_file)
            metrics["model_group"] = item["model_group"]
            metrics["model_display_name"] = item["model_display_name"]
            metrics["metric_source_file"] = str(metric_file)

            metric_tables.append(metrics)

        if not metric_tables:
            raise FileNotFoundError(
                "No model metric files were found. Please train at least one model first."
            )

        all_metrics = pd.concat(
            metric_tables,
            ignore_index=True,
            sort=False,
        )

        return all_metrics

    def build_leaderboard(self, all_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Build ranked leaderboard for all available splits.
        """
        required_columns = [
            "model_name",
            "split",
            self.config.primary_metric,
        ]

        missing_columns = [
            column for column in required_columns
            if column not in all_metrics.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Metrics table is missing required columns: {missing_columns}"
            )

        leaderboard = all_metrics.copy()

        leaderboard["balanced_summary_score"] = self._compute_balanced_summary_score(
            leaderboard
        )

        sort_columns = [
            "split",
            self.config.primary_metric,
            "sensitivity",
            "specificity",
            "f1",
            "balanced_summary_score",
        ]

        available_sort_columns = [
            column for column in sort_columns
            if column in leaderboard.columns
        ]

        leaderboard = leaderboard.sort_values(
            by=available_sort_columns,
            ascending=[True] + [False] * (len(available_sort_columns) - 1),
        ).reset_index(drop=True)

        leaderboard["rank_within_split"] = (
            leaderboard
            .groupby("split")[self.config.primary_metric]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        preferred_columns = [
            "rank_within_split",
            "model_display_name",
            "model_name",
            "model_group",
            "split",
            "threshold",
            "auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "balanced_summary_score",
            "tn",
            "fp",
            "fn",
            "tp",
            "n_samples",
            "positive_rate",
            "metric_source_file",
        ]

        existing_columns = [
            column for column in preferred_columns
            if column in leaderboard.columns
        ]

        other_columns = [
            column for column in leaderboard.columns
            if column not in existing_columns
        ]

        return leaderboard[existing_columns + other_columns]

    def _compute_balanced_summary_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a simple summary score.

        This is not a formal clinical metric.
        It is only used to help sort and review models.

        Formula:
        0.50 * AUC
        0.20 * sensitivity
        0.20 * specificity
        0.10 * F1
        """
        score = pd.Series([0.0] * len(df), index=df.index)

        if "auc" in df.columns:
            score += 0.50 * df["auc"].fillna(0)

        if "sensitivity" in df.columns:
            score += 0.20 * df["sensitivity"].fillna(0)

        if "specificity" in df.columns:
            score += 0.20 * df["specificity"].fillna(0)

        if "f1" in df.columns:
            score += 0.10 * df["f1"].fillna(0)

        return score.round(4)

    def split_leaderboards(
        self,
        leaderboard: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build validation and test leaderboard tables.
        """
        result = {}

        for split_name in ["train", "validation", "test"]:
            split_df = leaderboard[
                leaderboard["split"] == split_name
            ].copy()

            if not split_df.empty:
                split_df = split_df.sort_values(
                    by=[
                        self.config.primary_metric,
                        "sensitivity",
                        "specificity",
                        "f1",
                    ],
                    ascending=[False, False, False, False],
                ).reset_index(drop=True)

                split_df["rank_within_split"] = range(1, len(split_df) + 1)

            result[split_name] = split_df

        return result

    def select_best_model(
        self,
        leaderboard: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select best model based on validation split.

        Test split is not used for model selection.
        """
        selection_split = self.config.selection_split

        selection_df = leaderboard[
            leaderboard["split"] == selection_split
        ].copy()

        if selection_df.empty:
            raise ValueError(
                f"No rows were found for selection split: {selection_split}"
            )

        selection_df = selection_df.sort_values(
            by=[
                self.config.primary_metric,
                "sensitivity",
                "specificity",
                "f1",
                "balanced_summary_score",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

        best_row = selection_df.iloc[[0]].copy()
        best_row["selection_rule"] = (
            f"Best model selected by {selection_split} "
            f"{self.config.primary_metric}, then sensitivity, specificity, and F1."
        )

        best_row["selection_warning"] = (
            "Do not select models based on repeated test-set comparisons. "
            "The test set should remain a held-out final evaluation set."
        )

        return best_row


def save_model_leaderboard_outputs(
    leaderboard: pd.DataFrame,
    split_tables: Dict[str, pd.DataFrame],
    best_model_summary: pd.DataFrame,
    table_dir: Path,
    report_dir: Path,
) -> Dict[str, Path]:
    """
    Save leaderboard outputs.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "leaderboard_all": table_dir / "model_leaderboard_all.csv",
        "leaderboard_validation": table_dir / "model_leaderboard_validation.csv",
        "leaderboard_test": table_dir / "model_leaderboard_test.csv",
        "best_model_summary": table_dir / "model_best_model_summary.csv",
        "report": report_dir / "model_leaderboard_report.txt",
    }

    leaderboard.to_csv(
        output_paths["leaderboard_all"],
        index=False,
        encoding="utf-8-sig",
    )

    split_tables.get("validation", pd.DataFrame()).to_csv(
        output_paths["leaderboard_validation"],
        index=False,
        encoding="utf-8-sig",
    )

    split_tables.get("test", pd.DataFrame()).to_csv(
        output_paths["leaderboard_test"],
        index=False,
        encoding="utf-8-sig",
    )

    best_model_summary.to_csv(
        output_paths["best_model_summary"],
        index=False,
        encoding="utf-8-sig",
    )

    report_text = build_model_leaderboard_report(
        leaderboard=leaderboard,
        split_tables=split_tables,
        best_model_summary=best_model_summary,
    )

    output_paths["report"].write_text(report_text, encoding="utf-8")

    return output_paths


def build_model_leaderboard_report(
    leaderboard: pd.DataFrame,
    split_tables: Dict[str, pd.DataFrame],
    best_model_summary: pd.DataFrame,
) -> str:
    """
    Build plain text model leaderboard report.
    """
    validation_table = split_tables.get("validation", pd.DataFrame())
    test_table = split_tables.get("test", pd.DataFrame())

    display_columns = [
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

    validation_columns = [
        column for column in display_columns
        if column in validation_table.columns
    ]

    test_columns = [
        column for column in display_columns
        if column in test_table.columns
    ]

    best_columns = [
        column for column in [
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
            "selection_warning",
        ]
        if column in best_model_summary.columns
    ]

    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Unified Model Leaderboard Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Current Best Model")
    lines.append("-" * 70)
    lines.append(best_model_summary[best_columns].to_string(index=False))
    lines.append("")
    lines.append("2. Validation Leaderboard")
    lines.append("-" * 70)

    if validation_table.empty:
        lines.append("No validation leaderboard is available.")
    else:
        lines.append(validation_table[validation_columns].to_string(index=False))

    lines.append("")
    lines.append("3. Test Leaderboard")
    lines.append("-" * 70)

    if test_table.empty:
        lines.append("No test leaderboard is available.")
    else:
        lines.append(test_table[test_columns].to_string(index=False))

    lines.append("")
    lines.append("4. All Available Metrics")
    lines.append("-" * 70)
    lines.append(f"Total metric rows: {len(leaderboard)}")
    lines.append(f"Available models: {leaderboard['model_display_name'].nunique()}")
    lines.append("")
    lines.append("5. Interpretation Notes")
    lines.append("-" * 70)
    lines.append("- Validation performance should be used for model selection.")
    lines.append("- Test performance should be used as held-out final evaluation.")
    lines.append("- AUC evaluates discrimination, but it does not assess calibration.")
    lines.append("- Sensitivity is important when missing high-risk patients is costly.")
    lines.append("- Specificity is important when false alarms create clinical burden.")
    lines.append("- Calibration and DCA will be added in later steps.")
    lines.append("=" * 70)

    return "\n".join(lines)