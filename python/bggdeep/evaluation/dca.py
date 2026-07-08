"""
BggDeepLearning Decision Curve Analysis module

File:
python/bggdeep/evaluation/dca.py

Purpose:
1. Load prediction files from available models
2. Calculate net benefit across threshold probabilities
3. Compare model strategies with treat-all and treat-none strategies
4. Plot DCA curves for validation and test sets
5. Save DCA tables and report

Clinical meaning:
Decision Curve Analysis evaluates whether using a prediction model
adds clinical value across different risk thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DCAConfig:
    """
    Configuration for Decision Curve Analysis.

    threshold_min:
        Minimum threshold probability.

    threshold_max:
        Maximum threshold probability.

    n_thresholds:
        Number of threshold points.

    selected_thresholds:
        Thresholds that will be highlighted in summary table.
    """

    threshold_min: float = 0.01
    threshold_max: float = 0.99
    n_thresholds: int = 99
    selected_thresholds: tuple[float, ...] = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
    )


@dataclass
class DCAPredictionSpec:
    """
    One prediction file specification.
    """

    model_key: str
    model_display_name: str
    split: str
    prediction_file: Path


class DecisionCurveAnalyzer:
    """
    Decision Curve Analysis calculator.
    """

    def __init__(self, config: DCAConfig) -> None:
        self.config = config

    def get_thresholds(self) -> np.ndarray:
        """
        Generate threshold probabilities.
        """
        return np.linspace(
            self.config.threshold_min,
            self.config.threshold_max,
            self.config.n_thresholds,
        )

    def load_prediction_file(self, prediction_file: Path) -> pd.DataFrame:
        """
        Load prediction CSV and validate required columns.
        """
        if not prediction_file.exists():
            raise FileNotFoundError(f"Prediction file was not found: {prediction_file}")

        df = pd.read_csv(prediction_file)

        required_columns = [
            "true_label",
            "predicted_probability",
            "predicted_label",
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Prediction file is missing required columns: {missing_columns}"
            )

        return df

    def calculate_model_net_benefit(
        self,
        y_true: pd.Series,
        y_probability: pd.Series,
        threshold: float,
    ) -> dict[str, float | int]:
        """
        Calculate net benefit for one model at one threshold.

        Formula:
            Net Benefit = TP / N - FP / N * threshold / (1 - threshold)
        """
        if threshold <= 0 or threshold >= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        y_true_array = y_true.astype(int).to_numpy()
        y_probability_array = y_probability.astype(float).to_numpy()

        y_pred = (y_probability_array >= threshold).astype(int)

        n = len(y_true_array)

        tp = int(((y_pred == 1) & (y_true_array == 1)).sum())
        fp = int(((y_pred == 1) & (y_true_array == 0)).sum())
        tn = int(((y_pred == 0) & (y_true_array == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_array == 1)).sum())

        threshold_odds = threshold / (1.0 - threshold)

        net_benefit = (tp / n) - (fp / n) * threshold_odds

        positive_rate = float(y_true_array.mean())

        standardized_net_benefit = (
            net_benefit / positive_rate
            if positive_rate > 0
            else np.nan
        )

        return {
            "threshold": round(float(threshold), 4),
            "net_benefit": round(float(net_benefit), 6),
            "standardized_net_benefit": round(float(standardized_net_benefit), 6)
            if not np.isnan(standardized_net_benefit)
            else np.nan,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_samples": int(n),
            "positive_rate": round(float(positive_rate), 6),
        }

    def calculate_treat_all_net_benefit(
        self,
        y_true: pd.Series,
        threshold: float,
    ) -> dict[str, float | int]:
        """
        Calculate net benefit for treat-all strategy.

        Treat-all means every patient is considered high risk.
        """
        if threshold <= 0 or threshold >= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        y_true_array = y_true.astype(int).to_numpy()
        n = len(y_true_array)

        prevalence = float(y_true_array.mean())
        threshold_odds = threshold / (1.0 - threshold)

        net_benefit = prevalence - (1.0 - prevalence) * threshold_odds

        standardized_net_benefit = (
            net_benefit / prevalence
            if prevalence > 0
            else np.nan
        )

        return {
            "threshold": round(float(threshold), 4),
            "net_benefit": round(float(net_benefit), 6),
            "standardized_net_benefit": round(float(standardized_net_benefit), 6)
            if not np.isnan(standardized_net_benefit)
            else np.nan,
            "tp": int(y_true_array.sum()),
            "fp": int(n - y_true_array.sum()),
            "tn": 0,
            "fn": 0,
            "n_samples": int(n),
            "positive_rate": round(float(prevalence), 6),
        }

    def calculate_treat_none_net_benefit(
        self,
        y_true: pd.Series,
        threshold: float,
    ) -> dict[str, float | int]:
        """
        Calculate net benefit for treat-none strategy.

        Treat-none means no patient is considered high risk.
        Net benefit is always 0.
        """
        y_true_array = y_true.astype(int).to_numpy()
        n = len(y_true_array)

        return {
            "threshold": round(float(threshold), 4),
            "net_benefit": 0.0,
            "standardized_net_benefit": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": int((y_true_array == 0).sum()),
            "fn": int((y_true_array == 1).sum()),
            "n_samples": int(n),
            "positive_rate": round(float(y_true_array.mean()), 6),
        }

    def analyze_one_prediction_file(
        self,
        spec: DCAPredictionSpec,
    ) -> pd.DataFrame:
        """
        Calculate DCA points for one model prediction file.
        """
        df = self.load_prediction_file(spec.prediction_file)

        y_true = df["true_label"]
        y_probability = df["predicted_probability"]

        rows = []

        for threshold in self.get_thresholds():
            values = self.calculate_model_net_benefit(
                y_true=y_true,
                y_probability=y_probability,
                threshold=float(threshold),
            )

            values.update(
                {
                    "strategy": "model",
                    "model_key": spec.model_key,
                    "model_display_name": spec.model_display_name,
                    "split": spec.split,
                    "prediction_file": str(spec.prediction_file),
                }
            )

            rows.append(values)

        return pd.DataFrame(rows)

    def analyze_reference_strategies(
        self,
        y_true: pd.Series,
        split: str,
    ) -> pd.DataFrame:
        """
        Calculate treat-all and treat-none DCA points.
        """
        rows = []

        for threshold in self.get_thresholds():
            treat_all_values = self.calculate_treat_all_net_benefit(
                y_true=y_true,
                threshold=float(threshold),
            )
            treat_all_values.update(
                {
                    "strategy": "treat_all",
                    "model_key": "treat_all",
                    "model_display_name": "Treat All",
                    "split": split,
                    "prediction_file": "",
                }
            )
            rows.append(treat_all_values)

            treat_none_values = self.calculate_treat_none_net_benefit(
                y_true=y_true,
                threshold=float(threshold),
            )
            treat_none_values.update(
                {
                    "strategy": "treat_none",
                    "model_key": "treat_none",
                    "model_display_name": "Treat None",
                    "split": split,
                    "prediction_file": "",
                }
            )
            rows.append(treat_none_values)

        return pd.DataFrame(rows)


def build_available_dca_specs(prediction_dir: Path) -> List[DCAPredictionSpec]:
    """
    Build DCA prediction specs for available models.
    """
    candidates = [
        {
            "model_key": "logistic",
            "model_display_name": "Logistic Regression",
            "split": "validation",
            "file_name": "logistic_val_predictions.csv",
        },
        {
            "model_key": "logistic",
            "model_display_name": "Logistic Regression",
            "split": "test",
            "file_name": "logistic_test_predictions.csv",
        },
        {
            "model_key": "random_forest",
            "model_display_name": "Random Forest",
            "split": "validation",
            "file_name": "random_forest_val_predictions.csv",
        },
        {
            "model_key": "random_forest",
            "model_display_name": "Random Forest",
            "split": "test",
            "file_name": "random_forest_test_predictions.csv",
        },
        {
            "model_key": "gradient_boosting",
            "model_display_name": "Gradient Boosting",
            "split": "validation",
            "file_name": "gradient_boosting_val_predictions.csv",
        },
        {
            "model_key": "gradient_boosting",
            "model_display_name": "Gradient Boosting",
            "split": "test",
            "file_name": "gradient_boosting_test_predictions.csv",
        },
    ]

    specs: List[DCAPredictionSpec] = []

    for item in candidates:
        prediction_file = prediction_dir / item["file_name"]

        if prediction_file.exists():
            specs.append(
                DCAPredictionSpec(
                    model_key=item["model_key"],
                    model_display_name=item["model_display_name"],
                    split=item["split"],
                    prediction_file=prediction_file,
                )
            )

    if not specs:
        raise FileNotFoundError(
            "No prediction files were found. Please train at least one model first."
        )

    return specs


def build_reference_dca_tables(
    analyzer: DecisionCurveAnalyzer,
    specs: List[DCAPredictionSpec],
) -> List[pd.DataFrame]:
    """
    Build reference DCA tables for each available split.

    The first available prediction file from each split is used to obtain y_true.
    """
    reference_tables = []

    used_splits = set()

    for spec in specs:
        if spec.split in used_splits:
            continue

        df = analyzer.load_prediction_file(spec.prediction_file)
        y_true = df["true_label"]

        reference_table = analyzer.analyze_reference_strategies(
            y_true=y_true,
            split=spec.split,
        )

        reference_tables.append(reference_table)
        used_splits.add(spec.split)

    return reference_tables


def plot_dca_curve(
    dca_points: pd.DataFrame,
    split: str,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot DCA curves for one split.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_df = dca_points[
        dca_points["split"] == split
    ].copy()

    if split_df.empty:
        raise ValueError(f"No DCA points found for split: {split}")

    plt.figure(figsize=(9, 7))

    for name, group_df in split_df.groupby("model_display_name"):
        group_df = group_df.sort_values(by="threshold")

        if name == "Treat None":
            plt.plot(
                group_df["threshold"],
                group_df["net_benefit"],
                linestyle="--",
                linewidth=1.5,
                label=name,
            )
        elif name == "Treat All":
            plt.plot(
                group_df["threshold"],
                group_df["net_benefit"],
                linestyle=":",
                linewidth=2,
                label=name,
            )
        else:
            plt.plot(
                group_df["threshold"],
                group_df["net_benefit"],
                linewidth=2,
                label=name,
            )

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_selected_threshold_summary(
    dca_points: pd.DataFrame,
    selected_thresholds: tuple[float, ...],
) -> pd.DataFrame:
    """
    Build summary table at selected threshold probabilities.

    It uses nearest available threshold point.
    """
    rows = []

    for split in sorted(dca_points["split"].unique()):
        split_df = dca_points[dca_points["split"] == split].copy()

        for threshold in selected_thresholds:
            threshold_values = split_df["threshold"].astype(float)
            nearest_index = (threshold_values - threshold).abs().idxmin()
            nearest_threshold = float(split_df.loc[nearest_index, "threshold"])

            threshold_df = split_df[
                split_df["threshold"].astype(float) == nearest_threshold
            ].copy()

            threshold_df = threshold_df.sort_values(
                by="net_benefit",
                ascending=False,
            ).reset_index(drop=True)

            threshold_df["selected_threshold"] = threshold
            threshold_df["nearest_available_threshold"] = nearest_threshold
            threshold_df["rank_at_threshold"] = range(1, len(threshold_df) + 1)

            rows.append(threshold_df)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True, sort=False)

    preferred_columns = [
        "split",
        "selected_threshold",
        "nearest_available_threshold",
        "rank_at_threshold",
        "model_display_name",
        "strategy",
        "net_benefit",
        "standardized_net_benefit",
        "tp",
        "fp",
        "tn",
        "fn",
        "n_samples",
        "positive_rate",
    ]

    existing_columns = [
        column for column in preferred_columns
        if column in result.columns
    ]

    return result[existing_columns]


def build_dca_report(
    dca_points: pd.DataFrame,
    selected_threshold_summary: pd.DataFrame,
    output_paths: Dict[str, Path],
) -> str:
    """
    Build plain text DCA report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Decision Curve Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. What is DCA?")
    lines.append("-" * 70)
    lines.append("Decision Curve Analysis evaluates clinical net benefit across threshold probabilities.")
    lines.append("It compares using a model against treating all patients or treating no patients.")
    lines.append("")
    lines.append("2. Net Benefit Formula")
    lines.append("-" * 70)
    lines.append("Net Benefit = TP / N - FP / N * threshold / (1 - threshold)")
    lines.append("")
    lines.append("3. Selected Threshold Summary")
    lines.append("-" * 70)

    display_columns = [
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

    existing_columns = [
        column for column in display_columns
        if column in selected_threshold_summary.columns
    ]

    if selected_threshold_summary.empty:
        lines.append("No selected threshold summary is available.")
    else:
        lines.append(
            selected_threshold_summary[existing_columns]
            .head(80)
            .to_string(index=False)
        )

    lines.append("")
    lines.append("4. Generated Files")
    lines.append("-" * 70)

    for name, path in output_paths.items():
        lines.append(f"{name}: {path}")

    lines.append("")
    lines.append("5. Interpretation Notes")
    lines.append("-" * 70)
    lines.append("- A model is clinically useful at a threshold if it has higher net benefit than Treat All and Treat None.")
    lines.append("- Threshold probability reflects the risk level at which clinical action would be considered.")
    lines.append("- DCA should be interpreted together with discrimination, calibration, and clinical feasibility.")
    lines.append("- This simulated dataset is for engineering demonstration only.")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_dca_outputs(
    dca_points: pd.DataFrame,
    selected_threshold_summary: pd.DataFrame,
    table_dir: Path,
    report_dir: Path,
    output_paths: Dict[str, Path],
) -> Dict[str, Path]:
    """
    Save DCA result tables and report.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dca_points_file = table_dir / "dca_net_benefit_points.csv"
    selected_summary_file = table_dir / "dca_selected_threshold_summary.csv"
    report_file = report_dir / "dca_report.txt"

    dca_points.to_csv(
        dca_points_file,
        index=False,
        encoding="utf-8-sig",
    )

    selected_threshold_summary.to_csv(
        selected_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    all_output_paths = dict(output_paths)
    all_output_paths["dca_net_benefit_points"] = dca_points_file
    all_output_paths["dca_selected_threshold_summary"] = selected_summary_file
    all_output_paths["dca_report"] = report_file

    report_text = build_dca_report(
        dca_points=dca_points,
        selected_threshold_summary=selected_threshold_summary,
        output_paths=all_output_paths,
    )

    report_file.write_text(report_text, encoding="utf-8")

    return all_output_paths