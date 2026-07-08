"""
BggDeepLearning model calibration evaluation module

File:
python/bggdeep/evaluation/calibration.py

Purpose:
1. Load model prediction files
2. Compute Brier score
3. Compute calibration curve points
4. Plot single-model calibration curves
5. Plot multi-model calibration comparison curves
6. Save calibration tables and report

Clinical meaning:
A model with good calibration gives predicted probabilities that match
observed event rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


@dataclass
class CalibrationConfig:
    """
    Configuration for calibration evaluation.
    """

    n_bins: int = 10
    strategy: str = "uniform"


@dataclass
class ModelPredictionSpec:
    """
    One model prediction file specification.
    """

    model_key: str
    model_display_name: str
    split: str
    prediction_file: Path


class CalibrationEvaluator:
    """
    Evaluate calibration for binary classification model predictions.
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

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

    def evaluate_one_model(
        self,
        spec: ModelPredictionSpec,
    ) -> Tuple[dict[str, object], pd.DataFrame]:
        """
        Evaluate one model prediction file.
        """
        df = self.load_prediction_file(spec.prediction_file)

        y_true = df["true_label"]
        y_prob = df["predicted_probability"]

        brier = brier_score_loss(y_true, y_prob)

        observed_rate, predicted_mean = calibration_curve(
            y_true,
            y_prob,
            n_bins=self.config.n_bins,
            strategy=self.config.strategy,
        )

        curve_points = pd.DataFrame(
            {
                "model_key": spec.model_key,
                "model_display_name": spec.model_display_name,
                "split": spec.split,
                "bin_index": range(1, len(observed_rate) + 1),
                "mean_predicted_probability": predicted_mean,
                "observed_event_rate": observed_rate,
            }
        )

        metrics = {
            "model_key": spec.model_key,
            "model_display_name": spec.model_display_name,
            "split": spec.split,
            "n_samples": int(len(df)),
            "positive_rate": round(float(y_true.mean()), 4),
            "brier_score": round(float(brier), 6),
            "n_bins_requested": self.config.n_bins,
            "n_bins_returned": int(len(observed_rate)),
            "strategy": self.config.strategy,
            "prediction_file": str(spec.prediction_file),
        }

        return metrics, curve_points


def build_available_prediction_specs(prediction_dir: Path) -> List[ModelPredictionSpec]:
    """
    Build prediction file specs for available baseline models.
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

    specs: List[ModelPredictionSpec] = []

    for item in candidates:
        prediction_file = prediction_dir / item["file_name"]

        if prediction_file.exists():
            specs.append(
                ModelPredictionSpec(
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


def plot_single_calibration_curve(
    curve_points: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot calibration curve for one model and one split.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(
        curve_points["mean_predicted_probability"],
        curve_points["observed_event_rate"],
        marker="o",
        linewidth=2,
        label="Model",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Perfect calibration",
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Event Rate")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_multi_model_calibration_curve(
    all_curve_points: pd.DataFrame,
    split: str,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot calibration curves of multiple models for one split.
    """
    split_points = all_curve_points[
        all_curve_points["split"] == split
    ].copy()

    if split_points.empty:
        raise ValueError(f"No calibration curve points found for split: {split}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))

    for model_name, model_df in split_points.groupby("model_display_name"):
        model_df = model_df.sort_values(by="mean_predicted_probability")

        plt.plot(
            model_df["mean_predicted_probability"],
            model_df["observed_event_rate"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Perfect calibration",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Event Rate")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_calibration_report(
    metrics_df: pd.DataFrame,
    output_paths: Dict[str, Path],
) -> str:
    """
    Build plain text calibration report.
    """
    validation_metrics = metrics_df[
        metrics_df["split"] == "validation"
    ].copy()

    test_metrics = metrics_df[
        metrics_df["split"] == "test"
    ].copy()

    validation_metrics = validation_metrics.sort_values(
        by="brier_score",
        ascending=True,
    )

    test_metrics = test_metrics.sort_values(
        by="brier_score",
        ascending=True,
    )

    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Model Calibration Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. What is calibration?")
    lines.append("-" * 70)
    lines.append("Calibration evaluates whether predicted probabilities match observed event rates.")
    lines.append("For example, among patients predicted to have 70% risk, about 70% should actually experience the outcome.")
    lines.append("")
    lines.append("2. Brier Score")
    lines.append("-" * 70)
    lines.append("Brier score measures the mean squared difference between predicted probabilities and true labels.")
    lines.append("Lower Brier score is better.")
    lines.append("")
    lines.append("3. Validation Calibration Metrics")
    lines.append("-" * 70)

    if validation_metrics.empty:
        lines.append("No validation calibration metrics are available.")
    else:
        lines.append(
            validation_metrics[
                [
                    "model_display_name",
                    "split",
                    "n_samples",
                    "positive_rate",
                    "brier_score",
                    "n_bins_returned",
                ]
            ].to_string(index=False)
        )

    lines.append("")
    lines.append("4. Test Calibration Metrics")
    lines.append("-" * 70)

    if test_metrics.empty:
        lines.append("No test calibration metrics are available.")
    else:
        lines.append(
            test_metrics[
                [
                    "model_display_name",
                    "split",
                    "n_samples",
                    "positive_rate",
                    "brier_score",
                    "n_bins_returned",
                ]
            ].to_string(index=False)
        )

    lines.append("")
    lines.append("5. Generated Files")
    lines.append("-" * 70)

    for name, path in output_paths.items():
        lines.append(f"{name}: {path}")

    lines.append("")
    lines.append("6. Interpretation Notes")
    lines.append("-" * 70)
    lines.append("- AUC evaluates discrimination, not calibration.")
    lines.append("- A model can have high AUC but poor calibration.")
    lines.append("- In clinical risk prediction, calibration is important because clinicians may act on predicted probabilities.")
    lines.append("- Lower Brier score indicates better overall probability accuracy.")
    lines.append("- Later steps will add calibration improvement and Decision Curve Analysis.")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_calibration_outputs(
    metrics_df: pd.DataFrame,
    curve_points_df: pd.DataFrame,
    table_dir: Path,
    report_dir: Path,
    output_paths: Dict[str, Path],
) -> Dict[str, Path]:
    """
    Save calibration tables and report.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = table_dir / "model_calibration_metrics.csv"
    curve_points_file = table_dir / "model_calibration_curve_points.csv"
    report_file = report_dir / "model_calibration_report.txt"

    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    curve_points_df.to_csv(curve_points_file, index=False, encoding="utf-8-sig")

    all_output_paths = dict(output_paths)
    all_output_paths["calibration_metrics"] = metrics_file
    all_output_paths["calibration_curve_points"] = curve_points_file
    all_output_paths["calibration_report"] = report_file

    report_text = build_calibration_report(
        metrics_df=metrics_df,
        output_paths=all_output_paths,
    )

    report_file.write_text(report_text, encoding="utf-8")

    return all_output_paths