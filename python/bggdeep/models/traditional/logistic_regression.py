"""
BggDeepLearning logistic regression baseline model

File:
python/bggdeep/models/traditional/logistic_regression.py

Purpose:
1. Train a logistic regression baseline model
2. Evaluate train / validation / test performance
3. Save metrics, predictions, model, and report

Clinical task:
poor_outcome prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class LogisticBaselineConfig:
    """
    Configuration for logistic regression baseline.
    """

    model_name: str = "logistic_regression_baseline"
    random_state: int = 42
    max_iter: int = 2000
    class_weight: str | None = "balanced"
    threshold: float = 0.5


class LogisticBaselineTrainer:
    """
    Trainer for logistic regression baseline model.
    """

    def __init__(self, config: LogisticBaselineConfig) -> None:
        self.config = config
        self.model = LogisticRegression(
            max_iter=config.max_iter,
            random_state=config.random_state,
            class_weight=config.class_weight,
            solver="lbfgs",
        )

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> LogisticRegression:
        """
        Train logistic regression model.
        """
        self.model.fit(x_train, y_train)
        return self.model

    def predict_probability(self, x: pd.DataFrame) -> pd.Series:
        """
        Predict positive class probability.
        """
        probabilities = self.model.predict_proba(x)[:, 1]
        return pd.Series(probabilities, name="predicted_probability")

    def predict_label(self, probabilities: pd.Series) -> pd.Series:
        """
        Convert probabilities to binary labels.
        """
        labels = (probabilities >= self.config.threshold).astype(int)
        return pd.Series(labels, name="predicted_label")

    def evaluate(
        self,
        y_true: pd.Series,
        predicted_probability: pd.Series,
        predicted_label: pd.Series,
        split_name: str,
    ) -> Dict[str, float | str | int]:
        """
        Evaluate model performance.
        """
        auc = roc_auc_score(y_true, predicted_probability)
        accuracy = accuracy_score(y_true, predicted_label)
        precision = precision_score(y_true, predicted_label, zero_division=0)
        recall = recall_score(y_true, predicted_label, zero_division=0)
        f1 = f1_score(y_true, predicted_label, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predicted_label,
            labels=[0, 1],
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "model_name": self.config.model_name,
            "split": split_name,
            "threshold": self.config.threshold,
            "auc": round(float(auc), 4),
            "accuracy": round(float(accuracy), 4),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "n_samples": int(len(y_true)),
            "positive_rate": round(float(y_true.mean()), 4),
        }

    def evaluate_split(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        split_name: str,
    ) -> Tuple[Dict[str, float | str | int], pd.DataFrame]:
        """
        Predict and evaluate one dataset split.
        """
        probabilities = self.predict_probability(x)
        labels = self.predict_label(probabilities)

        metrics = self.evaluate(
            y_true=y,
            predicted_probability=probabilities,
            predicted_label=labels,
            split_name=split_name,
        )

        predictions = pd.DataFrame(
            {
                "split": split_name,
                "true_label": y.reset_index(drop=True),
                "predicted_probability": probabilities.reset_index(drop=True),
                "predicted_label": labels.reset_index(drop=True),
            }
        )

        return metrics, predictions


def load_processed_tabular_data(processed_data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load processed train / validation / test datasets.
    """
    files = {
        "x_train": processed_data_dir / "tabular_train_features.csv",
        "x_val": processed_data_dir / "tabular_val_features.csv",
        "x_test": processed_data_dir / "tabular_test_features.csv",
        "y_train": processed_data_dir / "tabular_train_labels.csv",
        "y_val": processed_data_dir / "tabular_val_labels.csv",
        "y_test": processed_data_dir / "tabular_test_labels.csv",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required processed data file was not found: {path}\n"
                "Please run: python scripts\\python\\run_tabular_preprocessing.py"
            )

    return {
        name: pd.read_csv(path)
        for name, path in files.items()
    }


def save_logistic_outputs(
    trainer: LogisticBaselineTrainer,
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    output_model_dir: Path,
    output_table_dir: Path,
    output_prediction_dir: Path,
    output_report_dir: Path,
) -> Dict[str, Path]:
    """
    Save model, metrics, predictions, and report.
    """
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_model_dir / "logistic_regression_baseline.joblib"
    metrics_file = output_table_dir / "logistic_baseline_metrics.csv"
    confusion_file = output_table_dir / "logistic_baseline_confusion_matrix.csv"
    val_prediction_file = output_prediction_dir / "logistic_val_predictions.csv"
    test_prediction_file = output_prediction_dir / "logistic_test_predictions.csv"
    report_file = output_report_dir / "logistic_baseline_report.txt"

    joblib.dump(trainer.model, model_file)

    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(confusion_file, index=False, encoding="utf-8-sig")
    val_predictions.to_csv(val_prediction_file, index=False, encoding="utf-8-sig")
    test_predictions.to_csv(test_prediction_file, index=False, encoding="utf-8-sig")

    report_text = build_logistic_report(metrics_df, confusion_df)
    report_file.write_text(report_text, encoding="utf-8")

    return {
        "model_file": model_file,
        "metrics_file": metrics_file,
        "confusion_file": confusion_file,
        "val_prediction_file": val_prediction_file,
        "test_prediction_file": test_prediction_file,
        "report_file": report_file,
    }


def build_confusion_matrix_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build confusion matrix table from metrics.
    """
    rows = []

    for _, row in metrics_df.iterrows():
        rows.extend(
            [
                {
                    "split": row["split"],
                    "cell": "true_negative",
                    "count": int(row["tn"]),
                },
                {
                    "split": row["split"],
                    "cell": "false_positive",
                    "count": int(row["fp"]),
                },
                {
                    "split": row["split"],
                    "cell": "false_negative",
                    "count": int(row["fn"]),
                },
                {
                    "split": row["split"],
                    "cell": "true_positive",
                    "count": int(row["tp"]),
                },
            ]
        )

    return pd.DataFrame(rows)


def build_logistic_report(
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> str:
    """
    Build plain text report for logistic regression baseline.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Logistic Regression Baseline Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Model")
    lines.append("-" * 70)
    lines.append("Model: Logistic Regression")
    lines.append("Task: poor_outcome prediction")
    lines.append("Threshold: 0.5")
    lines.append("")
    lines.append("2. Metrics")
    lines.append("-" * 70)
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("3. Confusion Matrix Cells")
    lines.append("-" * 70)
    lines.append(confusion_df.to_string(index=False))
    lines.append("")
    lines.append("4. Notes")
    lines.append("-" * 70)
    lines.append("This is the first baseline model.")
    lines.append("The model uses preprocessed train features and labels.")
    lines.append("Validation and test sets were transformed by the training-fitted preprocessor.")
    lines.append("Potential leakage columns were excluded during preprocessing.")
    lines.append("=" * 70)

    return "\n".join(lines)