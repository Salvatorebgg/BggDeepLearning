"""
BggDeepLearning gradient boosting baseline model

File:
python/bggdeep/models/traditional/gradient_boosting.py

Purpose:
1. Train a Gradient Boosting baseline model
2. Evaluate train / validation / test performance
3. Save metrics, predictions, feature importance, model, and report
4. Compare Gradient Boosting with Logistic Regression and Random Forest if available

Clinical task:
poor_outcome prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class GradientBoostingBaselineConfig:
    """
    Configuration for Gradient Boosting baseline.
    """

    model_name: str = "gradient_boosting_baseline"
    random_state: int = 42

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 2
    subsample: float = 0.9

    threshold: float = 0.5
    use_balanced_sample_weight: bool = True


class GradientBoostingBaselineTrainer:
    """
    Trainer for Gradient Boosting baseline model.
    """

    def __init__(self, config: GradientBoostingBaselineConfig) -> None:
        self.config = config
        self.model = GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            subsample=config.subsample,
            random_state=config.random_state,
        )

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting model.

        GradientBoostingClassifier does not have class_weight parameter.
        Therefore, we optionally use balanced sample weights.
        """
        sample_weight = None

        if self.config.use_balanced_sample_weight:
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

        self.model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
        )

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

    def build_feature_importance_table(
        self,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        Build Gradient Boosting feature importance table.
        """
        importances = self.model.feature_importances_

        table = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": importances,
            }
        )

        table = table.sort_values(
            by="importance",
            ascending=False,
        ).reset_index(drop=True)

        table["rank"] = range(1, len(table) + 1)

        return table[["rank", "feature_name", "importance"]]


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


def build_confusion_matrix_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build confusion matrix table from metrics.
    """
    rows = []

    for _, row in metrics_df.iterrows():
        rows.extend(
            [
                {"split": row["split"], "cell": "true_negative", "count": int(row["tn"])},
                {"split": row["split"], "cell": "false_positive", "count": int(row["fp"])},
                {"split": row["split"], "cell": "false_negative", "count": int(row["fn"])},
                {"split": row["split"], "cell": "true_positive", "count": int(row["tp"])},
            ]
        )

    return pd.DataFrame(rows)


def build_three_model_comparison_table(
    gradient_boosting_metrics: pd.DataFrame,
    table_dir: Path,
) -> pd.DataFrame:
    """
    Build model comparison table using available model metrics.

    It will include:
    1. Logistic Regression, if logistic_baseline_metrics.csv exists
    2. Random Forest, if random_forest_baseline_metrics.csv exists
    3. Gradient Boosting
    """
    metric_tables = []

    candidate_files = [
        table_dir / "logistic_baseline_metrics.csv",
        table_dir / "random_forest_baseline_metrics.csv",
    ]

    for file_path in candidate_files:
        if file_path.exists():
            metric_tables.append(pd.read_csv(file_path))

    metric_tables.append(gradient_boosting_metrics)

    comparison = pd.concat(
        metric_tables,
        ignore_index=True,
        sort=False,
    )

    comparison = comparison.sort_values(
        by=["split", "auc"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return comparison


def save_gradient_boosting_outputs(
    trainer: GradientBoostingBaselineTrainer,
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    model_comparison_df: pd.DataFrame,
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    output_model_dir: Path,
    output_table_dir: Path,
    output_prediction_dir: Path,
    output_report_dir: Path,
) -> Dict[str, Path]:
    """
    Save model, metrics, predictions, feature importance, comparison, and report.
    """
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_model_dir / "gradient_boosting_baseline.joblib"
    metrics_file = output_table_dir / "gradient_boosting_baseline_metrics.csv"
    confusion_file = output_table_dir / "gradient_boosting_baseline_confusion_matrix.csv"
    feature_importance_file = output_table_dir / "gradient_boosting_feature_importance.csv"
    model_comparison_file = output_table_dir / "model_comparison_metrics.csv"
    val_prediction_file = output_prediction_dir / "gradient_boosting_val_predictions.csv"
    test_prediction_file = output_prediction_dir / "gradient_boosting_test_predictions.csv"
    report_file = output_report_dir / "gradient_boosting_baseline_report.txt"
    comparison_report_file = output_report_dir / "model_comparison_report.txt"

    joblib.dump(trainer.model, model_file)

    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(confusion_file, index=False, encoding="utf-8-sig")
    feature_importance_df.to_csv(feature_importance_file, index=False, encoding="utf-8-sig")
    model_comparison_df.to_csv(model_comparison_file, index=False, encoding="utf-8-sig")
    val_predictions.to_csv(val_prediction_file, index=False, encoding="utf-8-sig")
    test_predictions.to_csv(test_prediction_file, index=False, encoding="utf-8-sig")

    report_file.write_text(
        build_gradient_boosting_report(
            metrics_df=metrics_df,
            confusion_df=confusion_df,
            feature_importance_df=feature_importance_df,
        ),
        encoding="utf-8",
    )

    comparison_report_file.write_text(
        build_model_comparison_report(model_comparison_df),
        encoding="utf-8",
    )

    return {
        "model_file": model_file,
        "metrics_file": metrics_file,
        "confusion_file": confusion_file,
        "feature_importance_file": feature_importance_file,
        "model_comparison_file": model_comparison_file,
        "val_prediction_file": val_prediction_file,
        "test_prediction_file": test_prediction_file,
        "report_file": report_file,
        "comparison_report_file": comparison_report_file,
    }


def build_gradient_boosting_report(
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
) -> str:
    """
    Build plain text report for Gradient Boosting baseline.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Gradient Boosting Baseline Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Model")
    lines.append("-" * 70)
    lines.append("Model: Gradient Boosting Classifier")
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
    lines.append("4. Top 30 Feature Importances")
    lines.append("-" * 70)
    lines.append(feature_importance_df.head(30).to_string(index=False))
    lines.append("")
    lines.append("5. Notes")
    lines.append("-" * 70)
    lines.append("Gradient Boosting is a sequential tree ensemble model.")
    lines.append("It can capture nonlinear relationships and complex feature interactions.")
    lines.append("Feature importance is model-specific and should not be interpreted as causal evidence.")
    lines.append("=" * 70)

    return "\n".join(lines)


def build_model_comparison_report(model_comparison_df: pd.DataFrame) -> str:
    """
    Build model comparison report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Model Comparison Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Compared models:")
    lines.append("- Logistic Regression baseline, if available")
    lines.append("- Random Forest baseline, if available")
    lines.append("- Gradient Boosting baseline")
    lines.append("")
    lines.append("Metrics:")
    lines.append("-" * 70)
    lines.append(model_comparison_df.to_string(index=False))
    lines.append("")
    lines.append("Notes:")
    lines.append("- Compare validation performance first for model selection.")
    lines.append("- Test performance should be treated as final held-out evaluation.")
    lines.append("- AUC measures discrimination, but calibration and DCA are still needed later.")
    lines.append("- If training AUC is much higher than validation/test AUC, overfitting may exist.")
    lines.append("=" * 70)

    return "\n".join(lines)