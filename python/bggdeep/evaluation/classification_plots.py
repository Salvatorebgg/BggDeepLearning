"""
BggDeepLearning classification plotting utilities

File:
python/bggdeep/evaluation/classification_plots.py

Purpose:
1. Plot ROC curve
2. Plot Precision-Recall curve
3. Plot confusion matrix
4. Plot feature importance
5. Save model evaluation figures

This module uses matplotlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_roc_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    output_path: Path,
    title: str,
) -> float:
    """
    Plot ROC curve and return AUC.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Reference")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return float(roc_auc)


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    output_path: Path,
    title: str,
) -> float:
    """
    Plot Precision-Recall curve and return average precision.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {average_precision:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return float(average_precision)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    title: str,
) -> Dict[str, int]:
    """
    Plot confusion matrix and return TN/FP/FN/TP.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_feature_importance(
    feature_importance_file: Path,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Plot top N feature importances.

    Expected input columns:
    - feature_name
    - importance
    """
    if not feature_importance_file.exists():
        raise FileNotFoundError(
            f"Feature importance file was not found: {feature_importance_file}\n"
            "Please run: python scripts\\python\\train_random_forest_baseline.py"
        )

    feature_importance_df = pd.read_csv(feature_importance_file)

    required_columns = ["feature_name", "importance"]

    missing_columns = [
        column for column in required_columns
        if column not in feature_importance_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Feature importance file is missing required columns: {missing_columns}"
        )

    top_features = (
        feature_importance_df
        .sort_values(by="importance", ascending=False)
        .head(top_n)
        .copy()
    )

    top_features = top_features.sort_values(by="importance", ascending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.barh(
        top_features["feature_name"],
        top_features["importance"],
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return top_features.sort_values(by="importance", ascending=False).reset_index(drop=True)


def load_prediction_file(prediction_file: Path) -> pd.DataFrame:
    """
    Load prediction CSV file.
    """
    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file was not found: {prediction_file}\n"
            "Please run the corresponding model training script first."
        )

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


def plot_binary_classification_evaluation(
    prediction_file: Path,
    figure_dir: Path,
    model_prefix: str,
    split_name: str,
) -> Dict[str, object]:
    """
    Plot ROC curve, PR curve, and confusion matrix for one split.
    """
    predictions = load_prediction_file(prediction_file)

    y_true = predictions["true_label"]
    y_score = predictions["predicted_probability"]
    y_pred = predictions["predicted_label"]

    roc_file = figure_dir / f"{model_prefix}_{split_name}_roc_curve.png"
    pr_file = figure_dir / f"{model_prefix}_{split_name}_pr_curve.png"
    confusion_file = figure_dir / f"{model_prefix}_{split_name}_confusion_matrix.png"

    roc_auc = plot_roc_curve(
        y_true=y_true,
        y_score=y_score,
        output_path=roc_file,
        title=f"{model_prefix} {split_name} ROC Curve",
    )

    average_precision = plot_precision_recall_curve(
        y_true=y_true,
        y_score=y_score,
        output_path=pr_file,
        title=f"{model_prefix} {split_name} Precision-Recall Curve",
    )

    confusion_values = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=confusion_file,
        title=f"{model_prefix} {split_name} Confusion Matrix",
    )

    result = {
        "split": split_name,
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(average_precision, 4),
        "roc_file": roc_file,
        "pr_file": pr_file,
        "confusion_file": confusion_file,
    }

    result.update(confusion_values)

    return result


def build_plot_report(
    plot_results: List[Dict[str, object]],
) -> str:
    """
    Build plain text report for Logistic baseline figures.

    This function is kept for compatibility with Step 22.
    """
    return build_model_plot_report(
        plot_results=plot_results,
        model_display_name="Logistic Baseline",
        extra_lines=[
            "This report was generated by the generic classification plotting module.",
        ],
    )


def build_model_plot_report(
    plot_results: List[Dict[str, object]],
    model_display_name: str,
    extra_lines: List[str] | None = None,
) -> str:
    """
    Build plain text report for model figures.
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"BggDeepLearning {model_display_name} Plot Report")
    lines.append("=" * 70)
    lines.append("")

    for result in plot_results:
        lines.append(f"Split: {result['split']}")
        lines.append("-" * 70)
        lines.append(f"ROC AUC: {result['roc_auc']}")
        lines.append(f"Average Precision: {result['average_precision']}")
        lines.append(f"TN: {result['tn']}")
        lines.append(f"FP: {result['fp']}")
        lines.append(f"FN: {result['fn']}")
        lines.append(f"TP: {result['tp']}")
        lines.append(f"ROC figure: {result['roc_file']}")
        lines.append(f"PR figure: {result['pr_file']}")
        lines.append(f"Confusion matrix figure: {result['confusion_file']}")
        lines.append("")

    lines.append("Notes:")
    lines.append("- ROC curve evaluates discrimination across thresholds.")
    lines.append("- PR curve is useful when the positive class is relatively uncommon.")
    lines.append("- Confusion matrix uses the predicted_label saved during model training.")

    if extra_lines:
        for line in extra_lines:
            lines.append(f"- {line}")

    lines.append("=" * 70)

    return "\n".join(lines)
def plot_multi_model_roc_comparison(
    prediction_files: Dict[str, Path],
    output_path: Path,
    title: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Plot ROC curves for multiple models in one figure.

    prediction_files:
        Dictionary where key is model display name and value is prediction CSV path.

    Expected prediction CSV columns:
    - true_label
    - predicted_probability
    - predicted_label
    """
    rows = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))

    for model_name, prediction_file in prediction_files.items():
        if not prediction_file.exists():
            continue

        predictions = load_prediction_file(prediction_file)

        y_true = predictions["true_label"]
        y_score = predictions["predicted_probability"]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} AUC = {roc_auc:.4f}",
        )

        rows.append(
            {
                "split": split_name,
                "model_name": model_name,
                "prediction_file": prediction_file,
                "roc_auc": round(float(roc_auc), 4),
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No valid prediction files were found for ROC comparison."
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Reference",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    result_df = pd.DataFrame(rows)
    result_df["roc_figure"] = str(output_path)

    return result_df