"""
BggDeepLearning SHAP explainability module

File:
python/bggdeep/evaluation/shap_explain.py

Purpose:
1. Load trained tree-based models
2. Load processed test data
3. Compute SHAP values
4. Generate global SHAP importance tables
5. Generate SHAP bar plots and beeswarm plots

Clinical note:
SHAP explains model behavior, not causal effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SHAPExplainabilityConfig:
    """
    SHAP explainability configuration.
    """

    max_samples: int = 200
    random_state: int = 42
    top_n: int = 30


@dataclass
class SHAPModelSpec:
    """
    One model specification for SHAP explanation.
    """

    model_key: str
    model_display_name: str
    model_file: Path


class SHAPExplainabilityRunner:
    """
    Run SHAP explainability for trained models.
    """

    def __init__(self, config: SHAPExplainabilityConfig) -> None:
        self.config = config

    def load_test_features(self, processed_data_dir: Path) -> pd.DataFrame:
        """
        Load processed test features.
        """
        feature_file = processed_data_dir / "tabular_test_features.csv"

        if not feature_file.exists():
            raise FileNotFoundError(
                f"Test feature file was not found: {feature_file}\n"
                "Please run: python scripts\\python\\run_tabular_preprocessing.py"
            )

        return pd.read_csv(feature_file)

    def sample_features(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Sample test features to speed up SHAP.
        """
        if len(x_test) <= self.config.max_samples:
            return x_test.copy()

        return x_test.sample(
            n=self.config.max_samples,
            random_state=self.config.random_state,
        ).reset_index(drop=True)

    def load_model(self, model_file: Path):
        """
        Load trained model.
        """
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file was not found: {model_file}"
            )

        return joblib.load(model_file)

    def compute_shap_values(
        self,
        model,
        x_sample: pd.DataFrame,
    ) -> Tuple[np.ndarray, object]:
        """
        Compute SHAP values for binary classification.

        Handles common SHAP output formats:
        1. list[class0_values, class1_values]
        2. ndarray with shape (n_samples, n_features)
        3. ndarray with shape (n_samples, n_features, n_classes)
        """
        import shap

        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(x_sample)

        if isinstance(raw_values, list):
            if len(raw_values) >= 2:
                shap_values = raw_values[1]
            else:
                shap_values = raw_values[0]
        else:
            shap_values = raw_values

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            if shap_values.shape[2] >= 2:
                shap_values = shap_values[:, :, 1]
            else:
                shap_values = shap_values[:, :, 0]

        if shap_values.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP value shape: {shap_values.shape}"
            )

        return shap_values, explainer

    def build_global_importance_table(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        model_key: str,
        model_display_name: str,
    ) -> pd.DataFrame:
        """
        Build global SHAP importance table using mean absolute SHAP value.
        """
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {
                "model_key": model_key,
                "model_display_name": model_display_name,
                "feature_name": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )

        importance_df = importance_df.sort_values(
            by="mean_abs_shap",
            ascending=False,
        ).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df[
            [
                "model_key",
                "model_display_name",
                "rank",
                "feature_name",
                "mean_abs_shap",
            ]
        ]

    def plot_shap_bar(
        self,
        importance_df: pd.DataFrame,
        output_path: Path,
        title: str,
    ) -> None:
        """
        Plot simple SHAP global importance bar chart.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        top_df = importance_df.head(self.config.top_n).copy()
        top_df = top_df.sort_values(by="mean_abs_shap", ascending=True)

        plt.figure(figsize=(9, 8))
        plt.barh(
            top_df["feature_name"],
            top_df["mean_abs_shap"],
        )
        plt.xlabel("Mean absolute SHAP value")
        plt.ylabel("Feature")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def plot_shap_beeswarm(
        self,
        shap_values: np.ndarray,
        x_sample: pd.DataFrame,
        output_path: Path,
        title: str,
    ) -> None:
        """
        Plot SHAP beeswarm summary plot.
        """
        import shap

        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure()
        shap.summary_plot(
            shap_values,
            x_sample,
            max_display=self.config.top_n,
            show=False,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def explain_one_model(
        self,
        spec: SHAPModelSpec,
        x_sample: pd.DataFrame,
        table_dir: Path,
        figure_dir: Path,
    ) -> Dict[str, object]:
        """
        Run SHAP explanation for one model.
        """
        model = self.load_model(spec.model_file)

        shap_values, _ = self.compute_shap_values(
            model=model,
            x_sample=x_sample,
        )

        importance_df = self.build_global_importance_table(
            shap_values=shap_values,
            feature_names=x_sample.columns.tolist(),
            model_key=spec.model_key,
            model_display_name=spec.model_display_name,
        )

        importance_file = (
            table_dir / f"shap_{spec.model_key}_global_importance.csv"
        )

        bar_file = (
            figure_dir / f"shap_{spec.model_key}_bar.png"
        )

        beeswarm_file = (
            figure_dir / f"shap_{spec.model_key}_beeswarm.png"
        )

        importance_file.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(
            importance_file,
            index=False,
            encoding="utf-8-sig",
        )

        self.plot_shap_bar(
            importance_df=importance_df,
            output_path=bar_file,
            title=f"{spec.model_display_name} SHAP Global Importance",
        )

        self.plot_shap_beeswarm(
            shap_values=shap_values,
            x_sample=x_sample,
            output_path=beeswarm_file,
            title=f"{spec.model_display_name} SHAP Beeswarm Plot",
        )

        return {
            "model_key": spec.model_key,
            "model_display_name": spec.model_display_name,
            "importance_df": importance_df,
            "importance_file": importance_file,
            "bar_file": bar_file,
            "beeswarm_file": beeswarm_file,
            "n_samples_used": len(x_sample),
            "n_features": x_sample.shape[1],
        }


def build_available_shap_model_specs(model_dir: Path) -> list[SHAPModelSpec]:
    """
    Build available SHAP model specs.
    """
    candidates = [
        SHAPModelSpec(
            model_key="random_forest",
            model_display_name="Random Forest",
            model_file=model_dir / "random_forest_baseline.joblib",
        ),
        SHAPModelSpec(
            model_key="gradient_boosting",
            model_display_name="Gradient Boosting",
            model_file=model_dir / "gradient_boosting_baseline.joblib",
        ),
    ]

    specs = [
        spec for spec in candidates
        if spec.model_file.exists()
    ]

    if not specs:
        raise FileNotFoundError(
            "No tree-based model files were found. "
            "Please run Random Forest or Gradient Boosting training first."
        )

    return specs


def build_shap_report(
    results: list[Dict[str, object]],
    combined_importance_file: Path,
) -> str:
    """
    Build SHAP explainability report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning SHAP Explainability Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Purpose")
    lines.append("-" * 70)
    lines.append("This report summarizes global SHAP explanations for tree-based models.")
    lines.append("SHAP values quantify how much each feature contributes to model predictions.")
    lines.append("")
    lines.append("2. Models")
    lines.append("-" * 70)

    for result in results:
        lines.append(f"Model: {result['model_display_name']}")
        lines.append(f"Samples used: {result['n_samples_used']}")
        lines.append(f"Features: {result['n_features']}")
        lines.append(f"Importance table: {result['importance_file']}")
        lines.append(f"Bar figure: {result['bar_file']}")
        lines.append(f"Beeswarm figure: {result['beeswarm_file']}")
        lines.append("")

    lines.append("3. Top Features")
    lines.append("-" * 70)

    for result in results:
        lines.append(f"Model: {result['model_display_name']}")
        importance_df = result["importance_df"]
        lines.append(
            importance_df[
                [
                    "rank",
                    "feature_name",
                    "mean_abs_shap",
                ]
            ].head(20).to_string(index=False)
        )
        lines.append("")

    lines.append("4. Combined Output")
    lines.append("-" * 70)
    lines.append(f"Combined importance file: {combined_importance_file}")
    lines.append("")
    lines.append("5. Interpretation Notes")
    lines.append("-" * 70)
    lines.append("- Higher mean absolute SHAP value means stronger global model influence.")
    lines.append("- SHAP explains model behavior, not causal relationships.")
    lines.append("- One-hot encoded feature names may contain prefixes such as num__ or cat__.")
    lines.append("- Clinical interpretation should combine SHAP with domain knowledge and external validation.")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_shap_outputs(
    results: list[Dict[str, object]],
    table_dir: Path,
    report_dir: Path,
) -> Dict[str, Path]:
    """
    Save combined SHAP output table and report.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    combined_importance_file = (
        table_dir / "shap_model_global_importance_all.csv"
    )

    report_file = report_dir / "shap_explainability_report.txt"

    combined_importance = pd.concat(
        [result["importance_df"] for result in results],
        ignore_index=True,
        sort=False,
    )

    combined_importance.to_csv(
        combined_importance_file,
        index=False,
        encoding="utf-8-sig",
    )

    report_text = build_shap_report(
        results=results,
        combined_importance_file=combined_importance_file,
    )

    report_file.write_text(report_text, encoding="utf-8")

    return {
        "combined_importance_file": combined_importance_file,
        "report_file": report_file,
    }