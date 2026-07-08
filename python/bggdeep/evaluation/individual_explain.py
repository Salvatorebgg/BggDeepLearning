"""
BggDeepLearning individual prediction explanation module

File:
python/bggdeep/evaluation/individual_explain.py

Purpose:
1. Load one trained tree-based model
2. Select one patient/sample from processed data
3. Predict individual risk probability
4. Compute SHAP values for that individual
5. Generate SHAP waterfall plot
6. Generate individual risk explanation table and text report

Clinical note:
Individual SHAP explains how the trained model made one prediction.
It does not prove causal relationships.
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
class IndividualExplanationConfig:
    """
    Individual explanation configuration.
    """

    model_key: str = "gradient_boosting"
    split: str = "test"
    sample_index: int | None = 0
    patient_id: str | None = None
    top_n: int = 20


@dataclass
class IndividualModelSpec:
    """
    Model file specification.
    """

    model_key: str
    model_display_name: str
    model_file_name: str


class IndividualPredictionExplainer:
    """
    Explain one individual prediction using SHAP.
    """

    SUPPORTED_MODELS: Dict[str, IndividualModelSpec] = {
        "random_forest": IndividualModelSpec(
            model_key="random_forest",
            model_display_name="Random Forest",
            model_file_name="random_forest_baseline.joblib",
        ),
        "gradient_boosting": IndividualModelSpec(
            model_key="gradient_boosting",
            model_display_name="Gradient Boosting",
            model_file_name="gradient_boosting_baseline.joblib",
        ),
    }

    def __init__(self, config: IndividualExplanationConfig) -> None:
        self.config = config

    def get_model_spec(self) -> IndividualModelSpec:
        """
        Get model spec by model key.
        """
        if self.config.model_key not in self.SUPPORTED_MODELS:
            supported = ", ".join(self.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Unsupported model_key: {self.config.model_key}. "
                f"Supported models: {supported}"
            )

        return self.SUPPORTED_MODELS[self.config.model_key]

    def load_model(self, model_dir: Path):
        """
        Load trained model.
        """
        spec = self.get_model_spec()
        model_file = model_dir / spec.model_file_name

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file was not found: {model_file}\n"
                "Please train the model first."
            )

        return joblib.load(model_file)

    def get_split_file_names(self) -> Dict[str, str]:
        """
        Get processed data file names for the selected split.
        """
        split = self.config.split

        if split not in ["train", "validation", "test"]:
            raise ValueError(
                "split must be one of: train, validation, test"
            )

        split_prefix = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }[split]

        return {
            "features": f"tabular_{split_prefix}_features.csv",
            "labels": f"tabular_{split_prefix}_labels.csv",
            "ids": f"tabular_{split_prefix}_ids.csv",
        }

    def load_split_data(self, processed_data_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load features, labels, and IDs for selected split.
        """
        file_names = self.get_split_file_names()

        feature_file = processed_data_dir / file_names["features"]
        label_file = processed_data_dir / file_names["labels"]
        id_file = processed_data_dir / file_names["ids"]

        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file was not found: {feature_file}")

        if not label_file.exists():
            raise FileNotFoundError(f"Label file was not found: {label_file}")

        if not id_file.exists():
            raise FileNotFoundError(f"ID file was not found: {id_file}")

        return {
            "features": pd.read_csv(feature_file),
            "labels": pd.read_csv(label_file),
            "ids": pd.read_csv(id_file),
        }

    def select_one_sample(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        ids: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, int]:
        """
        Select one sample by patient_id or sample_index.
        """
        if self.config.patient_id is not None:
            matched_index = self.find_patient_index(
                ids=ids,
                patient_id=self.config.patient_id,
            )
        else:
            if self.config.sample_index is None:
                raise ValueError(
                    "Either sample_index or patient_id must be provided."
                )

            matched_index = int(self.config.sample_index)

        if matched_index < 0 or matched_index >= len(features):
            raise IndexError(
                f"sample_index is out of range: {matched_index}. "
                f"Available range: 0 to {len(features) - 1}"
            )

        x_one = features.iloc[[matched_index]].copy()
        y_one = labels.iloc[matched_index].copy()
        id_one = ids.iloc[matched_index].copy()

        return x_one, y_one, id_one, matched_index

    @staticmethod
    def find_patient_index(ids: pd.DataFrame, patient_id: str) -> int:
        """
        Find row index by patient id.

        This function tries common ID columns:
        - study_patient_code
        - patient_id
        - encounter_code
        """
        candidate_columns = [
            "study_patient_code",
            "patient_id",
            "encounter_code",
        ]

        for column in candidate_columns:
            if column in ids.columns:
                matched = ids.index[ids[column].astype(str) == str(patient_id)].tolist()

                if matched:
                    return int(matched[0])

        raise ValueError(
            f"patient_id was not found in ID table: {patient_id}. "
            f"Available ID columns: {ids.columns.tolist()}"
        )

    @staticmethod
    def predict_probability(model, x_one: pd.DataFrame) -> float:
        """
        Predict positive class probability.
        """
        if not hasattr(model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba.")

        probability = model.predict_proba(x_one)[:, 1][0]

        return float(probability)

    @staticmethod
    def predict_label(probability: float, threshold: float = 0.5) -> int:
        """
        Convert probability into binary label.
        """
        return int(probability >= threshold)

    @staticmethod
    def compute_shap_values_for_one(model, x_one: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values for one sample.

        Handles common SHAP output formats:
        1. list[class0_values, class1_values]
        2. ndarray with shape (n_samples, n_features)
        3. ndarray with shape (n_samples, n_features, n_classes)
        """
        import shap

        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(x_one)

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

        if shap_values.ndim == 2:
            shap_one = shap_values[0]
        elif shap_values.ndim == 1:
            shap_one = shap_values
        else:
            raise ValueError(
                f"Unexpected SHAP value shape: {shap_values.shape}"
            )

        expected_value = explainer.expected_value

        if isinstance(expected_value, list):
            if len(expected_value) >= 2:
                base_value = expected_value[1]
            else:
                base_value = expected_value[0]
        else:
            expected_array = np.asarray(expected_value)

            if expected_array.ndim == 0:
                base_value = float(expected_array)
            elif expected_array.ndim >= 1 and len(expected_array) >= 2:
                base_value = float(expected_array[1])
            else:
                base_value = float(expected_array[0])

        return shap_one, float(base_value)

    @staticmethod
    def clean_feature_name(feature_name: str) -> str:
        """
        Convert processed feature names into more readable labels.
        """
        text = str(feature_name)

        replacements = {
            "num__": "",
            "cat__": "",
            "remainder__": "",
            "_": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def build_individual_explanation_table(
        self,
        x_one: pd.DataFrame,
        shap_one: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build individual SHAP explanation table.
        """
        rows = []

        for feature_name, feature_value, shap_value in zip(
            x_one.columns.tolist(),
            x_one.iloc[0].tolist(),
            shap_one.tolist(),
        ):
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_label": self.clean_feature_name(feature_name),
                    "feature_value": feature_value,
                    "shap_value": shap_value,
                    "abs_shap_value": abs(shap_value),
                    "risk_direction": "increase_risk" if shap_value > 0 else "decrease_risk",
                }
            )

        df = pd.DataFrame(rows)

        df = df.sort_values(
            by="abs_shap_value",
            ascending=False,
        ).reset_index(drop=True)

        df["rank"] = range(1, len(df) + 1)

        return df[
            [
                "rank",
                "feature_name",
                "feature_label",
                "feature_value",
                "shap_value",
                "abs_shap_value",
                "risk_direction",
            ]
        ]

    def plot_waterfall(
        self,
        shap_one: np.ndarray,
        base_value: float,
        x_one: pd.DataFrame,
        output_path: Path,
        title: str,
    ) -> None:
        """
        Plot SHAP waterfall plot for one sample.
        """
        import shap

        output_path.parent.mkdir(parents=True, exist_ok=True)

        explanation = shap.Explanation(
            values=shap_one,
            base_values=base_value,
            data=x_one.iloc[0].to_numpy(),
            feature_names=x_one.columns.tolist(),
        )

        plt.figure()
        shap.plots.waterfall(
            explanation,
            max_display=self.config.top_n,
            show=False,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def safe_text(value) -> str:
        """
        Convert value to safe text.
        """
        if pd.isna(value):
            return "NA"

        return str(value)

    def build_text_report(
        self,
        model_display_name: str,
        split: str,
        sample_index: int,
        id_one: pd.Series,
        y_one: pd.Series,
        probability: float,
        predicted_label: int,
        base_value: float,
        explanation_table: pd.DataFrame,
        table_file: Path,
        waterfall_file: Path,
    ) -> str:
        """
        Build individual risk explanation report.
        """
        outcome_value = "unknown"

        if "poor_outcome" in y_one.index:
            outcome_value = self.safe_text(y_one["poor_outcome"])
        elif len(y_one) > 0:
            outcome_value = self.safe_text(y_one.iloc[0])

        id_lines = []

        for key, value in id_one.items():
            id_lines.append(f"{key}: {self.safe_text(value)}")

        top_increase = explanation_table[
            explanation_table["shap_value"] > 0
        ].head(10)

        top_decrease = explanation_table[
            explanation_table["shap_value"] < 0
        ].head(10)

        lines = []
        lines.append("=" * 70)
        lines.append("BggDeepLearning Individual Risk Explanation Report")
        lines.append("=" * 70)
        lines.append("")
        lines.append("1. Sample Information")
        lines.append("-" * 70)
        lines.append(f"Model: {model_display_name}")
        lines.append(f"Split: {split}")
        lines.append(f"Sample index: {sample_index}")
        lines.extend(id_lines)
        lines.append(f"Observed outcome: {outcome_value}")
        lines.append("")
        lines.append("2. Prediction")
        lines.append("-" * 70)
        lines.append(f"Predicted probability of poor_outcome: {probability:.6f}")
        lines.append(f"Predicted label at threshold 0.5: {predicted_label}")
        lines.append(f"SHAP base value: {base_value:.6f}")
        lines.append("")
        lines.append("3. Top Features Increasing Predicted Risk")
        lines.append("-" * 70)

        if top_increase.empty:
            lines.append("No positive SHAP contributors were found.")
        else:
            lines.append(
                top_increase[
                    [
                        "rank",
                        "feature_label",
                        "feature_value",
                        "shap_value",
                    ]
                ].to_string(index=False)
            )

        lines.append("")
        lines.append("4. Top Features Decreasing Predicted Risk")
        lines.append("-" * 70)

        if top_decrease.empty:
            lines.append("No negative SHAP contributors were found.")
        else:
            lines.append(
                top_decrease[
                    [
                        "rank",
                        "feature_label",
                        "feature_value",
                        "shap_value",
                    ]
                ].to_string(index=False)
            )

        lines.append("")
        lines.append("5. Output Files")
        lines.append("-" * 70)
        lines.append(f"Explanation table: {table_file}")
        lines.append(f"Waterfall figure: {waterfall_file}")
        lines.append("")
        lines.append("6. Interpretation Notes")
        lines.append("-" * 70)
        lines.append("Positive SHAP values push this individual prediction toward higher risk.")
        lines.append("Negative SHAP values push this individual prediction toward lower risk.")
        lines.append("SHAP explains model behavior, not causal relationships.")
        lines.append("This report is generated from simulated data and is not clinical evidence.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def build_output_stem(self, sample_index: int) -> str:
        """
        Build output filename stem.
        """
        return (
            f"individual_{self.config.model_key}_"
            f"{self.config.split}_index_{sample_index}"
        )

    def explain(
        self,
        processed_data_dir: Path,
        model_dir: Path,
        table_dir: Path,
        figure_dir: Path,
        report_dir: Path,
    ) -> Dict[str, Path | float | int | str]:
        """
        Run complete individual prediction explanation.
        """
        spec = self.get_model_spec()
        model = self.load_model(model_dir)

        data = self.load_split_data(processed_data_dir)

        x_one, y_one, id_one, sample_index = self.select_one_sample(
            features=data["features"],
            labels=data["labels"],
            ids=data["ids"],
        )

        probability = self.predict_probability(model, x_one)
        predicted_label = self.predict_label(probability)

        shap_one, base_value = self.compute_shap_values_for_one(
            model=model,
            x_one=x_one,
        )

        explanation_table = self.build_individual_explanation_table(
            x_one=x_one,
            shap_one=shap_one,
        )

        output_stem = self.build_output_stem(sample_index)

        table_file = table_dir / f"{output_stem}_explanation.csv"
        waterfall_file = figure_dir / f"{output_stem}_waterfall.png"
        report_file = report_dir / f"{output_stem}_report.txt"

        table_file.parent.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        explanation_table.to_csv(
            table_file,
            index=False,
            encoding="utf-8-sig",
        )

        self.plot_waterfall(
            shap_one=shap_one,
            base_value=base_value,
            x_one=x_one,
            output_path=waterfall_file,
            title=f"{spec.model_display_name} Individual SHAP Waterfall",
        )

        report_text = self.build_text_report(
            model_display_name=spec.model_display_name,
            split=self.config.split,
            sample_index=sample_index,
            id_one=id_one,
            y_one=y_one,
            probability=probability,
            predicted_label=predicted_label,
            base_value=base_value,
            explanation_table=explanation_table,
            table_file=table_file,
            waterfall_file=waterfall_file,
        )

        report_file.write_text(report_text, encoding="utf-8")

        return {
            "model_key": self.config.model_key,
            "model_display_name": spec.model_display_name,
            "split": self.config.split,
            "sample_index": sample_index,
            "probability": probability,
            "predicted_label": predicted_label,
            "table_file": table_file,
            "waterfall_file": waterfall_file,
            "report_file": report_file,
        }