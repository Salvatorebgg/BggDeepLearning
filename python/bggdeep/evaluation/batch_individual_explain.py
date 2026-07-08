"""
BggDeepLearning batch individual prediction explanation module

File:
python/bggdeep/evaluation/batch_individual_explain.py

Purpose:
1. Load model prediction results
2. Select Top N high-risk patients
3. Run individual SHAP explanation for each selected sample
4. Generate individual waterfall plots and reports
5. Save batch candidate table, output index table, and batch report

Clinical note:
This module explains model predictions for selected high-risk samples.
It does not provide causal interpretation or validated clinical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from bggdeep.evaluation.individual_explain import (
    IndividualExplanationConfig,
    IndividualPredictionExplainer,
)


@dataclass
class BatchHighRiskExplanationConfig:
    """
    Batch high-risk explanation configuration.
    """

    model_key: str = "gradient_boosting"
    split: str = "test"
    top_n_patients: int = 5
    risk_threshold: float | None = None
    top_n_features: int = 20


class BatchHighRiskExplainer:
    """
    Batch explainer for high-risk patients.
    """

    MODEL_DISPLAY_NAMES: Dict[str, str] = {
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }

    PREDICTION_FILES: Dict[str, Dict[str, str]] = {
        "random_forest": {
            "validation": "random_forest_val_predictions.csv",
            "test": "random_forest_test_predictions.csv",
        },
        "gradient_boosting": {
            "validation": "gradient_boosting_val_predictions.csv",
            "test": "gradient_boosting_test_predictions.csv",
        },
    }

    SPLIT_PREFIX: Dict[str, str] = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    def __init__(self, config: BatchHighRiskExplanationConfig) -> None:
        self.config = config

    def validate_config(self) -> None:
        """
        Validate config values.
        """
        if self.config.model_key not in self.MODEL_DISPLAY_NAMES:
            supported = ", ".join(self.MODEL_DISPLAY_NAMES.keys())
            raise ValueError(
                f"Unsupported model_key: {self.config.model_key}. "
                f"Supported models: {supported}"
            )

        if self.config.split not in ["validation", "test"]:
            raise ValueError(
                "For batch high-risk explanation, split must be validation or test."
            )

        if self.config.top_n_patients <= 0:
            raise ValueError("top_n_patients must be positive.")

        if self.config.risk_threshold is not None:
            if self.config.risk_threshold < 0 or self.config.risk_threshold > 1:
                raise ValueError("risk_threshold must be between 0 and 1.")

    def get_prediction_file(self, prediction_dir: Path) -> Path:
        """
        Get prediction file path for selected model and split.
        """
        self.validate_config()

        file_name = self.PREDICTION_FILES[self.config.model_key][self.config.split]
        prediction_file = prediction_dir / file_name

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Prediction file was not found: {prediction_file}\n"
                "Please train the selected model first."
            )

        return prediction_file

    def load_prediction_table(self, prediction_dir: Path) -> pd.DataFrame:
        """
        Load prediction table.
        """
        prediction_file = self.get_prediction_file(prediction_dir)
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
                f"Prediction table is missing required columns: {missing_columns}"
            )

        df = df.copy()
        df["sample_index"] = range(len(df))
        df["prediction_file"] = str(prediction_file)

        return df

    def load_ids_and_labels(
        self,
        processed_data_dir: Path,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load ID and label tables for selected split.
        """
        split_prefix = self.SPLIT_PREFIX[self.config.split]

        id_file = processed_data_dir / f"tabular_{split_prefix}_ids.csv"
        label_file = processed_data_dir / f"tabular_{split_prefix}_labels.csv"

        if not id_file.exists():
            raise FileNotFoundError(f"ID file was not found: {id_file}")

        if not label_file.exists():
            raise FileNotFoundError(f"Label file was not found: {label_file}")

        return {
            "ids": pd.read_csv(id_file),
            "labels": pd.read_csv(label_file),
        }

    def build_candidate_table(
        self,
        prediction_dir: Path,
        processed_data_dir: Path,
    ) -> pd.DataFrame:
        """
        Build high-risk candidate table.
        """
        predictions = self.load_prediction_table(prediction_dir)
        id_label_data = self.load_ids_and_labels(processed_data_dir)

        ids = id_label_data["ids"].copy()
        labels = id_label_data["labels"].copy()

        if len(predictions) != len(ids):
            raise ValueError(
                "Prediction table and ID table have different row counts. "
                f"Predictions: {len(predictions)}, IDs: {len(ids)}"
            )

        if len(predictions) != len(labels):
            raise ValueError(
                "Prediction table and label table have different row counts. "
                f"Predictions: {len(predictions)}, Labels: {len(labels)}"
            )

        candidate_table = predictions.copy()

        for column in ids.columns:
            candidate_table[column] = ids[column]

        if "poor_outcome" in labels.columns:
            candidate_table["observed_poor_outcome"] = labels["poor_outcome"]
        elif labels.shape[1] > 0:
            candidate_table["observed_poor_outcome"] = labels.iloc[:, 0]

        if self.config.risk_threshold is not None:
            candidate_table = candidate_table[
                candidate_table["predicted_probability"] >= self.config.risk_threshold
            ].copy()

        candidate_table = candidate_table.sort_values(
            by="predicted_probability",
            ascending=False,
        ).reset_index(drop=True)

        candidate_table["risk_rank"] = range(1, len(candidate_table) + 1)

        candidate_table = candidate_table.head(self.config.top_n_patients).copy()

        preferred_columns = [
            "risk_rank",
            "sample_index",
            "predicted_probability",
            "predicted_label",
            "true_label",
            "observed_poor_outcome",
            "study_patient_code",
            "patient_id",
            "encounter_code",
            "split",
            "prediction_file",
        ]

        existing_columns = [
            column for column in preferred_columns
            if column in candidate_table.columns
        ]

        other_columns = [
            column for column in candidate_table.columns
            if column not in existing_columns
        ]

        return candidate_table[existing_columns + other_columns]

    def build_output_stem(self) -> str:
        """
        Build batch output filename stem.
        """
        threshold_part = ""

        if self.config.risk_threshold is not None:
            threshold_text = str(self.config.risk_threshold).replace(".", "p")
            threshold_part = f"_threshold_{threshold_text}"

        return (
            f"batch_high_risk_{self.config.model_key}_"
            f"{self.config.split}_top{self.config.top_n_patients}"
            f"{threshold_part}"
        )

    def run_individual_explanations(
        self,
        candidate_table: pd.DataFrame,
        processed_data_dir: Path,
        model_dir: Path,
        table_dir: Path,
        figure_dir: Path,
        report_dir: Path,
    ) -> pd.DataFrame:
        """
        Run individual explanation for every selected candidate.
        """
        rows: List[dict] = []

        for _, candidate in candidate_table.iterrows():
            sample_index = int(candidate["sample_index"])

            individual_config = IndividualExplanationConfig(
                model_key=self.config.model_key,
                split=self.config.split,
                sample_index=sample_index,
                patient_id=None,
                top_n=self.config.top_n_features,
            )

            individual_explainer = IndividualPredictionExplainer(
                individual_config
            )

            result = individual_explainer.explain(
                processed_data_dir=processed_data_dir,
                model_dir=model_dir,
                table_dir=table_dir,
                figure_dir=figure_dir,
                report_dir=report_dir,
            )

            row = {
                "risk_rank": int(candidate["risk_rank"]),
                "sample_index": sample_index,
                "model_key": result["model_key"],
                "model_display_name": result["model_display_name"],
                "split": result["split"],
                "predicted_probability": float(result["probability"]),
                "predicted_label": int(result["predicted_label"]),
                "explanation_table": str(result["table_file"]),
                "waterfall_figure": str(result["waterfall_file"]),
                "individual_report": str(result["report_file"]),
            }

            for id_column in ["study_patient_code", "patient_id", "encounter_code"]:
                if id_column in candidate.index:
                    row[id_column] = candidate[id_column]

            rows.append(row)

        return pd.DataFrame(rows)

    def build_batch_report(
        self,
        candidate_table: pd.DataFrame,
        output_index: pd.DataFrame,
        candidate_file: Path,
        output_index_file: Path,
    ) -> str:
        """
        Build batch explanation report.
        """
        model_display_name = self.MODEL_DISPLAY_NAMES[self.config.model_key]

        lines = []
        lines.append("=" * 70)
        lines.append("BggDeepLearning Batch High-Risk Explanation Report")
        lines.append("=" * 70)
        lines.append("")
        lines.append("1. Batch Configuration")
        lines.append("-" * 70)
        lines.append(f"Model: {model_display_name}")
        lines.append(f"Model key: {self.config.model_key}")
        lines.append(f"Split: {self.config.split}")
        lines.append(f"Top N patients: {self.config.top_n_patients}")
        lines.append(f"Risk threshold: {self.config.risk_threshold}")
        lines.append(f"Top N SHAP features per patient: {self.config.top_n_features}")
        lines.append("")
        lines.append("2. Selected High-Risk Candidates")
        lines.append("-" * 70)

        if candidate_table.empty:
            lines.append("No candidates were selected.")
        else:
            display_columns = [
                "risk_rank",
                "sample_index",
                "predicted_probability",
                "predicted_label",
                "true_label",
                "observed_poor_outcome",
                "study_patient_code",
                "encounter_code",
            ]

            existing_columns = [
                column for column in display_columns
                if column in candidate_table.columns
            ]

            lines.append(candidate_table[existing_columns].to_string(index=False))

        lines.append("")
        lines.append("3. Generated Individual Explanation Outputs")
        lines.append("-" * 70)

        if output_index.empty:
            lines.append("No individual explanation outputs were generated.")
        else:
            display_columns = [
                "risk_rank",
                "sample_index",
                "model_display_name",
                "predicted_probability",
                "predicted_label",
                "waterfall_figure",
                "individual_report",
            ]

            existing_columns = [
                column for column in display_columns
                if column in output_index.columns
            ]

            lines.append(output_index[existing_columns].to_string(index=False))

        lines.append("")
        lines.append("4. Batch Output Files")
        lines.append("-" * 70)
        lines.append(f"Candidate table: {candidate_file}")
        lines.append(f"Output index table: {output_index_file}")
        lines.append("")
        lines.append("5. Interpretation Notes")
        lines.append("-" * 70)
        lines.append("Patients are ranked by predicted probability from high to low.")
        lines.append("Individual SHAP waterfall plots explain each selected model prediction.")
        lines.append("Positive SHAP values push an individual prediction toward higher risk.")
        lines.append("Negative SHAP values push an individual prediction toward lower risk.")
        lines.append("SHAP explains model behavior, not causal relationships.")
        lines.append("This batch analysis uses simulated data and is not clinical evidence.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def run(
        self,
        prediction_dir: Path,
        processed_data_dir: Path,
        model_dir: Path,
        table_dir: Path,
        figure_dir: Path,
        report_dir: Path,
    ) -> Dict[str, Path | int]:
        """
        Run complete batch high-risk explanation workflow.
        """
        self.validate_config()

        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        output_stem = self.build_output_stem()

        candidate_file = table_dir / f"{output_stem}_candidates.csv"
        output_index_file = table_dir / f"{output_stem}_outputs.csv"
        batch_report_file = report_dir / f"{output_stem}_report.txt"

        candidate_table = self.build_candidate_table(
            prediction_dir=prediction_dir,
            processed_data_dir=processed_data_dir,
        )

        candidate_table.to_csv(
            candidate_file,
            index=False,
            encoding="utf-8-sig",
        )

        if candidate_table.empty:
            output_index = pd.DataFrame()
        else:
            output_index = self.run_individual_explanations(
                candidate_table=candidate_table,
                processed_data_dir=processed_data_dir,
                model_dir=model_dir,
                table_dir=table_dir,
                figure_dir=figure_dir,
                report_dir=report_dir,
            )

        output_index.to_csv(
            output_index_file,
            index=False,
            encoding="utf-8-sig",
        )

        report_text = self.build_batch_report(
            candidate_table=candidate_table,
            output_index=output_index,
            candidate_file=candidate_file,
            output_index_file=output_index_file,
        )

        batch_report_file.write_text(report_text, encoding="utf-8")

        return {
            "candidate_file": candidate_file,
            "output_index_file": output_index_file,
            "batch_report_file": batch_report_file,
            "n_candidates": int(len(candidate_table)),
            "n_outputs": int(len(output_index)),
        }