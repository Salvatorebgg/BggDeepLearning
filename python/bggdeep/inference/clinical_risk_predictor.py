"""
BggDeepLearning clinical risk predictor

File:
python/bggdeep/inference/clinical_risk_predictor.py

Purpose:
1. Load saved preprocessing object
2. Load trained model
3. Transform one or multiple raw clinical input rows
4. Predict poor_outcome risk
5. Generate simple model explanation when possible
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


@dataclass
class ClinicalRiskPredictorConfig:
    """
    Clinical risk predictor configuration.
    """

    model_key: str = "gradient_boosting"
    threshold_low: float = 0.20
    threshold_high: float = 0.50
    top_n_explanation: int = 10


@dataclass
class ClinicalModelSpec:
    """
    Saved model specification.
    """

    model_key: str
    model_display_name: str
    model_file_name: str
    model_type: str


class ClinicalRiskPredictor:
    """
    Predictor used by the Streamlit clinical demo.
    """

    MODEL_SPECS: Dict[str, ClinicalModelSpec] = {
        "logistic": ClinicalModelSpec(
            model_key="logistic",
            model_display_name="Logistic Regression",
            model_file_name="logistic_regression_baseline.joblib",
            model_type="linear",
        ),
        "random_forest": ClinicalModelSpec(
            model_key="random_forest",
            model_display_name="Random Forest",
            model_file_name="random_forest_baseline.joblib",
            model_type="tree",
        ),
        "gradient_boosting": ClinicalModelSpec(
            model_key="gradient_boosting",
            model_display_name="Gradient Boosting",
            model_file_name="gradient_boosting_baseline.joblib",
            model_type="tree",
        ),
    }

    RAW_FEATURE_COLUMNS: List[str] = [
        "age",
        "sex",
        "bmi",
        "admission_type",
        "hypertension",
        "diabetes",
        "coronary_disease",
        "chronic_kidney_disease",
        "infection_suspected",
        "trauma_suspected",
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "oxygen_saturation",
        "temperature_c",
        "hemoglobin",
        "white_blood_cell",
        "platelet",
        "creatinine",
        "lactate",
        "c_reactive_protein",
        "shock_index",
    ]

    def __init__(
        self,
        config: ClinicalRiskPredictorConfig,
        model_dir: Path,
        preprocessor_file: Path,
    ) -> None:
        self.config = config
        self.model_dir = model_dir
        self.preprocessor_file = preprocessor_file

    def get_model_spec(self) -> ClinicalModelSpec:
        """
        Get model specification.
        """
        if self.config.model_key not in self.MODEL_SPECS:
            supported = ", ".join(self.MODEL_SPECS.keys())
            raise ValueError(
                f"Unsupported model_key: {self.config.model_key}. "
                f"Supported models: {supported}"
            )

        return self.MODEL_SPECS[self.config.model_key]

    def load_model(self):
        """
        Load saved model.
        """
        spec = self.get_model_spec()
        model_file = self.model_dir / spec.model_file_name

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file was not found: {model_file}\n"
                "Please train the selected model first."
            )

        return joblib.load(model_file)

    def load_preprocessor(self):
        """
        Load saved tabular preprocessor.
        """
        if not self.preprocessor_file.exists():
            raise FileNotFoundError(
                f"Preprocessor file was not found: {self.preprocessor_file}\n"
                "Please run: python scripts\\python\\run_tabular_preprocessing.py"
            )

        return joblib.load(self.preprocessor_file)

    def validate_raw_input(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure raw input has all expected columns.
        """
        df = raw_df.copy()

        for column in self.RAW_FEATURE_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan

        return df[self.RAW_FEATURE_COLUMNS]

    @staticmethod
    def get_transformed_feature_names(preprocessor, transformed_array, model) -> List[str]:
        """
        Get transformed feature names.

        Priority:
        1. preprocessor.get_feature_names_out()
        2. model.feature_names_in_
        3. generated feature names
        """
        n_features = transformed_array.shape[1]

        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                names = list(preprocessor.get_feature_names_out())
                if len(names) == n_features:
                    return names
            except Exception:
                pass

        if hasattr(model, "feature_names_in_"):
            names = list(model.feature_names_in_)
            if len(names) == n_features:
                return names

        return [f"feature_{i}" for i in range(n_features)]

    @staticmethod
    def align_to_model_features(x_processed: pd.DataFrame, model) -> pd.DataFrame:
        """
        Align processed feature DataFrame to model expected features.
        """
        if not hasattr(model, "feature_names_in_"):
            return x_processed

        expected_features = list(model.feature_names_in_)
        aligned = x_processed.copy()

        for column in expected_features:
            if column not in aligned.columns:
                aligned[column] = 0.0

        return aligned[expected_features]

    def transform_raw_input(self, raw_df: pd.DataFrame, preprocessor, model) -> pd.DataFrame:
        """
        Transform raw clinical rows into model-ready features.
        """
        raw_df = self.validate_raw_input(raw_df)

        transformed = preprocessor.transform(raw_df)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = self.get_transformed_feature_names(
            preprocessor=preprocessor,
            transformed_array=transformed,
            model=model,
        )

        x_processed = pd.DataFrame(
            transformed,
            columns=feature_names,
        )

        x_processed = self.align_to_model_features(
            x_processed=x_processed,
            model=model,
        )

        return x_processed

    @staticmethod
    def predict_probability(model, x_processed: pd.DataFrame) -> float:
        """
        Predict positive class probability for one row.
        """
        probability = model.predict_proba(x_processed)[:, 1][0]
        return float(probability)

    @staticmethod
    def predict_probabilities(model, x_processed: pd.DataFrame) -> np.ndarray:
        """
        Predict positive class probabilities for multiple rows.
        """
        return model.predict_proba(x_processed)[:, 1]

    def risk_group(self, probability: float) -> str:
        """
        Convert probability to Chinese risk group.
        """
        if probability < self.config.threshold_low:
            return "低风险"

        if probability < self.config.threshold_high:
            return "中风险"

        return "高风险"

    def build_linear_explanation(
        self,
        model,
        x_processed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build simple contribution table for linear model.
        """
        if not hasattr(model, "coef_"):
            return pd.DataFrame()

        coefficients = model.coef_[0]
        values = x_processed.iloc[0].to_numpy()
        contributions = coefficients * values

        df = pd.DataFrame(
            {
                "feature_name": x_processed.columns,
                "feature_value": values,
                "contribution": contributions,
                "abs_contribution": np.abs(contributions),
                "direction": [
                    "增加风险" if value > 0 else "降低风险"
                    for value in contributions
                ],
            }
        )

        df = df.sort_values(
            by="abs_contribution",
            ascending=False,
        ).reset_index(drop=True)

        df["rank"] = range(1, len(df) + 1)

        return df[
            [
                "rank",
                "feature_name",
                "feature_value",
                "contribution",
                "abs_contribution",
                "direction",
            ]
        ].head(self.config.top_n_explanation)

    def build_tree_shap_explanation(
        self,
        model,
        x_processed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build SHAP explanation table for tree model.

        If SHAP is not installed or fails, returns empty DataFrame.
        """
        try:
            import shap
        except Exception:
            return pd.DataFrame()

        try:
            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(x_processed)

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
                return pd.DataFrame()

            df = pd.DataFrame(
                {
                    "feature_name": x_processed.columns,
                    "feature_value": x_processed.iloc[0].to_numpy(),
                    "shap_value": shap_one,
                    "abs_shap_value": np.abs(shap_one),
                    "direction": [
                        "增加风险" if value > 0 else "降低风险"
                        for value in shap_one
                    ],
                }
            )

            df = df.sort_values(
                by="abs_shap_value",
                ascending=False,
            ).reset_index(drop=True)

            df["rank"] = range(1, len(df) + 1)

            return df[
                [
                    "rank",
                    "feature_name",
                    "feature_value",
                    "shap_value",
                    "abs_shap_value",
                    "direction",
                ]
            ].head(self.config.top_n_explanation)

        except Exception:
            return pd.DataFrame()

    def explain_prediction(
        self,
        model,
        x_processed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Explain one prediction based on model type.
        """
        spec = self.get_model_spec()

        if spec.model_type == "linear":
            return self.build_linear_explanation(
                model=model,
                x_processed=x_processed,
            )

        return self.build_tree_shap_explanation(
            model=model,
            x_processed=x_processed,
        )

    def predict_from_raw(self, raw_input: Dict[str, object]) -> Dict[str, object]:
        """
        Full prediction workflow from one raw clinical input.
        """
        raw_df = pd.DataFrame([raw_input])

        model = self.load_model()
        preprocessor = self.load_preprocessor()

        x_processed = self.transform_raw_input(
            raw_df=raw_df,
            preprocessor=preprocessor,
            model=model,
        )

        probability = self.predict_probability(
            model=model,
            x_processed=x_processed,
        )

        predicted_label = int(probability >= 0.5)
        risk_group = self.risk_group(probability)

        explanation_df = self.explain_prediction(
            model=model,
            x_processed=x_processed,
        )

        spec = self.get_model_spec()

        return {
            "model_key": spec.model_key,
            "model_display_name": spec.model_display_name,
            "probability": probability,
            "predicted_label": predicted_label,
            "risk_group": risk_group,
            "x_processed": x_processed,
            "explanation_df": explanation_df,
        }

    def predict_batch_from_raw_df(self, raw_df: pd.DataFrame) -> Dict[str, object]:
        """
        Batch prediction workflow from raw clinical input table.
        """
        model = self.load_model()
        preprocessor = self.load_preprocessor()

        x_processed = self.transform_raw_input(
            raw_df=raw_df,
            preprocessor=preprocessor,
            model=model,
        )

        probabilities = self.predict_probabilities(
            model=model,
            x_processed=x_processed,
        )

        result_df = raw_df.copy()
        result_df["predicted_probability"] = probabilities
        result_df["predicted_probability_percent"] = [
            f"{value:.1%}" for value in probabilities
        ]
        result_df["predicted_label"] = [
            int(value >= 0.5) for value in probabilities
        ]
        result_df["risk_group"] = [
            self.risk_group(float(value)) for value in probabilities
        ]
        result_df["model_key"] = self.config.model_key
        result_df["model_display_name"] = self.get_model_spec().model_display_name

        result_df = result_df.sort_values(
            by="predicted_probability",
            ascending=False,
        ).reset_index(drop=True)

        result_df["risk_rank"] = range(1, len(result_df) + 1)

        return {
            "prediction_df": result_df,
            "x_processed": x_processed,
        }