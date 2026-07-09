# -*- coding: utf-8 -*-

"""
BggDeepLearning FastAPI 推理引擎封装

File:
python/bggdeep/api/engine.py

Purpose:
- 封装 ClinicalRiskPredictor 为单例推理引擎
- 管理模型的加载、缓存和切换
- 供路由层调用，保持路由文件简洁
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from bggdeep.inference.clinical_risk_predictor import (
    ClinicalRiskPredictor,
    ClinicalRiskPredictorConfig,
)


@dataclass
class InferenceEngineConfig:
    """
    Inference engine configuration.
    """

    model_dir: Path
    preprocessor_file: Path
    threshold_low: float = 0.20
    threshold_high: float = 0.50
    top_n_explanation: int = 10


class InferenceEngine:
    """
    Singleton-style inference engine that wraps ClinicalRiskPredictor.

    Manages model loading, prediction, and explanation for the API layer.
    """

    # Available model keys
    AVAILABLE_MODELS: List[str] = [
        "gradient_boosting",
        "random_forest",
        "logistic",
        "deep_mlp",
    ]

    def __init__(self, config: InferenceEngineConfig) -> None:
        self.config = config
        self._predictors: Dict[str, ClinicalRiskPredictor] = {}

    def _get_predictor(self, model_key: str) -> ClinicalRiskPredictor:
        """
        Get or create a ClinicalRiskPredictor for the given model.

        Predictors are cached after first load.
        """
        if model_key not in self.AVAILABLE_MODELS:
            supported = ", ".join(self.AVAILABLE_MODELS)
            raise ValueError(
                f"Unsupported model_key: {model_key}. "
                f"Supported models: {supported}"
            )

        if model_key not in self._predictors:
            predictor_config = ClinicalRiskPredictorConfig(
                model_key=model_key,
                threshold_low=self.config.threshold_low,
                threshold_high=self.config.threshold_high,
                top_n_explanation=self.config.top_n_explanation,
            )

            predictor = ClinicalRiskPredictor(
                config=predictor_config,
                model_dir=self.config.model_dir,
                preprocessor_file=self.config.preprocessor_file,
            )

            # Pre-load model to validate it exists and to warm up
            predictor.load_model()
            predictor.load_preprocessor()

            self._predictors[model_key] = predictor

        return self._predictors[model_key]

    def predict_single(
        self,
        input_data: dict,
        model_key: str = "gradient_boosting",
    ) -> dict:
        """
        Predict for a single patient.

        Parameters
        ----------
        input_data:
            Dictionary with raw clinical features.

        model_key:
            Which model to use.

        Returns
        -------
        dict
            Prediction result with probability, risk group, explanation.
        """
        # Auto-calculate shock_index if not provided
        if input_data.get("shock_index") is None:
            hr = float(input_data.get("heart_rate", 80))
            sbp = float(input_data.get("systolic_bp", 120))
            input_data["shock_index"] = hr / sbp if sbp > 0 else 1.0

        predictor = self._get_predictor(model_key)
        raw_result = predictor.predict_from_raw(input_data)

        # Format explanation
        explanation = []
        expl_df = raw_result.get("explanation_df", pd.DataFrame())

        if not expl_df.empty:
            for _, row in expl_df.iterrows():
                item = {
                    "rank": int(row.get("rank", 0)),
                    "feature_name": str(row.get("feature_name", "")),
                    "feature_value": float(row.get("feature_value", 0.0)),
                    "direction": str(row.get("direction", "")),
                }

                if "shap_value" in row:
                    item["shap_value"] = float(row["shap_value"])
                elif "contribution" in row:
                    item["contribution"] = float(row["contribution"])

                explanation.append(item)

        prob = float(raw_result["probability"])

        return {
            "model_key": model_key,
            "model_display_name": str(raw_result["model_display_name"]),
            "probability": prob,
            "probability_percent": f"{prob:.1%}",
            "predicted_label": int(raw_result["predicted_label"]),
            "risk_group": str(raw_result["risk_group"]),
            "threshold_low": self.config.threshold_low,
            "threshold_high": self.config.threshold_high,
            "explanation": explanation,
        }

    def predict_batch(
        self,
        patients: List[dict],
        model_key: str = "gradient_boosting",
    ) -> dict:
        """
        Predict for multiple patients.

        Parameters
        ----------
        patients:
            List of patient dicts.

        model_key:
            Which model to use.

        Returns
        -------
        dict
            Batch prediction result with summary statistics.
        """
        # Auto-calculate shock_index for each patient
        for patient in patients:
            if patient.get("shock_index") is None:
                hr = float(patient.get("heart_rate", 80))
                sbp = float(patient.get("systolic_bp", 120))
                patient["shock_index"] = hr / sbp if sbp > 0 else 1.0

        raw_df = pd.DataFrame(patients)
        predictor = self._get_predictor(model_key)
        batch_result = predictor.predict_batch_from_raw_df(raw_df)
        pred_df = batch_result["prediction_df"]

        predictions = []
        high_risk = 0
        medium_risk = 0
        low_risk = 0

        for idx, (_, row) in enumerate(pred_df.iterrows()):
            risk_group = str(row["risk_group"])
            prob = float(row["predicted_probability"])

            if risk_group == "高风险":
                high_risk += 1
            elif risk_group == "中风险":
                medium_risk += 1
            else:
                low_risk += 1

            predictions.append({
                "index": idx + 1,
                "probability": prob,
                "probability_percent": f"{prob:.1%}",
                "risk_group": risk_group,
                "predicted_label": int(row["predicted_label"]),
            })

        return {
            "model_key": model_key,
            "model_display_name": str(predictor.get_model_spec().model_display_name),
            "total_patients": len(patients),
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "predictions": predictions,
        }

    def get_health_info(self) -> dict:
        """
        Return health check info with available models.
        """
        return {
            "status": "ok",
            "version": "0.1.0",
            "available_models": self.AVAILABLE_MODELS,
        }


def create_engine(
    model_dir: Optional[Path] = None,
    preprocessor_file: Optional[Path] = None,
) -> InferenceEngine:
    """
    Factory function to create the inference engine with project defaults.

    Uses the standard project paths if not explicitly provided.
    """
    if model_dir is None:
        from bggdeep.core.paths import get_project_paths, ensure_project_directories

        paths = get_project_paths()
        ensure_project_directories(paths)
        model_dir = paths.model_dir
        preprocessor_file = model_dir / "tabular_preprocessor.joblib"

    config = InferenceEngineConfig(
        model_dir=model_dir,
        preprocessor_file=preprocessor_file,
    )

    return InferenceEngine(config)
