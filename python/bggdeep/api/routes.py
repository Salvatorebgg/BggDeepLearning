# -*- coding: utf-8 -*-

"""
BggDeepLearning FastAPI 路由 — 临床风险预测

File:
python/bggdeep/api/routes.py

Purpose:
- POST /api/v1/predict/single   — 单个患者预测
- POST /api/v1/predict/batch    — 批量患者预测
- GET  /api/v1/health           — 健康检查
- GET  /api/v1/models           — 可用模型列表

Clinical note:
当前模型基于模拟数据训练，仅用于工程演示，不能用于真实临床决策。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from bggdeep.api.engine import InferenceEngine, create_engine
from bggdeep.api.schemas import (
    BatchPredictionInput,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PatientPredictionItem,
    SinglePredictionInput,
    SinglePredictionResponse,
)

# ─────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/v1", tags=["clinical-risk-prediction"])


# ─────────────────────────────────────────────────────────
# Lazy engine singleton
# ─────────────────────────────────────────────────────────

_engine: Optional[InferenceEngine] = None


def _get_project_root() -> Path:
    """
    Find project root from this file's location.
    """
    # python/bggdeep/api/routes.py -> project root is 3 levels up
    current = Path(__file__).resolve()
    return current.parents[3]


def _get_engine() -> InferenceEngine:
    """
    Get or create the inference engine singleton.
    """
    global _engine

    if _engine is None:
        project_root = _get_project_root()
        model_dir = project_root / "outputs" / "models"
        preprocessor_file = model_dir / "tabular_preprocessor.joblib"
        _engine = create_engine(
            model_dir=model_dir,
            preprocessor_file=preprocessor_file,
        )

    return _engine


# ─────────────────────────────────────────────────────────
# Health & Info
# ─────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查：确认 API 服务正常运行，并返回可用模型列表。
    """
    engine = _get_engine()
    return engine.get_health_info()


@router.get("/models")
async def list_models():
    """
    列出当前可用的预测模型。
    """
    engine = _get_engine()
    return {
        "models": [
            {
                "model_key": key,
                "description": {
                    "gradient_boosting": "梯度提升模型 Gradient Boosting",
                    "random_forest": "随机森林 Random Forest",
                    "logistic": "逻辑回归 Logistic Regression",
                    "deep_mlp": "深度神经网络 Deep MLP (PyTorch)",
                }.get(key, key),
            }
            for key in engine.AVAILABLE_MODELS
        ]
    }


# ─────────────────────────────────────────────────────────
# Single Prediction
# ─────────────────────────────────────────────────────────


@router.post(
    "/predict/single",
    response_model=SinglePredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_single(
    input_data: SinglePredictionInput,
    model_key: str = Query(
        default="gradient_boosting",
        description="模型键：gradient_boosting / random_forest / logistic / deep_mlp",
    ),
):
    """
    单个患者风险预测。

    传入临床特征 JSON，返回 poor_outcome 预测概率和解释。

    **临床警告**：当前模型基于模拟数据训练，不能用于真实临床决策。
    """
    try:
        engine = _get_engine()

        patient_dict = input_data.model_dump()
        result = engine.predict_single(
            input_data=patient_dict,
            model_key=model_key,
        )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"模型文件或预处理器文件缺失：{exc}。请先训练模型。",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"预测服务内部错误：{type(exc).__name__}: {exc}",
        )


# ─────────────────────────────────────────────────────────
# Batch Prediction
# ─────────────────────────────────────────────────────────


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_batch(
    batch_input: BatchPredictionInput,
    model_key: str = Query(
        default="gradient_boosting",
        description="模型键：gradient_boosting / random_forest / logistic / deep_mlp",
    ),
):
    """
    批量患者风险预测。

    传入患者列表 JSON，返回所有患者的预测结果和风险统计。

    最多支持一次传入 500 条患者记录。

    **临床警告**：当前模型基于模拟数据训练，不能用于真实临床决策。
    """
    try:
        engine = _get_engine()

        patients = [p.model_dump() for p in batch_input.patients]
        result = engine.predict_batch(
            patients=patients,
            model_key=model_key,
        )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"模型文件或预处理器文件缺失：{exc}。请先训练模型。",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"批量预测服务内部错误：{type(exc).__name__}: {exc}",
        )


# ─────────────────────────────────────────────────────────
# Patient Report DOCX Export
# ─────────────────────────────────────────────────────────


@router.post(
    "/predict/single/report",
    responses={
        200: {
            "description": "DOCX 报告文件",
            "content": {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {}
            },
        },
        400: {"model": ErrorResponse},
    },
)
async def predict_single_with_report(
    input_data: SinglePredictionInput,
    model_key: str = Query(
        default="gradient_boosting",
        description="模型键",
    ),
):
    """
    单个患者预测 + DOCX 报告导出。

    与 /predict/single 相同的预测逻辑，但返回 .docx 文件下载。

    **临床警告**：当前模型基于模拟数据训练，不能用于真实临床决策。
    """
    from fastapi.responses import Response

    try:
        import pandas as pd

        from bggdeep.reporting.streamlit_patient_docx import (
            build_patient_prediction_docx_bytes,
        )

        engine = _get_engine()
        patient_dict = input_data.model_dump()
        result = engine.predict_single(
            input_data=patient_dict,
            model_key=model_key,
        )

        # Build explanation display DataFrame
        expl_list = result.get("explanation", [])
        expl_df = pd.DataFrame(expl_list) if expl_list else pd.DataFrame()

        # Build raw input display dict with Chinese labels
        from bggdeep.inference.clinical_risk_predictor import (
            ClinicalRiskPredictor,
        )

        raw_input_display = {
            "年龄": patient_dict.get("age", ""),
            "性别": {"Male": "男", "Female": "女"}.get(
                str(patient_dict.get("sex", "")), ""
            ),
            "BMI": patient_dict.get("bmi", ""),
            "入院类型": {
                "Emergency": "急诊",
                "Elective": "择期",
                "ICU_transfer": "ICU转入",
            }.get(str(patient_dict.get("admission_type", "")), ""),
            "高血压": "是" if patient_dict.get("hypertension", 0) == 1 else "否",
            "糖尿病": "是" if patient_dict.get("diabetes", 0) == 1 else "否",
            "冠心病": "是" if patient_dict.get("coronary_disease", 0) == 1 else "否",
            "慢性肾病": "是" if patient_dict.get("chronic_kidney_disease", 0) == 1 else "否",
            "疑似感染": "是" if patient_dict.get("infection_suspected", 0) == 1 else "否",
            "疑似创伤": "是" if patient_dict.get("trauma_suspected", 0) == 1 else "否",
            "心率": f"{patient_dict.get('heart_rate', '')}",
            "收缩压": f"{patient_dict.get('systolic_bp', '')}",
            "舒张压": f"{patient_dict.get('diastolic_bp', '')}",
            "呼吸频率": f"{patient_dict.get('respiratory_rate', '')}",
            "血氧饱和度": f"{patient_dict.get('oxygen_saturation', '')}",
            "体温": f"{patient_dict.get('temperature_c', '')}",
            "血红蛋白": f"{patient_dict.get('hemoglobin', '')}",
            "白细胞": f"{patient_dict.get('white_blood_cell', '')}",
            "血小板": f"{patient_dict.get('platelet', '')}",
            "肌酐": f"{patient_dict.get('creatinine', '')}",
            "乳酸": f"{patient_dict.get('lactate', '')}",
            "C反应蛋白": f"{patient_dict.get('c_reactive_protein', '')}",
            "休克指数": f"{patient_dict.get('shock_index', '')}",
        }

        docx_bytes = build_patient_prediction_docx_bytes(
            model_name=result["model_display_name"],
            probability=result["probability"],
            risk_group=result["risk_group"],
            predicted_label=result["predicted_label"],
            raw_input_display=raw_input_display,
            explanation_display_df=expl_df,
        )

        filename = f"patient_prediction_report_{result['model_key']}.docx"

        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"报告生成失败：{type(exc).__name__}: {exc}",
        )
