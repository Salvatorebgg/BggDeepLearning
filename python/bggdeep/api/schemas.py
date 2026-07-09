# -*- coding: utf-8 -*-

"""
BggDeepLearning FastAPI 后端推理服务请求/响应模型（Pydantic Schemas）

File:
python/bggdeep/api/schemas.py

Purpose:
- 定义 API 请求和响应的数据结构
- 提供模型选择、阈值配置等参数校验
- 提供批量预测和批量解释的标准化输出格式
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────
# 单个患者预测
# ─────────────────────────────────────────────────────────


class SinglePredictionInput(BaseModel):
    """
    单个患者预测请求体。

    所有临床特征均以键值对形式传入。
    """

    age: float = Field(..., ge=18, le=100, description="年龄（岁）")
    sex: str = Field(..., pattern="^(Male|Female)$", description="性别：Male 或 Female")
    bmi: float = Field(..., ge=10.0, le=60.0, description="体重指数 BMI")
    admission_type: str = Field(
        ...,
        pattern="^(Emergency|Elective|ICU_transfer)$",
        description="入院类型：Emergency / Elective / ICU_transfer",
    )
    hypertension: int = Field(..., ge=0, le=1, description="高血压：0=否，1=是")
    diabetes: int = Field(..., ge=0, le=1, description="糖尿病：0=否，1=是")
    coronary_disease: int = Field(..., ge=0, le=1, description="冠心病：0=否，1=是")
    chronic_kidney_disease: int = Field(..., ge=0, le=1, description="慢性肾病：0=否，1=是")
    infection_suspected: int = Field(..., ge=0, le=1, description="疑似感染：0=否，1=是")
    trauma_suspected: int = Field(..., ge=0, le=1, description="疑似创伤：0=否，1=是")
    heart_rate: float = Field(..., ge=30.0, le=220.0, description="心率（次/分）")
    systolic_bp: float = Field(..., ge=50.0, le=250.0, description="收缩压（mmHg）")
    diastolic_bp: float = Field(..., ge=20.0, le=150.0, description="舒张压（mmHg）")
    respiratory_rate: float = Field(..., ge=6.0, le=60.0, description="呼吸频率（次/分）")
    oxygen_saturation: float = Field(..., ge=50.0, le=100.0, description="血氧饱和度（%）")
    temperature_c: float = Field(..., ge=30.0, le=43.0, description="体温（℃）")
    hemoglobin: float = Field(..., ge=3.0, le=25.0, description="血红蛋白（g/dL）")
    white_blood_cell: float = Field(..., ge=0.1, le=80.0, description="白细胞（×10^9/L）")
    platelet: float = Field(..., ge=5.0, le=1000.0, description="血小板（×10^9/L）")
    creatinine: float = Field(..., ge=10.0, le=1200.0, description="肌酐（μmol/L）")
    lactate: float = Field(..., ge=0.1, le=30.0, description="乳酸（mmol/L）")
    c_reactive_protein: float = Field(..., ge=0.0, le=500.0, description="C反应蛋白（mg/L）")
    shock_index: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="休克指数（心率/收缩压），不传则自动计算",
    )


class ExplanationItem(BaseModel):
    """
    单个特征解释条目。
    """

    rank: int = Field(..., description="重要性排名（1 为最重要）")
    feature_name: str = Field(..., description="预处理后的特征名")
    feature_value: float = Field(..., description="该特征的数值")
    contribution: Optional[float] = Field(default=None, description="线性贡献值")
    shap_value: Optional[float] = Field(default=None, description="SHAP 值")
    direction: str = Field(..., description="影响方向：增加风险 / 降低风险")


class SinglePredictionResponse(BaseModel):
    """
    单个患者预测响应体。
    """

    model_key: str = Field(..., description="模型键，如 random_forest")
    model_display_name: str = Field(..., description="模型中文显示名")
    probability: float = Field(..., description="poor_outcome 预测概率（0~1）")
    probability_percent: str = Field(..., description="预测风险百分比，如 23.4%")
    predicted_label: int = Field(..., description="预测标签：0=低风险，1=高风险")
    risk_group: str = Field(..., description="风险分层：低风险 / 中风险 / 高风险")
    threshold_low: float = Field(..., description="配置的低风险阈值")
    threshold_high: float = Field(..., description="配置的高风险阈值")
    explanation: List[ExplanationItem] = Field(
        default_factory=list,
        description="主要解释性因素列表",
    )


# ─────────────────────────────────────────────────────────
# 批量 CSV 预测
# ─────────────────────────────────────────────────────────


class BatchPredictionInput(BaseModel):
    """
    批量预测请求体（JSON 数组方式）。
    """

    patients: List[SinglePredictionInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="患者列表，最多 500 条",
    )


class PatientPredictionItem(BaseModel):
    """
    单条批量预测结果。
    """

    index: int = Field(..., description="输入顺序索引")
    probability: float = Field(..., description="预测概率")
    probability_percent: str = Field(..., description="预测风险百分比")
    risk_group: str = Field(..., description="风险分层")
    predicted_label: int = Field(..., description="预测标签")


class BatchPredictionResponse(BaseModel):
    """
    批量预测响应体。
    """

    model_key: str = Field(..., description="模型键")
    model_display_name: str = Field(..., description="模型中文名称")
    total_patients: int = Field(..., description="患者总数")
    high_risk_count: int = Field(..., description="高风险人数")
    medium_risk_count: int = Field(..., description="中风险人数")
    low_risk_count: int = Field(..., description="低风险人数")
    predictions: List[PatientPredictionItem] = Field(..., description="预测结果列表")


# ─────────────────────────────────────────────────────────
# 通用
# ─────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """
    健康检查响应。
    """

    status: str = Field(default="ok", description="服务状态")
    version: str = Field(..., description="API 版本")
    available_models: List[str] = Field(..., description="可用模型列表")


class ErrorResponse(BaseModel):
    """
    通用错误响应。
    """

    error: str = Field(..., description="错误类型")
    detail: str = Field(..., description="错误详情")
