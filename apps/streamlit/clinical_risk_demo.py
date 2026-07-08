# -*- coding: utf-8 -*-

"""
BggDeepLearning Streamlit clinical risk demo - Chinese dashboard version

File:
apps/streamlit/clinical_risk_demo.py

Usage:
streamlit run apps\\streamlit\\clinical_risk_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


def find_project_root() -> Path:
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / "configs" / "app.yaml").exists():
            return parent

    raise FileNotFoundError("Project root was not found. configs/app.yaml is missing.")


PROJECT_ROOT = find_project_root()
PYTHON_DIR = PROJECT_ROOT / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


from bggdeep.inference.clinical_risk_predictor import (  # noqa: E402
    ClinicalRiskPredictor,
    ClinicalRiskPredictorConfig,
)
from bggdeep.reporting.streamlit_patient_docx import (  # noqa: E402
    build_patient_prediction_docx_bytes,
)

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


MODEL_DISPLAY = {
    "gradient_boosting": "梯度提升模型 Gradient Boosting",
    "random_forest": "随机森林 Random Forest",
    "logistic": "逻辑回归 Logistic Regression",
}


FEATURE_LABEL_ZH = {
    "age": "年龄",
    "sex": "性别",
    "bmi": "BMI",
    "admission_type": "入院类型",
    "hypertension": "高血压",
    "diabetes": "糖尿病",
    "coronary_disease": "冠心病",
    "chronic_kidney_disease": "慢性肾病",
    "infection_suspected": "疑似感染",
    "trauma_suspected": "疑似创伤",
    "heart_rate": "心率",
    "systolic_bp": "收缩压",
    "diastolic_bp": "舒张压",
    "respiratory_rate": "呼吸频率",
    "oxygen_saturation": "血氧饱和度",
    "temperature_c": "体温",
    "hemoglobin": "血红蛋白",
    "white_blood_cell": "白细胞",
    "platelet": "血小板",
    "creatinine": "肌酐",
    "lactate": "乳酸",
    "c_reactive_protein": "C反应蛋白",
    "shock_index": "休克指数",
    "predicted_probability": "预测风险概率",
    "predicted_probability_percent": "预测风险百分比",
    "predicted_label": "预测标签",
    "risk_group": "风险分层",
    "model_key": "模型键",
    "model_display_name": "模型名称",
    "risk_rank": "风险排名",
}


METRIC_LABEL_ZH = {
    "rank_within_split": "排名",
    "rank_at_threshold": "阈值内排名",
    "model_display_name": "模型显示名称",
    "model_name": "模型名称",
    "model_key": "模型键",
    "split": "数据集",
    "auc": "AUC",
    "roc_auc": "ROC AUC",
    "accuracy": "准确率",
    "sensitivity": "敏感度",
    "specificity": "特异度",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1值",
    "balanced_summary_score": "综合评分",
    "selection_rule": "选择规则",
    "n_samples": "样本量",
    "positive_rate": "阳性率",
    "brier_score": "Brier分数",
    "n_bins_returned": "校准分箱数",
    "strategy": "策略",
    "selected_threshold": "选定阈值",
    "nearest_available_threshold": "最近可用阈值",
    "net_benefit": "净获益",
    "standardized_net_benefit": "标准化净获益",
    "tp": "真阳性",
    "fp": "假阳性",
    "tn": "真阴性",
    "fn": "假阴性",
    "feature_name": "特征名",
    "feature_label": "特征",
    "mean_abs_shap": "平均绝对SHAP值",
    "file_type": "文件类型",
    "description": "说明",
    "exists": "是否存在",
    "relative_path": "相对路径",
}

MODEL_NAME_ZH = {
    "Logistic Regression": "逻辑回归",
    "Random Forest": "随机森林",
    "Gradient Boosting": "梯度提升模型",
    "logistic": "逻辑回归",
    "random_forest": "随机森林",
    "gradient_boosting": "梯度提升模型",
}


SPLIT_NAME_ZH = {
    "train": "训练集",
    "validation": "验证集",
    "test": "测试集",
}


DIRECTION_ZH = {
    "increase_risk": "增加风险",
    "decrease_risk": "降低风险",
    "增加风险": "增加风险",
    "降低风险": "降低风险",
}



CHINESE_COLUMN_TO_MODEL_COLUMN = {
    "年龄": "age",
    "性别": "sex",
    "BMI": "bmi",
    "体重指数": "bmi",
    "入院类型": "admission_type",
    "高血压": "hypertension",
    "糖尿病": "diabetes",
    "冠心病": "coronary_disease",
    "慢性肾病": "chronic_kidney_disease",
    "疑似感染": "infection_suspected",
    "疑似创伤": "trauma_suspected",
    "心率": "heart_rate",
    "收缩压": "systolic_bp",
    "舒张压": "diastolic_bp",
    "呼吸频率": "respiratory_rate",
    "血氧饱和度": "oxygen_saturation",
    "体温": "temperature_c",
    "血红蛋白": "hemoglobin",
    "白细胞": "white_blood_cell",
    "血小板": "platelet",
    "肌酐": "creatinine",
    "乳酸": "lactate",
    "C反应蛋白": "c_reactive_protein",
    "c反应蛋白": "c_reactive_protein",
    "休克指数": "shock_index",
}


YES_NO_COLUMNS = [
    "hypertension",
    "diabetes",
    "coronary_disease",
    "chronic_kidney_disease",
    "infection_suspected",
    "trauma_suspected",
]


def calculate_shock_index(heart_rate: float, systolic_bp: float) -> float:
    if systolic_bp <= 0:
        return 0.0

    return heart_rate / systolic_bp


def normalize_binary_value(value) -> int:
    if pd.isna(value):
        return 0

    text = str(value).strip().lower()

    positive_values = {
        "1",
        "yes",
        "y",
        "true",
        "是",
        "有",
        "阳性",
        "是的",
        "高",
    }

    negative_values = {
        "0",
        "no",
        "n",
        "false",
        "否",
        "无",
        "阴性",
        "不是",
        "低",
    }

    if text in positive_values:
        return 1

    if text in negative_values:
        return 0

    try:
        return int(float(text) >= 0.5)
    except Exception:
        return 0


def normalize_uploaded_csv(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    rename_map = {}
    for column in df.columns:
        if column in CHINESE_COLUMN_TO_MODEL_COLUMN:
            rename_map[column] = CHINESE_COLUMN_TO_MODEL_COLUMN[column]

    df = df.rename(columns=rename_map)

    if "sex" in df.columns:
        df["sex"] = df["sex"].replace(
            {
                "男": "Male",
                "女": "Female",
                "男性": "Male",
                "女性": "Female",
                "male": "Male",
                "female": "Female",
                "M": "Male",
                "F": "Female",
                "m": "Male",
                "f": "Female",
            }
        )

    if "admission_type" in df.columns:
        df["admission_type"] = df["admission_type"].replace(
            {
                "急诊": "Emergency",
                "急诊入院": "Emergency",
                "择期": "Elective",
                "择期入院": "Elective",
                "ICU转入": "ICU_transfer",
                "ICU 转入": "ICU_transfer",
                "icu转入": "ICU_transfer",
                "Emergency": "Emergency",
                "Elective": "Elective",
                "ICU_transfer": "ICU_transfer",
            }
        )

    for column in YES_NO_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(normalize_binary_value)

    numeric_columns = [
        "age",
        "bmi",
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

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "shock_index" not in df.columns:
        if "heart_rate" in df.columns and "systolic_bp" in df.columns:
            df["shock_index"] = df.apply(
                lambda row: calculate_shock_index(
                    heart_rate=float(row["heart_rate"]),
                    systolic_bp=float(row["systolic_bp"]),
                )
                if pd.notna(row["heart_rate"]) and pd.notna(row["systolic_bp"])
                else 0.0,
                axis=1,
            )

    return df


def translate_feature_name(feature_name: str) -> str:
    text = str(feature_name)
    text = text.replace("num__", "")
    text = text.replace("cat__", "")
    text = text.replace("remainder__", "")

    for raw_name, zh_name in FEATURE_LABEL_ZH.items():
        if text == raw_name:
            return zh_name

        prefix = raw_name + "_"
        if text.startswith(prefix):
            category_value = text.replace(prefix, "")
            return f"{zh_name}={category_value}"

    return text.replace("_", " ")
def translate_model_name(value) -> str:
    """
    Translate model name into Chinese display name.
    """
    text = str(value)
    return MODEL_NAME_ZH.get(text, text)


def translate_split_name(value) -> str:
    """
    Translate split name into Chinese display name.
    """
    text = str(value)
    return SPLIT_NAME_ZH.get(text, text)


def translate_direction(value) -> str:
    """
    Translate explanation direction.
    """
    text = str(value)
    return DIRECTION_ZH.get(text, text)


def beautify_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate common values and columns for Streamlit display.
    """
    display_df = df.copy()

    if "model_display_name" in display_df.columns:
        display_df["model_display_name"] = display_df["model_display_name"].apply(
            translate_model_name
        )

    if "model_name" in display_df.columns:
        display_df["model_name"] = display_df["model_name"].apply(
            translate_model_name
        )

    if "model_key" in display_df.columns:
        display_df["model_key"] = display_df["model_key"].apply(
            translate_model_name
        )

    if "split" in display_df.columns:
        display_df["split"] = display_df["split"].apply(
            translate_split_name
        )

    if "direction" in display_df.columns:
        display_df["direction"] = display_df["direction"].apply(
            translate_direction
        )

    if "feature_name" in display_df.columns:
        display_df["feature_label"] = display_df["feature_name"].apply(
            translate_feature_name
        )

    display_df = translate_dataframe_columns(display_df)

    return display_df





def make_unique_columns(columns) -> list[str]:
    """
    Make column names unique for Streamlit/PyArrow display.
    """
    seen = {}
    unique_columns = []

    for column in columns:
        column_text = str(column)

        if column_text not in seen:
            seen[column_text] = 1
            unique_columns.append(column_text)
        else:
            seen[column_text] += 1
            unique_columns.append(f"{column_text}_{seen[column_text]}")

    return unique_columns


def translate_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()

    rename_map = {}
    rename_map.update(FEATURE_LABEL_ZH)
    rename_map.update(METRIC_LABEL_ZH)

    display_df = display_df.rename(columns=rename_map)
    display_df.columns = make_unique_columns(display_df.columns)

    return display_df


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def read_text_if_exists(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8", errors="ignore")

    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...... 内容较长，已截断预览。"

    return text


def show_missing_file(path: Path) -> None:
    st.warning(f"文件不存在：{path.relative_to(PROJECT_ROOT)}")


def show_image_if_exists(path: Path, caption: str, width: int | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        show_missing_file(path)


def show_table_if_exists(
    path: Path,
    title: str,
    max_rows: int = 30,
    add_feature_label: bool = False,
) -> pd.DataFrame:
    st.subheader(title)

    df = read_csv_if_exists(path)

    if df.empty:
        show_missing_file(path)
        return df

    display_df = df.copy()

    if add_feature_label and "feature_name" in display_df.columns:
        display_df["feature_label"] = display_df["feature_name"].apply(
            translate_feature_name
        )

    display_df = beautify_display_dataframe(display_df)

    st.dataframe(display_df.head(max_rows), width="stretch")

    return df


def show_report_preview(path: Path, title: str) -> None:
    st.subheader(title)

    if not path.exists():
        show_missing_file(path)
        return

    text = read_text_if_exists(path)

    st.text_area(
        label=str(path.relative_to(PROJECT_ROOT)),
        value=text,
        height=300,
    )


def build_raw_input_from_widgets() -> dict:
    st.sidebar.header("一、患者基本信息")

    age = st.sidebar.slider("年龄", min_value=18, max_value=100, value=65, step=1)

    sex_display = st.sidebar.selectbox(
        "性别",
        options=["男", "女"],
    )
    sex = {"男": "Male", "女": "Female"}[sex_display]

    bmi = st.sidebar.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=24.0,
        step=0.1,
    )

    admission_type_display = st.sidebar.selectbox(
        "入院类型",
        options=["急诊", "择期", "ICU转入"],
    )
    admission_type = {
        "急诊": "Emergency",
        "择期": "Elective",
        "ICU转入": "ICU_transfer",
    }[admission_type_display]

    st.sidebar.header("二、合并症")

    hypertension = int(st.sidebar.checkbox("高血压", value=True))
    diabetes = int(st.sidebar.checkbox("糖尿病", value=False))
    coronary_disease = int(st.sidebar.checkbox("冠心病", value=False))
    chronic_kidney_disease = int(st.sidebar.checkbox("慢性肾病", value=False))

    st.sidebar.header("三、临床怀疑")

    infection_suspected = int(st.sidebar.checkbox("疑似感染", value=True))
    trauma_suspected = int(st.sidebar.checkbox("疑似创伤", value=False))

    st.sidebar.header("四、生命体征")

    heart_rate = st.sidebar.number_input(
        "心率",
        min_value=30.0,
        max_value=220.0,
        value=105.0,
        step=1.0,
    )

    systolic_bp = st.sidebar.number_input(
        "收缩压",
        min_value=50.0,
        max_value=250.0,
        value=105.0,
        step=1.0,
    )

    diastolic_bp = st.sidebar.number_input(
        "舒张压",
        min_value=20.0,
        max_value=150.0,
        value=65.0,
        step=1.0,
    )

    respiratory_rate = st.sidebar.number_input(
        "呼吸频率",
        min_value=6.0,
        max_value=60.0,
        value=24.0,
        step=1.0,
    )

    oxygen_saturation = st.sidebar.number_input(
        "血氧饱和度",
        min_value=50.0,
        max_value=100.0,
        value=93.0,
        step=0.5,
    )

    temperature_c = st.sidebar.number_input(
        "体温",
        min_value=30.0,
        max_value=43.0,
        value=38.0,
        step=0.1,
    )

    st.sidebar.header("五、实验室指标")

    hemoglobin = st.sidebar.number_input(
        "血红蛋白",
        min_value=3.0,
        max_value=25.0,
        value=11.0,
        step=0.1,
    )

    white_blood_cell = st.sidebar.number_input(
        "白细胞",
        min_value=0.1,
        max_value=80.0,
        value=13.0,
        step=0.1,
    )

    platelet = st.sidebar.number_input(
        "血小板",
        min_value=5.0,
        max_value=1000.0,
        value=180.0,
        step=1.0,
    )

    creatinine = st.sidebar.number_input(
        "肌酐",
        min_value=10.0,
        max_value=1200.0,
        value=120.0,
        step=1.0,
    )

    lactate = st.sidebar.number_input(
        "乳酸",
        min_value=0.1,
        max_value=30.0,
        value=3.0,
        step=0.1,
    )

    c_reactive_protein = st.sidebar.number_input(
        "C反应蛋白",
        min_value=0.0,
        max_value=500.0,
        value=80.0,
        step=1.0,
    )

    shock_index = calculate_shock_index(
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
    )

    return {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "admission_type": admission_type,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "coronary_disease": coronary_disease,
        "chronic_kidney_disease": chronic_kidney_disease,
        "infection_suspected": infection_suspected,
        "trauma_suspected": trauma_suspected,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate,
        "oxygen_saturation": oxygen_saturation,
        "temperature_c": temperature_c,
        "hemoglobin": hemoglobin,
        "white_blood_cell": white_blood_cell,
        "platelet": platelet,
        "creatinine": creatinine,
        "lactate": lactate,
        "c_reactive_protein": c_reactive_protein,
        "shock_index": shock_index,
    }


def show_probability_card(probability: float, risk_group: str) -> None:
    st.subheader("预测结果")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="poor_outcome 预测风险",
            value=f"{probability:.1%}",
        )

    with col2:
        st.metric(
            label="风险分层",
            value=risk_group,
        )

    with col3:
        st.metric(
            label="预测标签",
            value=int(probability >= 0.5),
        )

    if risk_group == "高风险":
        st.error("模型提示：高风险。真实临床场景中应考虑加强监测或进一步评估。")
    elif risk_group == "中风险":
        st.warning("模型提示：中风险。建议结合临床情况进一步判断。")
    else:
        st.success("模型提示：低风险。")


def show_explanation_table(explanation_df: pd.DataFrame) -> None:
    st.subheader("模型解释")

    if explanation_df.empty:
        st.info(
            "当前没有可用的解释表。"
            "如果使用树模型，请确认已经安装 SHAP：python -m pip install shap"
        )
        return

    display_df = explanation_df.copy()

    if "feature_name" in display_df.columns:
        display_df["feature_label"] = display_df["feature_name"].apply(
            translate_feature_name
        )

    rename_map = {
        "rank": "排名",
        "feature_label": "特征",
        "feature_value": "特征值",
        "shap_value": "SHAP值",
        "abs_shap_value": "SHAP绝对值",
        "contribution": "线性贡献值",
        "abs_contribution": "贡献绝对值",
        "direction": "影响方向",
    }

    display_df = display_df.rename(columns=rename_map)

    preferred_columns = [
        "排名",
        "特征",
        "特征值",
        "SHAP值",
        "SHAP绝对值",
        "线性贡献值",
        "贡献绝对值",
        "影响方向",
    ]

    existing_columns = [
        column for column in preferred_columns
        if column in display_df.columns
    ]

    st.dataframe(
        display_df[existing_columns],
        width="stretch",
    )

    if "SHAP值" in display_df.columns:
        chart_df = display_df.copy()
        chart_df = chart_df.sort_values(by="SHAP绝对值", ascending=True)
        st.bar_chart(chart_df.set_index("特征")["SHAP值"])

    elif "线性贡献值" in display_df.columns:
        chart_df = display_df.copy()
        chart_df = chart_df.sort_values(by="贡献绝对值", ascending=True)
        st.bar_chart(chart_df.set_index("特征")["线性贡献值"])
def prepare_explanation_display_df(explanation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare explanation table for TXT and DOCX reports.
    """
    if explanation_df.empty:
        return pd.DataFrame()

    display_df = explanation_df.copy()

    if "feature_name" in display_df.columns:
        display_df["feature_label"] = display_df["feature_name"].apply(
            translate_feature_name
        )

    if "direction" in display_df.columns:
        display_df["direction"] = display_df["direction"].apply(
            translate_direction
        )

    if "shap_value" in display_df.columns:
        selected_columns = [
            "rank",
            "feature_label",
            "feature_value",
            "shap_value",
            "direction",
        ]

        existing_columns = [
            column for column in selected_columns
            if column in display_df.columns
        ]

        display_df = display_df[existing_columns].rename(
            columns={
                "rank": "排名",
                "feature_label": "特征",
                "feature_value": "特征值",
                "shap_value": "SHAP值",
                "direction": "影响方向",
            }
        )

    elif "contribution" in display_df.columns:
        selected_columns = [
            "rank",
            "feature_label",
            "feature_value",
            "contribution",
            "direction",
        ]

        existing_columns = [
            column for column in selected_columns
            if column in display_df.columns
        ]

        display_df = display_df[existing_columns].rename(
            columns={
                "rank": "排名",
                "feature_label": "特征",
                "feature_value": "特征值",
                "contribution": "线性贡献值",
                "direction": "影响方向",
            }
        )

    display_df.columns = make_unique_columns(display_df.columns)

    return display_df


def build_raw_input_display_dict(raw_input: dict) -> dict:
    """
    Convert raw input dictionary into Chinese display dictionary.
    """
    display_dict = {}

    for key, value in raw_input.items():
        label = FEATURE_LABEL_ZH.get(key, key)

        if key == "sex":
            value = {
                "Male": "男",
                "Female": "女",
            }.get(str(value), value)

        if key == "admission_type":
            value = {
                "Emergency": "急诊",
                "Elective": "择期",
                "ICU_transfer": "ICU转入",
            }.get(str(value), value)

        if key in YES_NO_COLUMNS:
            value = "是" if int(value) == 1 else "否"

        display_dict[label] = value

    return display_dict

def build_demo_template_df() -> pd.DataFrame:
    rows = [
        {
            "age": 65,
            "sex": "Male",
            "bmi": 24.0,
            "admission_type": "Emergency",
            "hypertension": 1,
            "diabetes": 0,
            "coronary_disease": 0,
            "chronic_kidney_disease": 0,
            "infection_suspected": 1,
            "trauma_suspected": 0,
            "heart_rate": 105,
            "systolic_bp": 105,
            "diastolic_bp": 65,
            "respiratory_rate": 24,
            "oxygen_saturation": 93,
            "temperature_c": 38.0,
            "hemoglobin": 11.0,
            "white_blood_cell": 13.0,
            "platelet": 180,
            "creatinine": 120,
            "lactate": 3.0,
            "c_reactive_protein": 80,
            "shock_index": 1.0,
        },
        {
            "age": 52,
            "sex": "Female",
            "bmi": 22.5,
            "admission_type": "Elective",
            "hypertension": 0,
            "diabetes": 0,
            "coronary_disease": 0,
            "chronic_kidney_disease": 0,
            "infection_suspected": 0,
            "trauma_suspected": 0,
            "heart_rate": 82,
            "systolic_bp": 128,
            "diastolic_bp": 78,
            "respiratory_rate": 18,
            "oxygen_saturation": 98,
            "temperature_c": 36.7,
            "hemoglobin": 13.2,
            "white_blood_cell": 7.2,
            "platelet": 230,
            "creatinine": 70,
            "lactate": 1.2,
            "c_reactive_protein": 5,
            "shock_index": 0.64,
        },
    ]

    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def show_single_prediction_tab(
    model_key: str,
    threshold_low: float,
    threshold_high: float,
    top_n_explanation: int,
) -> None:
    st.header("单个患者风险预测")

    raw_input = build_raw_input_from_widgets()

    st.subheader("当前输入")

    input_df = pd.DataFrame([raw_input])
    input_display_df = input_df.rename(columns=FEATURE_LABEL_ZH)
    st.dataframe(input_display_df, width="stretch")

    st.caption(
        f"自动计算：休克指数 = 心率 / 收缩压 = {raw_input['shock_index']:.3f}"
    )

    predict_button = st.button("开始预测", type="primary")

    if predict_button:
        try:
            predictor = ClinicalRiskPredictor(
                config=ClinicalRiskPredictorConfig(
                    model_key=model_key,
                    threshold_low=threshold_low,
                    threshold_high=threshold_high,
                    top_n_explanation=top_n_explanation,
                ),
                model_dir=MODEL_DIR,
                preprocessor_file=MODEL_DIR / "tabular_preprocessor.joblib",
            )

            result = predictor.predict_from_raw(raw_input)

            show_probability_card(
                probability=result["probability"],
                risk_group=result["risk_group"],
            )

            show_explanation_table(result["explanation_df"])

            show_current_patient_report_download(
                raw_input=raw_input,
                result=result,
            )

            with st.expander("查看模型处理后的特征"):
                processed_display_df = result["x_processed"].copy()
                processed_display_df.columns = [
                    translate_feature_name(column)
                    for column in processed_display_df.columns
                ]
                st.dataframe(
                    processed_display_df,
                    width="stretch",
                )

        except Exception as exc:
            st.error("预测失败。请检查模型文件、预处理器文件和输入变量。")
            st.exception(exc)

def build_current_patient_report_text(
    raw_input: dict,
    result: dict,
    explanation_df: pd.DataFrame,
) -> str:
    """
    Build downloadable current patient prediction report text.
    """
    lines = []

    probability = float(result["probability"])
    risk_group = str(result["risk_group"])
    predicted_label = int(result["predicted_label"])
    model_name = translate_model_name(result["model_display_name"])

    lines.append("=" * 70)
    lines.append("BggDeepLearning 当前患者风险预测报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append("一、模型信息")
    lines.append("-" * 70)
    lines.append(f"模型名称：{model_name}")
    lines.append("预测结局：poor_outcome")
    lines.append("数据说明：当前模型基于模拟数据训练，仅用于工程演示。")
    lines.append("")
    lines.append("二、预测结果")
    lines.append("-" * 70)
    lines.append(f"预测风险概率：{probability:.6f}")
    lines.append(f"预测风险百分比：{probability:.1%}")
    lines.append(f"风险分层：{risk_group}")
    lines.append(f"预测标签：{predicted_label}")
    lines.append("")
    lines.append("三、患者输入信息")
    lines.append("-" * 70)

    for key, value in raw_input.items():
        label = FEATURE_LABEL_ZH.get(key, key)
        lines.append(f"{label}：{value}")

    lines.append("")
    lines.append("四、主要解释性因素")
    lines.append("-" * 70)

    if explanation_df.empty:
        lines.append("当前没有可用的模型解释结果。")
    else:
        display_df = explanation_df.copy()

        if "feature_name" in display_df.columns:
            display_df["feature_label"] = display_df["feature_name"].apply(
                translate_feature_name
            )

        if "direction" in display_df.columns:
            display_df["direction"] = display_df["direction"].apply(
                translate_direction
            )

        if "shap_value" in display_df.columns:
            top_df = display_df[
                [
                    "rank",
                    "feature_label",
                    "feature_value",
                    "shap_value",
                    "direction",
                ]
            ].head(10)

            top_df = top_df.rename(
                columns={
                    "rank": "排名",
                    "feature_label": "特征",
                    "feature_value": "特征值",
                    "shap_value": "SHAP值",
                    "direction": "影响方向",
                }
            )

            lines.append(top_df.to_string(index=False))

        elif "contribution" in display_df.columns:
            top_df = display_df[
                [
                    "rank",
                    "feature_label",
                    "feature_value",
                    "contribution",
                    "direction",
                ]
            ].head(10)

            top_df = top_df.rename(
                columns={
                    "rank": "排名",
                    "feature_label": "特征",
                    "feature_value": "特征值",
                    "contribution": "线性贡献值",
                    "direction": "影响方向",
                }
            )

            lines.append(top_df.to_string(index=False))

    lines.append("")
    lines.append("五、解释说明")
    lines.append("-" * 70)
    lines.append("1. 预测风险概率表示模型预测该患者发生 poor_outcome 的概率。")
    lines.append("2. 风险分层由当前页面左侧设置的低风险阈值和高风险阈值决定。")
    lines.append("3. SHAP 或贡献值用于解释模型行为，不代表因果关系。")
    lines.append("4. 当前模型基于模拟数据训练，不能用于真实临床诊疗决策。")
    lines.append("=" * 70)

    return "\n".join(lines)


def show_current_patient_report_download(
    raw_input: dict,
    result: dict,
) -> None:
    """
    Show download buttons for current patient prediction reports.
    """
    explanation_display_df = prepare_explanation_display_df(
        result["explanation_df"]
    )

    report_text = build_current_patient_report_text(
        raw_input=raw_input,
        result=result,
        explanation_df=result["explanation_df"],
    )

    raw_input_display = build_raw_input_display_dict(raw_input)

    model_name = translate_model_name(result["model_display_name"])

    docx_bytes = build_patient_prediction_docx_bytes(
        model_name=model_name,
        probability=float(result["probability"]),
        risk_group=str(result["risk_group"]),
        predicted_label=int(result["predicted_label"]),
        raw_input_display=raw_input_display,
        explanation_display_df=explanation_display_df,
    )

    st.subheader("导出当前患者报告")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="下载当前患者预测报告 TXT",
            data=report_text.encode("utf-8-sig"),
            file_name="current_patient_prediction_report.txt",
            mime="text/plain",
        )

    with col2:
        st.download_button(
            label="下载当前患者预测报告 DOCX",
            data=docx_bytes,
            file_name="current_patient_prediction_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    with st.expander("预览当前患者预测报告 TXT"):
        st.text_area(
            label="报告内容",
            value=report_text,
            height=420,
        )

    with st.expander("预览当前患者解释表"):
        if explanation_display_df.empty:
            st.info("当前没有可用解释表。")
        else:
            st.dataframe(explanation_display_df, width="stretch")
def show_batch_prediction_tab(
    model_key: str,
    threshold_low: float,
    threshold_high: float,
    top_n_explanation: int,
) -> None:
    st.header("批量 CSV 上传预测")

    st.markdown(
        """
        你可以上传一个 CSV 文件进行批量预测。

        支持两种列名：

        1. 英文模型列名，例如 `age`, `sex`, `heart_rate`, `lactate`
        2. 中文列名，例如 `年龄`, `性别`, `心率`, `乳酸`

        性别可以写：`Male/Female` 或 `男/女`。  
        入院类型可以写：`Emergency/Elective/ICU_transfer` 或 `急诊/择期/ICU转入`。
        """
    )

    template_df = build_demo_template_df()

    st.download_button(
        label="下载批量预测 CSV 模板",
        data=dataframe_to_csv_bytes(template_df),
        file_name="clinical_risk_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "上传患者 CSV 文件",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("请先上传 CSV 文件，或下载模板后填写再上传。")
        return

    try:
        try:
            raw_csv = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_csv = pd.read_csv(uploaded_file, encoding="gbk")

        st.subheader("上传文件预览")
        st.dataframe(raw_csv.head(20), width="stretch")

        normalized_df = normalize_uploaded_csv(raw_csv)

        with st.expander("查看标准化后的模型输入"):
            standard_display_df = normalized_df.rename(columns=FEATURE_LABEL_ZH)
            st.dataframe(standard_display_df.head(50), width="stretch")

        run_batch_button = st.button("开始批量预测", type="primary")

        if run_batch_button:
            predictor = ClinicalRiskPredictor(
                config=ClinicalRiskPredictorConfig(
                    model_key=model_key,
                    threshold_low=threshold_low,
                    threshold_high=threshold_high,
                    top_n_explanation=top_n_explanation,
                ),
                model_dir=MODEL_DIR,
                preprocessor_file=MODEL_DIR / "tabular_preprocessor.joblib",
            )

            batch_result = predictor.predict_batch_from_raw_df(normalized_df)
            prediction_df = batch_result["prediction_df"]

            st.subheader("批量预测结果")

            display_df = prediction_df.rename(columns=FEATURE_LABEL_ZH)

            preferred_columns = [
                "风险排名",
                "预测风险概率",
                "预测风险百分比",
                "风险分层",
                "预测标签",
                "模型名称",
                "年龄",
                "性别",
                "入院类型",
                "心率",
                "收缩压",
                "血氧饱和度",
                "乳酸",
                "休克指数",
            ]

            existing_columns = [
                column for column in preferred_columns
                if column in display_df.columns
            ]

            other_columns = [
                column for column in display_df.columns
                if column not in existing_columns
            ]

            st.dataframe(
                display_df[existing_columns + other_columns],
                width="stretch",
            )

            high_risk_count = int((prediction_df["risk_group"] == "高风险").sum())
            intermediate_risk_count = int((prediction_df["risk_group"] == "中风险").sum())
            low_risk_count = int((prediction_df["risk_group"] == "低风险").sum())

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("总人数", len(prediction_df))

            with col2:
                st.metric("高风险人数", high_risk_count)

            with col3:
                st.metric("中风险人数", intermediate_risk_count)

            with col4:
                st.metric("低风险人数", low_risk_count)

            st.download_button(
                label="下载批量预测结果 CSV",
                data=dataframe_to_csv_bytes(display_df),
                file_name="clinical_risk_batch_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error("批量预测失败。请检查 CSV 列名、文件编码和变量取值。")
        st.exception(exc)


def show_dashboard_overview() -> None:
    st.header("模型评估总览")

    best_df = read_csv_if_exists(TABLE_DIR / "model_best_model_summary.csv")
    validation_df = read_csv_if_exists(TABLE_DIR / "model_leaderboard_validation.csv")
    test_df = read_csv_if_exists(TABLE_DIR / "model_leaderboard_test.csv")

    col1, col2, col3, col4 = st.columns(4)

    best_model_name = "暂无"
    best_auc = "暂无"

    if not best_df.empty:
        if "model_display_name" in best_df.columns:
            best_model_name = str(best_df.iloc[0]["model_display_name"])
        if "auc" in best_df.columns:
            best_auc = f"{float(best_df.iloc[0]['auc']):.3f}"

    with col1:
        st.metric("当前最佳模型", best_model_name)

    with col2:
        st.metric("验证集最佳 AUC", best_auc)

    with col3:
        st.metric("候选模型数量", len(validation_df) if not validation_df.empty else 0)

    with col4:
        st.metric("测试集记录数", len(test_df) if not test_df.empty else 0)

    show_table_if_exists(
        TABLE_DIR / "model_best_model_summary.csv",
        "当前最佳模型摘要",
        max_rows=5,
    )

    show_table_if_exists(
        TABLE_DIR / "model_leaderboard_validation.csv",
        "验证集模型排行榜",
        max_rows=20,
    )

    show_table_if_exists(
        TABLE_DIR / "model_leaderboard_test.csv",
        "测试集模型排行榜",
        max_rows=20,
    )


def show_dashboard_roc_pr() -> None:
    st.header("区分能力：ROC 与 PR 曲线")

    st.markdown(
        """
        ROC 曲线主要反映模型区分阳性和阴性样本的能力。  
        PR 曲线在类别不平衡时也很重要，尤其适合关注阳性事件预测性能。
        """
    )

    show_table_if_exists(
        TABLE_DIR / "model_comparison_roc_summary.csv",
        "ROC 对比摘要",
        max_rows=30,
    )

    roc_col1, roc_col2 = st.columns(2)

    with roc_col1:
        show_image_if_exists(
            FIGURE_DIR / "model_comparison_validation_roc_curve.png",
            "验证集 ROC 曲线对比",
        )

    with roc_col2:
        show_image_if_exists(
            FIGURE_DIR / "model_comparison_test_roc_curve.png",
            "测试集 ROC 曲线对比",
        )

    st.subheader("单模型 PR 曲线")

    pr_col1, pr_col2, pr_col3 = st.columns(3)

    with pr_col1:
        show_image_if_exists(
            FIGURE_DIR / "logistic_test_pr_curve.png",
            "Logistic Regression 测试集 PR 曲线",
        )

    with pr_col2:
        show_image_if_exists(
            FIGURE_DIR / "random_forest_test_pr_curve.png",
            "Random Forest 测试集 PR 曲线",
        )

    with pr_col3:
        show_image_if_exists(
            FIGURE_DIR / "gradient_boosting_test_pr_curve.png",
            "Gradient Boosting 测试集 PR 曲线",
        )


def show_dashboard_calibration_dca() -> None:
    st.header("校准能力与临床净获益")

    st.subheader("Calibration 校准评估")

    st.markdown(
        """
        Calibration 用来评估模型预测概率是否接近真实事件率。  
        Brier 分数越低，说明整体概率预测误差越小。
        """
    )

    show_table_if_exists(
        TABLE_DIR / "model_calibration_metrics.csv",
        "Calibration / Brier Score 摘要",
        max_rows=30,
    )

    cal_col1, cal_col2 = st.columns(2)

    with cal_col1:
        show_image_if_exists(
            FIGURE_DIR / "model_comparison_validation_calibration_curve.png",
            "验证集 Calibration 曲线",
        )

    with cal_col2:
        show_image_if_exists(
            FIGURE_DIR / "model_comparison_test_calibration_curve.png",
            "测试集 Calibration 曲线",
        )

    st.subheader("DCA 决策曲线分析")

    st.markdown(
        """
        DCA 用来评估模型在不同风险阈值下的临床净获益。  
        当模型曲线高于 Treat All 和 Treat None 时，说明该阈值范围内模型可能有临床实用价值。
        """
    )

    show_table_if_exists(
        TABLE_DIR / "dca_selected_threshold_summary.csv",
        "DCA 选定阈值摘要",
        max_rows=80,
    )

    dca_col1, dca_col2 = st.columns(2)

    with dca_col1:
        show_image_if_exists(
            FIGURE_DIR / "dca_validation_curve.png",
            "验证集 DCA 曲线",
        )

    with dca_col2:
        show_image_if_exists(
            FIGURE_DIR / "dca_test_curve.png",
            "测试集 DCA 曲线",
        )


def show_dashboard_shap() -> None:
    st.header("模型可解释性：SHAP 分析")

    st.markdown(
        """
        SHAP 用来解释模型预测时主要依赖哪些变量。  
        这里展示的是全局解释结果：整体上哪些特征对模型输出影响更大。

        注意：SHAP 解释的是模型行为，不代表因果关系。
        """
    )

    show_table_if_exists(
        TABLE_DIR / "shap_model_global_importance_all.csv",
        "SHAP 全局特征重要性合并表",
        max_rows=60,
        add_feature_label=True,
    )

    shap_bar_col1, shap_bar_col2 = st.columns(2)

    with shap_bar_col1:
        show_image_if_exists(
            FIGURE_DIR / "shap_random_forest_bar.png",
            "Random Forest SHAP 全局重要性条形图",
        )

    with shap_bar_col2:
        show_image_if_exists(
            FIGURE_DIR / "shap_gradient_boosting_bar.png",
            "Gradient Boosting SHAP 全局重要性条形图",
        )

    shap_bee_col1, shap_bee_col2 = st.columns(2)

    with shap_bee_col1:
        show_image_if_exists(
            FIGURE_DIR / "shap_random_forest_beeswarm.png",
            "Random Forest SHAP Beeswarm 图",
        )

    with shap_bee_col2:
        show_image_if_exists(
            FIGURE_DIR / "shap_gradient_boosting_beeswarm.png",
            "Gradient Boosting SHAP Beeswarm 图",
        )


def show_dashboard_reports() -> None:
    st.header("报告与关键文件")

    show_table_if_exists(
        TABLE_DIR / "comprehensive_model_report_file_index.csv",
        "综合报告文件索引",
        max_rows=100,
    )

    report_tab1, report_tab2, report_tab3, report_tab4 = st.tabs(
        [
            "综合模型报告",
            "DCA 报告",
            "SHAP 报告",
            "开发说明",
        ]
    )

    with report_tab1:
        show_report_preview(
            REPORT_DIR / "comprehensive_model_evaluation_report.txt",
            "综合模型评估报告预览",
        )

    with report_tab2:
        show_report_preview(
            REPORT_DIR / "dca_report.txt",
            "DCA 决策曲线分析报告预览",
        )

    with report_tab3:
        show_report_preview(
            REPORT_DIR / "shap_explainability_report.txt",
            "SHAP 可解释性报告预览",
        )

    with report_tab4:
        st.markdown(
            """
            当前仪表盘由前面多个模块共同支撑：

            - 数据预处理模块
            - Logistic Regression 模型
            - Random Forest 模型
            - Gradient Boosting 模型
            - ROC / PR / 混淆矩阵绘图模块
            - 模型排行榜模块
            - Calibration 模块
            - DCA 模块
            - SHAP 可解释性模块
            - Word / Markdown 报告导出模块

            仪表盘读取的是 `outputs` 文件夹下已经生成的表格、图片和报告。
            """
        )


def show_dashboard_tab() -> None:
    st.header("临床模型评估仪表盘")

    st.markdown(
        """
        这个页面集中展示当前临床预测模型的阶段性开发结果。  
        它主要用于项目内部查看、汇报和调试，不用于真实临床决策。
        """
    )

    artifact_tab1, artifact_tab2, artifact_tab3, artifact_tab4, artifact_tab5 = st.tabs(
        [
            "模型总览",
            "ROC / PR",
            "Calibration / DCA",
            "SHAP 解释",
            "报告与文件",
        ]
    )

    with artifact_tab1:
        show_dashboard_overview()

    with artifact_tab2:
        show_dashboard_roc_pr()

    with artifact_tab3:
        show_dashboard_calibration_dca()

    with artifact_tab4:
        show_dashboard_shap()

    with artifact_tab5:
        show_dashboard_reports()


def show_help_tab() -> None:
    st.header("使用说明")

    st.markdown(
        """
        ### 1. 单个患者预测

        在左侧输入患者指标，然后点击 **开始预测**。

        页面会输出：

        - poor_outcome 预测风险
        - 风险分层
        - 预测标签
        - 主要解释性特征

        ### 2. 批量 CSV 预测

        先下载模板，填写多名患者数据后上传 CSV。

        批量预测会输出：

        - 每名患者的预测风险
        - 风险分层
        - 风险排名
        - 可下载的预测结果 CSV

        ### 3. 模型评估仪表盘

        仪表盘会读取 `outputs` 文件夹中已经生成的结果，包括：

        - 模型排行榜
        - ROC 曲线
        - PR 曲线
        - Calibration 曲线
        - DCA 曲线
        - SHAP 图
        - 综合报告

        ### 4. 缺少文件时怎么办？

        补跑下面命令：

        ```powershell
        python scripts\\python\\build_model_leaderboard.py
        python scripts\\python\\plot_gradient_boosting_baseline.py
        python scripts\\python\\evaluate_model_calibration.py
        python scripts\\python\\run_decision_curve_analysis.py
        python scripts\\python\\run_shap_explainability.py
        python scripts\\python\\generate_comprehensive_model_report.py
        ```

        ### 5. 重要提示

        - 当前模型使用模拟数据训练。
        - 当前系统只适合工程演示。
        - 不能作为真实临床诊疗依据。
        - SHAP 解释模型行为，不代表因果关系。
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="BggDeepLearning 临床风险预测演示",
        page_icon="🧠",
        layout="wide",
    )

    st.title("BggDeepLearning 临床风险预测与模型评估仪表盘")

    st.markdown(
        """
        本页面用于演示结构化临床变量的 `poor_outcome` 风险预测，并集中展示模型评估结果。

        **重要提示：当前模型基于模拟数据训练，仅用于工程演示，不能用于真实临床决策。**
        """
    )

    st.sidebar.header("模型设置")

    model_key = st.sidebar.selectbox(
        "选择模型",
        options=[
            "gradient_boosting",
            "random_forest",
            "logistic",
        ],
        format_func=lambda value: MODEL_DISPLAY.get(value, value),
    )

    threshold_low = st.sidebar.slider(
        "低风险阈值",
        min_value=0.01,
        max_value=0.49,
        value=0.20,
        step=0.01,
    )

    threshold_high = st.sidebar.slider(
        "高风险阈值",
        min_value=0.20,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )

    if threshold_high <= threshold_low:
        st.sidebar.warning("高风险阈值应大于低风险阈值。")

    top_n_explanation = st.sidebar.slider(
        "解释性特征数量",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
    )

    tab_single, tab_batch, tab_dashboard, tab_help = st.tabs(
        [
            "单个患者预测",
            "批量 CSV 预测",
            "模型评估仪表盘",
            "使用说明",
        ]
    )

    with tab_single:
        show_single_prediction_tab(
            model_key=model_key,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            top_n_explanation=top_n_explanation,
        )

    with tab_batch:
        show_batch_prediction_tab(
            model_key=model_key,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            top_n_explanation=top_n_explanation,
        )

    with tab_dashboard:
        show_dashboard_tab()

    with tab_help:
        show_help_tab()


if __name__ == "__main__":
    main()