"""
BggDeepLearning clinical data simulation module

File:
python/bggdeep/data/simulation.py

Purpose:
1. Generate simulated clinical tabular data
2. Generate data dictionary
3. Save simulated data to CSV
4. Provide data for later cleaning, statistics, ML, and DL modules

Important:
This module only creates simulated research/demo data.
It does not represent a validated clinical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    """
    Configuration for simulated clinical data.

    n_patients:
        Number of simulated patients.

    seed:
        Random seed. Same seed means reproducible data.

    missing_rate:
        Basic missing value rate.
    """

    n_patients: int = 500
    seed: int = 42
    missing_rate: float = 0.03


class ClinicalDataSimulator:
    """
    Clinical tabular data simulator.

    This simulator creates a simple critical care / emergency style dataset.

    Outcome:
        poor_outcome
        0 = no severe adverse outcome
        1 = severe adverse outcome

    The generated variables include:
    1. Basic demographics
    2. Comorbidities
    3. Vital signs
    4. Laboratory tests
    5. Treatment indicators
    6. Outcome label
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Convert linear risk score to probability.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def generate(self) -> pd.DataFrame:
        """
        Generate simulated clinical data.

        Returns:
            pandas DataFrame
        """
        n = self.config.n_patients
        rng = self.rng

        patient_code = [f"SIM_PATIENT_{i + 1:05d}" for i in range(n)]
        encounter_code = [f"SIM_ENCOUNTER_{i + 1:05d}" for i in range(n)]

        age = np.clip(rng.normal(loc=62, scale=15, size=n), 18, 90).round(1)

        sex = rng.choice(
            ["male", "female"],
            size=n,
            p=[0.55, 0.45],
        )

        bmi = np.clip(rng.normal(loc=24.5, scale=4.2, size=n), 15, 42).round(1)

        admission_type = rng.choice(
            ["emergency", "elective", "transfer"],
            size=n,
            p=[0.62, 0.23, 0.15],
        )

        hypertension = rng.binomial(1, np.clip(0.15 + age / 150, 0.05, 0.80))
        diabetes = rng.binomial(1, np.clip(0.08 + age / 260, 0.03, 0.45))
        coronary_disease = rng.binomial(1, np.clip(0.04 + age / 320, 0.01, 0.35))
        chronic_kidney_disease = rng.binomial(1, np.clip(0.03 + age / 400, 0.01, 0.28))

        infection_suspected = rng.binomial(1, 0.34, size=n)
        trauma_suspected = rng.binomial(1, 0.18, size=n)

        heart_rate = np.clip(
            rng.normal(loc=86, scale=18, size=n)
            + infection_suspected * 10
            + trauma_suspected * 8,
            40,
            180,
        ).round(0)

        systolic_bp = np.clip(
            rng.normal(loc=118, scale=22, size=n)
            - trauma_suspected * 10
            - infection_suspected * 5,
            60,
            210,
        ).round(0)

        diastolic_bp = np.clip(
            systolic_bp * 0.62 + rng.normal(loc=0, scale=8, size=n),
            35,
            130,
        ).round(0)

        respiratory_rate = np.clip(
            rng.normal(loc=19, scale=5, size=n)
            + infection_suspected * 3,
            8,
            45,
        ).round(0)

        oxygen_saturation = np.clip(
            rng.normal(loc=96, scale=3.5, size=n)
            - infection_suspected * 2
            - trauma_suspected * 1,
            70,
            100,
        ).round(0)

        temperature_c = np.clip(
            rng.normal(loc=36.8, scale=0.7, size=n)
            + infection_suspected * 0.8,
            34.0,
            41.0,
        ).round(1)

        hemoglobin = np.clip(
            rng.normal(loc=125, scale=22, size=n)
            - trauma_suspected * 15,
            45,
            190,
        ).round(1)

        white_blood_cell = np.clip(
            rng.normal(loc=8.2, scale=3.2, size=n)
            + infection_suspected * 5.0,
            1.0,
            40.0,
        ).round(1)

        platelet = np.clip(
            rng.normal(loc=210, scale=70, size=n),
            20,
            600,
        ).round(0)

        creatinine = np.clip(
            rng.normal(loc=78, scale=28, size=n)
            + chronic_kidney_disease * 55,
            25,
            450,
        ).round(1)

        lactate = np.clip(
            rng.gamma(shape=2.2, scale=0.9, size=n)
            + trauma_suspected * 0.9
            + infection_suspected * 0.7,
            0.4,
            12.0,
        ).round(2)

        c_reactive_protein = np.clip(
            rng.gamma(shape=2.5, scale=12.0, size=n)
            + infection_suspected * 45,
            0.1,
            300,
        ).round(1)

        shock_index = np.round(heart_rate / np.maximum(systolic_bp, 1), 3)

        # 这里是一个演示用风险公式，不是真实医学模型
        risk_logit = (
            -5.2
            + 0.025 * (age - 60)
            + 0.018 * (heart_rate - 85)
            - 0.035 * (systolic_bp - 110)
            + 0.55 * (lactate - 2.0)
            + 0.012 * (c_reactive_protein - 20)
            + 0.35 * diabetes
            + 0.40 * chronic_kidney_disease
            + 0.45 * infection_suspected
            + 0.38 * trauma_suspected
            + 0.65 * (shock_index > 1.0)
            - 0.05 * (oxygen_saturation - 94)
        )

        poor_outcome_probability = self._sigmoid(risk_logit)
        poor_outcome = rng.binomial(1, poor_outcome_probability)

        vasopressor_used = rng.binomial(
            1,
            np.clip(0.05 + poor_outcome_probability * 0.45, 0, 0.75),
        )

        mechanical_ventilation = rng.binomial(
            1,
            np.clip(0.04 + poor_outcome_probability * 0.35, 0, 0.70),
        )

        df = pd.DataFrame(
            {
                "study_patient_code": patient_code,
                "encounter_code": encounter_code,
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
                "vasopressor_used": vasopressor_used,
                "mechanical_ventilation": mechanical_ventilation,
                "poor_outcome_probability_demo": np.round(poor_outcome_probability, 4),
                "poor_outcome": poor_outcome,
            }
        )

        df = self._add_missing_values(df)

        return df

    def _add_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a small amount of missing values.

        We do not add missing values to:
        1. patient code
        2. encounter code
        3. outcome
        4. demo probability
        """
        missing_candidates = [
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
        ]

        result = df.copy()

        for column in missing_candidates:
            mask = self.rng.random(len(result)) < self.config.missing_rate
            result.loc[mask, column] = np.nan

        return result


def build_data_dictionary() -> pd.DataFrame:
    """
    Build data dictionary for simulated clinical dataset.
    """
    rows = [
        ("study_patient_code", "string", "De-identified patient code", "", "identifier"),
        ("encounter_code", "string", "De-identified encounter code", "", "identifier"),
        ("age", "numeric", "Age at enrollment", "years", "feature"),
        ("sex", "categorical", "Biological sex in simulated data", "", "feature"),
        ("bmi", "numeric", "Body mass index", "kg/m2", "feature"),
        ("admission_type", "categorical", "Type of admission", "", "feature"),
        ("hypertension", "binary", "History of hypertension", "0/1", "feature"),
        ("diabetes", "binary", "History of diabetes", "0/1", "feature"),
        ("coronary_disease", "binary", "History of coronary disease", "0/1", "feature"),
        ("chronic_kidney_disease", "binary", "History of chronic kidney disease", "0/1", "feature"),
        ("infection_suspected", "binary", "Suspected infection at admission", "0/1", "feature"),
        ("trauma_suspected", "binary", "Suspected trauma at admission", "0/1", "feature"),
        ("heart_rate", "numeric", "Heart rate", "beats/min", "feature"),
        ("systolic_bp", "numeric", "Systolic blood pressure", "mmHg", "feature"),
        ("diastolic_bp", "numeric", "Diastolic blood pressure", "mmHg", "feature"),
        ("respiratory_rate", "numeric", "Respiratory rate", "breaths/min", "feature"),
        ("oxygen_saturation", "numeric", "Peripheral oxygen saturation", "%", "feature"),
        ("temperature_c", "numeric", "Body temperature", "Celsius", "feature"),
        ("hemoglobin", "numeric", "Hemoglobin", "g/L", "feature"),
        ("white_blood_cell", "numeric", "White blood cell count", "10^9/L", "feature"),
        ("platelet", "numeric", "Platelet count", "10^9/L", "feature"),
        ("creatinine", "numeric", "Serum creatinine", "umol/L", "feature"),
        ("lactate", "numeric", "Blood lactate", "mmol/L", "feature"),
        ("c_reactive_protein", "numeric", "C-reactive protein", "mg/L", "feature"),
        ("shock_index", "numeric", "Heart rate divided by systolic blood pressure", "", "derived_feature"),
        ("vasopressor_used", "binary", "Whether vasopressor was used", "0/1", "treatment"),
        ("mechanical_ventilation", "binary", "Whether mechanical ventilation was used", "0/1", "treatment"),
        ("poor_outcome_probability_demo", "numeric", "Demo probability used during simulation", "0-1", "simulation_reference"),
        ("poor_outcome", "binary", "Severe adverse clinical outcome", "0/1", "target"),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "variable_name",
            "variable_type",
            "description",
            "unit",
            "role",
        ],
    )


def summarize_simulated_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple summary table.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    outcome_rate = float(df["poor_outcome"].mean())

    missing_summary = df.isna().mean().sort_values(ascending=False)
    max_missing_rate = float(missing_summary.iloc[0])

    summary = pd.DataFrame(
        {
            "item": [
                "n_rows",
                "n_columns",
                "poor_outcome_rate",
                "max_missing_rate",
                "mean_age",
                "mean_lactate",
                "mean_shock_index",
            ],
            "value": [
                n_rows,
                n_cols,
                round(outcome_rate, 4),
                round(max_missing_rate, 4),
                round(float(df["age"].mean()), 2),
                round(float(df["lactate"].mean()), 2),
                round(float(df["shock_index"].mean()), 3),
            ],
        }
    )

    return summary


def save_simulation_outputs(
    df: pd.DataFrame,
    data_dictionary: pd.DataFrame,
    summary: pd.DataFrame,
    simulated_data_dir: Path,
    table_dir: Path,
) -> Tuple[Path, Path, Path]:
    """
    Save simulated data, data dictionary, and summary table.
    """
    simulated_data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    data_file = simulated_data_dir / "clinical_simulated_data.csv"
    dictionary_file = simulated_data_dir / "clinical_simulated_data_dictionary.csv"
    summary_file = table_dir / "clinical_simulated_summary.csv"

    df.to_csv(data_file, index=False, encoding="utf-8-sig")
    data_dictionary.to_csv(dictionary_file, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    return data_file, dictionary_file, summary_file