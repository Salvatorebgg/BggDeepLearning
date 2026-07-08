"""
BggDeepLearning tabular data preprocessing module

File:
python/bggdeep/data/preprocessing.py

Purpose:
1. Load clinical tabular data
2. Exclude identifier columns and leakage columns
3. Split data into train / validation / test sets
4. Fit preprocessing only on the training set
5. Apply preprocessing to validation and test sets
6. Save processed feature matrices, labels, feature names, and reports

Important:
To prevent data leakage:
- train/val/test split happens before fitting imputers, scalers, and encoders
- preprocessing is fitted only on the training set
- validation and test sets are only transformed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TabularPreprocessingConfig:
    """
    Configuration for tabular preprocessing.
    """

    target_column: str = "poor_outcome"

    id_columns: List[str] = field(
        default_factory=lambda: [
            "study_patient_code",
            "encounter_code",
        ]
    )

    leakage_columns: List[str] = field(
        default_factory=lambda: [
            "poor_outcome_probability_demo",
        ]
    )

    treatment_columns: List[str] = field(
        default_factory=lambda: [
            "vasopressor_used",
            "mechanical_ventilation",
        ]
    )

    exclude_treatment_columns: bool = True

    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42


class ClinicalTabularPreprocessor:
    """
    Clinical tabular data preprocessor.

    Main steps:
    1. Validate input data
    2. Select usable feature columns
    3. Split data into train/val/test
    4. Fit preprocessing on training data only
    5. Transform train/val/test data
    """

    def __init__(self, config: TabularPreprocessingConfig) -> None:
        self.config = config
        self.preprocessor: ColumnTransformer | None = None
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.feature_columns: List[str] = []
        self.excluded_columns: List[str] = []
        self.processed_feature_names: List[str] = []

    def run(self, df: pd.DataFrame) -> Dict[str, object]:
        """
        Run full preprocessing workflow.
        """
        self._validate_input_dataframe(df)

        feature_df, target_series, id_df = self._select_features_target_ids(df)

        (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
            id_train,
            id_val,
            id_test,
        ) = self._split_data(feature_df, target_series, id_df)

        self.numeric_columns, self.categorical_columns = self._infer_column_types(x_train)

        self.preprocessor = self._build_preprocessor(
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
        )

        x_train_processed = self.preprocessor.fit_transform(x_train)
        x_val_processed = self.preprocessor.transform(x_val)
        x_test_processed = self.preprocessor.transform(x_test)

        self.processed_feature_names = self._get_processed_feature_names()

        train_features = pd.DataFrame(
            x_train_processed,
            columns=self.processed_feature_names,
        )

        val_features = pd.DataFrame(
            x_val_processed,
            columns=self.processed_feature_names,
        )

        test_features = pd.DataFrame(
            x_test_processed,
            columns=self.processed_feature_names,
        )

        train_labels = pd.DataFrame({self.config.target_column: y_train.reset_index(drop=True)})
        val_labels = pd.DataFrame({self.config.target_column: y_val.reset_index(drop=True)})
        test_labels = pd.DataFrame({self.config.target_column: y_test.reset_index(drop=True)})

        split_summary = self._build_split_summary(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

        feature_table = self._build_feature_table()
        excluded_table = self._build_excluded_column_table()

        return {
            "train_features": train_features,
            "val_features": val_features,
            "test_features": test_features,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "test_labels": test_labels,
            "train_ids": id_train.reset_index(drop=True),
            "val_ids": id_val.reset_index(drop=True),
            "test_ids": id_test.reset_index(drop=True),
            "split_summary": split_summary,
            "feature_table": feature_table,
            "excluded_table": excluded_table,
            "preprocessor": self.preprocessor,
        }

    def _validate_input_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate basic input requirements.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column was not found: {self.config.target_column}"
            )

        target_unique_count = df[self.config.target_column].nunique(dropna=True)

        if target_unique_count < 2:
            raise ValueError(
                f"Target column must have at least two classes. "
                f"Current unique classes: {target_unique_count}"
            )

    def _select_features_target_ids(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Select feature columns, target column, and ID columns.
        """
        target_column = self.config.target_column

        excluded = set()
        excluded.add(target_column)

        for column in self.config.id_columns:
            if column in df.columns:
                excluded.add(column)

        for column in self.config.leakage_columns:
            if column in df.columns:
                excluded.add(column)

        if self.config.exclude_treatment_columns:
            for column in self.config.treatment_columns:
                if column in df.columns:
                    excluded.add(column)

        self.excluded_columns = sorted(list(excluded))

        feature_columns = [
            column for column in df.columns
            if column not in excluded
        ]

        self.feature_columns = feature_columns

        available_id_columns = [
            column for column in self.config.id_columns
            if column in df.columns
        ]

        if available_id_columns:
            id_df = df[available_id_columns].copy()
        else:
            id_df = pd.DataFrame({"row_index": range(len(df))})

        feature_df = df[feature_columns].copy()
        target_series = df[target_column].copy()

        return feature_df, target_series, id_df

    def _split_data(
        self,
        feature_df: pd.DataFrame,
        target_series: pd.Series,
        id_df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Split data into train / validation / test sets.

        Final ratio:
        train = 60%
        validation = 20%
        test = 20%
        """
        test_size = self.config.test_size
        val_size = self.config.val_size

        if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
            raise ValueError("Invalid split sizes. test_size + val_size must be < 1.")

        x_temp, x_test, y_temp, y_test, id_temp, id_test = train_test_split(
            feature_df,
            target_series,
            id_df,
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=target_series,
        )

        val_ratio_in_temp = val_size / (1.0 - test_size)

        x_train, x_val, y_train, y_val, id_train, id_val = train_test_split(
            x_temp,
            y_temp,
            id_temp,
            test_size=val_ratio_in_temp,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        return (
            x_train.reset_index(drop=True),
            x_val.reset_index(drop=True),
            x_test.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_val.reset_index(drop=True),
            y_test.reset_index(drop=True),
            id_train.reset_index(drop=True),
            id_val.reset_index(drop=True),
            id_test.reset_index(drop=True),
        )

    def _infer_column_types(self, x_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Infer numeric and categorical columns from training data.
        """
        numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()

        categorical_columns = [
            column for column in x_train.columns
            if column not in numeric_columns
        ]

        return numeric_columns, categorical_columns

    def _build_preprocessor(
        self,
        numeric_columns: List[str],
        categorical_columns: List[str],
    ) -> ColumnTransformer:
        """
        Build sklearn preprocessing pipeline.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )

        return preprocessor

    def _get_processed_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted.")

        names = self.preprocessor.get_feature_names_out()
        return [str(name) for name in names]

    def _build_split_summary(
        self,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Build train/val/test split summary.
        """
        rows = []

        for split_name, y in [
            ("train", y_train),
            ("validation", y_val),
            ("test", y_test),
        ]:
            rows.append(
                {
                    "split": split_name,
                    "n_rows": int(len(y)),
                    "positive_count": int((y == 1).sum()),
                    "negative_count": int((y == 0).sum()),
                    "positive_rate": round(float(y.mean()), 4),
                }
            )

        return pd.DataFrame(rows)

    def _build_feature_table(self) -> pd.DataFrame:
        """
        Build feature metadata table.
        """
        rows = []

        for column in self.numeric_columns:
            rows.append(
                {
                    "original_feature": column,
                    "feature_type": "numeric",
                    "preprocessing": "median_imputation + standard_scaling",
                }
            )

        for column in self.categorical_columns:
            rows.append(
                {
                    "original_feature": column,
                    "feature_type": "categorical",
                    "preprocessing": "most_frequent_imputation + one_hot_encoding",
                }
            )

        processed_rows = [
            {
                "processed_feature_name": name,
            }
            for name in self.processed_feature_names
        ]

        original_table = pd.DataFrame(rows)
        processed_table = pd.DataFrame(processed_rows)

        original_table["table_type"] = "original_features"
        processed_table["table_type"] = "processed_features"

        return pd.concat(
            [original_table, processed_table],
            ignore_index=True,
            sort=False,
        )

    def _build_excluded_column_table(self) -> pd.DataFrame:
        """
        Build excluded column table.
        """
        rows = []

        for column in self.excluded_columns:
            if column == self.config.target_column:
                reason = "target_column"
            elif column in self.config.id_columns:
                reason = "identifier_column"
            elif column in self.config.leakage_columns:
                reason = "potential_data_leakage"
            elif column in self.config.treatment_columns:
                reason = "excluded_treatment_column_for_early_prediction"
            else:
                reason = "excluded"

            rows.append(
                {
                    "excluded_column": column,
                    "reason": reason,
                }
            )

        return pd.DataFrame(rows)


def save_preprocessing_outputs(
    results: Dict[str, object],
    processed_data_dir: Path,
    table_dir: Path,
    report_dir: Path,
    model_dir: Path,
    config: TabularPreprocessingConfig,
) -> Dict[str, Path]:
    """
    Save processed datasets and preprocessing artifacts.
    """
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train_features": processed_data_dir / "tabular_train_features.csv",
        "val_features": processed_data_dir / "tabular_val_features.csv",
        "test_features": processed_data_dir / "tabular_test_features.csv",
        "train_labels": processed_data_dir / "tabular_train_labels.csv",
        "val_labels": processed_data_dir / "tabular_val_labels.csv",
        "test_labels": processed_data_dir / "tabular_test_labels.csv",
        "train_ids": processed_data_dir / "tabular_train_ids.csv",
        "val_ids": processed_data_dir / "tabular_val_ids.csv",
        "test_ids": processed_data_dir / "tabular_test_ids.csv",
        "split_summary": table_dir / "preprocessing_split_summary.csv",
        "feature_table": table_dir / "preprocessing_feature_table.csv",
        "excluded_table": table_dir / "preprocessing_excluded_columns.csv",
        "config_json": report_dir / "preprocessing_config.json",
        "report_txt": report_dir / "preprocessing_report.txt",
        "preprocessor_joblib": model_dir / "tabular_preprocessor.joblib",
    }

    dataframe_keys = [
        "train_features",
        "val_features",
        "test_features",
        "train_labels",
        "val_labels",
        "test_labels",
        "train_ids",
        "val_ids",
        "test_ids",
        "split_summary",
        "feature_table",
        "excluded_table",
    ]

    for key in dataframe_keys:
        dataframe = results[key]
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for key: {key}")

        dataframe.to_csv(
            output_paths[key],
            index=False,
            encoding="utf-8-sig",
        )

    with output_paths["config_json"].open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2, ensure_ascii=False)

    preprocessor = results["preprocessor"]
    joblib.dump(preprocessor, output_paths["preprocessor_joblib"])

    report_text = build_preprocessing_report(results, config, output_paths)
    output_paths["report_txt"].write_text(report_text, encoding="utf-8")

    return output_paths


def build_preprocessing_report(
    results: Dict[str, object],
    config: TabularPreprocessingConfig,
    output_paths: Dict[str, Path],
) -> str:
    """
    Build plain text preprocessing report.
    """
    split_summary = results["split_summary"]
    feature_table = results["feature_table"]
    excluded_table = results["excluded_table"]

    train_features = results["train_features"]
    val_features = results["val_features"]
    test_features = results["test_features"]

    lines = []
    lines.append("=" * 70)
    lines.append("BggDeepLearning Tabular Preprocessing Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Target")
    lines.append("-" * 70)
    lines.append(f"Target column: {config.target_column}")
    lines.append("")
    lines.append("2. Data Split")
    lines.append("-" * 70)
    lines.append(split_summary.to_string(index=False))
    lines.append("")
    lines.append("3. Processed Feature Shape")
    lines.append("-" * 70)
    lines.append(f"Train features shape: {train_features.shape}")
    lines.append(f"Validation features shape: {val_features.shape}")
    lines.append(f"Test features shape: {test_features.shape}")
    lines.append("")
    lines.append("4. Original Feature Processing Preview")
    lines.append("-" * 70)
    lines.append(feature_table.head(30).to_string(index=False))
    lines.append("")
    lines.append("5. Excluded Columns")
    lines.append("-" * 70)
    lines.append(excluded_table.to_string(index=False))
    lines.append("")
    lines.append("6. Output Files")
    lines.append("-" * 70)

    for name, path in output_paths.items():
        lines.append(f"{name}: {path}")

    lines.append("")
    lines.append("7. Leakage Prevention Notes")
    lines.append("-" * 70)
    lines.append("Train/validation/test split was performed before preprocessing.")
    lines.append("Imputation, scaling, and one-hot encoding were fitted only on training data.")
    lines.append("Validation and test sets were transformed using the training-fitted preprocessor.")
    lines.append("Identifier columns and potential leakage columns were excluded.")
    lines.append("=" * 70)

    return "\n".join(lines)