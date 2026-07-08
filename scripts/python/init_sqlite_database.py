"""
BggDeepLearning SQLite database initialization script

File:
scripts/python/init_sqlite_database.py

Purpose:
1. Locate project root
2. Read database/schema/create_core_tables.sql
3. Create database/sqlite/bgg_clinical.db
4. Execute SQL schema
5. Insert a small demo patient and encounter
6. Print table information

Usage:
python scripts\\python\\init_sqlite_database.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def find_project_root() -> Path:
    """
    Find project root by locating configs/app.yaml.
    """
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / "configs" / "app.yaml").exists():
            return parent

    raise FileNotFoundError("Project root was not found. configs/app.yaml is missing.")


def initialize_database(project_root: Path) -> Path:
    """
    Create SQLite database and execute schema file.
    """
    schema_file = project_root / "database" / "schema" / "create_core_tables.sql"
    sqlite_dir = project_root / "database" / "sqlite"
    database_file = sqlite_dir / "bgg_clinical.db"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file was not found: {schema_file}")

    sqlite_dir.mkdir(parents=True, exist_ok=True)

    schema_sql = schema_file.read_text(encoding="utf-8")

    with sqlite3.connect(database_file) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.executescript(schema_sql)
        connection.commit()

    return database_file


def insert_demo_data(database_file: Path) -> None:
    """
    Insert one demo patient, one encounter, and one prediction.

    The insert uses INSERT OR IGNORE to avoid duplicate errors
    when the script is run multiple times.
    """
    with sqlite3.connect(database_file) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")

        connection.execute(
            """
            INSERT OR IGNORE INTO patients (
                study_patient_code,
                sex,
                age_at_enrollment,
                height_cm,
                weight_kg,
                bmi,
                data_source,
                enrollment_date,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "DEMO_PATIENT_001",
                "male",
                68,
                170,
                72,
                24.9,
                "simulated",
                "2026-01-01",
                "Demo patient for database initialization.",
            ),
        )

        patient_id_row = connection.execute(
            """
            SELECT patient_id
            FROM patients
            WHERE study_patient_code = ?;
            """,
            ("DEMO_PATIENT_001",),
        ).fetchone()

        if patient_id_row is None:
            raise RuntimeError("Failed to retrieve demo patient_id.")

        patient_id = patient_id_row[0]

        connection.execute(
            """
            INSERT OR IGNORE INTO clinical_encounters (
                patient_id,
                encounter_code,
                encounter_type,
                department,
                admission_datetime,
                heart_rate,
                systolic_bp,
                diastolic_bp,
                respiratory_rate,
                oxygen_saturation,
                temperature_c,
                hemoglobin,
                white_blood_cell,
                platelet,
                creatinine,
                lactate,
                c_reactive_protein,
                poor_outcome
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                patient_id,
                "DEMO_ENCOUNTER_001",
                "emergency",
                "ICU",
                "2026-01-01 08:00:00",
                118,
                88,
                55,
                24,
                91,
                38.2,
                98,
                14.5,
                168,
                115,
                4.6,
                86,
                1,
            ),
        )

        encounter_id_row = connection.execute(
            """
            SELECT encounter_id
            FROM clinical_encounters
            WHERE encounter_code = ?;
            """,
            ("DEMO_ENCOUNTER_001",),
        ).fetchone()

        if encounter_id_row is None:
            raise RuntimeError("Failed to retrieve demo encounter_id.")

        encounter_id = encounter_id_row[0]

        connection.execute(
            """
            INSERT INTO model_predictions (
                encounter_id,
                model_name,
                model_version,
                prediction_task,
                predicted_probability,
                predicted_label,
                risk_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                encounter_id,
                "demo_rule_based_model",
                "0.1.0",
                "poor_outcome_prediction",
                0.76,
                1,
                "high",
            ),
        )

        connection.commit()


def print_database_summary(database_file: Path) -> None:
    """
    Print database tables and row counts.
    """
    with sqlite3.connect(database_file) as connection:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name;
            """
        ).fetchall()

        print("=" * 60)
        print("BggDeepLearning SQLite database initialized successfully")
        print("-" * 60)
        print(f"Database file: {database_file}")
        print("-" * 60)

        for table_row in tables:
            table_name = table_row[0]
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name};"
            ).fetchone()[0]
            print(f"Table: {table_name} | rows: {row_count}")

        print("=" * 60)


def main() -> None:
    project_root = find_project_root()
    database_file = initialize_database(project_root)
    insert_demo_data(database_file)
    print_database_summary(database_file)


if __name__ == "__main__":
    main()