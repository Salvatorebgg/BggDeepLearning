-- ============================================================
-- BggDeepLearning core database schema
-- File: database/schema/create_core_tables.sql
--
-- Database: SQLite
-- Purpose:
-- 1. Create basic patient table
-- 2. Create clinical encounter table
-- 3. Create model prediction table
--
-- Important:
-- This schema is for research/demo use.
-- Do not store direct patient identifiers such as name, phone,
-- ID card number, home address, or medical record number.
-- ============================================================


PRAGMA foreign_keys = ON;


-- ------------------------------------------------------------
-- Table 1: patients
-- Basic de-identified patient information
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,

    study_patient_code TEXT NOT NULL UNIQUE,

    sex TEXT,
    age_at_enrollment REAL,

    height_cm REAL,
    weight_kg REAL,
    bmi REAL,

    data_source TEXT,
    enrollment_date TEXT,

    notes TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ------------------------------------------------------------
-- Table 2: clinical_encounters
-- One patient can have multiple clinical encounters
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS clinical_encounters (
    encounter_id INTEGER PRIMARY KEY AUTOINCREMENT,

    patient_id INTEGER NOT NULL,
    encounter_code TEXT NOT NULL UNIQUE,

    encounter_type TEXT,
    department TEXT,

    admission_datetime TEXT,
    discharge_datetime TEXT,

    heart_rate REAL,
    systolic_bp REAL,
    diastolic_bp REAL,
    respiratory_rate REAL,
    oxygen_saturation REAL,
    temperature_c REAL,

    hemoglobin REAL,
    white_blood_cell REAL,
    platelet REAL,
    creatinine REAL,
    lactate REAL,
    c_reactive_protein REAL,

    poor_outcome INTEGER,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (patient_id)
        REFERENCES patients(patient_id)
        ON DELETE CASCADE
);


-- ------------------------------------------------------------
-- Table 3: model_predictions
-- Store model prediction results
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,

    encounter_id INTEGER NOT NULL,

    model_name TEXT NOT NULL,
    model_version TEXT,
    prediction_task TEXT NOT NULL,

    predicted_probability REAL,
    predicted_label INTEGER,
    risk_level TEXT,

    prediction_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (encounter_id)
        REFERENCES clinical_encounters(encounter_id)
        ON DELETE CASCADE
);


-- ------------------------------------------------------------
-- Indexes
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_patients_study_code
ON patients(study_patient_code);

CREATE INDEX IF NOT EXISTS idx_encounters_patient_id
ON clinical_encounters(patient_id);

CREATE INDEX IF NOT EXISTS idx_encounters_code
ON clinical_encounters(encounter_code);

CREATE INDEX IF NOT EXISTS idx_predictions_encounter_id
ON model_predictions(encounter_id);