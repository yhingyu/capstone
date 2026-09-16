"""Validate the repository structure and key Step 7 deliverables."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "environment.yml",
    "data/README.md",
    "notebooks/02_data_collection_and_understanding.ipynb",
    "notebooks/03_preprocessing_eda_feature_engineering.ipynb",
    "notebooks/04_model_implementation_and_comparison.ipynb",
    "notebooks/05_bias_fairness_analysis.ipynb",
    "models/final_energy_forecast_model.joblib",
    "models/feature_columns.json",
    "models/step4_run_metadata.json",
    "outputs/model_comparison.csv",
    "outputs/reports/01_Problem_Understanding_and_Framing.docx",
    "outputs/reports/07_Final_Capstone_Report_Household_Energy_Monitoring.docx",
    "outputs/reports/07_Final_Capstone_Report_Household_Energy_Monitoring.pdf",
    "presentations/06A_Technical_Presentation_Household_Energy.ipynb",
    "presentations/06B_Business_Presentation_Household_Energy.pptx",
    "Dockerfile",
    "docker-compose.yml",
    "requirements-app.txt",
    "app/app.py",
    "config/app.yaml",
    "docs/STEP8_DEPLOYMENT.md",
    "docs/MONITORING_AND_ROLLBACK.md",
    "data/sample/daily_readings_example.csv",
]

JSON_FILES = [
    "models/feature_columns.json",
    "models/selected_features.json",
    "models/step3_config.json",
    "models/step4_best_parameters.json",
    "models/step4_run_metadata.json",
]

CSV_FILES = [
    "outputs/model_comparison.csv",
    "outputs/tables/step4_final_test_metrics.csv",
]


def validate() -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing required file: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"Empty required file: {relative}")

    for relative in JSON_FILES:
        path = ROOT / relative
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as stream:
                json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid JSON {relative}: {exc}")

    for relative in CSV_FILES:
        path = ROOT / relative
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            if len(rows) < 2 or not rows[0]:
                errors.append(f"CSV has no data rows: {relative}")
        except (OSError, csv.Error) as exc:
            errors.append(f"Invalid CSV {relative}: {exc}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Step 7 repository validation FAILED:")
        for error in errors:
            print(f" - {error}")
        return 1
    print("Step 7 repository validation PASSED.")
    print(f"Checked {len(REQUIRED_FILES)} required deliverables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
