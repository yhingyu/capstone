"""Log the completed Step 4 run to a local MLflow tracking directory."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "models/step4_run_metadata.json"


def main() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("household-energy-capstone")
    with mlflow.start_run(run_name="step4-final-xgboost"):
        mlflow.log_param("selected_model", metadata["selected_model"])
        mlflow.log_param("selection_rule", metadata["selection_rule"])
        mlflow.log_params(metadata["configuration"])
        mlflow.log_metrics(
            {
                "test_mae_kwh": metadata["final_test_metrics"]["MAE_kWh"],
                "test_rmse_kwh": metadata["final_test_metrics"]["RMSE_kWh"],
                "test_mape_percent": metadata["final_test_metrics"]["MAPE_percent"],
                "test_r2": metadata["final_test_metrics"]["R2"],
                "warning_precision": metadata["warning_metrics"]["precision"],
                "warning_recall": metadata["warning_metrics"]["recall_coverage"],
            }
        )
        mlflow.log_artifact(str(METADATA), artifact_path="metadata")
        mlflow.log_artifact(str(ROOT / "models/feature_columns.json"), artifact_path="model-schema")
        mlflow.log_artifact(str(ROOT / "outputs/tables/step4_final_test_metrics.csv"), artifact_path="metrics")
    print("Logged Step 4 metadata to ./mlruns")


if __name__ == "__main__":
    main()
