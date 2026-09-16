# Step 8 — Dockerized Deployment and MLOps Proof of Concept

## Scope

This is a **local proof of concept**, not a production billing system. It loads the Step 4 XGBoost model and forecasts the next day's household electricity consumption. It does not reproduce an official Meralco bill, control appliances, or provide a dependable bill-shock alarm.

## Inputs

The dashboard accepts CSV or Excel files containing one of these schemas:

| Input type | Required columns | Optional columns |
|---|---|---|
| Daily consumption | `date`, `daily_kwh` (or `kwh`) | `voltage_std_v` |
| Cumulative meter | `date`/`timestamp`, `meter_reading_kwh` | — |
| Timestamped power | `timestamp`, `power_kw` | `voltage_v` |
| Original UCI | `Date`, `Time`, `Global_active_power` | `Voltage` |

Manual input accepts a date, daily kWh, and optional voltage variability. At least 30 consecutive daily consumption values are required because the model uses 30-day rolling features.

## Run with Docker

Prerequisites: Docker Desktop or Docker Engine with Compose.

```bash
git clone https://github.com/yhingyu/capstone.git
cd capstone
docker compose up --build
```

Open <http://localhost:8050>. Verify service health at <http://localhost:8050/health>.

Stop the application:

```bash
docker compose down
```

Prediction audit events are retained locally in `data/monitoring/predictions.jsonl` through a Docker bind mount.

## Run without Docker

```bash
conda activate household-energy-capstone
python -m pip install -r requirements-app.txt
python app/app.py
```

## Configuration

`config/app.yaml` controls:

- model and feature-schema paths;
- model version;
- prediction-band radius;
- default rate used only for an illustrative cost estimate;
- warning thresholds;
- monitoring-log location.

Changing warning thresholds does not change the saved Step 4 evaluation metrics. Any new thresholds must be validated and reported separately.

## Tests and quality checks

```bash
python -m pip install -r requirements-dev.txt
ruff check app tests src
pytest -q
python src/validate_repository.py
```

GitHub Actions performs syntax checks, linting, unit tests, and repository validation on pushes and pull requests.

## Local MLflow record

The completed Step 4 experiment can be registered locally:

```bash
python -m pip install -r requirements-mlops.txt
python src/log_step4_experiment.py
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open <http://localhost:5000>. The `mlruns/` directory is local and should not be committed.

## Recommended production setup (not implemented)

For a future Philippine pilot, use a managed container service such as AWS ECS/Fargate, Azure Container Apps, or Google Cloud Run behind TLS and identity-aware access. Store encrypted readings in a managed database, put the model in a versioned artifact registry, centralize logs and metrics, manage secrets outside the image, and use separate development/staging/production environments. Start with silent forecasts before enabling alerts.

Cloud deployment is deliberately out of scope for this POC.

## Demo media

The screencast/GIF will be added after the app is tested interactively. See `demo/README.md` for the recording checklist.
