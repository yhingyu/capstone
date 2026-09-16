"""Model loading, next-day forecasting, advisory logic, and local audit logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import build_next_day_features


@dataclass(frozen=True)
class Forecast:
    target_date: str
    predicted_kwh: float
    lower_kwh: float
    upper_kwh: float
    baseline_kwh: float
    high_use_threshold_kwh: float
    advisory: str
    utilization_percent: float
    estimated_daily_cost: float
    projected_30_day_cost: float
    warnings: tuple[str, ...]


def advisory_level(prediction: float, high_threshold: float, levels: dict[str, float]) -> tuple[str, float]:
    ratio = prediction / high_threshold if high_threshold > 0 else 0.0
    if ratio >= levels["high"]:
        return "HIGH", ratio
    if ratio >= levels["moderate"]:
        return "MODERATE", ratio
    if ratio >= levels["watch"]:
        return "WATCH", ratio
    return "NORMAL", ratio


class EnergyPredictor:
    def __init__(self, config: dict[str, Any], root: Path):
        self.config = config
        self.root = root
        model_path = root / config["model"]["path"]
        feature_path = root / config["model"]["feature_columns_path"]
        self.model = joblib.load(model_path)
        self.feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))

    def forecast(self, history: pd.DataFrame, rate_per_kwh: float) -> Forecast:
        result = build_next_day_features(
            history,
            self.feature_columns,
            training_origin=self.config["features"]["training_origin"],
            default_voltage_std_v=float(self.config["features"]["default_voltage_std_v"]),
        )
        prediction = max(0.0, float(self.model.predict(result.features)[0]))
        radius = float(self.config["model"]["prediction_interval_radius_kwh"])
        threshold = result.baseline_kwh * float(self.config["advisory"]["high_use_multiplier"])
        level, ratio = advisory_level(prediction, threshold, self.config["advisory"]["levels"])
        return Forecast(
            target_date=result.target_date.date().isoformat(),
            predicted_kwh=prediction,
            lower_kwh=max(0.0, prediction - radius),
            upper_kwh=prediction + radius,
            baseline_kwh=result.baseline_kwh,
            high_use_threshold_kwh=threshold,
            advisory=level,
            utilization_percent=ratio * 100,
            estimated_daily_cost=prediction * rate_per_kwh,
            projected_30_day_cost=prediction * rate_per_kwh * 30,
            warnings=result.warnings,
        )


def log_forecast(path: Path, forecast: Forecast, *, source: str, model_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "source": source,
        **asdict(forecast),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
