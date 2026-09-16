"""Feature engineering consistent with the Step 3 training notebook."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureResult:
    target_date: pd.Timestamp
    features: pd.DataFrame
    baseline_kwh: float
    warnings: tuple[str, ...]


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "daily_kwh"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = history.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["daily_kwh"] = pd.to_numeric(data["daily_kwh"], errors="coerce")
    if "voltage_std_v" not in data:
        data["voltage_std_v"] = np.nan
    data["voltage_std_v"] = pd.to_numeric(data["voltage_std_v"], errors="coerce")
    data = data.dropna(subset=["date", "daily_kwh"]).sort_values("date")
    data = data.drop_duplicates("date", keep="last").reset_index(drop=True)
    if (data["daily_kwh"] < 0).any():
        raise ValueError("Daily kWh values cannot be negative.")
    if len(data) < 30:
        raise ValueError("At least 30 valid daily readings are required for a forecast.")
    return data


def build_next_day_features(
    history: pd.DataFrame,
    feature_columns: list[str],
    *,
    training_origin: str = "2006-12-16",
    default_voltage_std_v: float = 3.0,
) -> FeatureResult:
    data = _validate_history(history)
    energy = data.set_index("date")["daily_kwh"]
    target_date = energy.index.max() + timedelta(days=1)
    prior = energy.reindex(pd.date_range(end=energy.index.max(), periods=30, freq="D"))
    if prior.isna().any():
        missing_days = int(prior.isna().sum())
        raise ValueError(
            f"The latest 30-day window has {missing_days} missing day(s). "
            "Provide consecutive daily readings."
        )

    row: dict[str, float] = {}
    for lag in (1, 7, 14, 21, 28):
        row[f"energy_lag_{lag}"] = float(energy.iloc[-lag])
    for window in (7, 14, 30):
        values = energy.iloc[-window:]
        row[f"energy_roll_mean_{window}"] = float(values.mean())
        row[f"energy_roll_min_{window}"] = float(values.min())
        row[f"energy_roll_max_{window}"] = float(values.max())

    row["energy_ewm_7"] = float(energy.ewm(span=7, adjust=False).mean().iloc[-1])
    row["energy_change_7d"] = float(energy.iloc[-1] - energy.iloc[-8])
    row["day_of_week"] = float(target_date.dayofweek)
    row["month"] = float(target_date.month)
    row["time_index"] = float((target_date - pd.Timestamp(training_origin)).days)
    row["dow_sin"] = float(np.sin(2 * np.pi * target_date.dayofweek / 7))
    row["year_cos"] = float(np.cos(2 * np.pi * target_date.dayofyear / 365.25))

    warnings: list[str] = []
    voltage = data.loc[data["date"] == energy.index.max(), "voltage_std_v"].iloc[-1]
    if pd.isna(voltage):
        voltage = default_voltage_std_v
        warnings.append(
            f"Latest voltage variability was unavailable; {default_voltage_std_v:.2f} V was used."
        )
    row["voltage_std_v_lag_1"] = float(voltage)

    missing_features = [name for name in feature_columns if name not in row]
    if missing_features:
        raise ValueError(f"Cannot construct model features: {', '.join(missing_features)}")

    features = pd.DataFrame([[row[name] for name in feature_columns]], columns=feature_columns)
    return FeatureResult(
        target_date=target_date,
        features=features,
        baseline_kwh=float(energy.iloc[-30:].mean()),
        warnings=tuple(warnings),
    )
