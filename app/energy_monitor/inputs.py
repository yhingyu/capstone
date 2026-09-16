"""Normalize uploaded household energy readings into daily observations."""

from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in frame.columns}


def read_upload(contents: str, filename: str) -> pd.DataFrame:
    if not contents or "," not in contents:
        raise ValueError("The uploaded file is empty or invalid.")
    _, encoded = contents.split(",", 1)
    payload = base64.b64decode(encoded)
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(payload))
    if lower_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(payload))
    raise ValueError("Upload a CSV or Excel (.xlsx/.xls) file.")


def normalize_readings(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if frame.empty:
        raise ValueError("The input file contains no rows.")
    columns = _column_map(frame)

    # Original UCI minute-level format.
    if {"date", "time", "global_active_power"}.issubset(columns):
        timestamp = pd.to_datetime(
            frame[columns["date"]].astype(str) + " " + frame[columns["time"]].astype(str),
            dayfirst=True,
            errors="coerce",
        )
        power = pd.to_numeric(frame[columns["global_active_power"]], errors="coerce")
        voltage = pd.to_numeric(frame[columns.get("voltage")], errors="coerce") if "voltage" in columns else np.nan
        parsed = pd.DataFrame({"timestamp": timestamp, "power_kw": power, "voltage_v": voltage})
        return _aggregate_power(parsed), "UCI minute-level readings"

    date_key = next((key for key in ("date", "timestamp", "datetime") if key in columns), None)
    if date_key is None:
        raise ValueError("No date or timestamp column was found.")
    timestamp = pd.to_datetime(frame[columns[date_key]], errors="coerce")

    daily_key = next((key for key in ("daily_kwh", "consumption_kwh", "kwh") if key in columns), None)
    if daily_key:
        result = pd.DataFrame(
            {
                "date": timestamp.dt.normalize(),
                "daily_kwh": pd.to_numeric(frame[columns[daily_key]], errors="coerce"),
            }
        )
        voltage_key = next((key for key in ("voltage_std_v", "voltage_std") if key in columns), None)
        result["voltage_std_v"] = (
            pd.to_numeric(frame[columns[voltage_key]], errors="coerce") if voltage_key else np.nan
        )
        return _finish_daily(result), "daily consumption readings"

    meter_key = next((key for key in ("meter_reading_kwh", "cumulative_kwh", "meter_kwh") if key in columns), None)
    if meter_key:
        meter = pd.DataFrame(
            {
                "date": timestamp.dt.normalize(),
                "meter": pd.to_numeric(frame[columns[meter_key]], errors="coerce"),
            }
        ).dropna().sort_values("date")
        meter = meter.drop_duplicates("date", keep="last")
        meter["daily_kwh"] = meter["meter"].diff()
        if (meter["daily_kwh"].dropna() < 0).any():
            raise ValueError("Cumulative meter readings decreased; check resets or data errors.")
        meter["voltage_std_v"] = np.nan
        return _finish_daily(meter[["date", "daily_kwh", "voltage_std_v"]]), "cumulative meter readings"

    power_key = next((key for key in ("power_kw", "active_power_kw", "kw") if key in columns), None)
    if power_key:
        voltage_key = next((key for key in ("voltage_v", "voltage") if key in columns), None)
        parsed = pd.DataFrame(
            {
                "timestamp": timestamp,
                "power_kw": pd.to_numeric(frame[columns[power_key]], errors="coerce"),
                "voltage_v": pd.to_numeric(frame[columns[voltage_key]], errors="coerce") if voltage_key else np.nan,
            }
        )
        return _aggregate_power(parsed), "timestamped power readings"

    raise ValueError(
        "No supported value column was found. Use daily_kwh, meter_reading_kwh, "
        "power_kw, or the original UCI columns."
    )


def _aggregate_power(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.dropna(subset=["timestamp", "power_kw"]).sort_values("timestamp")
    if len(data) < 2:
        raise ValueError("At least two timestamped power readings are required.")
    intervals = data["timestamp"].diff().dt.total_seconds().div(3600)
    typical_interval = float(intervals.dropna().median())
    if not 0 < typical_interval <= 24:
        raise ValueError("The timestamp interval is invalid or too sparse.")
    data["energy_kwh"] = data["power_kw"] * typical_interval
    data["date"] = data["timestamp"].dt.normalize()
    daily = data.groupby("date").agg(
        daily_kwh=("energy_kwh", "sum"),
        voltage_std_v=("voltage_v", "std"),
    ).reset_index()
    return _finish_daily(daily)


def _finish_daily(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.dropna(subset=["date", "daily_kwh"]).sort_values("date")
    result = result.drop_duplicates("date", keep="last").reset_index(drop=True)
    if (result["daily_kwh"] < 0).any():
        raise ValueError("Negative consumption values are not supported.")
    return result
