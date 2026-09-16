import pandas as pd

from energy_monitor.inputs import normalize_readings


def test_daily_input_alias():
    raw = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "kwh": [12.5, 13.0]})
    daily, source = normalize_readings(raw)
    assert source == "daily consumption readings"
    assert daily["daily_kwh"].tolist() == [12.5, 13.0]


def test_cumulative_meter_conversion():
    raw = pd.DataFrame(
        {"timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"], "meter_reading_kwh": [100, 112, 125]}
    )
    daily, source = normalize_readings(raw)
    assert source == "cumulative meter readings"
    assert daily["daily_kwh"].tolist() == [12.0, 13.0]
