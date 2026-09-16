import pandas as pd
import pytest

from energy_monitor.features import build_next_day_features


FEATURES = [
    "day_of_week", "dow_sin", "energy_change_7d", "energy_ewm_7", "energy_lag_1",
    "energy_lag_14", "energy_lag_21", "energy_lag_28", "energy_lag_7",
    "energy_roll_max_14", "energy_roll_max_30", "energy_roll_max_7",
    "energy_roll_mean_14", "energy_roll_mean_30", "energy_roll_mean_7",
    "energy_roll_min_14", "energy_roll_min_30", "energy_roll_min_7", "month",
    "time_index", "voltage_std_v_lag_1", "year_cos",
]


def history(days=35):
    return pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=days),
            "daily_kwh": [10.0 + index for index in range(days)],
            "voltage_std_v": [3.0] * days,
        }
    )


def test_feature_order_and_lags():
    result = build_next_day_features(history(), FEATURES)
    assert list(result.features.columns) == FEATURES
    assert result.features.loc[0, "energy_lag_1"] == 44.0
    assert result.features.loc[0, "energy_lag_7"] == 38.0
    assert result.target_date == pd.Timestamp("2010-02-05")


def test_requires_thirty_days():
    with pytest.raises(ValueError, match="At least 30"):
        build_next_day_features(history(29), FEATURES)


def test_rejects_gap_in_latest_window():
    data = history().drop(index=30)
    with pytest.raises(ValueError, match="missing day"):
        build_next_day_features(data, FEATURES)
