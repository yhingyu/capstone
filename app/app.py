"""Dash proof-of-concept for local household energy monitoring."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update

from energy_monitor.inputs import normalize_readings, read_upload
from energy_monitor.predictor import EnergyPredictor, log_forecast

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.getenv("ENERGY_CONFIG", ROOT / "config/app.yaml"))
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
PREDICTOR = EnergyPredictor(CONFIG, ROOT)

app = Dash(__name__, title="Energy Monitor POC", suppress_callback_exceptions=True)
server = app.server


@server.get("/health")
def health():
    return {
        "status": "ok",
        "service": "household-energy-monitor-poc",
        "model_version": CONFIG["model"]["version"],
    }


def sample_history() -> pd.DataFrame:
    return pd.read_csv(ROOT / CONFIG["app"]["sample_data_path"])


def history_figure(data: pd.DataFrame, forecast: dict | None = None) -> go.Figure:
    figure = go.Figure()
    if not data.empty:
        figure.add_trace(
            go.Scatter(x=data["date"], y=data["daily_kwh"], mode="lines+markers", name="Daily kWh")
        )
    if forecast:
        figure.add_trace(
            go.Scatter(
                x=[forecast["target_date"]],
                y=[forecast["predicted_kwh"]],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [forecast["upper_kwh"] - forecast["predicted_kwh"]],
                    "arrayminus": [forecast["predicted_kwh"] - forecast["lower_kwh"]],
                },
                mode="markers",
                marker={"size": 13, "color": "#ffb703"},
                name="Next-day forecast",
            )
        )
    figure.update_layout(
        template="plotly_dark",
        margin={"l": 40, "r": 20, "t": 35, "b": 40},
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def metric_card(label: str, value_id: str) -> html.Div:
    return html.Div([html.Span(label), html.Strong("—", id=value_id)], className="metric-card")


app.layout = html.Div(
    [
        dcc.Store(id="history-store", data=sample_history().to_dict("records")),
        dcc.Store(id="source-store", data="included demonstration data"),
        html.Header(
            [
                html.Div("POC", className="poc-badge"),
                html.H1(CONFIG["app"]["title"]),
                html.P(
                    "Local next-day energy forecast and advisory dashboard. "
                    "This prototype does not calculate or replace an official Meralco bill."
                ),
            ]
        ),
        html.Main(
            [
                html.Section(
                    [
                        html.H2("1. Provide readings"),
                        dcc.Upload(
                            id="upload-readings",
                            children=html.Div(["Drop a CSV/Excel file here or ", html.A("browse")]),
                            className="upload-box",
                            multiple=False,
                        ),
                        html.P(
                            "Supported: daily_kwh, cumulative meter_reading_kwh, power_kw, "
                            "or original UCI minute readings.",
                            className="hint",
                        ),
                        html.Div(id="upload-status", className="status-message"),
                        html.H3("Manual daily reading"),
                        html.Div(
                            [
                                dcc.DatePickerSingle(id="manual-date", display_format="YYYY-MM-DD"),
                                dcc.Input(id="manual-kwh", type="number", min=0, step=0.01, placeholder="Daily kWh"),
                                dcc.Input(
                                    id="manual-voltage", type="number", min=0, step=0.01,
                                    placeholder="Voltage std V (optional)",
                                ),
                                html.Button("Add / update", id="add-reading", n_clicks=0),
                                html.Button("Load sample", id="load-sample", n_clicks=0, className="secondary"),
                            ],
                            className="manual-grid",
                        ),
                        html.Div(id="manual-status", className="status-message"),
                        dash_table.DataTable(
                            id="history-table",
                            page_size=8,
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"backgroundColor": "#152238", "color": "#e8eef7", "border": "1px solid #31425f"},
                            style_header={"backgroundColor": "#22324d", "fontWeight": "bold"},
                        ),
                    ],
                    className="panel",
                ),
                html.Section(
                    [
                        html.H2("2. Forecast and advisory"),
                        html.Div(
                            [
                                html.Label("Electricity rate (PHP/kWh)"),
                                dcc.Input(
                                    id="rate-input", type="number", min=0, step=0.01,
                                    value=CONFIG["app"]["default_rate_php_per_kwh"],
                                ),
                                html.Button("Forecast next day", id="forecast-button", n_clicks=0),
                            ],
                            className="forecast-controls",
                        ),
                        html.Div(id="forecast-error", className="error-message"),
                        html.Div(
                            [
                                metric_card("Next-day forecast", "forecast-kwh"),
                                metric_card("90% prediction band", "forecast-band"),
                                metric_card("Advisory", "forecast-level"),
                                metric_card("Est. daily cost", "forecast-cost"),
                            ],
                            className="metric-grid",
                        ),
                        dcc.Graph(id="history-chart", figure=history_figure(sample_history())),
                        html.Div(id="forecast-explanation", className="explanation"),
                    ],
                    className="panel",
                ),
                html.Section(
                    [
                        html.H2("Responsible-use notice"),
                        html.Ul(
                            [
                                html.Li("Proof of concept trained on one French household, not Philippine households."),
                                html.Li("High-consumption warning recall was only 21.74% on the held-out test set."),
                                html.Li("Use forecasts as advisory information; do not use them for billing or disconnection."),
                                html.Li("Uploaded readings remain inside this local deployment."),
                            ]
                        ),
                    ],
                    className="notice",
                ),
            ]
        ),
        html.Footer(f"Model {CONFIG['model']['version']} · Local Docker POC"),
    ],
    className="page-shell",
)


@callback(
    Output("history-store", "data"),
    Output("source-store", "data"),
    Output("upload-status", "children"),
    Input("upload-readings", "contents"),
    Input("load-sample", "n_clicks"),
    State("upload-readings", "filename"),
    prevent_initial_call=True,
)
def load_history(contents, load_clicks, filename):
    from dash import ctx

    if ctx.triggered_id == "load-sample":
        data = sample_history()
        return data.to_dict("records"), "included demonstration data", f"Loaded {len(data)} sample days."
    if not contents:
        return no_update, no_update, no_update
    try:
        frame = read_upload(contents, filename or "")
        daily, source = normalize_readings(frame)
        return daily.to_dict("records"), source, f"Loaded {len(daily)} valid daily rows from {filename}."
    except Exception as exc:  # surfaced as a user-friendly validation message
        return no_update, no_update, f"Upload error: {exc}"


@callback(
    Output("history-store", "data", allow_duplicate=True),
    Output("source-store", "data", allow_duplicate=True),
    Output("manual-status", "children"),
    Input("add-reading", "n_clicks"),
    State("manual-date", "date"),
    State("manual-kwh", "value"),
    State("manual-voltage", "value"),
    State("history-store", "data"),
    prevent_initial_call=True,
)
def add_manual_reading(n_clicks, date, kwh, voltage, records):
    if not date or kwh is None:
        return no_update, no_update, "Enter a date and daily kWh value."
    data = pd.DataFrame(records or [], columns=["date", "daily_kwh", "voltage_std_v"])
    row = pd.DataFrame([{"date": date, "daily_kwh": float(kwh), "voltage_std_v": voltage}])
    data = pd.concat([data, row], ignore_index=True)
    data["date"] = pd.to_datetime(data["date"]).dt.date.astype(str)
    data = data.drop_duplicates("date", keep="last").sort_values("date")
    return data.to_dict("records"), "manual daily readings", f"Saved reading for {date}."


@callback(
    Output("history-table", "data"),
    Output("history-table", "columns"),
    Output("history-chart", "figure", allow_duplicate=True),
    Input("history-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def refresh_history(records):
    data = pd.DataFrame(records or [])
    columns = [{"name": column.replace("_", " ").title(), "id": column} for column in data.columns]
    return data.to_dict("records"), columns, history_figure(data)


@callback(
    Output("forecast-kwh", "children"),
    Output("forecast-band", "children"),
    Output("forecast-level", "children"),
    Output("forecast-cost", "children"),
    Output("forecast-explanation", "children"),
    Output("forecast-error", "children"),
    Output("history-chart", "figure", allow_duplicate=True),
    Input("forecast-button", "n_clicks"),
    State("history-store", "data"),
    State("source-store", "data"),
    State("rate-input", "value"),
    prevent_initial_call=True,
)
def forecast(n_clicks, records, source, rate):
    try:
        history = pd.DataFrame(records or [])
        result = PREDICTOR.forecast(history, float(rate or 0))
        payload = result.__dict__
        log_forecast(
            ROOT / CONFIG["monitoring"]["prediction_log_path"],
            result,
            source=source or "unknown",
            model_version=CONFIG["model"]["version"],
        )
        notes = [
            html.P(
                f"Forecast for {result.target_date}: {result.utilization_percent:.1f}% of the configured "
                f"high-use threshold ({result.high_use_threshold_kwh:.2f} kWh)."
            ),
            html.P(
                f"The 30-day baseline is {result.baseline_kwh:.2f} kWh. A 30-day projection at the "
                f"entered rate is PHP {result.projected_30_day_cost:,.2f}."
            ),
        ]
        notes.extend(html.P(warning, className="warning-note") for warning in result.warnings)
        return (
            f"{result.predicted_kwh:.2f} kWh",
            f"{result.lower_kwh:.2f}–{result.upper_kwh:.2f} kWh",
            result.advisory,
            f"PHP {result.estimated_daily_cost:,.2f}",
            notes,
            "",
            history_figure(history, payload),
        )
    except Exception as exc:
        return "—", "—", "—", "—", "", str(exc), history_figure(pd.DataFrame(records or []))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
