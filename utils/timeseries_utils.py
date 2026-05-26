from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose


def prepare_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Convierte un DataFrame en serie de tiempo con índice DatetimeIndex."""
    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col])
    out = out.sort_values(date_col)
    out = out.set_index(date_col)
    out.index.name = "Fecha"
    return out


def time_series_plot(ts: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """Grafica una serie de tiempo."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts[value_col],
            mode="lines+markers",
            name=value_col,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=value_col,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def decompose_to_frame(ts: pd.DataFrame, value_col: str, model: str, period: int) -> pd.DataFrame:
    """Aplica descomposición clásica y devuelve los componentes en un DataFrame."""
    result = seasonal_decompose(ts[value_col], model=model, period=period, extrapolate_trend="freq")
    return pd.DataFrame(
        {
            "observado": result.observed,
            "tendencia": result.trend,
            "estacionalidad": result.seasonal,
            "residuo": result.resid,
        }
    )


def decomposition_plot(components: pd.DataFrame) -> go.Figure:
    """Grafica componentes de la descomposición clásica."""
    fig = go.Figure()
    for col in components.columns:
        fig.add_trace(go.Scatter(x=components.index, y=components[col], mode="lines", name=col))
    fig.update_layout(
        title="Descomposición de la serie: observado, tendencia, estacionalidad y residuo",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def seasonal_naive_forecast(ts: pd.DataFrame, value_col: str, forecast_length: int, period: int = 12) -> pd.DataFrame:
    """Pronóstico de respaldo cuando AutoTS no está disponible."""
    y = ts[value_col].astype(float)
    if len(y) < period:
        base_values = np.repeat(y.iloc[-1], forecast_length)
        sigma = float(y.diff().dropna().std() or y.std() or 1)
    else:
        template = y.tail(period).to_numpy()
        reps = int(np.ceil(forecast_length / period))
        base_values = np.tile(template, reps)[:forecast_length]
        sigma = float(y.diff(period).dropna().std() or y.diff().dropna().std() or y.std() or 1)

    if len(y) >= 2 * period:
        trend = float((y.tail(period).to_numpy() - y.iloc[-2 * period:-period].to_numpy()).mean())
        base_values = base_values + np.arange(forecast_length) * (trend / max(period, 1))

    freq = pd.infer_freq(y.index) or "MS"
    future_index = pd.date_range(y.index.max() + pd.tseries.frequencies.to_offset(freq), periods=forecast_length, freq=freq)

    return pd.DataFrame(
        {
            "Fecha": future_index,
            "Prediccion": np.round(base_values, 3),
            "Limite_Inferior": np.round(base_values - 1.64 * sigma, 3),
            "Limite_Superior": np.round(base_values + 1.64 * sigma, 3),
        }
    )


def forecast_plot(history: pd.DataFrame, value_col: str, forecast: pd.DataFrame) -> go.Figure:
    """Grafica histórico, pronóstico e intervalo."""
    pred_col = "Prediccion" if "Prediccion" in forecast.columns else "Prediccion_Pasajeros"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history[value_col],
            mode="lines",
            name="Histórico",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["Fecha"],
            y=forecast[pred_col],
            mode="lines+markers",
            name="Pronóstico",
        )
    )
    if {"Limite_Inferior", "Limite_Superior"}.issubset(forecast.columns):
        fig.add_trace(
            go.Scatter(
                x=forecast["Fecha"],
                y=forecast["Limite_Superior"],
                mode="lines",
                name="Límite superior",
                line=dict(dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["Fecha"],
                y=forecast["Limite_Inferior"],
                mode="lines",
                name="Límite inferior",
                line=dict(dash="dot"),
            )
        )
    fig.update_layout(
        title="Histórico y pronóstico",
        xaxis_title="Fecha",
        yaxis_title=value_col,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
