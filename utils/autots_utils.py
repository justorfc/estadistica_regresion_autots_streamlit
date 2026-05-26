from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .timeseries_utils import seasonal_naive_forecast


def autots_available() -> bool:
    """Indica si la librería AutoTS está instalada en el entorno."""
    try:
        import autots  # noqa: F401
        return True
    except Exception:
        return False


def run_autots(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    forecast_length: int = 12,
    frequency: str = "MS",
    prediction_interval: float = 0.90,
    model_list: str = "superfast",
    max_generations: int = 1,
    num_validations: int = 2,
    ensemble: str | None = "auto",
) -> dict[str, Any]:
    """Entrena AutoTS y devuelve predicción, intervalos, leaderboard y metadatos."""
    from autots import AutoTS

    model = AutoTS(
        forecast_length=forecast_length,
        frequency=frequency,
        prediction_interval=prediction_interval,
        ensemble=ensemble,
        model_list=model_list,
        max_generations=max_generations,
        num_validations=num_validations,
        verbose=-1,
    )

    model = model.fit(
        df,
        date_col=date_col,
        value_col=value_col,
        id_col=None,
    )
    prediction = model.predict()

    forecast = prediction.forecast.copy()
    upper = prediction.upper_forecast.copy()
    lower = prediction.lower_forecast.copy()

    target_name = forecast.columns[0]
    out = pd.concat(
        [
            forecast.rename(columns={target_name: "Prediccion"}),
            lower.rename(columns={target_name: "Limite_Inferior"}),
            upper.rename(columns={target_name: "Limite_Superior"}),
        ],
        axis=1,
    ).reset_index(names="Fecha")

    results = model.results()
    keep = [c for c in ["ID", "Model", "ModelParameters", "Score", "MAE", "RMSE"] if c in results.columns]
    leaderboard = results[keep].sort_values("Score", ascending=True).head(15) if keep else results.head(15)

    return {
        "forecast": out,
        "leaderboard": leaderboard,
        "best_model_name": getattr(model, "best_model_name", "No disponible"),
        "best_model_params": getattr(model, "best_model_params", {}),
        "method": "AutoTS",
    }


def run_fallback_forecast(ts: pd.DataFrame, value_col: str, forecast_length: int, period: int = 12) -> dict[str, Any]:
    """Genera un pronóstico de respaldo para que la app no se detenga sin AutoTS."""
    forecast = seasonal_naive_forecast(ts, value_col=value_col, forecast_length=forecast_length, period=period)
    leaderboard = pd.DataFrame(
        {
            "Model": ["SeasonalNaiveFallback"],
            "Score": [None],
            "MAE": [None],
            "RMSE": [None],
            "ModelParameters": [json.dumps({"period": period, "note": "fallback local"})],
        }
    )
    return {
        "forecast": forecast,
        "leaderboard": leaderboard,
        "best_model_name": "SeasonalNaiveFallback",
        "best_model_params": {"period": period, "reason": "AutoTS no disponible o falló el entrenamiento"},
        "method": "Respaldo local",
    }
