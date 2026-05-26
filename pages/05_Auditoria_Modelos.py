from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_demo_forecast, load_demo_leaderboard
from utils.ui import info_box, page_title

page_title(
    'Evaluación de la "caja negra" y métricas de error',
    "Equivale a la quinta sesión: modelo ganador, parámetros, leaderboard, MAE y RMSE.",
)

artifacts = st.session_state.get("autots_artifacts")

if artifacts is None:
    info_box(
        "Modo demostrativo",
        "No se detectó un entrenamiento previo en esta sesión. Se muestran resultados demostrativos para explicar la auditoría.",
    )
    leaderboard = load_demo_leaderboard()
    best_name = str(leaderboard.iloc[0]["Model"])
    best_params = {"origen": "leaderboard_demo.csv", "uso": "demostración docente"}
    forecast = load_demo_forecast()
else:
    leaderboard = artifacts.get("leaderboard", pd.DataFrame())
    best_name = artifacts.get("best_model_name", "No disponible")
    best_params = artifacts.get("best_model_params", {})
    forecast = artifacts.get("forecast", pd.DataFrame())

st.subheader("1. Modelo ganador")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Mejor modelo", best_name)
with col2:
    st.markdown("**Parámetros principales**")
    try:
        st.json(best_params)
    except Exception:
        st.code(str(best_params), language="text")

st.subheader("2. Leaderboard de modelos evaluados")
if leaderboard.empty:
    st.warning("No hay leaderboard disponible.")
else:
    st.dataframe(leaderboard, width="stretch", hide_index=True)

    metric_cols = [c for c in ["MAE", "RMSE"] if c in leaderboard.columns]
    if metric_cols:
        top = leaderboard.head(5).copy()
        top["Modelo"] = top["Model"].astype(str) if "Model" in top.columns else top.index.astype(str)

        fig = go.Figure()
        if "MAE" in top.columns:
            fig.add_trace(go.Bar(x=top["Modelo"], y=top["MAE"], name="MAE"))
        if "RMSE" in top.columns:
            fig.add_trace(go.Bar(x=top["Modelo"], y=top["RMSE"], name="RMSE"))

        fig.update_layout(
            title="Comparación de MAE y RMSE en los modelos superiores",
            xaxis_title="Modelo",
            yaxis_title="Magnitud del error",
            barmode="group",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, width="stretch")

st.subheader("3. Interpretación técnica de métricas")
st.markdown(
    """
    - **MAE**: error absoluto medio. Es fácil de interpretar porque conserva la unidad original de la variable.
    - **RMSE**: raíz del error cuadrático medio. Penaliza más los errores grandes.
    - **Score**: métrica compuesta interna de AutoTS; en general, valores más bajos indican mejor desempeño relativo.
    - Un modelo no debe aceptarse solo porque aparece primero: debe verificarse si el pronóstico es coherente con el fenómeno.
    """
)

if not forecast.empty:
    st.subheader("4. Pronóstico disponible para exportación")
    st.dataframe(forecast, width="stretch", hide_index=True)
