from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.autots_utils import autots_available, run_autots, run_fallback_forecast
from utils.data_loader import load_flights, numeric_columns, read_uploaded_table
from utils.timeseries_utils import forecast_plot, prepare_time_series
from utils.ui import info_box, page_title

page_title(
    "Pronósticos automatizados con AutoTS",
    "Equivale a la cuarta sesión: configurar AutoML, entrenar, predecir e interpretar intervalos.",
)

info_box(
    "Recomendación para clase",
    "Use model_list='superfast', max_generations=1 y num_validations=1 o 2 para evitar tiempos largos en equipos de estudiantes.",
)

fuente = st.radio("Fuente de datos", ["Serie demostrativa de vuelos", "Cargar archivo propio"], horizontal=True)

if fuente == "Serie demostrativa de vuelos":
    raw = load_flights()
    date_col = "fecha"
    value_col = "passengers"
else:
    uploaded = st.file_uploader("Cargue un CSV o Excel", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.info("Cargue un archivo para continuar.")
        st.stop()
    raw = read_uploaded_table(uploaded)
    st.dataframe(raw.head(20), width="stretch")
    date_col = st.selectbox("Columna de fecha", raw.columns.tolist())
    nums = numeric_columns(raw)
    if not nums:
        st.error("Se requiere al menos una columna numérica.")
        st.stop()
    value_col = st.selectbox("Columna de valor", nums)

ts = prepare_time_series(raw, date_col=date_col, value_col=value_col)
autots_df = ts.reset_index().rename(columns={"Fecha": "fecha", value_col: "valor"})

st.subheader("Configuración del pronóstico")
col1, col2, col3 = st.columns(3)
with col1:
    forecast_length = st.number_input("Horizonte", min_value=1, max_value=60, value=12, step=1)
    frequency = st.selectbox("Frecuencia", ["MS", "D", "W", "M", "Q", "YS"], index=0)
with col2:
    prediction_interval = st.slider("Intervalo de predicción", 0.50, 0.99, 0.90, 0.01)
    model_list = st.selectbox("Lista de modelos", ["superfast", "fast", "default"], index=0)
with col3:
    max_generations = st.number_input("Generaciones", min_value=1, max_value=5, value=1, step=1)
    num_validations = st.number_input("Validaciones", min_value=1, max_value=5, value=2, step=1)

st.plotly_chart(forecast_plot(ts, value_col, pd.DataFrame({
    "Fecha": [],
    "Prediccion": [],
    "Limite_Inferior": [],
    "Limite_Superior": [],
})), width="stretch")

if not autots_available():
    st.warning("AutoTS no está instalado en este entorno. Se usará un pronóstico estacional de respaldo si ejecuta el entrenamiento.")

run_button = st.button("Entrenar modelo y generar pronóstico", type="primary")

if run_button:
    with st.spinner("Entrenando y generando pronóstico..."):
        try:
            if autots_available():
                artifacts = run_autots(
                    autots_df,
                    date_col="fecha",
                    value_col="valor",
                    forecast_length=int(forecast_length),
                    frequency=frequency,
                    prediction_interval=float(prediction_interval),
                    model_list=model_list,
                    max_generations=int(max_generations),
                    num_validations=int(num_validations),
                    ensemble="auto",
                )
            else:
                artifacts = run_fallback_forecast(ts, value_col=value_col, forecast_length=int(forecast_length), period=12)
        except Exception as exc:
            st.warning(f"AutoTS no pudo completar el entrenamiento. Se usará respaldo local. Detalle: {exc}")
            artifacts = run_fallback_forecast(ts, value_col=value_col, forecast_length=int(forecast_length), period=12)

    st.session_state["autots_artifacts"] = artifacts
    st.session_state["autots_history"] = {"ts": ts, "value_col": value_col}
    st.success(f"Pronóstico generado con método: {artifacts['method']}")

artifacts = st.session_state.get("autots_artifacts")
history = st.session_state.get("autots_history")

if artifacts and history:
    forecast = artifacts["forecast"].copy()
    st.subheader("Resultado del pronóstico")
    st.plotly_chart(forecast_plot(history["ts"], history["value_col"], forecast), width="stretch")
    st.dataframe(forecast, width="stretch", hide_index=True)

    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar pronóstico en CSV",
        data=csv,
        file_name="pronostico_autots.csv",
        mime="text/csv",
    )
else:
    st.info("Aún no hay pronóstico entrenado en esta sesión.")
