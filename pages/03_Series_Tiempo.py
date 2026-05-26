from __future__ import annotations

import streamlit as st

from utils.data_loader import load_flights, numeric_columns, read_uploaded_table
from utils.timeseries_utils import decompose_to_frame, decomposition_plot, prepare_time_series, time_series_plot
from utils.ui import page_title

page_title(
    "Introducción a series de tiempo y preprocesamiento",
    "Equivale a la tercera sesión: fecha, índice temporal, visualización y descomposición.",
)

fuente = st.radio("Fuente de datos", ["Serie demostrativa de vuelos", "Cargar archivo propio"], horizontal=True)

if fuente == "Serie demostrativa de vuelos":
    df = load_flights()
    date_col = "fecha"
    value_col = "passengers"
else:
    uploaded = st.file_uploader("Cargue un CSV o Excel con una columna de fecha", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.info("Cargue un archivo para continuar.")
        st.stop()
    df = read_uploaded_table(uploaded)
    st.dataframe(df.head(20), width="stretch")

    date_col = st.selectbox("Columna de fecha", df.columns.tolist())
    nums = numeric_columns(df)
    if not nums:
        st.error("Se requiere al menos una columna numérica para la serie.")
        st.stop()
    value_col = st.selectbox("Columna de valor", nums)

ts = prepare_time_series(df, date_col=date_col, value_col=value_col)

st.subheader("1. Serie estructurada")
col1, col2, col3 = st.columns(3)
col1.metric("Observaciones", len(ts))
col2.metric("Fecha inicial", str(ts.index.min().date()))
col3.metric("Fecha final", str(ts.index.max().date()))
st.dataframe(ts.head(15), width="stretch")

st.subheader("2. Visualización exploratoria")
st.plotly_chart(time_series_plot(ts, value_col=value_col, title="Serie de tiempo histórica"), width="stretch")

st.subheader("3. Descomposición clásica")
col1, col2 = st.columns([1, 1])
with col1:
    model = st.selectbox("Modelo de descomposición", ["multiplicative", "additive"], index=0)
with col2:
    period = st.number_input("Periodo estacional", min_value=2, max_value=60, value=12, step=1)

if len(ts) < 2 * period:
    st.warning("Para una descomposición estable se recomiendan al menos dos ciclos completos.")
else:
    try:
        components = decompose_to_frame(ts, value_col=value_col, model=model, period=int(period))
        st.plotly_chart(decomposition_plot(components), width="stretch")
        with st.expander("Tabla de componentes"):
            st.dataframe(components.round(3), width="stretch")
    except ValueError as exc:
        st.error(f"No fue posible descomponer la serie: {exc}")

with st.expander("Interpretación orientadora"):
    st.markdown(
        """
        - **Tendencia**: dirección general de largo plazo.
        - **Estacionalidad**: patrón que se repite cada cierto número de periodos.
        - **Residuo**: variación no explicada por tendencia ni estacionalidad.
        - Para AutoTS conviene tener una fecha válida, frecuencia razonablemente estable y valores numéricos sin vacíos críticos.
        """
    )

st.session_state["last_time_series"] = {
    "data": ts.reset_index(),
    "date_col": "Fecha",
    "value_col": value_col,
}
