from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils.data_loader import build_formula, load_anscombe, load_tips_demo, numeric_columns, read_uploaded_table
from utils.regression_utils import (
    coefficients_table,
    fit_ols,
    observed_vs_predicted,
    ols_metrics,
    residual_plot,
    scatter_with_regression_line,
)
from utils.ui import page_title, show_model_note

page_title(
    "Regresión lineal simple y múltiple",
    "Equivale a la primera sesión: visualizar, ajustar, interpretar y diagnosticar.",
)

tab_anscombe, tab_multiple = st.tabs(["Cuarteto de Anscombe", "Regresión múltiple"])

with tab_anscombe:
    st.subheader("1. Por qué debemos graficar antes de modelar")
    anscombe = load_anscombe()

    resumen = anscombe.groupby("dataset")[["x", "y"]].agg(["mean", "var"]).round(3)
    st.dataframe(resumen, width="stretch")

    fig_all = px.scatter(
        anscombe,
        x="x",
        y="y",
        facet_col="dataset",
        facet_col_wrap=2,
        trendline="ols",
        title="Cuarteto de Anscombe: misma estadística, patrones visuales diferentes",
    )
    st.plotly_chart(fig_all, width="stretch")

    dataset = st.selectbox("Seleccione un subconjunto para ajustar y ~ x", sorted(anscombe["dataset"].unique()))
    df_model = anscombe[anscombe["dataset"] == dataset].copy()
    model = fit_ols(df_model, "y ~ x")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Métricas del modelo**")
        st.dataframe(ols_metrics(model), width="stretch", hide_index=True)
    with col2:
        st.markdown("**Coeficientes**")
        st.dataframe(coefficients_table(model).round(4), width="stretch", hide_index=True)

    st.plotly_chart(
        scatter_with_regression_line(
            df_model,
            x_col="x",
            y_col="y",
            model=model,
            title=f"Ajuste lineal para el subconjunto {dataset}",
        ),
        width="stretch",
    )
    st.plotly_chart(residual_plot(model), width="stretch")
    show_model_note()

with tab_multiple:
    st.subheader("2. Regresión lineal múltiple con datos demostrativos o propios")

    fuente = st.radio("Fuente de datos", ["Datos demostrativos", "Cargar archivo propio"], horizontal=True)
    if fuente == "Datos demostrativos":
        df = load_tips_demo()
        st.caption("Dataset didáctico local tipo propinas: total de cuenta, tamaño del grupo y propina.")
    else:
        uploaded = st.file_uploader("Cargue un CSV o Excel", type=["csv", "xlsx", "xls"])
        if uploaded is None:
            st.info("Cargue un archivo para continuar.")
            st.stop()
        df = read_uploaded_table(uploaded)

    st.dataframe(df.head(20), width="stretch")

    nums = numeric_columns(df)
    if len(nums) < 2:
        st.error("Se requieren al menos dos columnas numéricas.")
        st.stop()

    default_target = "tip" if "tip" in nums else nums[-1]
    y_col = st.selectbox("Variable respuesta", nums, index=nums.index(default_target))
    x_options = [c for c in nums if c != y_col]
    default_predictors = [c for c in ["total_bill", "size"] if c in x_options] or x_options[: min(2, len(x_options))]
    predictors = st.multiselect("Variables predictoras", x_options, default=default_predictors)

    if not predictors:
        st.warning("Seleccione al menos una variable predictora.")
        st.stop()

    formula = build_formula(y_col, predictors)
    st.code(formula, language="text")

    model = fit_ols(df.dropna(subset=[y_col] + predictors), formula)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Métricas globales**")
        st.dataframe(ols_metrics(model).round(4), width="stretch", hide_index=True)
    with col2:
        st.markdown("**Coeficientes del modelo**")
        st.dataframe(coefficients_table(model).round(4), width="stretch", hide_index=True)

    st.plotly_chart(observed_vs_predicted(model, df.dropna(subset=[y_col] + predictors), y_col), width="stretch")
    st.plotly_chart(residual_plot(model), width="stretch")

    with st.expander("Interpretación orientadora"):
        st.markdown(
            """
            - **Coeficiente positivo**: al aumentar esa variable, la respuesta tiende a aumentar, manteniendo las demás constantes.
            - **Coeficiente negativo**: al aumentar esa variable, la respuesta tiende a disminuir, manteniendo las demás constantes.
            - **R² ajustado**: útil para comparar modelos con diferente número de predictores.
            - **Residuos**: deben revisarse para detectar curvatura, heterocedasticidad o valores atípicos.
            """
        )
    show_model_note()
