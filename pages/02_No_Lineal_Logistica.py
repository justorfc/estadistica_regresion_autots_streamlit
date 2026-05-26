from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_mpg_demo, load_titanic_demo
from utils.regression_utils import (
    coefficients_table,
    compare_models_table,
    fit_logit,
    fit_ols,
    logit_metrics,
    odds_ratios_table,
    residual_plot,
    scatter_with_regression_line,
)
from utils.ui import page_title, show_model_note

page_title(
    "Regresión no lineal y regresión logística",
    "Equivale a la segunda sesión: capturar curvaturas y modelar eventos binarios.",
)

tab_poly, tab_logit = st.tabs(["Regresión polinomial", "Regresión logística"])

with tab_poly:
    st.subheader("1. Modelo polinomial para fenómenos con curvatura")
    df = load_mpg_demo()

    st.markdown(
        "Ejemplo didáctico: relación entre potencia del motor (`horsepower`) y rendimiento (`mpg`). "
        "La idea es comparar un modelo lineal contra un modelo cuadrático."
    )
    st.dataframe(df.head(15), width="stretch")

    degree = st.slider("Grado polinomial", min_value=1, max_value=3, value=2)

    formulas = {
        "Lineal": "mpg ~ horsepower",
        "Cuadrático": "mpg ~ horsepower + I(horsepower**2)",
        "Cúbico": "mpg ~ horsepower + I(horsepower**2) + I(horsepower**3)",
    }
    selected_formula = {
        1: formulas["Lineal"],
        2: formulas["Cuadrático"],
        3: formulas["Cúbico"],
    }[degree]

    model_lineal = fit_ols(df, formulas["Lineal"])
    model_selected = fit_ols(df, selected_formula)

    st.code(selected_formula, language="text")
    st.plotly_chart(
        scatter_with_regression_line(
            df,
            x_col="horsepower",
            y_col="mpg",
            model=model_selected,
            title=f"Ajuste polinomial de grado {degree}",
            polynomial_degree=degree,
        ),
        width="stretch",
    )

    st.markdown("**Comparación de modelos**")
    st.dataframe(
        compare_models_table({"Lineal": model_lineal, f"Polinomial grado {degree}": model_selected}).round(4),
        width="stretch",
        hide_index=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Coeficientes**")
        st.dataframe(coefficients_table(model_selected).round(5), width="stretch", hide_index=True)
    with col2:
        st.plotly_chart(residual_plot(model_selected), width="stretch")

with tab_logit:
    st.subheader("2. Regresión logística para una respuesta binaria")
    df = load_titanic_demo()
    st.markdown(
        "Ejemplo didáctico binario: `survived` representa 1 = evento ocurrido y 0 = evento no ocurrido. "
        "En ingeniería puede reinterpretarse como falla/no falla, aceptación/rechazo o éxito/fracaso."
    )
    st.dataframe(df.head(15), width="stretch")

    formula = "survived ~ age + fare + pclass"
    st.code(formula, language="text")

    model = fit_logit(df, formula)
    probabilities = model.predict(df)
    threshold = st.slider("Umbral de clasificación", 0.10, 0.90, 0.50, 0.05)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Métricas de clasificación**")
        st.dataframe(logit_metrics(model, df["survived"], probabilities, threshold).round(4), width="stretch", hide_index=True)
    with col2:
        st.markdown("**Odds ratios**")
        st.dataframe(
            odds_ratios_table(model)[["termino", "odds_ratio", "OR_IC_2.5%", "OR_IC_97.5%", "p_valor"]].round(4),
            width="stretch",
            hide_index=True,
        )

    plot_df = df.copy()
    plot_df["probabilidad_predicha"] = probabilities
    plot_df["clasificacion"] = (probabilities >= threshold).astype(int)

    fig = px.scatter(
        plot_df,
        x="age",
        y="probabilidad_predicha",
        color="pclass",
        symbol="survived",
        title="Probabilidad estimada del evento según edad y clase/categoría",
        labels={"age": "Edad", "probabilidad_predicha": "Probabilidad predicha", "pclass": "Clase"},
    )
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="Umbral")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Interpretación orientadora de odds ratios"):
        st.markdown(
            """
            - Un **odds ratio mayor que 1** indica aumento en los momios del evento.
            - Un **odds ratio menor que 1** indica disminución en los momios del evento.
            - El modelo logístico estima probabilidades, pero la decisión final depende del umbral.
            - En problemas de falla de equipos, bajar o subir el umbral cambia falsos positivos y falsos negativos.
            """
        )

show_model_note()
