from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ui import info_box, page_title

page_title(
    "App multipágina de Estadística Aplicada",
    "Regresiones, series de tiempo, AutoTS, auditoría de modelos y despliegue en Streamlit Cloud.",
)

info_box(
    "Propósito académico",
    "Esta aplicación convierte los seis notebooks finales del semestre en un sistema interactivo "
    "para que los estudiantes exploren, ajusten, interpreten y desplieguen modelos estadísticos.",
)

st.markdown(
    """
    Esta app está pensada como proyecto final para las tres últimas semanas de la asignatura.
    El flujo reproduce la secuencia didáctica:

    1. Visualizar antes de modelar.
    2. Ajustar regresiones lineales, múltiples, polinomiales y logísticas.
    3. Preparar datos con estructura temporal.
    4. Ejecutar pronósticos automatizados con AutoTS.
    5. Auditar el modelo ganador mediante métricas.
    6. Exportar resultados y preparar el despliegue en Streamlit Cloud.
    """
)

cronograma = pd.DataFrame(
    [
        ["Semana 1", "Regresión lineal simple y múltiple", "OLS, fórmulas, R², p-valores, residuos"],
        ["Semana 1", "Regresión no lineal y logística", "Modelo polinomial, Logit, odds ratios"],
        ["Semana 2", "Series de tiempo", "Índice datetime, visualización, descomposición"],
        ["Semana 2", "AutoTS", "Pronóstico a 12 meses, intervalos, configuración superfast"],
        ["Semana 3", "Auditoría", "Modelo ganador, leaderboard, MAE, RMSE"],
        ["Semana 3", "Exportación", "CSV, GitHub, Streamlit Cloud"],
    ],
    columns=["Bloque", "Página", "Énfasis"],
)

st.subheader("Estructura de trabajo")
st.dataframe(cronograma, width="stretch", hide_index=True)

st.subheader("Estructura de carpetas sugerida")
st.code(
    """
estadistica_regresion_autots_streamlit/
├── app.py
├── pages/
│   ├── 00_Inicio.py
│   ├── 01_Regresion_Lineal.py
│   ├── 02_No_Lineal_Logistica.py
│   ├── 03_Series_Tiempo.py
│   ├── 04_AutoTS_Pronostico.py
│   ├── 05_Auditoria_Modelos.py
│   └── 06_Exportacion_Resultados.py
├── utils/
│   ├── data_loader.py
│   ├── regression_utils.py
│   ├── timeseries_utils.py
│   └── autots_utils.py
├── data/
├── requirements.txt
└── README.md
    """,
    language="text",
)

st.subheader("Cómo usar la app")
st.markdown(
    """
    - Use primero los datos demostrativos incluidos.
    - Luego cargue archivos propios en CSV o Excel.
    - En regresión, verifique que la variable respuesta sea numérica.
    - En series de tiempo, verifique que exista una columna de fecha y una columna numérica.
    - En AutoTS, use configuraciones ligeras en clase para evitar tiempos largos de entrenamiento.
    """
)

st.success("La app queda lista para ser extendida por grupos de estudiantes con nuevos datasets de ingeniería.")
