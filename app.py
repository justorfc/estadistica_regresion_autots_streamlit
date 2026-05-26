from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Estadística Aplicada: Regresiones y AutoTS",
    page_icon="📊",
    layout="wide",
)

pages = {
    "Inicio": [
        st.Page("pages/00_Inicio.py", title="Presentación del proyecto", icon="🏠"),
    ],
    "Semana 1 — Regresiones": [
        st.Page("pages/01_Regresion_Lineal.py", title="Regresión lineal simple y múltiple", icon="📈"),
        st.Page("pages/02_No_Lineal_Logistica.py", title="Regresión no lineal y logística", icon="🧮"),
    ],
    "Semana 2 — Series de tiempo": [
        st.Page("pages/03_Series_Tiempo.py", title="Preprocesamiento y descomposición", icon="🕒"),
        st.Page("pages/04_AutoTS_Pronostico.py", title="Pronóstico automatizado con AutoTS", icon="🤖"),
    ],
    "Semana 3 — Auditoría y despliegue": [
        st.Page("pages/05_Auditoria_Modelos.py", title="Caja negra y métricas", icon="🔍"),
        st.Page("pages/06_Exportacion_Resultados.py", title="Exportación y despliegue", icon="☁️"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
