from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import load_demo_forecast
from utils.ui import info_box, page_title

page_title(
    "Exportación de resultados y preparación para despliegue",
    "Equivale a la sexta sesión: CSV final, GitHub, requirements.txt y Streamlit Cloud.",
)

artifacts = st.session_state.get("autots_artifacts")

if artifacts is None:
    info_box(
        "Modo demostrativo",
        "No hay pronóstico entrenado en esta sesión. Se cargará un CSV demostrativo incluido en la app.",
    )
    forecast = load_demo_forecast()
else:
    forecast = artifacts.get("forecast", pd.DataFrame())

st.subheader("1. Archivo de pronóstico")
if forecast.empty:
    st.warning("No hay datos para exportar.")
else:
    st.dataframe(forecast, width="stretch", hide_index=True)

    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar pronóstico en CSV",
        data=csv,
        file_name="pronostico_autots.csv",
        mime="text/csv",
    )

st.subheader("2. Pasos de ejecución local en VSCode")
st.code(
    """
# 1. Entrar a la carpeta del proyecto
cd estadistica_regresion_autots_streamlit

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno en Windows PowerShell
.venv\\Scripts\\Activate.ps1

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar la app
streamlit run app.py
    """,
    language="powershell",
)

st.subheader("3. Pasos de publicación en GitHub y Streamlit Cloud")
st.markdown(
    """
    1. Crear un repositorio en GitHub.
    2. Subir la carpeta completa del proyecto.
    3. Verificar que existan `app.py`, `requirements.txt`, `pages/`, `utils/` y `data/`.
    4. Entrar a Streamlit Community Cloud.
    5. Seleccionar el repositorio.
    6. Definir `app.py` como archivo principal.
    7. Publicar y copiar el enlace público del dashboard.
    """
)

st.subheader("4. Lista mínima de archivos requeridos")
st.code(
    """
app.py
requirements.txt
pages/
utils/
data/
.streamlit/config.toml
    """,
    language="text",
)

with st.expander("Criterios sugeridos para revisar el proyecto final"):
    st.markdown(
        """
        - La app carga correctamente datos demostrativos o propios.
        - El estudiante interpreta los coeficientes del modelo de regresión.
        - El estudiante justifica la preparación de la serie de tiempo.
        - El pronóstico incluye tabla, gráfica e intervalo.
        - La auditoría muestra modelo ganador, leaderboard y métricas.
        - El repositorio de GitHub está organizado y el enlace de Streamlit Cloud funciona.
        """
    )
