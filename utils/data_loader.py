from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@st.cache_data(show_spinner=False)
def load_dataset(filename: str) -> pd.DataFrame:
    """Carga un archivo CSV incluido en la carpeta data/ del proyecto."""
    path = DATA_DIR / filename
    return pd.read_csv(path)


def load_anscombe() -> pd.DataFrame:
    """Carga el Cuarteto de Anscombe incluido como datos locales."""
    return load_dataset("anscombe.csv")


def load_tips_demo() -> pd.DataFrame:
    """Carga un conjunto didáctico para regresión múltiple tipo propinas."""
    return load_dataset("tips_demo.csv")


def load_mpg_demo() -> pd.DataFrame:
    """Carga un conjunto didáctico de potencia del motor y rendimiento."""
    return load_dataset("mpg_demo.csv")


def load_titanic_demo() -> pd.DataFrame:
    """Carga un conjunto didáctico para regresión logística binaria."""
    return load_dataset("titanic_demo.csv")


def load_flights() -> pd.DataFrame:
    """Carga la serie mensual clásica de pasajeros aéreos."""
    df = load_dataset("flights.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def load_demo_forecast() -> pd.DataFrame:
    """Carga un pronóstico demostrativo usado cuando no se ha entrenado AutoTS."""
    df = load_dataset("pronostico_autots_demo.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def load_demo_leaderboard() -> pd.DataFrame:
    """Carga una tabla demostrativa de métricas de modelos."""
    return load_dataset("leaderboard_demo.csv")


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Lee archivos CSV o Excel cargados desde la interfaz de Streamlit."""
    if uploaded_file is None:
        raise ValueError("No se recibió ningún archivo.")

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Formato no soportado. Use CSV, XLSX o XLS.")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas numéricas de un DataFrame."""
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve columnas categóricas o de texto."""
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convierte una columna a fecha y elimina filas con fechas no válidas."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col)
    return out


def build_formula(target: str, predictors: Iterable[str]) -> str:
    """Construye una fórmula de statsmodels estilo R: y ~ x1 + x2."""
    predictors = list(predictors)
    if not predictors:
        raise ValueError("Debe seleccionar al menos una variable predictora.")
    return f"{target} ~ " + " + ".join(predictors)
