from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score, confusion_matrix


def fit_ols(data: pd.DataFrame, formula: str):
    """Ajusta un modelo de regresión lineal mediante fórmulas de statsmodels."""
    return smf.ols(formula=formula, data=data).fit()


def fit_logit(data: pd.DataFrame, formula: str):
    """Ajusta un modelo logístico binario mediante fórmulas de statsmodels."""
    return smf.logit(formula=formula, data=data).fit(disp=False)


def coefficients_table(model) -> pd.DataFrame:
    """Resume coeficientes, errores estándar, estadísticos y valores p."""
    conf = model.conf_int()
    stat = getattr(model, "tvalues", None)
    if stat is None:
        stat = getattr(model, "zvalues", None)

    table = pd.DataFrame(
        {
            "coeficiente": model.params,
            "error_std": model.bse,
            "estadistico": stat,
            "p_valor": model.pvalues,
            "IC_2.5%": conf[0],
            "IC_97.5%": conf[1],
        }
    )
    return table.reset_index(names="termino")


def ols_metrics(model) -> pd.DataFrame:
    """Devuelve métricas globales de un modelo OLS."""
    return pd.DataFrame(
        {
            "métrica": ["R²", "R² ajustado", "AIC", "BIC", "n"],
            "valor": [
                model.rsquared,
                model.rsquared_adj,
                model.aic,
                model.bic,
                int(model.nobs),
            ],
        }
    )


def logit_metrics(model, y_true: pd.Series, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    """Calcula métricas básicas de clasificación para un modelo logístico."""
    y_pred = (probability >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame(
        {
            "métrica": ["Exactitud", "Verdaderos negativos", "Falsos positivos", "Falsos negativos", "Verdaderos positivos"],
            "valor": [acc, tn, fp, fn, tp],
        }
    )


def odds_ratios_table(model) -> pd.DataFrame:
    """Calcula razones de momios para interpretar un modelo logístico."""
    coef = coefficients_table(model)
    coef["odds_ratio"] = np.exp(coef["coeficiente"])
    coef["OR_IC_2.5%"] = np.exp(coef["IC_2.5%"])
    coef["OR_IC_97.5%"] = np.exp(coef["IC_97.5%"])
    return coef


def scatter_with_regression_line(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    model,
    title: str,
    polynomial_degree: int = 1,
) -> go.Figure:
    """Grafica dispersión y curva ajustada para modelos con una variable predictora."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode="markers",
            name="Datos observados",
            marker=dict(size=8, opacity=0.75),
        )
    )

    x_grid = np.linspace(data[x_col].min(), data[x_col].max(), 150)
    grid = pd.DataFrame({x_col: x_grid})
    y_hat = model.predict(grid)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_hat,
            mode="lines",
            name="Ajuste del modelo",
            line=dict(width=3),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title="Elemento",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def observed_vs_predicted(model, data: pd.DataFrame, y_col: str) -> go.Figure:
    """Compara valores observados y predichos."""
    fitted = model.fittedvalues
    observed = data[y_col]
    min_v = min(observed.min(), fitted.min())
    max_v = max(observed.max(), fitted.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=observed, y=fitted, mode="markers", name="Observación"))
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Línea ideal"))
    fig.update_layout(
        title="Valores observados vs. valores ajustados",
        xaxis_title="Observado",
        yaxis_title="Ajustado",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def residual_plot(model) -> go.Figure:
    """Grafica residuos contra valores ajustados para diagnóstico básico."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=model.fittedvalues,
            y=model.resid,
            mode="markers",
            name="Residuos",
        )
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Diagnóstico: residuos vs. valores ajustados",
        xaxis_title="Valores ajustados",
        yaxis_title="Residuos",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def compare_models_table(models: dict[str, object]) -> pd.DataFrame:
    """Compara modelos OLS usando métricas de ajuste."""
    rows = []
    for name, model in models.items():
        rows.append(
            {
                "modelo": name,
                "R2": model.rsquared,
                "R2_ajustado": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic,
            }
        )
    return pd.DataFrame(rows).sort_values("AIC")
