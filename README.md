<<<<<<< HEAD
# App multipágina de Estadística Aplicada: Regresiones y AutoTS

Aplicación desarrollada en Streamlit para cerrar el semestre de **Estadística Aplicada con Python y R** mediante un proyecto integrado de regresiones, series de tiempo, AutoTS, auditoría de modelos y despliegue web.

## 1. Objetivo

Convertir los seis notebooks de Google Colab en una aplicación web multipágina que permita:

- Ajustar e interpretar regresión lineal simple.
- Ajustar e interpretar regresión lineal múltiple.
- Comparar modelos lineales y polinomiales.
- Ajustar regresión logística para respuestas binarias.
- Preparar series de tiempo con índice temporal.
- Aplicar descomposición clásica.
- Entrenar pronósticos automatizados con AutoTS.
- Revisar el modelo ganador, el leaderboard y las métricas MAE/RMSE.
- Exportar pronósticos a CSV.
- Preparar el despliegue en Streamlit Cloud.

## 2. Estructura del proyecto

```text
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
```

## 3. Instalación local en Windows con VSCode

Abra PowerShell en la carpeta donde desea trabajar y ejecute:

```powershell
cd C:\
git clone <URL_DEL_REPOSITORIO>
cd estadistica_regresion_autots_streamlit

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Si PowerShell bloquea la activación del entorno virtual, ejecute una sola vez:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego cierre y abra de nuevo PowerShell.

## 4. Datos incluidos

La carpeta `data/` incluye datos locales para evitar depender de descargas externas:

- `anscombe.csv`: Cuarteto de Anscombe.
- `tips_demo.csv`: datos didácticos para regresión múltiple.
- `mpg_demo.csv`: datos didácticos para regresión polinomial.
- `titanic_demo.csv`: datos didácticos para regresión logística.
- `flights.csv`: serie mensual clásica de pasajeros.
- `pronostico_autots_demo.csv`: pronóstico demostrativo.
- `leaderboard_demo.csv`: auditoría demostrativa.

## 5. Despliegue en Streamlit Cloud

1. Cree un repositorio en GitHub.
2. Suba todos los archivos de esta carpeta.
3. Entre a Streamlit Community Cloud.
4. Cree una nueva app desde el repositorio.
5. Seleccione `app.py` como archivo principal.
6. Publique la aplicación.

## 6. Recomendación docente

Para clase, use en AutoTS:

- `model_list = "superfast"`
- `max_generations = 1`
- `num_validations = 1` o `2`
- `forecast_length = 12`

Esto permite que la app sea útil en equipos de estudiantes con recursos limitados.

## 7. Extensión sugerida para estudiantes

Cada grupo puede reemplazar los datos demostrativos por un dataset propio de ingeniería agrícola, agroindustrial o civil, y luego documentar:

- Problema técnico.
- Variable respuesta.
- Variables predictoras.
- Justificación del modelo.
- Métricas de ajuste o error.
- Interpretación técnica del pronóstico.
- Enlace público del dashboard.
=======
# estadistica_regresion_autots_streamlit
>>>>>>> cba47362ee6239f22547ab8837c18b7453009528
