# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # CRISP-DM — Fase 2: Comprensión de los datos
#
# Este cuaderno documenta el estado inicial de los datos observados y permite
# identificar problemas antes de cualquier transformación. No se generan datos
# sintéticos ni se eliminan registros automáticamente. Las banderas de valores
# atípicos se usan para revisión con criterio meteorológico/agronómico: una lluvia
# extrema o una temperatura inusual puede ser un evento real y relevante para la
# investigación, no un error.

# %% [markdown]
# ## 1. Dependencias, rutas y trazabilidad
#
# El bloque localiza la raíz del proyecto tanto al ejecutarse desde `notebooks/`
# como desde la raíz. La ruta configurada corresponde a un archivo que existe en
# el repositorio actual. Si ya se cuenta con una tabla consolidada de indicadores,
# cambie **solo** `DATASET_PATH` por esa tabla; conserve los datos crudos sin
# sobrescribirlos.

# %%
from pathlib import Path
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")


def locate_project_root() -> Path:
    """Encuentra la carpeta que contiene Dataset/ o data/."""
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "Dataset").exists() or (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró la raíz del proyecto. Ejecute el notebook desde el repositorio."
    )


PROJECT_ROOT = locate_project_root()
RAW_DATA_DIR = PROJECT_ROOT / "Dataset" / "Datos"
if not RAW_DATA_DIR.exists():
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"

# Archivo observado disponible actualmente. Sustituya esta ruta por la tabla
# consolidada final cuando reúna las variables de precipitación, temperatura,
# ENSO y el indicador de sequía en una sola granularidad temporal.
DATASET_PATH = RAW_DATA_DIR / "Precipitacion" / "chicoral_consolidado.csv"
SHEET_NAME = 0  # Solo se usa si DATASET_PATH es .xlsx o .xls

REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SAVE_FIGURES = False  # True para exportar PNG de alta resolución al informe

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"No existe el archivo configurado: {DATASET_PATH}")

print(f"Raíz del proyecto: {PROJECT_ROOT}")
print(f"Archivo en análisis: {DATASET_PATH}")

# %% [markdown]
# ## 2. Inventario y carga reproducible
#
# Se listan los insumos sin modificarlos y se aplica una carga según extensión.
# Los nombres de columnas se normalizan a `snake_case` sin alterar los valores;
# esta decisión evita fallos posteriores por tildes, espacios o saltos de línea.

# %%
available_files = sorted(
    p.relative_to(PROJECT_ROOT).as_posix()
    for p in RAW_DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}
)
display(Markdown(f"**Archivos tabulares detectados: {len(available_files)}**"))
display(pd.DataFrame({"archivo": available_files}))


def read_tabular_dataset(path: Path, sheet_name=0) -> pd.DataFrame:
    """Carga CSV, Excel o Parquet sin inferir transformaciones de negocio."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # sep=None detecta automáticamente coma, punto y coma o tabulador.
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Formato no soportado: {suffix}")


def normalize_column_name(column_name: object) -> str:
    """Convierte un encabezado a snake_case conservando su significado."""
    normalized = unicodedata.normalize("NFKD", str(column_name))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", normalized).strip("_").lower()
    return normalized or "columna_sin_nombre"


df_raw = read_tabular_dataset(DATASET_PATH, sheet_name=SHEET_NAME)
df = df_raw.copy()
df.columns = [normalize_column_name(column) for column in df.columns]

if df.columns.duplicated().any():
    duplicates = df.columns[df.columns.duplicated()].tolist()
    raise ValueError(f"Encabezados duplicados tras normalizar: {duplicates}")

print(f"Dimensiones (filas, columnas): {df.shape}")
display(df.head())
display(df.sample(min(5, len(df)), random_state=42) if len(df) else df)

# %% [markdown]
# ## 3. Estructura inicial y diccionario de datos conceptual
#
# El diccionario clasifica las columnas como temporales, numéricas continuas o
# categóricas. Los códigos e identificadores se tratan como categóricos aunque
# estén almacenados como números: su magnitud no representa una distancia física.
# Revise esta clasificación antes de emplear cualquier variable en el modelo.

# %%
date_name_pattern = r"fecha|date|datetime|timestamp|periodo|period|anio_mes|year_month"
identifier_pattern = r"(^id$|_id$|codigo|code|estacion|station|municipio|municipality)"


def classify_variable(name: str, series: pd.Series) -> str:
    if re.search(date_name_pattern, name, flags=re.IGNORECASE):
        return "Temporal"
    if re.search(identifier_pattern, name, flags=re.IGNORECASE):
        return "Categórica (identificador)"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Temporal"
    if pd.api.types.is_numeric_dtype(series):
        return "Numérica continua"
    return "Categórica"


data_dictionary = pd.DataFrame(
    {
        "variable": df.columns,
        "tipo_original": [str(dtype) for dtype in df.dtypes],
        "clasificacion_conceptual": [classify_variable(col, df[col]) for col in df.columns],
        "no_nulos": [df[col].notna().sum() for col in df.columns],
        "nulos": [df[col].isna().sum() for col in df.columns],
        "n_unicos": [df[col].nunique(dropna=True) for col in df.columns],
        "ejemplo": [df[col].dropna().iloc[0] if df[col].notna().any() else pd.NA for col in df.columns],
    }
)
data_dictionary["porcentaje_nulos"] = 100 * data_dictionary["nulos"] / len(df) if len(df) else 0
display(data_dictionary.sort_values(["clasificacion_conceptual", "variable"]))

print("\nSalida de df.info():")
df.info(show_counts=True)

# %% [markdown]
# ## 4. Calidad de datos: nulos, duplicados y valores atípicos
#
# Se reporta el conteo completo de valores ausentes por columna, duplicados
# exactos y duplicados por fecha cuando existe una fecha identificable. Para los
# valores atípicos se aplica la regla IQR (1.5 × rango intercuartílico) por cada
# variable continua. Esta regla es una *alerta estadística*, no una autorización
# para borrar observaciones climáticas extremas.

# %%
missing_summary = pd.DataFrame(
    {
        "nulos": df.isna().sum(),
        "porcentaje_nulos": 100 * df.isna().mean(),
        "tipo": df.dtypes.astype(str),
    }
).sort_values(["nulos", "porcentaje_nulos"], ascending=False)
display(Markdown("### Valores nulos por variable"))
display(missing_summary)

exact_duplicate_rows = int(df.duplicated().sum())
print(f"Filas duplicadas exactas: {exact_duplicate_rows:,}")

date_candidates = [column for column in df.columns if re.search(date_name_pattern, column)]
DATE_COL = date_candidates[0] if date_candidates else None
if DATE_COL is not None:
    parsed_date = pd.to_datetime(df[DATE_COL], errors="coerce")
    print(f"Columna temporal candidata: '{DATE_COL}'")
    print(f"Fechas no interpretables: {parsed_date.isna().sum():,}")
    print(f"Filas con fecha duplicada: {parsed_date.duplicated(keep=False).sum():,}")
else:
    print("No se detectó una columna temporal. Defínala manualmente antes del análisis de series.")

numeric_columns = [
    column
    for column in df.select_dtypes(include="number").columns
    if "identificador" not in classify_variable(column, df[column]).lower()
]


def iqr_outlier_summary(data: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve límites IQR y una máscara de observaciones señaladas."""
    summary_rows = []
    mask = pd.DataFrame(False, index=data.index, columns=columns)
    for column in columns:
        values = data[column].dropna()
        if values.empty:
            lower_bound = upper_bound = float("nan")
            outlier_count = 0
        else:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask[column] = (data[column] < lower_bound) | (data[column] > upper_bound)
            outlier_count = int(mask[column].sum())
        summary_rows.append(
            {
                "variable": column,
                "q1": q1 if not values.empty else float("nan"),
                "q3": q3 if not values.empty else float("nan"),
                "limite_inferior_iqr": lower_bound,
                "limite_superior_iqr": upper_bound,
                "outliers_iqr": outlier_count,
                "porcentaje_outliers": 100 * outlier_count / len(data) if len(data) else 0,
            }
        )
    return pd.DataFrame(summary_rows).sort_values("outliers_iqr", ascending=False), mask


outlier_summary, outlier_mask = iqr_outlier_summary(df, numeric_columns)
display(Markdown("### Valores atípicos IQR por variable numérica"))
display(outlier_summary)
print(f"Filas señaladas en al menos una variable: {outlier_mask.any(axis=1).sum():,}")

if numeric_columns:
    flagged_columns = [DATE_COL] if DATE_COL else []
    flagged_columns += numeric_columns
    display(Markdown("### Muestra de registros para validación de dominio"))
    display(df.loc[outlier_mask.any(axis=1), flagged_columns].head(20))

# %% [markdown]
# ## 5. EDA: distribuciones de indicadores
#
# Los histogramas describen asimetrías, acumulaciones en cero y rangos físicos.
# Ajuste `EDA_COLUMNS` si una columna numérica es un identificador o si desea
# concentrarse únicamente en los indicadores que formarán parte del modelo.

# %%
EDA_COLUMNS = [column for column in numeric_columns if df[column].nunique(dropna=True) > 1]

if not EDA_COLUMNS:
    raise ValueError("No hay variables numéricas continuas con variación para graficar.")

n_cols = 3
n_rows = (len(EDA_COLUMNS) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows))
axes = axes.flatten()

for axis, column in zip(axes, EDA_COLUMNS):
    sns.histplot(data=df, x=column, bins="auto", kde=True, ax=axis, color="#2A6F97")
    axis.set_title(f"Distribución: {column}")
    axis.set_xlabel(column)

for axis in axes[len(EDA_COLUMNS):]:
    axis.remove()

fig.suptitle("Distribuciones de variables climáticas e indicadores", y=1.02, fontsize=14)
fig.tight_layout()
if SAVE_FIGURES:
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_FIGURES_DIR / "02_histogramas_indicadores.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. EDA: comportamiento temporal
#
# La fecha se convierte solo para el análisis visual. Para datos diarios se
# recomienda inspeccionar una agregación mensual; **la agregación debe respetar
# la semántica del indicador**. Por ejemplo, precipitación diaria suele agregarse
# con suma mensual, mientras que temperatura suele agregarse con promedio.

# %%
PLOT_FREQUENCY = "MS"  # "D", "W", "MS", "QS" o "YS" según la granularidad analítica
PLOT_AGGREGATION = "mean"  # Cambiar a "sum" para acumulados diarios de precipitación

if DATE_COL is None:
    raise ValueError("Defina DATE_COL para generar las gráficas de series de tiempo.")

time_df = df[[DATE_COL, *EDA_COLUMNS]].copy()
time_df[DATE_COL] = pd.to_datetime(time_df[DATE_COL], errors="coerce")
time_df = time_df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

# Si existen varias observaciones en una misma fecha, se agregan para la gráfica;
# esta operación no reemplaza la decisión de consolidación del dataset final.
time_series = (
    time_df.groupby(DATE_COL, as_index=True)[EDA_COLUMNS]
    .mean(numeric_only=True)
    .resample(PLOT_FREQUENCY)
    .agg(PLOT_AGGREGATION)
)

fig, axes = plt.subplots(len(EDA_COLUMNS), 1, figsize=(14, 3.5 * len(EDA_COLUMNS)), sharex=True)
axes = [axes] if len(EDA_COLUMNS) == 1 else axes
for axis, column in zip(axes, EDA_COLUMNS):
    axis.plot(time_series.index, time_series[column], linewidth=1.1, color="#1D3557")
    axis.set_title(f"Serie temporal: {column} ({PLOT_AGGREGATION}, {PLOT_FREQUENCY})")
    axis.set_ylabel(column)
axes[-1].set_xlabel("Fecha")
fig.tight_layout()
if SAVE_FIGURES:
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_FIGURES_DIR / "02_series_temporales.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. EDA: correlación y posible multicolinealidad
#
# Pearson mide asociación lineal y Spearman asociación monótona basada en rangos.
# Ninguna correlación prueba causalidad. Como guía, correlaciones absolutas altas
# entre predictores (por ejemplo, ≥ 0.85) justifican revisar redundancia, rezagos
# y la elección de variables antes del entrenamiento de la LSTM.

# %%
CORRELATION_METHODS = ("pearson", "spearman")
HIGH_CORRELATION_THRESHOLD = 0.85

if len(EDA_COLUMNS) < 2:
    print("Se requieren al menos dos variables numéricas para calcular una matriz de correlación.")
else:
    for method in CORRELATION_METHODS:
        correlation = df[EDA_COLUMNS].corr(method=method)
        plt.figure(figsize=(max(7, len(EDA_COLUMNS) * 1.1), max(5, len(EDA_COLUMNS) * 0.9)))
        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.4,
        )
        plt.title(f"Matriz de correlación {method.title()}")
        plt.tight_layout()
        if SAVE_FIGURES:
            REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                REPORTS_FIGURES_DIR / f"02_correlacion_{method}.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

        upper_triangle = correlation.where(
            np.triu(np.ones(correlation.shape), k=1).astype(bool)
        )
        high_pairs = (
            upper_triangle.stack()
            .rename("correlacion")
            .loc[lambda values: values.abs() >= HIGH_CORRELATION_THRESHOLD]
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )
        print(f"Pares con |correlación {method}| ≥ {HIGH_CORRELATION_THRESHOLD}:")
        display(high_pairs.to_frame())

# %% [markdown]
# ## 8. Cierre de la fase de comprensión
#
# Registre en el informe: periodo cubierto, estación/municipio, granularidad,
# tasa de nulos, duplicados, atípicos validados y decisiones sobre indicadores.
# Antes de la Fase 3 se debe disponer de una tabla con una fila por fecha y las
# columnas numéricas de predictores más el indicador objetivo de sequía. No es
# metodológicamente válido etiquetar automáticamente `valor` (precipitación) como
# sequía sin definir y justificar un indicador —por ejemplo SPI/SPEI— y su horizonte.
