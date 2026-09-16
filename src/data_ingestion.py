"""Utilidades para carga e integración de datos."""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Carga un archivo CSV en un DataFrame de pandas."""
    return pd.read_csv(path, **kwargs)


def merge_dataframes(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
    """Une dos DataFrames usando una llave común."""
    return left.merge(right, on=on, how=how)
