"""Funciones de limpieza y transformación de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fill_missing_values(dataframe: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Imputa valores nulos con un método de pandas (por defecto forward fill)."""
    return dataframe.fillna(method=method)


def scale_features(values: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Escala variables numéricas al rango [0, 1]."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Crea secuencias temporales para entrenamiento LSTM."""
    x_data, y_data = [], []
    for index in range(len(values) - sequence_length):
        x_data.append(values[index : index + sequence_length])
        y_data.append(values[index + sequence_length])
    return np.array(x_data), np.array(y_data)
