"""Funciones de visualización estandarizadas para reportes."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_series(values, title: str, xlabel: str = "Tiempo", ylabel: str = "Valor"):
    """Grafica una serie temporal simple."""
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(values)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    return figure, axis
