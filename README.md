# Proyecto_de_Grado

Desarrollo de un ecosistema de datos interactivo que integra indicadores y métodos predictivos LSTM para el análisis de sequías en zonas rurales, usando la metodología CRISP-DM.

## Estructura del proyecto

```text
proyecto-sequias-lstm/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   └── 03_processed/
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── lstm_model.py
│   └── visualization.py
└── reports/
    └── figures/
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Flujo recomendado (CRISP-DM)

1. Documentar objetivos de negocio en `notebooks/01_business_understanding.ipynb`.
2. Analizar los datos en `notebooks/02_data_understanding.ipynb`.
3. Preparar e imputar datos en `notebooks/03_data_preparation.ipynb` y `src/preprocessing.py`.
4. Entrenar el modelo en `notebooks/04_modeling.ipynb` con `src/lstm_model.py`.
5. Evaluar resultados en `notebooks/05_evaluation.ipynb`.
