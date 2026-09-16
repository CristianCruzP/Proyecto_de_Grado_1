"""Definición del modelo LSTM."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


class DroughtLSTMModel:
    """Modelo LSTM configurable para predicción de sequías."""

    def __init__(self, sequence_length: int, n_features: int, units: int = 64):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential(
            [
                LSTM(self.units, input_shape=(self.sequence_length, self.n_features)),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def fit(self, x_train, y_train, **kwargs):
        return self.model.fit(x_train, y_train, **kwargs)

    def predict(self, x_data, **kwargs):
        return self.model.predict(x_data, **kwargs)
