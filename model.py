"""
model.py
--------
Defines deep learning and baseline machine learning architectures for
time-series stock price forecasting:
1. Stacked LSTM (2 LSTM layers with Dropout and Dense output).
2. Stacked GRU (2 GRU layers with Dropout and Dense output).
3. Simple Feedforward Neural Network (Flatten + Dense layers with Dropout).
4. Linear Regression baseline (Scikit-Learn).
"""

from sklearn.linear_model import LinearRegression


def build_lstm_model(
    input_shape: tuple,
    lstm_units_1: int = 64,
    lstm_units_2: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
):
    """
    Constructs a 2-layer stacked LSTM network in Keras.
    
    Architecture:
    - Input: (window_size, num_features)
    - LSTM 1: lstm_units_1 memory cells, return_sequences=True (passes sequential representations)
    - Dropout 1: Zeroes activations at dropout_rate to prevent overfitting on price noise
    - LSTM 2: lstm_units_2 memory cells, return_sequences=False (condenses into context vector)
    - Dropout 2: Additional regularization layer
    - Dense(1): Linear scalar output for predicted normalized Close price
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=input_shape, name="input_window"),
        LSTM(units=lstm_units_1, return_sequences=True, name="lstm_layer_1"),
        Dropout(rate=dropout_rate, name="dropout_1"),
        LSTM(units=lstm_units_2, return_sequences=False, name="lstm_layer_2"),
        Dropout(rate=dropout_rate, name="dropout_2"),
        Dense(units=1, activation="linear", name="dense_output")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model


def build_gru_model(
    input_shape: tuple,
    gru_units_1: int = 64,
    gru_units_2: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
):
    """
    Constructs a 2-layer stacked GRU (Gated Recurrent Unit) network.
    
    GRU is computationally lighter than LSTM (2 gates: Reset & Update, vs LSTM's 3 gates).
    Often trains faster and performs comparably on small to medium-sized financial datasets.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dropout, Dense, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=input_shape, name="input_window"),
        GRU(units=gru_units_1, return_sequences=True, name="gru_layer_1"),
        Dropout(rate=dropout_rate, name="dropout_1"),
        GRU(units=gru_units_2, return_sequences=False, name="gru_layer_2"),
        Dropout(rate=dropout_rate, name="dropout_2"),
        Dense(units=1, activation="linear", name="dense_output")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model


def build_feedforward_model(
    input_shape: tuple,
    dense_units_1: int = 64,
    dense_units_2: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
):
    """
    Constructs a standard Multi-Layer Perceptron (MLP / ANN) feedforward network.
    
    Flattens the 2D window into a 1D vector and passes through dense linear layers.
    Does NOT maintain an internal recurrent hidden state across time steps.
    Used to benchmark whether recurrent temporal recurrence is genuinely beneficial.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=input_shape, name="input_window"),
        Flatten(name="flatten_window"),
        Dense(units=dense_units_1, activation="relu", name="dense_1"),
        Dropout(rate=dropout_rate, name="dropout_1"),
        Dense(units=dense_units_2, activation="relu", name="dense_2"),
        Dropout(rate=dropout_rate, name="dropout_2"),
        Dense(units=1, activation="linear", name="dense_output")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model


def build_linear_regression():
    """
    Constructs a standard Ordinary Least Squares Linear Regression baseline.
    Operates on flattened sliding window feature representations.
    """
    return LinearRegression()
