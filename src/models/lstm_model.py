"""
Deep LSTM Neural Network Forecasting Model Module
Implements multi-feature LSTM sequence modeling using TensorFlow/Keras with recursive multi-step forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os

# Suppress TensorFlow GPU & INFO warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMForecaster:
    """
    Recurrent Neural Network Forecaster based on Long Short-Term Memory (LSTM) cells.
    """
    def __init__(
        self,
        lookback: int = 14,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        dense_units: int = 16,
        epochs: int = 70,
        batch_size: int = 16,
        learning_rate: float = 0.005,
        random_state: int = 42
    ):
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model: Optional[Sequential] = None
        self.feature_scaler: Optional[MinMaxScaler] = None
        self.target_scaler: Optional[MinMaxScaler] = None
        self.train_dates: Optional[pd.DatetimeIndex] = None
        self.feature_cols: List[str] = []
        self.train_history: Optional[Dict[str, Any]] = None
        self.residual_std: float = 0.0
        self.last_train_window: Optional[np.ndarray] = None

    def _build_sequences(
        self,
        features_scaled: np.ndarray,
        target_scaled: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D sequence tensors (samples, timesteps, features).
        """
        X, y = [], []
        for i in range(self.lookback, len(features_scaled)):
            X.append(features_scaled[i - self.lookback:i, :])
            y.append(target_scaled[i])
        return np.array(X), np.array(y)

    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        verbose: int = 0
    ) -> 'LSTMForecaster':
        """
        Fit LSTM model on target y and optional exogenous features.
        """
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        self.train_dates = y.index
        
        # Prepare feature matrix: sales target + exog features
        df_feat = pd.DataFrame({'sales': y.values}, index=y.index)
        if exog is not None:
            for col in exog.columns:
                df_feat[col] = exog[col].values
        
        self.feature_cols = df_feat.columns.tolist()

        # Scalers
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_features = self.feature_scaler.fit_transform(df_feat.values)
        scaled_target = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        X_train, y_train = self._build_sequences(scaled_features, scaled_target)

        # Store last sequence window for forecasting
        self.last_train_window = scaled_features[-self.lookback:, :]

        # Build Neural Network
        n_features = scaled_features.shape[1]
        model = Sequential()
        model.add(Input(shape=(self.lookback, n_features)))
        
        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1)
            model.add(LSTM(units, return_sequences=return_seq, activation='tanh'))
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dense(1))

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(
            monitor='loss',
            patience=12,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=verbose,
            callbacks=[early_stop],
            shuffle=False
        )

        self.model = model
        self.train_history = history.history

        # Compute training residuals to estimate prediction intervals
        in_sample_pred_scaled = model.predict(X_train, verbose=0)
        in_sample_pred = self.target_scaler.inverse_transform(in_sample_pred_scaled).flatten()
        actual_in_sample = y.values[self.lookback:]
        residuals = actual_in_sample - in_sample_pred
        self.residual_std = float(np.std(residuals))

        return self

    def predict(
        self,
        steps: int,
        exog_future: Optional[pd.DataFrame] = None,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Perform recursive multi-step forecasting into the future.
        """
        if self.model is None or self.last_train_window is None:
            raise ValueError("Model has not been fitted.")

        current_window = self.last_train_window.copy()  # shape (lookback, n_features)
        predictions_scaled = []

        # Prepare future exogenous values if available
        if exog_future is not None:
            future_exog_vals = exog_future.values
        else:
            future_exog_vals = np.zeros((steps, len(self.feature_cols) - 1))

        for step in range(steps):
            # Input sequence shape (1, lookback, n_features)
            X_step = current_window.reshape(1, self.lookback, -1)
            next_pred_scaled = self.model.predict(X_step, verbose=0)[0, 0]
            predictions_scaled.append(next_pred_scaled)

            # Build next feature row for recursive sliding
            next_row = np.zeros(len(self.feature_cols))
            next_row[0] = next_pred_scaled  # Predicted sales
            if len(self.feature_cols) > 1:
                if step < len(future_exog_vals):
                    next_row[1:] = future_exog_vals[step, :len(self.feature_cols) - 1]
                else:
                    next_row[1:] = current_window[-1, 1:]  # Repeat last observed

            # Slide window forward
            current_window = np.vstack([current_window[1:], next_row])

        # Inverse transform predictions
        pred_scaled_arr = np.array(predictions_scaled).reshape(-1, 1)
        yhat = self.target_scaler.inverse_transform(pred_scaled_arr).flatten()
        yhat = np.clip(yhat, 0, None)

        # Confidence intervals based on expanding residual uncertainty
        uncertainty_multiplier = np.sqrt(np.arange(1, steps + 1))
        lower_bound = np.clip(yhat - 1.96 * self.residual_std * (0.8 + 0.1 * uncertainty_multiplier), 0, None)
        upper_bound = np.clip(yhat + 1.96 * self.residual_std * (0.8 + 0.1 * uncertainty_multiplier), 0, None)

        # Dates
        last_date = self.train_dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)

        forecast_df = pd.DataFrame({
            'yhat': yhat,
            'yhat_lower': lower_bound,
            'yhat_upper': upper_bound
        }, index=future_dates)
        forecast_df.index.name = 'Date'
        return forecast_df

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return network architecture parameters.
        """
        return {
            'model_name': 'LSTM Recurrent Neural Network',
            'lookback_window': self.lookback,
            'lstm_layers': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'residual_std': self.residual_std,
            'feature_count': len(self.feature_cols)
        }
