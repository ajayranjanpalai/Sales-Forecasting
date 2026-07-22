"""
ARIMA / SARIMAX Forecasting Model Module
Implements Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX).
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
import pmdarima as pm
import warnings

warnings.filterwarnings('ignore')


class SARIMAXForecaster:
    """
    SARIMAX Time Series Forecasting Model with exogenous variable support.
    """
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.fitted_model: Optional[SARIMAXResultsWrapper] = None
        self.train_index: Optional[pd.DatetimeIndex] = None
        self.exog_cols: Optional[List[str]] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None

    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        auto_tune: bool = False,
        seasonal_period: int = 7
    ) -> 'SARIMAXForecaster':
        """
        Fit SARIMAX model to training series y, optionally using auto_arima.
        """
        self.train_index = y.index
        if exog is not None:
            self.exog_cols = exog.columns.tolist()
            exog_matrix = exog.values
        else:
            self.exog_cols = None
            exog_matrix = None

        if auto_tune:
            try:
                auto_model = pm.auto_arima(
                    y,
                    X=exog_matrix,
                    start_p=0, max_p=3,
                    start_q=0, max_q=3,
                    m=seasonal_period,
                    seasonal=True,
                    d=1, D=0,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                self.order = auto_model.order
                self.seasonal_order = auto_model.seasonal_order
            except Exception:
                # Fallback to default orders
                self.order = (1, 1, 1)
                self.seasonal_order = (1, 0, 1, seasonal_period)

        model = SARIMAX(
            endog=y,
            exog=exog_matrix,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        self.fitted_model = model.fit(disp=False, maxiter=200)
        self.aic = float(self.fitted_model.aic)
        self.bic = float(self.fitted_model.bic)
        return self

    def predict(
        self,
        steps: int,
        exog_future: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
        start_date: Optional[pd.Timestamp] = None,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Generate out-of-sample forecast with confidence intervals.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        exog_pred = exog_future.values if exog_future is not None else None
        
        forecast_res = self.fitted_model.get_forecast(steps=steps, exog=exog_pred)
        predicted_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=alpha)

        # Generate future date index if not provided
        if start_date is None and self.train_index is not None:
            last_date = self.train_index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)
        elif start_date is not None:
            future_dates = pd.date_range(start=start_date, periods=steps, freq=freq)
        else:
            future_dates = range(steps)

        # Ensure no negative sales predictions
        yhat = np.clip(np.array(predicted_mean), a_min=0, a_max=None)
        lower_bound = np.clip(np.array(conf_int.iloc[:, 0]), a_min=0, a_max=None)
        upper_bound = np.clip(np.array(conf_int.iloc[:, 1]), a_min=0, a_max=None)

        forecast_df = pd.DataFrame({
            'yhat': yhat,
            'yhat_lower': lower_bound,
            'yhat_upper': upper_bound
        }, index=future_dates)
        forecast_df.index.name = 'Date'
        return forecast_df

    def predict_in_sample(self, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Return in-sample fitted values.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted.")
        exog_mat = exog.values if exog is not None else None
        fitted_vals = self.fitted_model.fittedvalues
        return pd.Series(np.clip(fitted_vals, 0, None), index=self.train_index)

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return model parameters, summary statistics, and information criteria.
        """
        if self.fitted_model is None:
            return {}
        return {
            'model_name': 'SARIMAX',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.aic,
            'bic': self.bic,
            'exog_features': self.exog_cols
        }
