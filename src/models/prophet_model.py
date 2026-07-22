"""
Facebook Prophet Forecasting Model Module
Decomposes trend, weekly seasonality, and exogenous promotional regressors with Bayesian uncertainty estimation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from prophet import Prophet
import logging

# Suppress Prophet internal verbose logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)


class ProphetForecaster:
    """
    Prophet Forecasting Model for Retail Time-Series.
    """
    def __init__(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.95,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        daily_seasonality: bool = False
    ):
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.model: Optional[Prophet] = None
        self.regressors: List[str] = []
        self.train_dates: Optional[pd.DatetimeIndex] = None

    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        regressor_cols: Optional[List[str]] = None
    ) -> 'ProphetForecaster':
        """
        Fit Prophet on target series y with optional exogenous regressors.
        """
        self.train_dates = y.index
        
        # Prepare Prophet format dataframe: 'ds', 'y'
        df_p = pd.DataFrame({
            'ds': pd.to_datetime(y.index),
            'y': y.values
        })
        
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        
        # Add custom monthly seasonality
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Add exogenous regressors
        self.regressors = []
        if exog is not None:
            cols = regressor_cols if regressor_cols is not None else exog.columns.tolist()
            for col in cols:
                if col in exog.columns:
                    df_p[col] = exog[col].values
                    self.model.add_regressor(col)
                    self.regressors.append(col)

        self.model.fit(df_p)
        return self

    def predict(
        self,
        steps: int,
        exog_future: Optional[pd.DataFrame] = None,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Generate future predictions with uncertainty intervals and components.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq=freq, include_history=False)
        
        # Attach future regressors
        if len(self.regressors) > 0:
            if exog_future is not None:
                for col in self.regressors:
                    if col in exog_future.columns:
                        future[col] = exog_future[col].values[:len(future)]
                    else:
                        future[col] = 0.0
            else:
                for col in self.regressors:
                    future[col] = 0.0

        forecast = self.model.predict(future)
        
        # Format output dataframe
        forecast_df = pd.DataFrame({
            'yhat': np.clip(forecast['yhat'].values, 0, None),
            'yhat_lower': np.clip(forecast['yhat_lower'].values, 0, None),
            'yhat_upper': np.clip(forecast['yhat_upper'].values, 0, None),
            'trend': forecast['trend'].values,
            'weekly': forecast['weekly'].values if 'weekly' in forecast else 0.0,
            'monthly': forecast['monthly'].values if 'monthly' in forecast else 0.0
        }, index=pd.to_datetime(forecast['ds']))
        forecast_df.index.name = 'Date'
        
        # Include regressor effects if present
        for col in self.regressors:
            if col in forecast.columns:
                forecast_df[f'{col}_effect'] = forecast[col].values
                
        return forecast_df

    def predict_in_sample(self, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict over training timeline to evaluate in-sample fit and residuals.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")

        df_p = pd.DataFrame({'ds': pd.to_datetime(self.train_dates)})
        if len(self.regressors) > 0 and exog is not None:
            for col in self.regressors:
                if col in exog.columns:
                    df_p[col] = exog[col].values

        forecast = self.model.predict(df_p)
        res_df = pd.DataFrame({
            'yhat': np.clip(forecast['yhat'].values, 0, None),
            'yhat_lower': np.clip(forecast['yhat_lower'].values, 0, None),
            'yhat_upper': np.clip(forecast['yhat_upper'].values, 0, None),
            'trend': forecast['trend'].values
        }, index=self.train_dates)
        res_df.index.name = 'Date'
        return res_df

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return model metadata and configuration details.
        """
        return {
            'model_name': 'Facebook Prophet',
            'growth': self.growth,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'regressors': self.regressors
        }
