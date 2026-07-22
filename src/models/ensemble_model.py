"""
Ensemble Time-Series Forecaster Module
Blends predictions from ARIMA/SARIMAX, Prophet, and LSTM using weighted multi-model aggregation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


class EnsembleForecaster:
    """
    Weighted Ensemble Forecast combining statistical (SARIMAX), structural (Prophet), and deep sequence (LSTM) models.
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        # Default balanced weights favoring robust structural and sequential components
        self.weights = weights or {
            'Prophet': 0.40,
            'SARIMAX': 0.30,
            'LSTM': 0.30
        }
        # Normalize weights
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def blend_forecasts(self, forecast_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine individual model forecast DataFrames into an ensemble forecast.
        
        Parameters:
            forecast_dict (dict): Dictionary mapping model name to forecast DataFrame (containing 'yhat', 'yhat_lower', 'yhat_upper').
            
        Returns:
            pd.DataFrame: Blended ensemble forecast with confidence intervals.
        """
        active_models = [m for m in self.weights.keys() if m in forecast_dict]
        if not active_models:
            raise ValueError(f"No matching models found in forecast_dict. Available weights: {list(self.weights.keys())}")

        # Re-normalize weights for active models
        sub_weights = {m: self.weights[m] for m in active_models}
        sub_total = sum(sub_weights.values())
        sub_weights = {k: v / sub_total for k, v in sub_weights.items()}

        sample_df = next(iter(forecast_dict.values()))
        blended_yhat = np.zeros(len(sample_df))
        blended_lower = np.zeros(len(sample_df))
        blended_upper = np.zeros(len(sample_df))

        for model_name, w in sub_weights.items():
            df_m = forecast_dict[model_name]
            blended_yhat += w * df_m['yhat'].values
            blended_lower += w * df_m['yhat_lower'].values
            blended_upper += w * df_m['yhat_upper'].values

        ensemble_df = pd.DataFrame({
            'yhat': np.clip(blended_yhat, 0, None),
            'yhat_lower': np.clip(blended_lower, 0, None),
            'yhat_upper': np.clip(blended_upper, 0, None)
        }, index=sample_df.index)
        ensemble_df.index.name = 'Date'
        
        # Attach individual model columns for comparison
        for model_name in active_models:
            ensemble_df[f'yhat_{model_name.lower()}'] = forecast_dict[model_name]['yhat'].values

        return ensemble_df
