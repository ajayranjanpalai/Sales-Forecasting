"""
Feature Engineering Module for Sales Forecasting
Builds trend, seasonal, calendar, promotional, and autoregressive lag features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from statsmodels.tsa.seasonal import seasonal_decompose


def add_calendar_and_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate calendar and cyclical trigonometric seasonal features.
    """
    df = df.copy()
    dates = df.index
    
    # Calendar features
    df['dayofweek'] = dates.dayofweek
    df['day'] = dates.day
    df['month'] = dates.month
    df['quarter'] = dates.quarter
    df['dayofyear'] = dates.dayofyear
    df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    df['is_month_start'] = dates.is_month_start.astype(int)
    df['is_month_end'] = dates.is_month_end.astype(int)
    
    # Cyclical Sine & Cosine Encodings
    # Day of Week (period = 7)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    
    # Month of Year (period = 12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Day of Month (period = 31)
    df['dom_sin'] = np.sin(2 * np.pi * df['day'] / 31.0)
    df['dom_cos'] = np.cos(2 * np.pi * df['day'] / 31.0)
    
    return df


def add_trend_and_promotional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate linear trend indicators, promotional interaction features, and discount momentum.
    """
    df = df.copy()
    
    # Linear trend index
    n_records = len(df)
    df['trend_step'] = np.arange(n_records)
    df['trend_norm'] = np.linspace(0, 1, n_records)
    
    # Promotional flags & intensity
    if 'discount_rate' in df.columns:
        df['is_high_promo'] = (df['discount_rate'] >= 25.0).astype(int)
        df['promo_intensity'] = pd.cut(
            df['discount_rate'],
            bins=[-1, 5, 15, 25, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Interaction between promotions and weekends
        if 'is_weekend' in df.columns:
            df['promo_weekend_interaction'] = df['discount_rate'] * df['is_weekend']
        
        # Promotional momentum (lag and rolling average discount)
        df['discount_lag_1'] = df['discount_rate'].shift(1).bfill()
        df['discount_rolling_7'] = df['discount_rate'].rolling(7, min_periods=1).mean()
    
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, 
    target_col: str = 'sales', 
    lags: List[int] = [1, 2, 3, 7, 14, 21],
    windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """
    Add autoregressive lag and rolling statistics on the target series.
    Shifts by 1 step to avoid lookahead leakage.
    """
    df = df.copy()
    
    # Autoregressive Lags
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
        
    # Rolling Statistics (lagged by 1 day)
    shifted_target = df[target_col].shift(1)
    for w in windows:
        df[f'rolling_mean_{w}'] = shifted_target.rolling(window=w, min_periods=1).mean()
        df[f'rolling_std_{w}'] = shifted_target.rolling(window=w, min_periods=1).std().fillna(0)
        df[f'rolling_min_{w}'] = shifted_target.rolling(window=w, min_periods=1).min()
        df[f'rolling_max_{w}'] = shifted_target.rolling(window=w, min_periods=1).max()
    
    # Exponential moving averages
    df['ema_7'] = shifted_target.ewm(span=7, adjust=False).mean()
    df['ema_14'] = shifted_target.ewm(span=14, adjust=False).mean()
    
    return df


def build_full_feature_matrix(
    df_ts: pd.DataFrame, 
    target_col: str = 'sales',
    drop_na_rows: bool = True
) -> pd.DataFrame:
    """
    Execute complete end-to-end feature pipeline.
    """
    data = add_calendar_and_seasonal_features(df_ts)
    data = add_trend_and_promotional_features(data)
    data = add_lag_and_rolling_features(data, target_col=target_col)
    
    if drop_na_rows:
        data = data.dropna()
        
    return data


def perform_seasonal_decomposition(
    series: pd.Series, 
    period: int = 7, 
    model: str = 'additive'
) -> Dict[str, pd.Series]:
    """
    Decompose time-series into Observed, Trend, Seasonal, and Residual components.
    """
    # Ensure regular frequency
    s = series.copy()
    decomp = seasonal_decompose(s, model=model, period=period, extrapolate_trend='freq')
    
    return {
        'observed': decomp.observed,
        'trend': decomp.trend,
        'seasonal': decomp.seasonal,
        'residual': decomp.resid
    }
