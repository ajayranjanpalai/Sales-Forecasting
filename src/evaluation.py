"""
Model Evaluation and Diagnostic Module
Computes performance metrics (RMSE, MAE, MAPE, R2, MDA) and residual diagnostics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import skew, kurtosis


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive time-series performance metrics.
    
    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Model predicted values.
        
    Returns:
        dict: RMSE, MAE, MAPE (%), R2, and Mean Directional Accuracy (%).
    """
    actual = np.array(y_true).astype(float)
    predicted = np.array(y_pred).astype(float)

    # Filter out potential NaNs
    valid_mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if len(actual) == 0:
        return {'RMSE': 0.0, 'MAE': 0.0, 'MAPE (%)': 0.0, 'R2': 0.0, 'MDA (%)': 0.0}

    # RMSE
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    
    # MAE
    mae = float(mean_absolute_error(actual, predicted))
    
    # MAPE (handling divide by zero if actual is 0)
    denom = np.where(actual == 0, 1e-6, actual)
    mape = float(np.mean(np.abs((actual - predicted) / denom)) * 100.0)
    
    # R2
    r2 = float(r2_score(actual, predicted))
    
    # Mean Directional Accuracy (MDA)
    if len(actual) > 1:
        actual_diff = np.diff(actual)
        pred_diff = np.diff(predicted)
        mda = float(np.mean((np.sign(actual_diff) == np.sign(pred_diff))) * 100.0)
    else:
        mda = 100.0

    return {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE (%)': round(mape, 2),
        'R2': round(r2, 4),
        'MDA (%)': round(mda, 2)
    }


def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparative leaderboard ranked by lowest RMSE.
    """
    df = pd.DataFrame(results_dict).T
    if 'RMSE' in df.columns:
        df = df.sort_values('RMSE', ascending=True)
    df.index.name = 'Model'
    return df


def calculate_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Perform statistical diagnostics on model residuals (errors).
    """
    actual = np.array(y_true).astype(float)
    predicted = np.array(y_pred).astype(float)
    residuals = actual - predicted

    # Ljung-Box test for autocorrelation in residuals
    try:
        lb_res = acorr_ljungbox(residuals, lags=[7, 14], return_df=True)
        lb_pvalue_lag7 = float(lb_res.loc[7, 'lb_pvalue'])
    except Exception:
        lb_pvalue_lag7 = None

    return {
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'skewness': float(skew(residuals)),
        'kurtosis': float(kurtosis(residuals)),
        'ljung_box_pvalue_lag7': lb_pvalue_lag7,
        'has_autocorrelated_residuals': (lb_pvalue_lag7 is not None and lb_pvalue_lag7 < 0.05)
    }
