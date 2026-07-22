"""
Main CLI Pipeline for Sales Forecasting
Executes end-to-end data loading, seasonal decomposition, model training, evaluation, future forecasting, and reporting.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.data_loader import load_raw_data, aggregate_time_series, get_dataset_summary
from src.feature_engineering import build_full_feature_matrix, perform_seasonal_decomposition
from src.models.arima_model import SARIMAXForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.evaluation import calculate_metrics, compare_models, calculate_residual_diagnostics
from src.visualizer import plot_forecast_vs_actual, plot_multi_model_comparison


def run_sales_forecasting_pipeline(
    data_path: str = 'ecommerce_dataset_updated.csv',
    output_dir: str = 'outputs',
    test_days: int = 30,
    forecast_horizon: int = 30,
    category: str = 'All'
):
    """
    Execute complete end-to-end sales forecasting pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=================================================================")
    print("           E-COMMERCE SALES FORECASTING ENGINE                   ")
    print("=================================================================")

    # 1. Load Data
    print(f"\n[1/6] Loading data from {data_path}...")
    df_raw = load_raw_data(data_path)
    summary = get_dataset_summary(df_raw)
    print(f" -> Total Transactions: {summary['total_transactions']:,}")
    print(f" -> Date Span: {summary['start_date']} to {summary['end_date']} ({summary['total_days_span']} days)")
    print(f" -> Total Historical Revenue: Rs. {summary['total_revenue']:,.2f}")
    print(f" -> Categories: {', '.join(summary['categories'])}")

    # 2. Resample Time-Series
    print(f"\n[2/6] Aggregating daily time-series (Filter: {category})...")
    df_daily = aggregate_time_series(df_raw, freq='D', category=None if category == 'All' else category)
    print(f" -> Continuous Daily Observations: {len(df_daily)}")

    # 3. Seasonal Decomposition
    print("\n[3/6] Performing classical seasonal decomposition (Period=7)...")
    decomp = perform_seasonal_decomposition(df_daily['sales'], period=7)
    
    # Save static decomposition plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(decomp['observed'], color='#0284C7', label='Observed')
    axes[0].set_title('Observed Daily Sales')
    axes[1].plot(decomp['trend'], color='#DC2626', label='Trend')
    axes[1].set_title('Underlying Trend')
    axes[2].plot(decomp['seasonal'], color='#7C3AED', label='Weekly Seasonality')
    axes[2].set_title('Weekly Seasonality (7-day pattern)')
    axes[3].scatter(decomp['residual'].index, decomp['residual'], color='#475569', s=10, label='Residual')
    axes[3].set_title('Residual Noise')
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    decomp_plot_path = os.path.join(output_dir, 'seasonal_decomposition.png')
    fig.savefig(decomp_plot_path, dpi=200)
    plt.close(fig)
    print(f" -> Saved decomposition plot to {decomp_plot_path}")

    # 4. Feature Matrix & Train/Test Split
    print(f"\n[4/6] Engineering lag, rolling, calendar, and promotional features...")
    df_features = build_full_feature_matrix(df_daily, target_col='sales')
    
    # Holdout Split
    train_df = df_features.iloc[:-test_days].copy()
    test_df = df_features.iloc[-test_days:].copy()
    print(f" -> Training Period: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')} ({len(train_df)} days)")
    print(f" -> Holdout Test Period: {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')} ({len(test_df)} days)")

    # 5. Model Training & Holdout Evaluation
    print("\n[5/6] Training & Evaluating Forecasting Models...")
    eval_predictions = {}
    metrics_results = {}
    
    exog_cols = ['discount_rate', 'is_weekend']

    # --- Model A: SARIMAX ---
    print(" -> Training SARIMAX (p=1, d=1, q=1) x (P=1, D=0, Q=1, s=7)...")
    sarimax = SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
    sarimax.fit(train_df['sales'], exog=train_df[exog_cols])
    sarimax_test_pred = sarimax.predict(steps=test_days, exog_future=test_df[exog_cols], start_date=test_df.index[0])
    eval_predictions['SARIMAX'] = sarimax_test_pred['yhat'].values
    metrics_results['SARIMAX'] = calculate_metrics(test_df['sales'].values, sarimax_test_pred['yhat'].values)

    # --- Model B: Facebook Prophet ---
    print(" -> Training Facebook Prophet (Weekly + Monthly seasonality + Promotional regressor)...")
    prophet = ProphetForecaster()
    prophet.fit(train_df['sales'], exog=train_df[exog_cols])
    prophet_test_pred = prophet.predict(steps=test_days, exog_future=test_df[exog_cols])
    eval_predictions['Prophet'] = prophet_test_pred['yhat'].values
    metrics_results['Prophet'] = calculate_metrics(test_df['sales'].values, prophet_test_pred['yhat'].values)

    # --- Model C: Deep LSTM Neural Network ---
    print(" -> Training Deep LSTM Sequential Network (64 -> Dropout -> 32 -> Dense)...")
    lstm = LSTMForecaster(lookback=14, epochs=60, batch_size=16)
    lstm.fit(train_df['sales'], exog=train_df[exog_cols], verbose=0)
    lstm_test_pred = lstm.predict(steps=test_days, exog_future=test_df[exog_cols])
    eval_predictions['LSTM'] = lstm_test_pred['yhat'].values
    metrics_results['LSTM'] = calculate_metrics(test_df['sales'].values, lstm_test_pred['yhat'].values)

    # --- Model D: XGBoost Regressor ---
    print(" -> Training XGBoost ML Regressor with lag & rolling dynamics...")
    xgb_forecaster = XGBoostForecaster(n_estimators=120, max_depth=3)
    xgb_forecaster.fit(train_df, target_col='sales')
    xgb_test_pred = xgb_forecaster.predict(steps=test_days)
    eval_predictions['XGBoost'] = xgb_test_pred['yhat'].values
    metrics_results['XGBoost'] = calculate_metrics(test_df['sales'].values, xgb_test_pred['yhat'].values)

    # --- Model E: Weighted Ensemble ---
    print(" -> Blending forecasts with Weighted Ensemble Forecaster...")
    ensemble = EnsembleForecaster(weights={'Prophet': 0.40, 'SARIMAX': 0.30, 'LSTM': 0.30})
    ensemble_test_pred = ensemble.blend_forecasts({
        'SARIMAX': sarimax_test_pred,
        'Prophet': prophet_test_pred,
        'LSTM': lstm_test_pred
    })
    eval_predictions['Ensemble'] = ensemble_test_pred['yhat'].values
    metrics_results['Ensemble'] = calculate_metrics(test_df['sales'].values, ensemble_test_pred['yhat'].values)

    # Comparison Leaderboard
    leaderboard = compare_models(metrics_results)
    print("\n================ MODEL PERFORMANCE LEADERBOARD ================")
    print(leaderboard.to_string())
    leaderboard_csv_path = os.path.join(output_dir, 'model_evaluation_leaderboard.csv')
    leaderboard.to_csv(leaderboard_csv_path)

    # Save Holdout Evaluation Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_df.index, test_df['sales'], color='black', linewidth=2.5, marker='o', label='Actual Sales')
    ax.plot(test_df.index, eval_predictions['Prophet'], label='Prophet', linestyle='--', color='#8B5CF6')
    ax.plot(test_df.index, eval_predictions['SARIMAX'], label='SARIMAX', linestyle='--', color='#10B981')
    ax.plot(test_df.index, eval_predictions['LSTM'], label='LSTM', linestyle='--', color='#EC4899')
    ax.plot(test_df.index, eval_predictions['Ensemble'], label='Ensemble', linewidth=2, color='#06B6D4')
    ax.set_title(f'Model Holdout Evaluation Comparison (Last {test_days} Days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales Revenue (Rs.)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, 'holdout_model_comparison.png')
    fig.savefig(comp_plot_path, dpi=200)
    plt.close(fig)
    print(f" -> Saved comparison plot to {comp_plot_path}")

    # 6. Future Forecasting (Out-of-sample)
    print(f"\n[6/6] Generating Future {forecast_horizon}-Day Out-of-Sample Forecasts...")
    
    # Fit models on the complete dataset features
    full_prophet = ProphetForecaster()
    full_prophet.fit(df_features['sales'], exog=df_features[exog_cols])
    
    # Create future exogenous features (assuming average promotional discount)
    future_dates = pd.date_range(start=df_features.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    future_exog = pd.DataFrame({
        'discount_rate': [float(df_features['discount_rate'].mean())] * forecast_horizon,
        'is_weekend': (future_dates.dayofweek >= 5).astype(int)
    }, index=future_dates)
    
    future_forecast = full_prophet.predict(steps=forecast_horizon, exog_future=future_exog)
    forecast_csv_path = os.path.join(output_dir, 'future_sales_forecast.csv')
    future_forecast.to_csv(forecast_csv_path)
    print(f" -> Saved future forecast to {forecast_csv_path}")

    # Plot future trajectory
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_features.index[-90:], df_features['sales'].iloc[-90:], color='#0284C7', label='Historical Sales (Last 90d)')
    ax.plot(future_forecast.index, future_forecast['yhat'], color='#F59E0B', linestyle='--', linewidth=2.5, label='Future Forecast (Prophet)')
    ax.fill_between(future_forecast.index, future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='#F59E0B', alpha=0.2, label='95% Confidence Band')
    ax.set_title(f'Future {forecast_horizon}-Day Sales Forecast Trajectory', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales Revenue (Rs.)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    plt.tight_layout()
    traj_plot_path = os.path.join(output_dir, 'future_forecast_trajectory.png')
    fig.savefig(traj_plot_path, dpi=200)
    plt.close(fig)
    print(f" -> Saved future trajectory plot to {traj_plot_path}")

    print("\n=================================================================")
    print(" Pipeline execution finished successfully!                       ")
    print(f" Outputs directory: {os.path.abspath(output_dir)}               ")
    print("=================================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="E-Commerce Sales Forecasting CLI")
    parser.add_argument('--data', type=str, default='ecommerce_dataset_updated.csv', help='Path to CSV dataset')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--test_days', type=int, default=30, help='Number of days for holdout test set')
    parser.add_argument('--horizon', type=int, default=30, help='Future forecast horizon in days')
    parser.add_argument('--category', type=str, default='All', help='Specific product category or All')
    
    args = parser.parse_args()
    run_sales_forecasting_pipeline(
        data_path=args.data,
        output_dir=args.output,
        test_days=args.test_days,
        forecast_horizon=args.horizon,
        category=args.category
    )
