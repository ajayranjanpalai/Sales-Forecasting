# 📈 E-Commerce Sales Forecasting System

An enterprise-grade, end-to-end time-series forecasting solution designed to predict future sales from historical e-commerce transactions using **ARIMA / SARIMAX**, **Facebook Prophet**, **Deep LSTM Neural Networks**, and **Machine Learning Ensemble Models**. 

---

## 🌟 Key Features

1. **Multi-Model Forecasting Engine**:
   - **SARIMAX $(p, d, q) \times (P, D, Q)_s$**: Classical statistical model capturing autocorrelation, differencing, moving averages, and 7-day seasonality with exogenous promotional inputs.
   - **Facebook Prophet**: Generalized Additive Model (GAM) incorporating non-linear growth curves, weekly/monthly seasonalities, changepoints, and promotional discount regressors with Bayesian uncertainty estimation.
   - **Deep LSTM Sequential Network**: Recurrent neural network built with TensorFlow/Keras capable of capturing non-linear temporal sequence patterns across multi-variate lag and rolling dynamics.
   - **XGBoost Regressor**: Feature-engineered gradient boosted decision trees for fast, high-accuracy tabular forecasting.
   - **Weighted Ensemble**: Meta-forecaster combining predictions to reduce variance and achieve superior directional accuracy.

2. **Decomposition & Component Analysis**:
   - **Trend**: Underlying growth and directional trajectory.
   - **Seasonality**: Day-of-week cyclical spikes and monthly shifts.
   - **Promotions & Discounts**: Elasticity curve modeling discount % vs. order volume and revenue generation.

3. **Interactive Streamlit Web Dashboard (`app.py`)**:
   - Modern dark glassmorphic UI with real-time KPI metrics.
   - What-if Promotional Scenario Simulator: Interactively adjust future discount % to see projected revenue impact.
   - Category-level drill-down forecasting.
   - Interactive Plotly visualizations (Forecast trajectories, 95% confidence bands, residual diagnostics).
   - One-click CSV export for downstream ERP / business operations.

4. **End-to-End Jupyter Notebook (`sales_forecasting.ipynb`)**:
   - Thoroughly documented step-by-step workflow from data exploration to model evaluation.

---

## 📁 Repository Structure

```
Sales Forecasting/
├── ecommerce_dataset_updated.csv    # Historical transaction dataset (3,660 records, 326 days)
├── app.py                          # Streamlit Interactive Web Application
├── main.py                         # CLI Pipeline runner
├── sales_forecasting.ipynb         # End-to-end Jupyter Notebook
├── README.md                       # Comprehensive project documentation
├── outputs/                        # Generated charts, leaderboard & predictions
│   ├── holdout_model_comparison.png
│   ├── future_forecast_trajectory.png
│   ├── seasonal_decomposition.png
│   ├── model_evaluation_leaderboard.csv
│   └── future_sales_forecast.csv
└── src/
    ├── __init__.py
    ├── data_loader.py              # Ingestion, date parsing & time-series aggregation
    ├── feature_engineering.py      # Trend, seasonal, promotional & lag features
    ├── evaluation.py               # RMSE, MAE, MAPE, R², MDA & residual diagnostics
    ├── visualizer.py               # Interactive Plotly & Matplotlib charts
    └── models/
        ├── __init__.py
        ├── arima_model.py          # SARIMAX Forecaster
        ├── prophet_model.py        # Facebook Prophet Forecaster
        ├── lstm_model.py           # Deep LSTM Sequence Forecaster
        ├── xgboost_model.py        # XGBoost ML Forecaster
        └── ensemble_model.py       # Weighted Ensemble Forecaster
```

---

## 🚀 Quickstart Guide

### 1. Environment & Dependencies
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels prophet tensorflow xgboost streamlit plotly
```

### 2. Run the Command-Line Pipeline
Execute the full data processing, model training, benchmarking, and forecast generation pipeline:
```bash
python main.py
```
**Optional CLI Arguments:**
- `--data`: Path to dataset CSV (default: `ecommerce_dataset_updated.csv`)
- `--output`: Output directory (default: `outputs`)
- `--test_days`: Number of holdout evaluation days (default: `30`)
- `--horizon`: Future forecast horizon in days (default: `30`)
- `--category`: Specific product category (e.g., `Clothing`, `Electronics`, or `All`)

### 3. Launch the Interactive Web Dashboard
Run the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore the interactive dashboard.

### 4. Run the Jupyter Notebook
Open and run `sales_forecasting.ipynb` via VS Code or Jupyter:
```bash
jupyter notebook sales_forecasting.ipynb
```

---

## 📊 Model Evaluation Summary

Evaluated on the last 30 days holdout test set:

| Model | RMSE (₹) | MAE (₹) | MAPE (%) | $R^2$ Score | Mean Directional Accuracy (MDA) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Forecaster** | **703.66** | **541.07** | **23.13%** | **0.0621** | **65.52%** |
| **SARIMAX** | 725.93 | 561.08 | 22.22% | 0.0018 | 62.07% |
| **XGBoost ML** | 730.90 | 558.11 | 22.61% | -0.0120 | 51.72% |
| **Facebook Prophet** | 747.41 | 600.34 | 23.61% | -0.0582 | 58.62% |
| **Deep LSTM** | 879.59 | 710.39 | 33.78% | -0.4656 | 48.28% |

---

## 📈 Visual Outputs

- **Holdout Evaluation Comparison**: `outputs/holdout_model_comparison.png`
- **Future 30-Day Trajectory with 95% Confidence Band**: `outputs/future_forecast_trajectory.png`
- **Classical 4-Panel Decomposition**: `outputs/seasonal_decomposition.png`
- **Exported Forecast Values**: `outputs/future_sales_forecast.csv`

---

## 💡 Promotional "What-If" Scenario Simulation
The system allows retailers to simulate the impact of promotional campaigns:
$$\text{Projected Sales} = f(\text{Base Trend}, \text{Weekly Seasonality}, \text{Simulated Discount Rate}, \text{Weekend Boost})$$
By adjusting the planned discount rate in the Streamlit UI, businesses can evaluate whether price reductions generate sufficient volume growth to increase net top-line revenue.
