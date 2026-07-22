"""
Streamlit Web Application - E-Commerce Sales Forecasting Dashboard
Interactive enterprise-grade time-series forecasting studio featuring SARIMAX, Prophet, LSTM, and Ensemble models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Add root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw_data, aggregate_time_series, get_dataset_summary, get_category_pivot
from src.feature_engineering import build_full_feature_matrix, perform_seasonal_decomposition
from src.models.arima_model import SARIMAXForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.evaluation import calculate_metrics, compare_models, calculate_residual_diagnostics
from src.visualizer import (
    plot_forecast_vs_actual, 
    plot_multi_model_comparison, 
    plot_seasonal_decomposition_plotly,
    plot_promotional_elasticity,
    plot_category_breakdown_sunburst,
    PALETTE
)

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting AI Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Dark Theme & Typography */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #1E293B 100%);
        border: 1px solid #312E81;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-desc {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 13px;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 12px;
        color: #10B981;
        margin-top: 2px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #090D16;
        border-right: 1px solid #1E293B;
    }
    
    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    csv_path = 'ecommerce_dataset_updated.csv'
    if os.path.exists(csv_path):
        return load_raw_data(csv_path)
    return None


# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("### **Forecasting Controls**")
    st.markdown("---")
    
    df_raw = load_data()
    if df_raw is None:
        st.error("⚠️ Dataset `ecommerce_dataset_updated.csv` not found!")
        st.stop()
        
    categories = ['All'] + sorted(df_raw['Category'].dropna().unique().tolist())
    selected_category = st.selectbox("🎯 Target Category", categories, index=0)
    
    resample_freq = st.selectbox(
        "⏱️ Aggregation Frequency",
        options=['Daily (D)', 'Weekly (W)', 'Monthly (M)'],
        index=0
    )
    freq_code = {'Daily (D)': 'D', 'Weekly (W)': 'W-MON', 'Monthly (M)': 'ME'}[resample_freq]
    
    st.markdown("---")
    st.markdown("### **Model Configuration**")
    
    model_choice = st.selectbox(
        "🤖 Forecasting Algorithm",
        ["Prophet", "SARIMAX", "LSTM Neural Net", "XGBoost ML", "Weighted Ensemble"],
        index=0
    )
    
    forecast_horizon = st.slider("📅 Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=7)
    test_split_days = st.slider("🧪 Holdout Test Size (Days)", min_value=14, max_value=60, value=30, step=1)
    
    st.markdown("---")
    st.markdown("### **💡 Promotional Simulation**")
    simulated_discount = st.slider(
        "Simulated Future Discount Rate (%)", 
        min_value=0, max_value=50, value=int(df_raw['Discount (%)'].mean()), step=1,
        help="Adjust planned promotional discount % to simulate impact on predicted sales."
    )

# Prepare Base Time Series
df_ts = aggregate_time_series(df_raw, freq=freq_code, category=None if selected_category == 'All' else selected_category)
summary = get_dataset_summary(df_raw)

# Header Banner
st.markdown(f"""
<div class="header-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="header-title">E-Commerce Sales Forecasting Studio</h1>
            <p class="header-desc">Multi-model time-series intelligence powered by ARIMA/SARIMAX, Facebook Prophet, LSTM Neural Networks & Machine Learning.</p>
        </div>
        <div>
            <span class="badge-pill">326 Continuous Days</span>
            <span class="badge-pill">7 Categories</span>
            <span class="badge-pill">Promotional Regressors</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main KPI Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_rev = df_ts['sales'].sum()
avg_daily_rev = df_ts['sales'].mean()
total_orders = df_ts['transactions'].sum()
avg_discount = df_ts['discount_rate'].mean()
peak_sales = df_ts['sales'].max()

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Revenue</div>
        <div class="metric-value">₹{total_rev:,.0f}</div>
        <div class="metric-delta">{selected_category} Scope</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Daily Avg Revenue</div>
        <div class="metric-value">₹{avg_daily_rev:,.0f}</div>
        <div class="metric-delta">± ₹{df_ts['sales'].std():,.0f} std dev</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Volume</div>
        <div class="metric-value">{total_orders:,}</div>
        <div class="metric-delta">Avg ₹{total_rev/max(1, total_orders):,.1f} / order</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Discount</div>
        <div class="metric-value">{avg_discount:.1f}%</div>
        <div class="metric-delta">Promotional Intensity</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Single Day Peak</div>
        <div class="metric-value">₹{peak_sales:,.0f}</div>
        <div class="metric-delta">Highest Recorded Sales</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab_overview, tab_decomposition, tab_forecast, tab_eval, tab_export = st.tabs([
    "📊 Executive Overview",
    "📈 Trend & Seasonality",
    "🔮 Forecasting Studio",
    "🏆 Model Comparison & Diagnostics",
    "💾 Export Forecast Data"
])

# ================= TAB 1: EXECUTIVE OVERVIEW =================
with tab_overview:
    st.subheader("Historical Sales Trajectory")
    
    # Interactive Time Series Plot
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_ts.index,
        y=df_ts['sales'],
        mode='lines',
        name='Daily Sales (₹)',
        line=dict(color='#38BDF8', width=2),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.08)'
    ))
    # 7-day Moving Average
    df_ts['ma_7'] = df_ts['sales'].rolling(7, min_periods=1).mean()
    fig_hist.add_trace(go.Scatter(
        x=df_ts.index,
        y=df_ts['ma_7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='#F59E0B', width=2.5)
    ))
    fig_hist.update_layout(
        template="plotly_dark",
        plot_bgcolor=PALETTE['card_bg'],
        paper_bgcolor=PALETTE['background'],
        xaxis=dict(title="Date", showgrid=True, gridcolor=PALETTE['grid']),
        yaxis=dict(title="Revenue (₹)", showgrid=True, gridcolor=PALETTE['grid'], tickprefix="₹"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(plot_promotional_elasticity(df_raw), use_container_width=True)
    with col_c2:
        st.plotly_chart(plot_category_breakdown_sunburst(df_raw), use_container_width=True)

# ================= TAB 2: DECOMPOSITION & DIAGNOSTICS =================
with tab_decomposition:
    st.subheader("Time-Series Structural Decomposition")
    st.markdown("Decomposing observed sales into **Long-term Trend**, **Cyclical Seasonality (7-day period)**, and **Residual Stochastic Noise**.")
    
    decomp_res = perform_seasonal_decomposition(df_ts['sales'], period=7)
    fig_decomp = plot_seasonal_decomposition_plotly(decomp_res)
    st.plotly_chart(fig_decomp, use_container_width=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        # Day of Week Distribution
        df_dow = df_ts.copy()
        df_dow['Day'] = df_dow.index.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = df_dow.groupby('Day')['sales'].mean().reindex(days_order).reset_index()
        
        fig_dow = px.bar(
            dow_avg, x='Day', y='sales',
            title="<b>Average Sales by Day of Week</b>",
            color='sales',
            color_continuous_scale='Viridis',
            template='plotly_dark'
        )
        fig_dow.update_layout(
            plot_bgcolor=PALETTE['card_bg'],
            paper_bgcolor=PALETTE['background'],
            yaxis=dict(tickprefix="₹")
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        
    with col_s2:
        # Category pivot heatmap over months
        cat_pivot = get_category_pivot(df_raw, freq='ME')
        cat_pivot.index = cat_pivot.index.strftime('%b %Y')
        fig_heat = px.imshow(
            cat_pivot.T,
            labels=dict(x="Month", y="Category", color="Revenue (₹)"),
            title="<b>Monthly Category Revenue Heatmap</b>",
            color_continuous_scale='Magma',
            template='plotly_dark'
        )
        fig_heat.update_layout(
            plot_bgcolor=PALETTE['card_bg'],
            paper_bgcolor=PALETTE['background']
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ================= TAB 3: FORECASTING STUDIO =================
with tab_forecast:
    st.subheader(f"🔮 Future {forecast_horizon}-Day Forecast ({model_choice})")
    
    # Feature Engineering
    df_features = build_full_feature_matrix(df_ts, target_col='sales')
    exog_cols = ['discount_rate', 'is_weekend']
    
    # Future exogenous dataframe
    future_dates = pd.date_range(start=df_features.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    future_exog = pd.DataFrame({
        'discount_rate': [float(simulated_discount)] * forecast_horizon,
        'is_weekend': (future_dates.dayofweek >= 5).astype(int)
    }, index=future_dates)
    
    with st.spinner(f"Training {model_choice} model and generating forecast..."):
        if model_choice == "Prophet":
            model = ProphetForecaster()
            model.fit(df_features['sales'], exog=df_features[exog_cols])
            forecast_df = model.predict(steps=forecast_horizon, exog_future=future_exog)
        elif model_choice == "SARIMAX":
            model = SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
            model.fit(df_features['sales'], exog=df_features[exog_cols])
            forecast_df = model.predict(steps=forecast_horizon, exog_future=future_exog)
        elif model_choice == "LSTM Neural Net":
            model = LSTMForecaster(lookback=14, epochs=50, batch_size=16)
            model.fit(df_features['sales'], exog=df_features[exog_cols], verbose=0)
            forecast_df = model.predict(steps=forecast_horizon, exog_future=future_exog)
        elif model_choice == "XGBoost ML":
            model = XGBoostForecaster(n_estimators=120, max_depth=3)
            model.fit(df_features, target_col='sales')
            forecast_df = model.predict(steps=forecast_horizon)
        else: # Weighted Ensemble
            p_model = ProphetForecaster().fit(df_features['sales'], exog=df_features[exog_cols])
            s_model = SARIMAXForecaster(order=(1,1,1), seasonal_order=(1,0,1,7)).fit(df_features['sales'], exog=df_features[exog_cols])
            l_model = LSTMForecaster(lookback=14, epochs=40, batch_size=16).fit(df_features['sales'], exog=df_features[exog_cols], verbose=0)
            
            p_pred = p_model.predict(steps=forecast_horizon, exog_future=future_exog)
            s_pred = s_model.predict(steps=forecast_horizon, exog_future=future_exog)
            l_pred = l_model.predict(steps=forecast_horizon, exog_future=future_exog)
            
            ensemble = EnsembleForecaster()
            forecast_df = ensemble.blend_forecasts({'Prophet': p_pred, 'SARIMAX': s_pred, 'LSTM': l_pred})

    # Forecast Plot
    fig_f = plot_forecast_vs_actual(
        history_df=df_ts.iloc[-90:], 
        forecast_df=forecast_df, 
        model_name=model_choice,
        title=f"{selected_category} Sales Forecast ({forecast_horizon} Days Ahead with {simulated_discount}% Promo Discount)"
    )
    st.plotly_chart(fig_f, use_container_width=True)
    
    # Forecast Summary Cards
    fc_col1, fc_col2, fc_col3 = st.columns(3)
    projected_revenue = forecast_df['yhat'].sum()
    daily_projected_avg = forecast_df['yhat'].mean()
    growth_rate = ((daily_projected_avg - avg_daily_rev) / max(1, avg_daily_rev)) * 100.0
    
    with fc_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Horizon Revenue</div>
            <div class="metric-value">₹{projected_revenue:,.0f}</div>
            <div class="metric-delta">Next {forecast_horizon} Days</div>
        </div>
        """, unsafe_allow_html=True)
    with fc_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Daily Average</div>
            <div class="metric-value">₹{daily_projected_avg:,.0f}</div>
            <div class="metric-delta">vs Historical ₹{avg_daily_rev:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with fc_col3:
        delta_color = "#10B981" if growth_rate >= 0 else "#EF4444"
        delta_sign = "+" if growth_rate >= 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Growth Delta</div>
            <div class="metric-value" style="color: {delta_color};">{delta_sign}{growth_rate:.2f}%</div>
            <div class="metric-delta">Driven by Seasonality & {simulated_discount}% Discount</div>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: MODEL EVALUATION & DIAGNOSTICS =================
with tab_eval:
    st.subheader(f"🏆 Multi-Model Benchmark (Holdout Test: Last {test_split_days} Days)")
    
    # Train/Test Split
    train_df = df_features.iloc[:-test_split_days].copy()
    test_df = df_features.iloc[-test_split_days:].copy()
    
    with st.spinner("Evaluating all models on holdout test set..."):
        # Fit models on train
        sarimax = SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(train_df['sales'], exog=train_df[exog_cols])
        prophet = ProphetForecaster().fit(train_df['sales'], exog=train_df[exog_cols])
        lstm = LSTMForecaster(lookback=14, epochs=50, batch_size=16).fit(train_df['sales'], exog=train_df[exog_cols], verbose=0)
        xgb_m = XGBoostForecaster(n_estimators=120, max_depth=3).fit(train_df, target_col='sales')
        
        # Predict on test
        s_pred = sarimax.predict(steps=test_split_days, exog_future=test_df[exog_cols], start_date=test_df.index[0])
        p_pred = prophet.predict(steps=test_split_days, exog_future=test_df[exog_cols])
        l_pred = lstm.predict(steps=test_split_days, exog_future=test_df[exog_cols])
        x_pred = xgb_m.predict(steps=test_split_days)
        
        ens = EnsembleForecaster()
        ens_pred = ens.blend_forecasts({'SARIMAX': s_pred, 'Prophet': p_pred, 'LSTM': l_pred})
        
        test_preds = {
            'SARIMAX': s_pred['yhat'].values,
            'Prophet': p_pred['yhat'].values,
            'LSTM': l_pred['yhat'].values,
            'XGBoost': x_pred['yhat'].values,
            'Ensemble': ens_pred['yhat'].values
        }
        
        metrics = {k: calculate_metrics(test_df['sales'].values, v) for k, v in test_preds.items()}
        leaderboard = compare_models(metrics)

    # Multi-model Comparison Plot
    fig_comp = plot_multi_model_comparison(test_df, test_preds)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Leaderboard Table
    st.markdown("#### 🥇 Model Performance Leaderboard")
    st.dataframe(leaderboard.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='#065F46').highlight_max(subset=['R2', 'MDA (%)'], color='#065F46'), use_container_width=True)
    
    # Residual Analysis
    st.markdown("#### 🔍 Residual Error Diagnostics")
    res_col1, res_col2 = st.columns(2)
    selected_eval_model = res_col1.selectbox("Select Model for Residual Diagnostics", list(test_preds.keys()), index=0)
    residuals = test_df['sales'].values - test_preds[selected_eval_model]
    
    with res_col1:
        fig_res_time = go.Figure()
        fig_res_time.add_trace(go.Scatter(x=test_df.index, y=residuals, mode='lines+markers', line=dict(color='#F43F5E'), name='Residuals'))
        fig_res_time.add_hline(y=0, line_dash='dash', line_color='#94A3B8')
        fig_res_time.update_layout(title=f"<b>{selected_eval_model} Residuals Over Time</b>", template="plotly_dark", plot_bgcolor=PALETTE['card_bg'], paper_bgcolor=PALETTE['background'])
        st.plotly_chart(fig_res_time, use_container_width=True)
        
    with res_col2:
        fig_res_hist = px.histogram(residuals, nbins=15, title=f"<b>{selected_eval_model} Residual Error Distribution</b>", template="plotly_dark", color_discrete_sequence=['#818CF8'])
        fig_res_hist.update_layout(plot_bgcolor=PALETTE['card_bg'], paper_bgcolor=PALETTE['background'], xaxis_title="Prediction Error (₹)")
        st.plotly_chart(fig_res_hist, use_container_width=True)

# ================= TAB 5: EXPORT FORECAST DATA =================
with tab_export:
    st.subheader("💾 Export & Download Forecast Datasets")
    st.markdown("Download projected forecasts, confidence bounds, and model evaluation metrics for downstream ERP/CRM integration.")
    
    export_df = forecast_df.reset_index()
    export_df.columns = [c.replace('_', ' ').title() for c in export_df.columns]
    
    st.dataframe(export_df, use_container_width=True)
    
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_bytes,
        file_name=f"sales_forecast_{selected_category}_{forecast_horizon}d.csv",
        mime="text/csv"
    )
