"""
Visualization Module for Sales Forecasting
Generates publication-quality Plotly interactive charts and Matplotlib diagnostic plots.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List


# Color palette for consistent dark/light themes
PALETTE = {
    'primary': '#6366F1',     # Indigo
    'actual': '#3B82F6',      # Vibrant Blue
    'forecast': '#F59E0B',    # Amber / Gold
    'ci_fill': 'rgba(245, 158, 11, 0.15)',
    'sarimax': '#10B981',     # Emerald Green
    'prophet': '#8B5CF6',     # Purple
    'lstm': '#EC4899',        # Pink
    'xgboost': '#F97316',     # Orange
    'ensemble': '#06B6D4',    # Cyan
    'background': '#0F172A',
    'card_bg': '#1E293B',
    'text': '#F8FAFC',
    'grid': '#334155'
}


def plot_forecast_vs_actual(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_name: str = 'Forecasting Model',
    title: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive Plotly figure displaying historical sales, predicted forecast, and confidence bounds.
    """
    fig = go.Figure()

    # 1. Historical Actual Sales
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['sales'],
        mode='lines',
        name='Historical Actual Sales',
        line=dict(color=PALETTE['actual'], width=2.2),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Sales</b>: ₹%{y:,.2f}<extra></extra>'
    ))

    # 2. Forecast Confidence Interval (Upper & Lower bounds)
    if 'yhat_upper' in forecast_df.columns and 'yhat_lower' in forecast_df.columns:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=PALETTE['ci_fill'],
            name='95% Confidence Band',
            hoverinfo='skip'
        ))

    # 3. Forecasted Trajectory
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['yhat'],
        mode='lines+markers',
        name=f'{model_name} Forecast',
        line=dict(color=PALETTE['forecast'], width=2.8, dash='dash'),
        marker=dict(size=4, color=PALETTE['forecast']),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Predicted</b>: ₹%{y:,.2f}<extra></extra>'
    ))

    # Vertical split line demarcating forecast horizon
    split_date_str = pd.to_datetime(forecast_df.index[0]).strftime('%Y-%m-%d')
    try:
        fig.add_vline(
            x=split_date_str,
            line_width=1.5,
            line_dash="dot",
            line_color="#94A3B8"
        )
    except Exception:
        pass

    plot_title = title or f"Sales Forecast Trajectory - {model_name}"
    fig.update_layout(
        title=dict(text=f"<b>{plot_title}</b>", font=dict(size=18, color=PALETTE['text'])),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor=PALETTE['grid'],
            tickformat="%b %d, %Y",
            color=PALETTE['text']
        ),
        yaxis=dict(
            title="Sales Revenue (₹)",
            showgrid=True,
            gridcolor=PALETTE['grid'],
            tickprefix="₹",
            color=PALETTE['text']
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=PALETTE['text'])
        ),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor=PALETTE['card_bg'],
        paper_bgcolor=PALETTE['background']
    )
    return fig


def plot_multi_model_comparison(
    test_df: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    title: str = "Model Forecast Comparison vs Ground Truth (Test Set)"
) -> go.Figure:
    """
    Overlay forecasts from multiple models against actual test data.
    """
    fig = go.Figure()

    # Actual Test Series
    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df['sales'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='#FFFFFF', width=3),
        marker=dict(size=5, color='#FFFFFF'),
        hovertemplate='<b>Actual</b>: ₹%{y:,.2f}<extra></extra>'
    ))

    color_map = {
        'SARIMAX': PALETTE['sarimax'],
        'Prophet': PALETTE['prophet'],
        'LSTM': PALETTE['lstm'],
        'XGBoost': PALETTE['xgboost'],
        'Ensemble': PALETTE['ensemble']
    }

    for name, preds in predictions_dict.items():
        c = color_map.get(name, '#FBBF24')
        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=preds,
            mode='lines',
            name=f'{name}',
            line=dict(color=c, width=2.2, dash='solid' if name == 'Ensemble' else 'dash'),
            hovertemplate=f'<b>{name}</b>: ₹%{{y:,.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, color=PALETTE['text'])),
        xaxis=dict(title="Date", showgrid=True, gridcolor=PALETTE['grid'], color=PALETTE['text']),
        yaxis=dict(title="Sales Revenue (₹)", showgrid=True, gridcolor=PALETTE['grid'], tickprefix="₹", color=PALETTE['text']),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=PALETTE['text'])),
        template="plotly_dark",
        plot_bgcolor=PALETTE['card_bg'],
        paper_bgcolor=PALETTE['background']
    )
    return fig


def plot_seasonal_decomposition_plotly(decomp_dict: Dict[str, pd.Series]) -> go.Figure:
    """
    Interactive 4-panel seasonal decomposition plot (Observed, Trend, Seasonal, Residual).
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Observed Daily Sales", "Underlying Trend Component", "Weekly Seasonal Component", "Residual Noise")
    )

    idx = decomp_dict['observed'].index

    # 1. Observed
    fig.add_trace(
        go.Scatter(x=idx, y=decomp_dict['observed'], mode='lines', line=dict(color='#38BDF8', width=1.8), name='Observed'),
        row=1, col=1
    )
    # 2. Trend
    fig.add_trace(
        go.Scatter(x=idx, y=decomp_dict['trend'], mode='lines', line=dict(color='#F43F5E', width=2.5), name='Trend'),
        row=2, col=1
    )
    # 3. Seasonal
    fig.add_trace(
        go.Scatter(x=idx, y=decomp_dict['seasonal'], mode='lines', line=dict(color='#A855F7', width=1.8), name='Seasonal'),
        row=3, col=1
    )
    # 4. Residual
    fig.add_trace(
        go.Scatter(x=idx, y=decomp_dict['residual'], mode='markers', marker=dict(color='#E2E8F0', size=3, opacity=0.7), name='Residual'),
        row=4, col=1
    )

    fig.update_layout(
        height=750,
        title_text="<b>Classical Time-Series Additive Decomposition</b>",
        title_font=dict(size=18, color=PALETTE['text']),
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor=PALETTE['card_bg'],
        paper_bgcolor=PALETTE['background']
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE['grid'], color=PALETTE['text'])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE['grid'], color=PALETTE['text'])
    return fig


def plot_promotional_elasticity(df_raw: pd.DataFrame) -> go.Figure:
    """
    Interactive plot showing the relationship between Discount % and Final Revenue.
    """
    # Group by discount tier
    disc_summary = df_raw.groupby('Discount (%)').agg(
        total_revenue=('Final_Price(Rs.)', 'sum'),
        order_count=('User_ID', 'count'),
        avg_revenue_per_order=('Final_Price(Rs.)', 'mean')
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar for total volume
    fig.add_trace(
        go.Bar(
            x=disc_summary['Discount (%)'],
            y=disc_summary['order_count'],
            name="Order Count (Volume)",
            marker_color='rgba(99, 102, 241, 0.7)',
            hovertemplate='<b>Discount</b>: %{x}%<br><b>Orders</b>: %{y}<extra></extra>'
        ),
        secondary_y=False
    )

    # Line for total revenue
    fig.add_trace(
        go.Scatter(
            x=disc_summary['Discount (%)'],
            y=disc_summary['total_revenue'],
            name="Total Revenue (₹)",
            mode='lines+markers',
            line=dict(color='#F59E0B', width=3),
            marker=dict(size=8, color='#F59E0B'),
            hovertemplate='<b>Discount</b>: %{x}%<br><b>Revenue</b>: ₹%{y:,.2f}<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=dict(text="<b>Promotional Impact: Discount Rate vs Order Volume & Revenue</b>", font=dict(size=18, color=PALETTE['text'])),
        xaxis=dict(title="Discount Percentage (%)", showgrid=True, gridcolor=PALETTE['grid'], color=PALETTE['text']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=PALETTE['text'])),
        template="plotly_dark",
        plot_bgcolor=PALETTE['card_bg'],
        paper_bgcolor=PALETTE['background']
    )
    fig.update_yaxes(title_text="Order Count", secondary_y=False, showgrid=True, gridcolor=PALETTE['grid'], color=PALETTE['text'])
    fig.update_yaxes(title_text="Total Revenue (₹)", secondary_y=True, showgrid=False, tickprefix="₹", color=PALETTE['text'])
    return fig


def plot_category_breakdown_sunburst(df_raw: pd.DataFrame) -> go.Figure:
    """
    Sunburst chart showing sales distribution across Categories and Payment Methods.
    """
    fig = px.sunburst(
        df_raw,
        path=['Category', 'Payment_Method'],
        values='Final_Price(Rs.)',
        color='Category',
        color_discrete_sequence=px.colors.qualitative.Prism,
        title="<b>Sales Distribution by Category & Payment Method</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE['background'],
        margin=dict(t=50, l=10, r=10, b=10)
    )
    return fig
