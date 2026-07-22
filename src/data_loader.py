"""
Data Loader and Preprocessing Module for Sales Forecasting
Handles CSV loading, datetime normalization, missing dates reindexing, and multi-frequency aggregations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any


def load_raw_data(filepath: str = 'ecommerce_dataset_updated.csv') -> pd.DataFrame:
    """
    Load raw e-commerce transaction data, parse dates, and sort chronologically.
    
    Parameters:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned transaction DataFrame.
    """
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Parse Purchase_Date
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], format='%d-%m-%Y', errors='coerce')
    
    # Check for invalid dates
    null_dates = df['Purchase_Date'].isna().sum()
    if null_dates > 0:
        df = df.dropna(subset=['Purchase_Date'])
    
    # Sort chronologically
    df = df.sort_values('Purchase_Date').reset_index(drop=True)
    
    # Ensure numeric columns are strictly float/int
    df['Price (Rs.)'] = pd.to_numeric(df['Price (Rs.)'], errors='coerce')
    df['Discount (%)'] = pd.to_numeric(df['Discount (%)'], errors='coerce')
    df['Final_Price(Rs.)'] = pd.to_numeric(df['Final_Price(Rs.)'], errors='coerce')
    
    # Calculate discount amount
    df['Discount_Amount'] = df['Price (Rs.)'] - df['Final_Price(Rs.)']
    
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary KPIs and distribution metrics from raw transaction data.
    """
    min_date = df['Purchase_Date'].min()
    max_date = df['Purchase_Date'].max()
    total_days = (max_date - min_date).days + 1
    
    summary = {
        'total_transactions': len(df),
        'total_revenue': float(df['Final_Price(Rs.)'].sum()),
        'total_gross_value': float(df['Price (Rs.)'].sum()),
        'total_discount_given': float(df['Discount_Amount'].sum()),
        'avg_discount_rate': float(df['Discount (%)'].mean()),
        'avg_order_value': float(df['Final_Price(Rs.)'].mean()),
        'unique_users': int(df['User_ID'].nunique()),
        'unique_products': int(df['Product_ID'].nunique()),
        'categories': sorted(df['Category'].dropna().unique().tolist()),
        'payment_methods': df['Payment_Method'].value_counts().to_dict(),
        'start_date': min_date.strftime('%Y-%m-%d'),
        'end_date': max_date.strftime('%Y-%m-%d'),
        'total_days_span': int(total_days)
    }
    return summary


def aggregate_time_series(
    df: pd.DataFrame, 
    freq: str = 'D', 
    category: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate individual transactions into regular time-series intervals.
    
    Parameters:
        df (pd.DataFrame): Raw transaction DataFrame.
        freq (str): Resampling frequency: 'D' (Daily), 'W' (Weekly), 'M' (Monthly).
        category (str, optional): Filter by a specific category if provided.
        
    Returns:
        pd.DataFrame: Resampled time-series indexed by datetime.
    """
    data = df.copy()
    if category and category != 'All':
        data = data[data['Category'] == category]
    
    min_date = df['Purchase_Date'].min()
    max_date = df['Purchase_Date'].max()
    
    # Set index for resampling
    data = data.set_index('Purchase_Date')
    
    # Define aggregation mapping
    agg_dict = {
        'Final_Price(Rs.)': ['sum', 'mean'],
        'Price (Rs.)': 'sum',
        'Discount (%)': ['mean', 'max'],
        'Discount_Amount': 'sum',
        'User_ID': 'count'
    }
    
    resampled = data.resample(freq).agg(agg_dict)
    
    # Flatten MultiIndex columns
    resampled.columns = [
        'sales',                # Final_Price sum
        'avg_order_value',      # Final_Price mean
        'gross_sales',          # Price sum
        'discount_rate',        # Discount (%) mean
        'max_discount',         # Discount (%) max
        'discount_amount',      # Discount_Amount sum
        'transactions'          # User_ID count
    ]
    
    # Ensure full date continuity without gaps
    if freq == 'D':
        full_idx = pd.date_range(min_date, max_date, freq='D')
        resampled = resampled.reindex(full_idx)
    elif freq == 'W':
        full_idx = pd.date_range(min_date, max_date, freq='W-MON')
        resampled = resampled.reindex(full_idx)
    elif freq == 'M' or freq == 'ME':
        full_idx = pd.date_range(min_date, max_date, freq='ME')
        resampled = resampled.reindex(full_idx)
    
    # Fill zeros for periods with 0 transactions
    resampled['sales'] = resampled['sales'].fillna(0.0)
    resampled['gross_sales'] = resampled['gross_sales'].fillna(0.0)
    resampled['discount_rate'] = resampled['discount_rate'].fillna(0.0)
    resampled['max_discount'] = resampled['max_discount'].fillna(0.0)
    resampled['discount_amount'] = resampled['discount_amount'].fillna(0.0)
    resampled['transactions'] = resampled['transactions'].fillna(0).astype(int)
    resampled['avg_order_value'] = resampled['avg_order_value'].fillna(0.0)
    
    resampled.index.name = 'Date'
    return resampled


def get_category_pivot(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """
    Create a pivot table of sales by category across time.
    """
    pivot = df.pivot_table(
        index='Purchase_Date',
        columns='Category',
        values='Final_Price(Rs.)',
        aggfunc='sum',
        fill_value=0.0
    )
    
    min_date = df['Purchase_Date'].min()
    max_date = df['Purchase_Date'].max()
    full_idx = pd.date_range(min_date, max_date, freq=freq)
    pivot = pivot.reindex(full_idx, fill_value=0.0)
    pivot.index.name = 'Date'
    return pivot
