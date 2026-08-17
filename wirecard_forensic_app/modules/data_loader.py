"""Data loading and validation module for Wirecard Forensic Analysis.

Data source: Wirecard AG Annual Reports 2015–2018, Q3 2019 Statement.
2019 full-year and 2020 insolvency data are estimates based on scandal timeline.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

def load_financial_data():
    """Load the master financial dataset."""
    try:
        df = pd.read_csv("data/wirecard_data.csv")
        df['Year'] = df['Year'].astype(int)
        return df
    except Exception as e:
        st.error(f"Error loading financial data: {e}")
        return pd.DataFrame()

def load_share_prices():
    """Load monthly share price data."""
    try:
        df = pd.read_csv("data/share_prices.csv", parse_dates=['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading share price data: {e}")
        return pd.DataFrame()

def validate_data(df):
    """Validate dataset quality and return report."""
    report = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_records': int(df.duplicated().sum()),
        'missing_pct': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        'years_available': sorted(df['Year'].dropna().unique().tolist()) if 'Year' in df.columns else [],
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'outliers': {}
    }

    # Detect outliers using IQR method for key metrics
    for col in ['Revenue', 'EBITDA', 'Net_Income', 'Operating_Cash_Flow']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            report['outliers'][col] = len(outliers)

    return report

def get_data_summary(df):
    """Return high-level data summary."""
    summary = {
        'period': f"{df['Year'].min()} - {df['Year'].max()}" if 'Year' in df.columns else "N/A",
        'total_revenue': df['Revenue'].sum() if 'Revenue' in df.columns else 0,
        'peak_revenue': df['Revenue'].max() if 'Revenue' in df.columns else 0,
        'peak_revenue_year': df.loc[df['Revenue'].idxmax(), 'Year'] if 'Revenue' in df.columns else None,
        'latest_net_income': df['Net_Income'].iloc[-1] if 'Net_Income' in df.columns else 0,
        'latest_year': df['Year'].iloc[-1] if 'Year' in df.columns else None,
    }
    return summary
