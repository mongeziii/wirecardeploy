"""Financial forecasting module."""
import pandas as pd
import numpy as np

def build_forecast_model(historical_df, assumptions):
    """Build a 5-year three-statement financial forecast."""
    years = list(range(2021, 2026))
    last_year = historical_df.iloc[-1]

    # Base revenue from last actual year (2020 was distorted, use 2019 as base and apply decline)
    base_revenue = historical_df[historical_df['Year'] == 2019]['Revenue'].values[0]

    forecast = pd.DataFrame()
    forecast['Year'] = years

    # Income Statement
    revenue = []
    for i, year in enumerate(years):
        rev = base_revenue * ((1 + assumptions['revenue_growth']/100) ** (i+1))
        # Apply scandal impact decay
        if i == 0:
            rev *= 0.4  # Immediate scandal impact
        elif i == 1:
            rev *= 0.55
        elif i == 2:
            rev *= 0.70
        revenue.append(rev)

    forecast['Revenue'] = revenue
    forecast['EBITDA'] = forecast['Revenue'] * (assumptions['ebitda_margin'] / 100)
    forecast['D_A'] = forecast['Revenue'] * 0.04  # Approximate D&A as % of revenue
    forecast['EBIT'] = forecast['EBITDA'] - forecast['D_A']
    forecast['Interest_Expense'] = forecast['Revenue'] * 0.015
    forecast['EBT'] = forecast['EBIT'] - forecast['Interest_Expense']
    forecast['Tax'] = forecast['EBT'] * (assumptions['tax_rate'] / 100)
    forecast['Net_Income'] = forecast['EBT'] - forecast['Tax']

    # Balance Sheet
    forecast['Total_Assets'] = forecast['Revenue'] * 1.5  # Asset turnover assumption
    forecast['Current_Assets'] = forecast['Revenue'] * 0.6
    forecast['Cash'] = forecast['Revenue'] * 0.15
    forecast['Receivables'] = forecast['Revenue'] * 0.25
    forecast['Total_Liabilities'] = forecast['Total_Assets'] * 0.65
    forecast['Current_Liabilities'] = forecast['Revenue'] * 0.35
    forecast['Long_Term_Debt'] = forecast['Total_Liabilities'] * 0.45
    forecast['Total_Equity'] = forecast['Total_Assets'] - forecast['Total_Liabilities']
    forecast['Shares_Outstanding'] = last_year['Shares_Outstanding']

    # Cash Flow
    forecast['Operating_Cash_Flow'] = forecast['Net_Income'] + forecast['D_A'] - (forecast['Receivables'] * 0.05)
    forecast['Capex'] = forecast['Revenue'] * (assumptions['capex_pct'] / 100)
    forecast['Free_Cash_Flow'] = forecast['Operating_Cash_Flow'] - forecast['Capex']
    forecast['Investing_Cash_Flow'] = -forecast['Capex']
    forecast['Financing_Cash_Flow'] = -forecast['Net_Income'] * (assumptions['dividend_payout'] / 100)

    # Margins
    forecast['EBITDA_Margin'] = forecast['EBITDA'] / forecast['Revenue'] * 100
    forecast['EBIT_Margin'] = forecast['EBIT'] / forecast['Revenue'] * 100
    forecast['Profit_Margin'] = forecast['Net_Income'] / forecast['Revenue'] * 100

    return forecast.round(2)

def build_no_scenario_forecast(historical_df, assumptions):
    """Build hypothetical no-scandal forecast."""
    years = list(range(2021, 2026))
    base_revenue = historical_df[historical_df['Year'] == 2019]['Revenue'].values[0]

    forecast = pd.DataFrame()
    forecast['Year'] = years

    revenue = []
    for i, year in enumerate(years):
        rev = base_revenue * ((1 + assumptions['revenue_growth']/100) ** (i+1))
        revenue.append(rev)

    forecast['Revenue'] = revenue
    forecast['EBITDA'] = forecast['Revenue'] * (assumptions['ebitda_margin'] / 100)
    forecast['D_A'] = forecast['Revenue'] * 0.04
    forecast['EBIT'] = forecast['EBITDA'] - forecast['D_A']
    forecast['Interest_Expense'] = forecast['Revenue'] * 0.012
    forecast['EBT'] = forecast['EBIT'] - forecast['Interest_Expense']
    forecast['Tax'] = forecast['EBT'] * (assumptions['tax_rate'] / 100)
    forecast['Net_Income'] = forecast['EBT'] - forecast['Tax']
    forecast['Operating_Cash_Flow'] = forecast['Net_Income'] * 1.1 + forecast['D_A']
    forecast['Capex'] = forecast['Revenue'] * (assumptions['capex_pct'] / 100)
    forecast['Free_Cash_Flow'] = forecast['Operating_Cash_Flow'] - forecast['Capex']
    forecast['Total_Assets'] = forecast['Revenue'] * 1.3
    forecast['Total_Equity'] = forecast['Total_Assets'] * 0.45
    forecast['Total_Liabilities'] = forecast['Total_Assets'] - forecast['Total_Equity']
    forecast['Cash'] = forecast['Revenue'] * 0.18
    forecast['EBITDA_Margin'] = forecast['EBITDA'] / forecast['Revenue'] * 100
    forecast['EBIT_Margin'] = forecast['EBIT'] / forecast['Revenue'] * 100
    forecast['Profit_Margin'] = forecast['Net_Income'] / forecast['Revenue'] * 100

    return forecast.round(2)

def get_default_assumptions():
    """Return default forecasting assumptions."""
    return {
        'revenue_growth': 12.0,
        'ebitda_margin': 28.0,
        'ebit_margin': 24.0,
        'tax_rate': 25.0,
        'capex_pct': 6.0,
        'working_capital_pct': 8.0,
        'interest_rate': 3.5,
        'debt_growth': 5.0,
        'dividend_payout': 20.0,
        'terminal_growth': 2.0,
        'wacc': 9.0,
        'risk_free_rate': 1.5,
        'beta': 1.2,
        'market_premium': 6.0,
    }
