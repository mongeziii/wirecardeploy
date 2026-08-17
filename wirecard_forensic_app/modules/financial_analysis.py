"""Financial analysis calculations and ratios."""
import pandas as pd
import numpy as np

def calculate_all_ratios(df):
    """Calculate comprehensive financial ratios."""
    ratios = pd.DataFrame()
    ratios['Year'] = df['Year']

    # Profitability
    ratios['EBITDA_Margin'] = df['EBITDA'] / df['Revenue'] * 100
    ratios['EBIT_Margin'] = df['EBIT'] / df['Revenue'] * 100
    ratios['Profit_Margin'] = df['Net_Income'] / df['Revenue'] * 100
    ratios['Cash_Flow_Margin'] = df['Operating_Cash_Flow'] / df['Revenue'] * 100
    ratios['ROA'] = df['Net_Income'] / df['Total_Assets'] * 100
    ratios['ROE'] = df['Net_Income'] / df['Total_Equity'] * 100
    ratios['ROIC'] = df['EBIT'] * (1 - 0.25) / (df['Total_Equity'] + df['Total_Liabilities'] - df['Cash']) * 100

    # Liquidity
    ratios['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities']
    ratios['Quick_Ratio'] = (df['Current_Assets'] - df['Inventory']) / df['Current_Liabilities']
    ratios['Cash_Ratio'] = df['Cash'] / df['Current_Liabilities']

    # Leverage
    ratios['Debt_to_Equity'] = df['Total_Liabilities'] / df['Total_Equity']
    ratios['Debt_to_Assets'] = df['Total_Liabilities'] / df['Total_Assets']
    ratios['Interest_Coverage'] = df['EBIT'] / df['Interest_Expense']
    ratios['Net_Debt'] = (df['Short_Term_Debt'] + df['Long_Term_Debt']) - df['Cash']
    ratios['Net_Debt_to_EBITDA'] = ratios['Net_Debt'] / df['EBITDA']

    # Efficiency
    ratios['Asset_Turnover'] = df['Revenue'] / df['Total_Assets']
    ratios['Receivables_Turnover'] = df['Revenue'] / df['Receivables']
    ratios['Days_Sales_Outstanding'] = 365 / ratios['Receivables_Turnover']
    ratios['Inventory_Turnover'] = df['COGS'] / df['Inventory']

    # Growth
    ratios['Revenue_Growth'] = df['Revenue'].pct_change() * 100
    ratios['EBITDA_Growth'] = df['EBITDA'].pct_change() * 100
    ratios['Net_Income_Growth'] = df['Net_Income'].pct_change() * 100
    ratios['Receivables_Growth'] = df['Receivables'].pct_change() * 100

    # Earnings Quality
    ratios['Earnings_Quality'] = df['Operating_Cash_Flow'] / df['Net_Income']
    ratios['Accruals_Ratio'] = (df['Net_Income'] - df['Operating_Cash_Flow']) / df['Total_Assets']
    ratios['OCF_to_EBITDA'] = df['Operating_Cash_Flow'] / df['EBITDA']

    # Per Share
    ratios['EPS'] = df['Net_Income'] / df['Shares_Outstanding']
    ratios['Book_Value_Per_Share'] = df['Total_Equity'] / df['Shares_Outstanding']
    ratios['FCF_Per_Share'] = df['Free_Cash_Flow'] / df['Shares_Outstanding']

    return ratios.round(2)

def get_ratio_status(value, ratio_name):
    """Return status color based on ratio thresholds."""
    thresholds = {
        'Current_Ratio': (1.5, 1.0),
        'Quick_Ratio': (1.0, 0.8),
        'Debt_to_Equity': (1.0, 2.0),
        'ROA': (5, 2),
        'ROE': (10, 5),
        'EBITDA_Margin': (15, 10),
        'Profit_Margin': (10, 5),
        'Cash_Flow_Margin': (10, 5),
        'Interest_Coverage': (3, 1.5),
        'Earnings_Quality': (0.8, 0.5),
    }

    if ratio_name not in thresholds:
        return 'neutral'

    good, warning = thresholds[ratio_name]

    # For debt ratios, lower is better
    if ratio_name in ['Debt_to_Equity', 'Net_Debt_to_EBITDA', 'Days_Sales_Outstanding', 'Accruals_Ratio']:
        if value <= good:
            return 'good'
        elif value <= warning:
            return 'warning'
        else:
            return 'danger'
    else:
        if value >= good:
            return 'good'
        elif value >= warning:
            return 'warning'
        else:
            return 'danger'

def format_millions(value):
    """Format value in millions with € symbol."""
    return f"€{value:,.0f}M"

def format_billions(value):
    """Format value in billions with € symbol."""
    return f"€{value/1000:,.2f}B"
