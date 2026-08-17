"""Valuation module: DCF, DDM, and Relative Valuation."""
import pandas as pd
import numpy as np

def calculate_dcf(forecast_df, assumptions, historical_df):
    """Calculate DCF valuation."""
    fcf = forecast_df['Free_Cash_Flow'].values
    wacc = assumptions['wacc'] / 100
    terminal_growth = assumptions['terminal_growth'] / 100

    # Present value of forecast period FCF
    pv_fcf = 0
    for i, f in enumerate(fcf):
        pv_fcf += f / ((1 + wacc) ** (i + 1))

    # Terminal value
    terminal_value = fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** len(fcf))

    enterprise_value = pv_fcf + pv_terminal

    # Net debt from last historical year
    last_hist = historical_df.iloc[-1]
    net_debt = (last_hist['Short_Term_Debt'] + last_hist['Long_Term_Debt']) - last_hist['Cash']
    equity_value = enterprise_value - net_debt
    shares = last_hist['Shares_Outstanding']
    value_per_share = equity_value / shares

    return {
        'PV_FCF': round(pv_fcf, 2),
        'Terminal_Value': round(terminal_value, 2),
        'PV_Terminal': round(pv_terminal, 2),
        'Enterprise_Value': round(enterprise_value, 2),
        'Net_Debt': round(net_debt, 2),
        'Equity_Value': round(equity_value, 2),
        'Value_Per_Share': round(value_per_share, 2),
        'Shares_Outstanding': shares,
        'WACC': assumptions['wacc'],
        'Terminal_Growth': assumptions['terminal_growth']
    }

def calculate_ddm(forecast_df, assumptions, historical_df):
    """Calculate Dividend Discount Model valuation."""
    # Wirecard paid no dividends, so DDM is not applicable
    # We'll use a residual income approach instead

    last_hist = historical_df.iloc[-1]
    book_value = last_hist['Total_Equity']
    shares = last_hist['Shares_Outstanding']
    bv_per_share = book_value / shares

    cost_of_equity = assumptions['risk_free_rate']/100 + assumptions['beta'] * (assumptions['market_premium']/100)

    # Residual income = Net Income - (Equity * Cost of Equity)
    ri = []
    for _, row in forecast_df.iterrows():
        ri.append(row['Net_Income'] - (row['Total_Equity'] * cost_of_equity))

    pv_ri = sum(ri[i] / ((1 + cost_of_equity) ** (i+1)) for i in range(len(ri)))
    terminal_ri = ri[-1] * (1 + assumptions['terminal_growth']/100) / (cost_of_equity - assumptions['terminal_growth']/100)
    pv_terminal_ri = terminal_ri / ((1 + cost_of_equity) ** len(ri))

    value_per_share = bv_per_share + (pv_ri + pv_terminal_ri) / shares

    return {
        'Book_Value_Per_Share': round(bv_per_share, 2),
        'Cost_of_Equity': round(cost_of_equity * 100, 2),
        'PV_Residual_Income': round(pv_ri, 2),
        'PV_Terminal_RI': round(pv_terminal_ri, 2),
        'Value_Per_Share': round(value_per_share, 2)
    }

def calculate_relative_valuation(historical_df):
    """Calculate relative valuation using comparable multiples."""
    last_hist = historical_df.iloc[-1]

    # Use 2019 data (pre-scandal) for more realistic multiples
    ref = historical_df[historical_df['Year'] == 2019].iloc[0]

    # Industry comparable multiples (fintech/payment processors)
    ev_ebitda_multiple = 15.0  # Industry average
    pe_multiple = 25.0
    ps_multiple = 4.0

    ev = ref['EBITDA'] * ev_ebitda_multiple
    net_debt = (ref['Short_Term_Debt'] + ref['Long_Term_Debt']) - ref['Cash']
    equity_value = ev - net_debt

    pe_value = ref['Net_Income'] * pe_multiple
    ps_value = ref['Revenue'] * ps_multiple

    shares = ref['Shares_Outstanding']

    return {
        'EV_EBITDA_Multiple': ev_ebitda_multiple,
        'PE_Multiple': pe_multiple,
        'PS_Multiple': ps_multiple,
        'EV_EBITDA_Value': round(ev / shares, 2),
        'PE_Value': round(pe_value / shares, 2),
        'PS_Value': round(ps_value / shares, 2),
        'Average_Value': round(((ev / shares) + (pe_value / shares) + (ps_value / shares)) / 3, 2),
        'Enterprise_Value': round(ev, 2),
        'Equity_Value': round(equity_value, 2)
    }

def calculate_wacc(assumptions, historical_df):
    """Calculate WACC components."""
    last_hist = historical_df.iloc[-1]

    # Cost of Equity (CAPM)
    rf = assumptions['risk_free_rate']
    beta = assumptions['beta']
    mrp = assumptions['market_premium']
    cost_of_equity = rf + beta * mrp

    # Cost of Debt
    total_debt = last_hist['Short_Term_Debt'] + last_hist['Long_Term_Debt']
    interest = last_hist['Interest_Expense']
    cost_of_debt = (interest / total_debt * 100) if total_debt > 0 else 0
    tax_rate = assumptions['tax_rate']
    after_tax_cod = cost_of_debt * (1 - tax_rate/100)

    # Weights
    equity = last_hist['Total_Equity']
    debt = total_debt
    total_capital = equity + debt

    w_e = equity / total_capital if total_capital > 0 else 0
    w_d = debt / total_capital if total_capital > 0 else 0

    wacc = w_e * cost_of_equity + w_d * after_tax_cod

    return {
        'Risk_Free_Rate': rf,
        'Beta': beta,
        'Market_Risk_Premium': mrp,
        'Cost_of_Equity': round(cost_of_equity, 2),
        'Cost_of_Debt': round(cost_of_debt, 2),
        'After_Tax_Cost_of_Debt': round(after_tax_cod, 2),
        'Equity_Weight': round(w_e * 100, 1),
        'Debt_Weight': round(w_d * 100, 1),
        'WACC': round(wacc, 2)
    }

def sensitivity_analysis(forecast_df, assumptions, historical_df):
    """Generate DCF sensitivity heatmap data."""
    wacc_range = np.arange(assumptions['wacc'] - 3, assumptions['wacc'] + 3.5, 0.5)
    tg_range = np.arange(assumptions['terminal_growth'] - 1.5, assumptions['terminal_growth'] + 2, 0.5)

    sensitivity = pd.DataFrame(index=[f"{w:.1f}%" for w in wacc_range], 
                                columns=[f"{tg:.1f}%" for tg in tg_range])

    for w in wacc_range:
        for tg in tg_range:
            temp_assumptions = assumptions.copy()
            temp_assumptions['wacc'] = w
            temp_assumptions['terminal_growth'] = tg
            result = calculate_dcf(forecast_df, temp_assumptions, historical_df)
            sensitivity.loc[f"{w:.1f}%", f"{tg:.1f}%"] = result['Value_Per_Share']

    return sensitivity.astype(float).round(2)
