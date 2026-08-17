"""Forensic analysis and fraud detection indicators."""
import pandas as pd
import numpy as np

def calculate_forensic_risk_score(df):
    """Calculate overall forensic risk score (0-100, higher = more risk)."""
    scores = {}

    # Earnings Quality (weight: 25%)
    eq = df['Operating_Cash_Flow'].iloc[-1] / df['Net_Income'].iloc[-1]
    if eq < 0.5:
        scores['earnings_quality'] = 100
    elif eq < 0.8:
        scores['earnings_quality'] = 75
    elif eq < 1.0:
        scores['earnings_quality'] = 50
    else:
        scores['earnings_quality'] = 25

    # Cash Flow Divergence (weight: 20%)
    ni_growth = df['Net_Income'].pct_change().mean() * 100
    ocf_growth = df['Operating_Cash_Flow'].pct_change().mean() * 100
    divergence = abs(ni_growth - ocf_growth)
    scores['cash_flow_divergence'] = min(100, divergence * 5)

    # Receivables Growth vs Revenue Growth (weight: 20%)
    rec_growth = df['Receivables'].pct_change().mean() * 100
    rev_growth = df['Revenue'].pct_change().mean() * 100
    if rec_growth > rev_growth * 1.5:
        scores['receivables_growth'] = 100
    elif rec_growth > rev_growth:
        scores['receivables_growth'] = 75
    elif rec_growth > rev_growth * 0.8:
        scores['receivables_growth'] = 50
    else:
        scores['receivables_growth'] = 25

    # Profitability vs Cash Generation (weight: 20%)
    profit_margin = df['Net_Income'].iloc[-1] / df['Revenue'].iloc[-1]
    cf_margin = df['Operating_Cash_Flow'].iloc[-1] / df['Revenue'].iloc[-1]
    if profit_margin > cf_margin * 2:
        scores['profitability_vs_cash'] = 100
    elif profit_margin > cf_margin * 1.5:
        scores['profitability_vs_cash'] = 75
    elif profit_margin > cf_margin:
        scores['profitability_vs_cash'] = 50
    else:
        scores['profitability_vs_cash'] = 25

    # Governance / Leverage (weight: 15%)
    de = df['Total_Liabilities'].iloc[-1] / df['Total_Equity'].iloc[-1]
    if de > 10:
        scores['governance'] = 100
    elif de > 5:
        scores['governance'] = 75
    elif de > 2:
        scores['governance'] = 50
    else:
        scores['governance'] = 25

    # Weighted average
    weights = {
        'earnings_quality': 0.25,
        'cash_flow_divergence': 0.20,
        'receivables_growth': 0.20,
        'profitability_vs_cash': 0.20,
        'governance': 0.15
    }

    total_score = sum(scores[k] * weights[k] for k in scores)
    return round(total_score, 1), scores

def get_forensic_indicators(df):
    """Get detailed forensic warning indicators."""
    indicators = []

    # Indicator 1: Earnings Quality
    eq = df['Operating_Cash_Flow'].iloc[-1] / df['Net_Income'].iloc[-1]
    if eq < 0.5:
        status = "Significant Warning"
        level = "high"
    elif eq < 0.8:
        status = "Potential Warning"
        level = "medium"
    else:
        status = "Acceptable"
        level = "low"

    indicators.append({
        'name': 'Earnings Quality',
        'score': round(eq, 2),
        'status': status,
        'level': level,
        'explanation': f'Operating Cash Flow / Net Income = {eq:.2f}. Values below 0.8 indicate potential earnings manipulation.',
        'implication': 'Net income may not be supported by actual cash generation.'
    })

    # Indicator 2: Cash Flow Divergence
    ni_cagr = ((df['Net_Income'].iloc[-1] / df['Net_Income'].iloc[0]) ** (1/5) - 1) * 100
    ocf_cagr = ((df['Operating_Cash_Flow'].iloc[-1] / df['Operating_Cash_Flow'].iloc[0]) ** (1/5) - 1) * 100
    divergence = ni_cagr - ocf_cagr

    if divergence > 30:
        status = "Significant Warning"
        level = "high"
    elif divergence > 15:
        status = "Potential Warning"
        level = "medium"
    else:
        status = "Acceptable"
        level = "low"

    indicators.append({
        'name': 'Cash Flow Divergence',
        'score': f"{divergence:.1f}%",
        'status': status,
        'level': level,
        'explanation': f'Net Income CAGR ({ni_cagr:.1f}%) vs OCF CAGR ({ocf_cagr:.1f}%). Large divergence suggests accounting irregularities.',
        'implication': 'Reported profits growing faster than cash collections.'
    })

    # Indicator 3: Receivables Growth
    rec_growth = df['Receivables'].pct_change().mean() * 100
    rev_growth = df['Revenue'].pct_change().mean() * 100
    ratio = rec_growth / rev_growth if rev_growth > 0 else 999

    if ratio > 1.5:
        status = "Significant Warning"
        level = "high"
    elif ratio > 1.0:
        status = "Potential Warning"
        level = "medium"
    else:
        status = "Acceptable"
        level = "low"

    indicators.append({
        'name': 'Receivables Growth',
        'score': f"{rec_growth:.1f}%",
        'status': status,
        'level': level,
        'explanation': f'Receivables growing at {rec_growth:.1f}% vs Revenue at {rev_growth:.1f}%. Ratio: {ratio:.2f}x.',
        'implication': 'Potential revenue inflation through fictitious receivables.'
    })

    # Indicator 4: Profitability vs Cash
    pm = df['Net_Income'].iloc[-1] / df['Revenue'].iloc[-1] * 100
    cfm = df['Operating_Cash_Flow'].iloc[-1] / df['Revenue'].iloc[-1] * 100
    gap = pm - cfm

    if gap > 15:
        status = "Significant Warning"
        level = "high"
    elif gap > 8:
        status = "Potential Warning"
        level = "medium"
    else:
        status = "Acceptable"
        level = "low"

    indicators.append({
        'name': 'Profitability vs Cash Generation',
        'score': f"{gap:.1f}pp",
        'status': status,
        'level': level,
        'explanation': f'Profit margin ({pm:.1f}%) exceeds cash flow margin ({cfm:.1f}%) by {gap:.1f}pp.',
        'implication': 'Earnings may include non-cash or fictitious items.'
    })

    # Indicator 5: Governance
    de = df['Total_Liabilities'].iloc[-1] / df['Total_Equity'].iloc[-1]
    if de > 10:
        status = "High Risk"
        level = "high"
    elif de > 5:
        status = "Elevated Risk"
        level = "medium"
    else:
        status = "Acceptable"
        level = "low"

    indicators.append({
        'name': 'Governance & Leverage',
        'score': f"{de:.1f}x",
        'status': status,
        'level': level,
        'explanation': f'Debt-to-Equity ratio of {de:.1f}x indicates extreme leverage and governance risk.',
        'implication': 'High leverage amplifies risk and may indicate off-balance-sheet obligations.'
    })

    return indicators

def get_beneish_scores(df):
    """Calculate Beneish M-Score components (simplified)."""
    m_score_components = {}

    # Days Sales Outstanding Index (DSRI)
    dso_2018 = (df['Receivables'].iloc[-2] / df['Revenue'].iloc[-2]) * 365
    dso_2019 = (df['Receivables'].iloc[-1] / df['Revenue'].iloc[-1]) * 365
    m_score_components['DSRI'] = dso_2019 / dso_2018 if dso_2018 > 0 else 1

    # Gross Margin Index (GMI)
    gm_2018 = df['Gross_Profit'].iloc[-2] / df['Revenue'].iloc[-2]
    gm_2019 = df['Gross_Profit'].iloc[-1] / df['Revenue'].iloc[-1]
    m_score_components['GMI'] = gm_2018 / gm_2019 if gm_2019 > 0 else 1

    # Asset Quality Index (AQI)
    aqi_2018 = (df['Total_Assets'].iloc[-2] - df['Current_Assets'].iloc[-2]) / df['Total_Assets'].iloc[-2]
    aqi_2019 = (df['Total_Assets'].iloc[-1] - df['Current_Assets'].iloc[-1]) / df['Total_Assets'].iloc[-1]
    m_score_components['AQI'] = aqi_2019 / aqi_2018 if aqi_2018 > 0 else 1

    # Sales Growth Index (SGI)
    m_score_components['SGI'] = df['Revenue'].iloc[-1] / df['Revenue'].iloc[-2]

    # Depreciation Index (DEPI)
    dep_2018 = df['D_A'].iloc[-2] / (df['D_A'].iloc[-2] + df['PPE'].iloc[-2])
    dep_2019 = df['D_A'].iloc[-1] / (df['D_A'].iloc[-1] + df['PPE'].iloc[-1])
    m_score_components['DEPI'] = dep_2018 / dep_2019 if dep_2019 > 0 else 1

    # Simplified M-Score
    m_score = (-4.84 + 0.92*m_score_components['DSRI'] + 0.528*m_score_components['GMI'] + 
               0.404*m_score_components['AQI'] + 0.892*m_score_components['SGI'] + 
               0.115*m_score_components['DEPI'] - 0.172)

    m_score_components['M_Score'] = m_score
    return m_score_components


def calculate_altman_zscore(df):
    """Calculate Altman Z-Score for bankruptcy prediction.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value Equity / Total Liabilities
    X5 = Revenue / Total Assets

    Zones:
    Z > 2.99 = Safe (Green)
    1.81 < Z <= 2.99 = Grey Zone (Warning)
    Z <= 1.81 = Distress Zone (Red)
    """
    z_scores = pd.DataFrame()
    z_scores['Year'] = df['Year']

    # X1: Working Capital / Total Assets
    working_capital = df['Current_Assets'] - df['Current_Liabilities']
    X1 = working_capital / df['Total_Assets']

    # X2: Retained Earnings / Total Assets
    X2 = df['Retained_Earnings'] / df['Total_Assets']

    # X3: EBIT / Total Assets
    X3 = df['EBIT'] / df['Total_Assets']

    # X4: Market Value of Equity / Total Liabilities
    # Approximate market value using share price * shares outstanding
    # For simplicity, use Book Value if share price not available per row
    # We'll compute based on available data
    if 'Share_Price' in df.columns:
        mkt_equity = df['Share_Price'] * df['Shares_Outstanding']
    else:
        mkt_equity = df['Total_Equity']  # fallback to book value
    X4 = mkt_equity / df['Total_Liabilities']

    # X5: Revenue / Total Assets
    X5 = df['Revenue'] / df['Total_Assets']

    # Z-Score
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    z_scores['X1_WC_TA'] = X1
    z_scores['X2_RE_TA'] = X2
    z_scores['X3_EBIT_TA'] = X3
    z_scores['X4_MVE_TL'] = X4
    z_scores['X5_Sales_TA'] = X5
    z_scores['Z_Score'] = Z

    def zone(z):
        if z > 2.99:
            return 'Safe Zone'
        elif z > 1.81:
            return 'Grey Zone'
        else:
            return 'Distress Zone'

    z_scores['Zone'] = Z.apply(zone)
    return z_scores.round(3)
