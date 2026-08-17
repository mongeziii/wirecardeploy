"""Recommendations module."""
import pandas as pd

def get_recommendations(forensic_score, risk_indicators, governance_risks):
    """Generate dynamic recommendations based on analysis."""

    # Determine overall risk level
    if forensic_score >= 70:
        overall_risk = "Critical"
    elif forensic_score >= 50:
        overall_risk = "High"
    elif forensic_score >= 30:
        overall_risk = "Medium"
    else:
        overall_risk = "Low"

    # Investor Recommendations
    investor = {
        'key_lessons': [
            'Cash flow divergence is a critical red flag — always verify OCF against reported earnings',
            'Rapid receivables growth outpacing revenue suggests potential revenue inflation',
            'High intangible assets relative to physical assets increases audit risk',
            'Regulatory protection of management (BaFin vs short sellers) can mask fraud',
            'Third-party trust accounts in opaque jurisdictions are high-risk structures',
        ],
        'recommended_actions': [
            'Implement mandatory cash flow verification for all growth investments',
            'Require independent verification of material off-balance-sheet items',
            'Establish position limits for companies with governance red flags',
            'Diversify exposure to avoid concentration in single high-growth fintech',
            'Demand quarterly third-party reconciliation of trust accounts',
        ]
    }

    # Regulator Recommendations
    regulator = {
        'key_lessons': [
            'Market manipulation investigations should not automatically target short sellers',
            'Auditor rotation and enhanced scrutiny of high-risk jurisdictions needed',
            'Real-time transaction monitoring can detect anomalies before annual audits',
            'Whistleblower protection mechanisms were insufficient in Wirecard case',
            'Cross-border regulatory cooperation gaps allowed fraud to persist',
        ],
        'recommended_actions': [
            'Establish independent forensic audit unit for systemically important fintechs',
            'Mandate direct bank confirmation for all material cash balances',
            'Implement mandatory auditor rotation every 5 years',
            'Create EU-wide fintech supervision framework with data sharing',
            'Strengthen whistleblower protections and anonymous reporting channels',
        ]
    }

    # Management Recommendations
    management = {
        'key_lessons': [
            'Aggressive growth targets can incentivize earnings manipulation',
            'Complex corporate structures with third-party partners obscure accountability',
            'Board oversight was inadequate — independent directors lacked payment expertise',
            'Internal controls over trust accounts were effectively non-existent',
            'Management override of controls was not detected or prevented',
        ],
        'recommended_actions': [
            'Separate CEO and Chair roles with independent board majority',
            'Implement zero-tolerance policy for management override of controls',
            'Establish direct reporting line from Internal Audit to Board Audit Committee',
            'Require monthly independent reconciliation of all third-party accounts',
            'Link executive compensation to verified cash metrics, not just revenue/earnings',
        ]
    }

    return {
        'overall_risk': overall_risk,
        'investor': investor,
        'regulator': regulator,
        'management': management
    }

def get_shareholder_impact(historical_df, share_price_df):
    """Calculate shareholder value destruction metrics."""
    peak_price = share_price_df['Share_Price'].max()
    peak_date = share_price_df.loc[share_price_df['Share_Price'].idxmax(), 'Date']

    # Pre-scandal price (end of 2018)
    pre_scandal = share_price_df[share_price_df['Date'] <= '2018-12-31']['Share_Price'].iloc[-1]

    # Post-scandal price (end of 2020)
    post_scandal = share_price_df['Share_Price'].iloc[-1]

    decline_pct = ((post_scandal - peak_price) / peak_price) * 100

    # Market cap change
    last_shares = historical_df['Shares_Outstanding'].iloc[-1] * 1e6
    peak_mcap = peak_price * last_shares
    post_mcap = post_scandal * last_shares
    mcap_loss = peak_mcap - post_mcap

    return {
        'peak_price': round(peak_price, 2),
        'peak_date': peak_date.strftime('%Y-%m-%d'),
        'pre_scandal_price': round(pre_scandal, 2),
        'post_scandal_price': round(post_scandal, 2),
        'decline_pct': round(decline_pct, 1),
        'peak_market_cap': round(peak_mcap / 1e9, 2),
        'post_market_cap': round(post_mcap / 1e9, 2),
        'value_destroyed': round(mcap_loss / 1e9, 2)
    }
