"""Scenario analysis module."""
import pandas as pd
import numpy as np
from modules.valuation import calculate_dcf

def build_scenarios(historical_df, base_assumptions):
    """Build Bull, Base, and Bear scenarios."""
    scenarios = {}

    # Bear Case: Low growth, low margin, high WACC
    bear = base_assumptions.copy()
    bear['revenue_growth'] = base_assumptions['revenue_growth'] * 0.5
    bear['ebitda_margin'] = base_assumptions['ebitda_margin'] * 0.85
    bear['wacc'] = base_assumptions['wacc'] + 2.0
    bear['terminal_growth'] = max(0.5, base_assumptions['terminal_growth'] - 1.0)
    scenarios['Bear'] = bear

    # Base Case: As provided
    scenarios['Base'] = base_assumptions.copy()

    # Bull Case: Higher growth, higher margin, lower WACC
    bull = base_assumptions.copy()
    bull['revenue_growth'] = base_assumptions['revenue_growth'] * 1.5
    bull['ebitda_margin'] = min(40, base_assumptions['ebitda_margin'] * 1.15)
    bull['wacc'] = max(6.0, base_assumptions['wacc'] - 1.5)
    bull['terminal_growth'] = base_assumptions['terminal_growth'] + 0.5
    scenarios['Bull'] = bull

    return scenarios

def run_scenario_analysis(forecast_df, scenarios, historical_df):
    """Run valuation for all scenarios."""
    from modules.forecasting import build_forecast_model

    results = {}
    for name, assumptions in scenarios.items():
        # Build scenario-specific forecast
        scenario_forecast = build_forecast_model(historical_df, assumptions)
        dcf = calculate_dcf(scenario_forecast, assumptions, historical_df)

        results[name] = {
            'assumptions': assumptions,
            'forecast': scenario_forecast,
            'dcf': dcf,
            'enterprise_value': dcf['Enterprise_Value'],
            'equity_value': dcf['Equity_Value'],
            'value_per_share': dcf['Value_Per_Share']
        }

    return results
