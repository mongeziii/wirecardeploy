# Wirecard AG — Financial Forensic Analysis & Business Intelligence Platform

A complete, professional Python web application for financial forensic investigation, business intelligence, forecasting, machine learning, and valuation of Wirecard AG.

## Data Source

- **2015–2018**: Wirecard AG Annual Reports (consolidated financial statements)
- **Q3 2019**: Wirecard Q3 2019 Quarterly Statement
- **2019 full-year & 2020**: Estimates based on scandal timeline and insolvency filings

All annual data is in EUR million. Original 2015–2017 statements were in kEUR and have been standardised.

## Features

- **Executive Dashboard** — KPI cards, interactive filters, and comprehensive visualizations
- **Financial Analysis** — Income statement, balance sheet, cash flow, and ratio analysis
- **Forensic Warning Signs** — Risk scoring, earnings quality analysis, Beneish M-Score, Altman Z-Score
- **Governance & Scandal Timeline** — Investigation timeline and governance risk dashboard
- **Forecasting** — 5-year three-statement financial model with assumption controls
- **Machine Learning** — Linear Regression, Random Forest, and Gradient Boosting share price prediction
- **Valuation** — DCF, DDM/Residual Income, and relative valuation with sensitivity heatmaps
- **Scenario Analysis** — Bull, Base, and Bear cases with shareholder value impact
- **Recommendations** — Dynamic recommendations for investors, regulators, and management
- **Data Quality** — Validation reports and outlier detection
- **Downloads** — Export all analytical results

## Design

- Dark navy financial investigation theme
- Custom CSS styling (no default Streamlit appearance)
- Professional typography and card-based layouts
- Responsive Plotly charts with consistent color palette
- Bloomberg Terminal × Audit Dashboard aesthetic

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd wirecard_forensic_app
streamlit run app.py
```

## Project Structure

```
wirecard_forensic_app/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── financial_analysis.py
│   ├── forensic_analysis.py
│   ├── governance.py
│   ├── forecasting.py
│   ├── machine_learning.py
│   ├── valuation.py
│   ├── scenario_analysis.py
│   └── recommendations.py
├── assets/
│   └── style.css
└── data/
    ├── wirecard_data.csv
    └── share_prices.csv
```

## License

Academic / Educational Use
