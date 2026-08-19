"""
Wirecard AG — Financial Forensic Analysis & Business Intelligence Platform
A complete professional web application for financial forensic investigation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import sys
import base64
from io import BytesIO

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_loader import load_financial_data, load_share_prices, validate_data, get_data_summary
from modules.data_cleaning import clean_dataframe
from modules.financial_analysis import calculate_all_ratios, get_ratio_status, format_millions, format_billions
from modules.forensic_analysis import calculate_forensic_risk_score, get_forensic_indicators, get_beneish_scores
from modules.governance import get_scandal_timeline, get_governance_risks
from modules.forecasting import build_forecast_model, build_no_scenario_forecast, get_default_assumptions
from modules.machine_learning import train_models, get_best_model
from modules.valuation import calculate_dcf, calculate_ddm, calculate_relative_valuation, calculate_wacc, sensitivity_analysis
from modules.scenario_analysis import build_scenarios, run_scenario_analysis
from modules.recommendations import get_recommendations, get_shareholder_impact

# Page config
st.set_page_config(
    page_title="Wirecard AG — Financial Forensic Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Color palette
COLORS = {
    'navy': '#071A2B',
    'dark_navy': '#0D2942',
    'blue': '#1677C8',
    'light_blue': '#3FA9F5',
    'white': '#FFFFFF',
    'light_grey': '#F3F5F7',
    'dark_text': '#1F2933',
    'warning': '#F59E0B',
    'danger': '#DC2626',
    'success': '#16A34A',
    'text_muted': '#A0B4C8'
}

# Plotly template
pio.templates.default = "plotly_dark"

def apply_chart_style(fig, title=""):
    """Apply consistent styling to Plotly charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS['white'], family="Inter, sans-serif"), x=0.5, xanchor='center'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=COLORS['text_muted']),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text_muted'])),
        xaxis=dict(gridcolor='rgba(26, 74, 122, 0.3)', linecolor='rgba(26, 74, 122, 0.5)', tickfont=dict(color=COLORS['text_muted'])),
        yaxis=dict(gridcolor='rgba(26, 74, 122, 0.3)', linecolor='rgba(26, 74, 122, 0.5)', tickfont=dict(color=COLORS['text_muted'])),
        margin=dict(l=60, r=40, t=80, b=60),
        hoverlabel=dict(bgcolor=COLORS['dark_navy'], font=dict(color=COLORS['white'])),
    )
    return fig

# Load data
@st.cache_data
def get_data():
    df = load_financial_data()
    prices = load_share_prices()
    return df, prices

df, share_prices = get_data()
df_clean = clean_dataframe(df)
ratios = calculate_all_ratios(df_clean)

# Session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid #1a4a7a; margin-bottom: 20px;">
        <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF; letter-spacing: 0.05em;">WIRECARD AG</div>
        <div style="font-size: 0.7rem; color: #3FA9F5; text-transform: uppercase; letter-spacing: 0.15em; margin-top: 4px;">Forensic Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("Dashboard", "📊"),
        ("Financial Analysis", "📈"),
        ("Forensic Warning Signs", "⚠️"),
        ("Governance & Scandal", "🏛️"),
        ("Forecasting", "🔮"),
        ("Machine Learning", "🤖"),
        ("Valuation", "💎"),
        ("Scenario Analysis", "📋"),
        ("Recommendations", "✅"),
        ("Data Quality", "🔬"),
        ("Downloads", "💾"),
    ]

    for item, icon in nav_items:
        btn_type = "primary" if st.session_state.current_page == item else "secondary"
        if st.button(f"{icon} {item}", key=f"nav_{item}", type=btn_type, use_container_width=True):
            st.session_state.current_page = item
            st.rerun()

    st.markdown("""
    <div style="position: absolute; bottom: 20px; left: 20px; right: 20px; text-align: center; color: #5a7a9a; font-size: 0.75rem;">
        <div style="border-top: 1px solid #1a4a7a; padding-top: 12px;">
            Financial Forensic<br>Investigation Platform<br><br>
            <span style="color: #3FA9F5;">v2.0.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="header-title">WIRECARD AG — FINANCIAL FORENSIC ANALYSIS</div>
            <div class="header-subtitle">Financial Forensic Analysis & Business Intelligence</div>
            <div class="header-nav">Financial Performance | Governance | Scandal Impact | Forecasting | Machine Learning | Valuation</div>
        </div>
        <div style="text-align: right;">
            <div style="color: #3FA9F5; font-size: 0.85rem; font-weight: 600;">Financial Forensic Investigation Platform</div>
            <div style="color: #5a7a9a; font-size: 0.75rem; margin-top: 4px;">Data Status: <span style="color: #16A34A;">● Validated</span></div>
            <div style="color: #5a7a9a; font-size: 0.75rem;">Period: 2015 — 2020</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Filter Bar (shown on most pages)
def render_filter_bar():
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        year_filter = st.multiselect("Year", options=sorted(df['Year'].unique()), default=sorted(df['Year'].unique()), key="filter_year")
    with col2:
        metric_filter = st.selectbox("Financial Metric", ["All", "Revenue", "EBITDA", "EBIT", "Net Income", "Cash Flow"], key="filter_metric")
    with col3:
        period_filter = st.selectbox("Analysis Period", ["Full Period", "Pre-Scandal (2015-2018)", "Scandal Period (2019-2020)"], key="filter_period")
    with col4:
        scenario_filter = st.selectbox("Scenario", ["Actual", "Base Case", "Bear Case", "Bull Case"], key="filter_scenario")
    with col5:
        source_filter = st.selectbox("Data Source", ["Consolidated", "Annual Reports", "Forensic Restatement"], key="filter_source")
    return year_filter, metric_filter, period_filter, scenario_filter, source_filter

# Helper: KPI Card
def kpi_card(title, value, subtitle="", status="neutral"):
    status_class = status if status in ["positive", "negative", "warning"] else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value {status_class}">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper: Risk Card
def risk_card(title, score, status, level, explanation, implication):
    st.markdown(f"""
    <div class="risk-card {level}">
        <div class="risk-title">{title}</div>
        <div class="risk-status {level}">{status}</div>
        <div style="color: #FFFFFF; font-size: 1.4rem; font-weight: 700; margin: 8px 0;">{score}</div>
        <div style="color: #A0B4C8; font-size: 0.85rem; line-height: 1.5;">{explanation}</div>
        <div style="color: #3FA9F5; font-size: 0.8rem; margin-top: 8px; font-style: italic;">💡 {implication}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper: Scenario Card
def scenario_card(title, ev, eq_val, vps, upside, case_type="base"):
    st.markdown(f"""
    <div class="scenario-card {case_type}">
        <div class="scenario-title">{title}</div>
        <div style="color: #FFFFFF; font-size: 1.6rem; font-weight: 700; margin: 12px 0;">€{vps:.2f}</div>
        <div style="color: #A0B4C8; font-size: 0.85rem;">Value per Share</div>
        <div style="margin-top: 16px; border-top: 1px solid #1a4a7a; padding-top: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #A0B4C8; font-size: 0.8rem;">Enterprise Value</span>
                <span style="color: #FFFFFF; font-weight: 600;">€{ev/1000:.2f}B</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #A0B4C8; font-size: 0.8rem;">Equity Value</span>
                <span style="color: #FFFFFF; font-weight: 600;">€{eq_val/1000:.2f}B</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #A0B4C8; font-size: 0.8rem;">Upside/Downside</span>
                <span style="color: {'#16A34A' if upside > 0 else '#DC2626'}; font-weight: 600;">{upside:+.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Helper: Recommendation Panel
def rec_panel(title, lessons, actions):
    lessons_html = "\n".join([f"<li>{l}</li>" for l in lessons])
    actions_html = "\n".join([f"<li>{a}</li>" for a in actions])
    st.markdown(f"""
    <div class="rec-panel">
        <h4>📌 {title} — Key Lessons</h4>
        <ul>{lessons_html}</ul>
        <h4>✅ {title} — Recommended Actions</h4>
        <ul>{actions_html}</ul>
    </div>
    """, unsafe_allow_html=True)

# Helper: Download link
def get_download_link(data, filename, label):
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
    elif isinstance(data, str):
        b64 = base64.b64encode(data.encode()).decode()
    else:
        b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;"><div class="download-card"><div class="download-icon">📥</div><div class="download-title">{label}</div><div class="download-desc">{filename}</div></div></a>'
    return href

# Filter data based on selections
def filter_data(df, years):
    if years:
        return df[df['Year'].isin(years)]
    return df

# ==================== DASHBOARD PAGE ====================
def render_dashboard():
    st.markdown("<h1 style='text-align: center; margin-bottom: 8px;'>WIRECARD AG — FINANCIAL FORENSIC ANALYSIS</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #3FA9F5; margin-bottom: 32px;'>Executive Overview</h3>", unsafe_allow_html=True)

    year_filter, metric_filter, period_filter, scenario_filter, source_filter = render_filter_bar()
    filtered_df = filter_data(df_clean, year_filter)

    # KPI Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latest_rev = filtered_df['Revenue'].iloc[-1] if len(filtered_df) > 0 else 0
        kpi_card("Revenue", format_billions(latest_rev), "Historical trend", "positive")
    with col2:
        latest_ebitda = filtered_df['EBITDA'].iloc[-1] if len(filtered_df) > 0 else 0
        kpi_card("EBITDA", format_billions(latest_ebitda), "Financial metric", "neutral")
    with col3:
        latest_ni = filtered_df['Net_Income'].iloc[-1] if len(filtered_df) > 0 else 0
        status = "negative" if latest_ni < 0 else "positive"
        kpi_card("Earnings After Tax", format_millions(latest_ni), "Net income", status)
    with col4:
        latest_ocf = filtered_df['Operating_Cash_Flow'].iloc[-1] if len(filtered_df) > 0 else 0
        kpi_card("Operating Cash Flow", format_millions(latest_ocf), "Cash generation", "neutral")

    # KPI Cards Row 2
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latest_ebit = filtered_df['EBIT'].iloc[-1] if len(filtered_df) > 0 else 0
        kpi_card("EBIT", format_millions(latest_ebit), "Operating profit", "neutral")
    with col2:
        ebitda_margin = (latest_ebitda / latest_rev * 100) if latest_rev > 0 else 0
        kpi_card("EBITDA Margin", f"{ebitda_margin:.1f}%", "Profitability", "positive" if ebitda_margin > 20 else "warning")
    with col3:
        eq_ratio = filtered_df['Earnings_Quality_Ratio'].iloc[-1] if len(filtered_df) > 0 and 'Earnings_Quality_Ratio' in filtered_df.columns else 0
        status = "positive" if eq_ratio > 0.8 else "warning" if eq_ratio > 0.5 else "negative"
        kpi_card("Earnings Quality", f"{eq_ratio:.2f}", "OCF / Net Income", status)
    with col4:
        risk_score, _ = calculate_forensic_risk_score(df_clean)
        status = "negative" if risk_score >= 70 else "warning" if risk_score >= 50 else "positive"
        kpi_card("Forensic Risk Score", f"{risk_score:.0f}/100", "Risk assessment", status)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Charts Grid
    col1, col2 = st.columns(2)

    with col1:
        # Revenue Trend
        fig = px.line(filtered_df, x='Year', y='Revenue', markers=True, 
                      color_discrete_sequence=[COLORS['light_blue']])
        fig = apply_chart_style(fig, "Revenue Trend")
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

        # EBITDA / EBIT
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['EBITDA'], 
                                 mode='lines+markers', name='EBITDA', line=dict(color=COLORS['light_blue'], width=3)))
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['EBIT'], 
                                 mode='lines+markers', name='EBIT', line=dict(color=COLORS['blue'], width=3)))
        fig = apply_chart_style(fig, "EBITDA vs EBIT")
        st.plotly_chart(fig, use_container_width=True)

        # Margin Analysis
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['EBITDA_Margin'], 
                                 mode='lines+markers', name='EBITDA Margin', line=dict(color=COLORS['light_blue'], width=3)))
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['EBIT_Margin'], 
                                 mode='lines+markers', name='EBIT Margin', line=dict(color=COLORS['blue'], width=3)))
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Profit_Margin'], 
                                 mode='lines+markers', name='Profit Margin', line=dict(color=COLORS['success'], width=3)))
        fig = apply_chart_style(fig, "Margin Analysis")
        fig.update_layout(yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Earnings vs Cash Flow
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Net_Income'], 
                                 mode='lines+markers', name='Net Income', line=dict(color=COLORS['light_blue'], width=3)))
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Operating_Cash_Flow'], 
                                 mode='lines+markers', name='Operating CF', line=dict(color=COLORS['success'], width=3)))
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Free_Cash_Flow'], 
                                 mode='lines+markers', name='Free CF', line=dict(color=COLORS['warning'], width=3)))
        fig = apply_chart_style(fig, "Earnings vs Cash Flow")
        st.plotly_chart(fig, use_container_width=True)

        # Financial Activity
        fig = go.Figure()
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Operating_Cash_Flow'], name='Operating', marker_color=COLORS['success']))
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Investing_Cash_Flow'], name='Investing', marker_color=COLORS['warning']))
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Financing_Cash_Flow'], name='Financing', marker_color=COLORS['light_blue']))
        fig = apply_chart_style(fig, "Cash Flow Activity")
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        # Assets vs Equity & Liabilities
        fig = go.Figure()
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Total_Equity'], name='Equity', marker_color=COLORS['success']))
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Total_Liabilities'], name='Liabilities', marker_color=COLORS['danger']))
        fig = apply_chart_style(fig, "Assets = Equity + Liabilities")
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)


    # APAC Revenue Concentration
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Geographic Revenue Concentration — Asia-Pacific TPA Segment")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Revenue'], name='Total Revenue', marker_color=COLORS['dark_navy'], opacity=0.3))
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['APAC_Revenue'], name='APAC Revenue', marker_color=COLORS['danger']))
        fig = apply_chart_style(fig, "APAC Revenue vs Total Revenue")
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['APAC_Revenue_Pct'], mode='lines+markers', 
                                 name='APAC % of Total', line=dict(color=COLORS['danger'], width=4), 
                                 fill='tozeroy', fillcolor='rgba(220, 38, 38, 0.2)'))
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS['warning'], annotation_text="Suspicious Threshold")
        fig = apply_chart_style(fig, "APAC Revenue as % of Total")
        fig.update_layout(yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)

    st.warning("⚠️ APAC revenue grew from 25% to 42% of total revenue (2015–2019), concentrated in opaque third-party acquiring partners. This segment was later identified as the primary source of fictitious revenue.")

# ==================== FINANCIAL ANALYSIS PAGE ====================
def render_financial_analysis():
    st.markdown("<h1>Financial Performance Analysis</h1>", unsafe_allow_html=True)

    year_filter, metric_filter, period_filter, scenario_filter, source_filter = render_filter_bar()
    filtered_df = filter_data(df_clean, year_filter)
    filtered_ratios = filter_data(ratios, year_filter)

    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        kpi_card("Revenue", format_billions(filtered_df['Revenue'].iloc[-1]), "Latest", "positive")
    with col2:
        kpi_card("EBITDA", format_millions(filtered_df['EBITDA'].iloc[-1]), "Operating", "neutral")
    with col3:
        kpi_card("EBIT", format_millions(filtered_df['EBIT'].iloc[-1]), "Pre-interest", "neutral")
    with col4:
        status = "negative" if filtered_df['Net_Income'].iloc[-1] < 0 else "positive"
        kpi_card("Net Income", format_millions(filtered_df['Net_Income'].iloc[-1]), "After tax", status)
    with col5:
        kpi_card("Operating CF", format_millions(filtered_df['Operating_Cash_Flow'].iloc[-1]), "Cash generated", "neutral")
    with col6:
        kpi_card("Free CF", format_millions(filtered_df['Free_Cash_Flow'].iloc[-1]), "After capex", "neutral")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Ratios", "Trends"])

    with tabs[0]:
        st.subheader("Income Statement Summary")
        inc_cols = ['Year', 'Revenue', 'COGS', 'Gross_Profit', 'Operating_Expenses', 'EBITDA', 'D_A', 'EBIT', 'Interest_Expense', 'EBT', 'Tax', 'Net_Income']
        inc_data = filtered_df[inc_cols] if all(c in filtered_df.columns for c in inc_cols) else filtered_df
        st.dataframe(inc_data.style.format("{:,.0f}").background_gradient(subset=['Revenue', 'Net_Income'], cmap='Blues'), use_container_width=True)

        fig = px.bar(filtered_df, x='Year', y=['Revenue', 'EBITDA', 'Net_Income'], barmode='group',
                     color_discrete_sequence=[COLORS['light_blue'], COLORS['blue'], COLORS['success']])
        fig = apply_chart_style(fig, "Income Statement Overview")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Balance Sheet Summary")
        bs_cols = ['Year', 'Total_Assets', 'Current_Assets', 'Non_Current_Assets', 'Total_Liabilities', 'Current_Liabilities', 'Non_Current_Liabilities', 'Total_Equity']
        bs_data = filtered_df[bs_cols] if all(c in filtered_df.columns for c in bs_cols) else filtered_df
        st.dataframe(bs_data.style.format("{:,.0f}").background_gradient(subset=['Total_Assets', 'Total_Equity'], cmap='Blues'), use_container_width=True)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Asset Composition", "Liability & Equity Structure"))
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Current_Assets'], name='Current Assets', marker_color=COLORS['light_blue']), row=1, col=1)
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Non_Current_Assets'], name='Non-Current Assets', marker_color=COLORS['blue']), row=1, col=1)
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Total_Equity'], name='Equity', marker_color=COLORS['success']), row=1, col=2)
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Total_Liabilities'], name='Liabilities', marker_color=COLORS['danger']), row=1, col=2)
        fig.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color=COLORS['text_muted']))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Cash Flow Statement")
        cf_cols = ['Year', 'Operating_Cash_Flow', 'Capex', 'Free_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow']
        cf_data = filtered_df[cf_cols] if all(c in filtered_df.columns for c in cf_cols) else filtered_df
        st.dataframe(cf_data.style.format("{:,.0f}").background_gradient(subset=['Free_Cash_Flow'], cmap='RdYlGn'), use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Waterfall(
            x=filtered_df['Year'],
            y=filtered_df['Operating_Cash_Flow'],
            name="Operating CF",
            measure=["relative"] * len(filtered_df),
            text=[f"{v:,.0f}" for v in filtered_df['Operating_Cash_Flow']],
            textposition="outside",
            connector={"line": {"color": COLORS['text_muted']}},
            decreasing={"marker": {"color": COLORS['danger']}},
            increasing={"marker": {"color": COLORS['success']}}
        ))
        fig = apply_chart_style(fig, "Operating Cash Flow Waterfall")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Financial Ratios")
        ratio_display = filtered_ratios[['Year', 'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity', 'ROA', 'ROE', 
                                         'EBITDA_Margin', 'EBIT_Margin', 'Profit_Margin', 'Cash_Flow_Margin', 
                                         'Earnings_Quality', 'Asset_Turnover', 'Receivables_Turnover']]

        def color_status(val, col_name):
            status = get_ratio_status(val, col_name)
            if status == 'good':
                return 'background-color: rgba(22, 163, 74, 0.3); color: #16A34A'
            elif status == 'warning':
                return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B'
            elif status == 'danger':
                return 'background-color: rgba(220, 38, 38, 0.3); color: #DC2626'
            return ''

        styled = ratio_display.style
        for col in ratio_display.columns:
            if col != 'Year':
                styled = styled.applymap(lambda x: color_status(x, col), subset=[col])

        st.dataframe(styled.format("{:.2f}"), use_container_width=True)

        # Ratio trends
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Profitability Ratios", "Liquidity Ratios", "Leverage Ratios", "Efficiency Ratios"))
        fig.add_trace(go.Scatter(x=filtered_ratios['Year'], y=filtered_ratios['ROA'], name='ROA', line=dict(color=COLORS['light_blue'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_ratios['Year'], y=filtered_ratios['ROE'], name='ROE', line=dict(color=COLORS['blue'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_ratios['Year'], y=filtered_ratios['Current_Ratio'], name='Current', line=dict(color=COLORS['success'])), row=1, col=2)
        fig.add_trace(go.Scatter(x=filtered_ratios['Year'], y=filtered_ratios['Quick_Ratio'], name='Quick', line=dict(color=COLORS['warning'])), row=1, col=2)
        fig.add_trace(go.Scatter(x=filtered_ratios['Year'], y=filtered_ratios['Debt_to_Equity'], name='D/E', line=dict(color=COLORS['danger'])), row=2, col=1)
        fig.add_trace(go.Scatter(x=filtered_ratios['Year'], y=filtered_ratios['Asset_Turnover'], name='Asset Turnover', line=dict(color=COLORS['light_blue'])), row=2, col=2)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text_muted']), height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Trend Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Revenue_Growth'], mode='lines+markers', name='Revenue Growth', line=dict(color=COLORS['light_blue'], width=3)))
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Receivables_Growth'], mode='lines+markers', name='Receivables Growth', line=dict(color=COLORS['danger'], width=3)))
        fig = apply_chart_style(fig, "Revenue vs Receivables Growth")
        fig.update_layout(yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)

        # Earnings Quality Trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Earnings_Quality_Ratio'], mode='lines+markers', 
                                 name='Earnings Quality (OCF/NI)', line=dict(color=COLORS['warning'], width=3)))
        fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS['success'], annotation_text="Healthy Threshold")
        fig.add_hline(y=0.8, line_dash="dash", line_color=COLORS['warning'], annotation_text="Warning Threshold")
        fig = apply_chart_style(fig, "Earnings Quality Ratio Trend")
        st.plotly_chart(fig, use_container_width=True)

# ==================== FORENSIC WARNING SIGNS PAGE ====================
def render_forensic():
    st.markdown("<h1>Financial Forensic Warning Signs</h1>", unsafe_allow_html=True)

    risk_score, score_components = calculate_forensic_risk_score(df_clean)
    indicators = get_forensic_indicators(df_clean)
    beneish = get_beneish_scores(df_clean)

    # Risk Score Gauge
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="indicator-container">
            <div class="indicator-value" style="color: {'#DC2626' if risk_score >= 70 else '#F59E0B' if risk_score >= 50 else '#3FA9F5'};">{risk_score:.0f}</div>
            <div class="indicator-label">Forensic Risk Score / 100</div>
            <div class="indicator-status {'critical' if risk_score >= 70 else 'high' if risk_score >= 50 else 'medium'}">
                {'CRITICAL RISK' if risk_score >= 70 else 'HIGH RISK' if risk_score >= 50 else 'MEDIUM RISK'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'font': {'size': 48, 'color': COLORS['white']}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': COLORS['text_muted']},
                'bar': {'color': COLORS['danger'] if risk_score >= 70 else COLORS['warning'] if risk_score >= 50 else COLORS['light_blue']},
                'bgcolor': COLORS['dark_navy'],
                'borderwidth': 2,
                'bordercolor': COLORS['dark_navy'],
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(22, 163, 74, 0.2)'},
                    {'range': [30, 50], 'color': 'rgba(59, 130, 246, 0.2)'},
                    {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(220, 38, 38, 0.2)'}
                ],
                'threshold': {'line': {'color': COLORS['danger'], 'width': 4}, 'thickness': 0.75, 'value': risk_score}
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Component Breakdown")
        comp_df = pd.DataFrame({
            'Component': ['Earnings Quality', 'Cash Flow Divergence', 'Receivables Growth', 'Profitability vs Cash', 'Governance'],
            'Score': [score_components['earnings_quality'], score_components['cash_flow_divergence'], 
                     score_components['receivables_growth'], score_components['profitability_vs_cash'], 
                     score_components['governance']],
            'Weight': ['25%', '20%', '20%', '20%', '15%']
        })
        fig = px.bar(comp_df, x='Component', y='Score', color='Score', 
                     color_continuous_scale=['#16A34A', '#F59E0B', '#DC2626'],
                     text='Weight')
        fig = apply_chart_style(fig, "Risk Score Components")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Warning Cards
    st.subheader("Forensic Warning Indicators")
    cols = st.columns(2)
    for i, ind in enumerate(indicators):
        with cols[i % 2]:
            risk_card(ind['name'], ind['score'], ind['status'], ind['level'], ind['explanation'], ind['implication'])

    st.markdown("<hr>", unsafe_allow_html=True)
    # Earnings Quality Section
    st.subheader("Earnings Quality Analysis")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_clean['Year'], y=df_clean['Net_Income'], name='Net Income', marker_color=COLORS['light_blue']))
    fig.add_trace(go.Bar(x=df_clean['Year'], y=df_clean['Operating_Cash_Flow'], name='Operating Cash Flow', marker_color=COLORS['success']))
    fig = apply_chart_style(fig, "Earnings vs Cash Generation")
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    latest_eq = df_clean['Earnings_Quality_Ratio'].iloc[-1]
    if latest_eq < 0.5:
        st.error("**CRITICAL:** Operating cash flow is significantly below reported net income. This is a strong indicator of earnings manipulation or fictitious revenue. The company may be booking revenue without corresponding cash collections.")
    elif latest_eq < 0.8:
        st.warning("**WARNING:** Earnings quality is deteriorating. A substantial portion of reported earnings is not supported by operating cash flows. Investigate accruals and revenue recognition policies.")
    else:
        st.success("**ACCEPTABLE:** Earnings are reasonably supported by operating cash flows. Monitor for trends.")

    # Beneish M-Score
    st.subheader("Beneish M-Score Analysis")
    beneish_df = pd.DataFrame([beneish])
    st.dataframe(beneish_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn_r'), use_container_width=True)
    st.info("M-Score > -1.78 suggests high probability of earnings manipulation. Wirecard's indicators show elevated risk across multiple dimensions.")


    # Altman Z-Score Section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Altman Z-Score — Bankruptcy Prediction")
    from modules.forensic_analysis import calculate_altman_zscore
    z_scores = calculate_altman_zscore(df_clean)

    col1, col2 = st.columns([2, 1])
    with col1:
        def z_color(val):
            if val > 2.99:
                return 'background-color: rgba(22, 163, 74, 0.3); color: #16A34A'
            elif val > 1.81:
                return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B'
            else:
                return 'background-color: rgba(220, 38, 38, 0.3); color: #DC2626'

        styled_z = z_scores[['Year', 'X1_WC_TA', 'X2_RE_TA', 'X3_EBIT_TA', 'X4_MVE_TL', 'X5_Sales_TA', 'Z_Score', 'Zone']].style
        styled_z = styled_z.applymap(z_color, subset=['Z_Score'])
        st.dataframe(styled_z.format({"X1_WC_TA": "{:.3f}", "X2_RE_TA": "{:.3f}", "X3_EBIT_TA": "{:.3f}", 
                                      "X4_MVE_TL": "{:.3f}", "X5_Sales_TA": "{:.3f}", "Z_Score": "{:.3f}"}), use_container_width=True)
        st.info("Z > 2.99 = Safe Zone | 1.81–2.99 = Grey Zone | < 1.81 = Distress Zone. Wirecard's Z-Score plunged into distress territory by 2020.")

    with col2:
        fig = go.Figure()
        colors = ['#16A34A' if z > 2.99 else '#F59E0B' if z > 1.81 else '#DC2626' for z in z_scores['Z_Score']]
        fig.add_trace(go.Bar(x=z_scores['Year'], y=z_scores['Z_Score'], marker_color=colors, name='Z-Score'))
        fig.add_hline(y=2.99, line_dash="dash", line_color="#16A34A", annotation_text="Safe Threshold")
        fig.add_hline(y=1.81, line_dash="dash", line_color="#DC2626", annotation_text="Distress Threshold")
        fig = apply_chart_style(fig, "Altman Z-Score Trend")
        st.plotly_chart(fig, use_container_width=True)

# ==================== GOVERNANCE & SCANDAL PAGE ====================
def render_governance():
    st.markdown("<h1>WIRECARD AG — Governance & Scandal Timeline</h1>", unsafe_allow_html=True)

    timeline = get_scandal_timeline()
    governance_risks = get_governance_risks()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Investigation Timeline")
        for _, event in timeline.iterrows():
            color_map = {'high': '#DC2626', 'medium': '#F59E0B', 'low': '#16A34A'}
            cat_map = {'growth': '#3FA9F5', 'regulatory': '#F59E0B', 'audit': '#DC2626', 'scandal': '#DC2626', 'collapse': '#DC2626'}
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0D2942 0%, #0a1f35 100%); border-left: 4px solid {color_map.get(event['risk_level'], '#3FA9F5')}; 
                        border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #3FA9F5; font-weight: 600; font-size: 0.85rem;">{event['date']}</span>
                    <span style="background-color: {cat_map.get(event['category'], '#1a4a7a')}; color: #FFFFFF; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem; text-transform: uppercase;">{event['category']}</span>
                </div>
                <div style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem; margin-bottom: 6px;">{event['event']}</div>
                <div style="color: #A0B4C8; font-size: 0.85rem; line-height: 1.5;"><strong>Governance:</strong> {event['governance_significance']}</div>
                <div style="color: #A0B4C8; font-size: 0.85rem; line-height: 1.5; margin-top: 4px;"><strong>Financial:</strong> {event['financial_significance']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("Governance Risk Dashboard")
        for _, risk in governance_risks.iterrows():
            color = '#DC2626' if risk['score'] >= 90 else '#F59E0B' if risk['score'] >= 70 else '#3FA9F5'
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0D2942 0%, #0a1f35 100%); border: 1px solid #1a4a7a; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #FFFFFF; font-weight: 600;">{risk['risk']}</span>
                    <span style="color: {color}; font-weight: 700; font-size: 1.2rem;">{risk['score']}/100</span>
                </div>
                <div style="background-color: #1a3a5c; border-radius: 4px; height: 6px; margin-bottom: 8px;">
                    <div style="background-color: {color}; width: {risk['score']}%; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="color: #A0B4C8; font-size: 0.8rem; line-height: 1.4;">{risk['description']}</div>
                <div style="color: #3FA9F5; font-size: 0.75rem; margin-top: 6px; font-style: italic;">✓ {risk['mitigation']}</div>
            </div>
            """, unsafe_allow_html=True)

# ==================== FORECASTING PAGE ====================
def render_forecasting():
    st.markdown("<h1>Five-Year Financial Forecast Model</h1>", unsafe_allow_html=True)

    assumptions = get_default_assumptions()

    st.subheader("Assumption Control Panel")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        assumptions['revenue_growth'] = st.slider("Revenue Growth %", 0.0, 30.0, 12.0, 0.5)
        assumptions['ebitda_margin'] = st.slider("EBITDA Margin %", 10.0, 50.0, 28.0, 1.0)
    with col2:
        assumptions['ebit_margin'] = st.slider("EBIT Margin %", 5.0, 45.0, 24.0, 1.0)
        assumptions['tax_rate'] = st.slider("Tax Rate %", 15.0, 40.0, 25.0, 1.0)
    with col3:
        assumptions['capex_pct'] = st.slider("Capex %", 1.0, 20.0, 6.0, 0.5)
        assumptions['working_capital_pct'] = st.slider("WC %", 1.0, 20.0, 8.0, 0.5)
    with col4:
        assumptions['interest_rate'] = st.slider("Interest Rate %", 0.5, 10.0, 3.5, 0.25)
        assumptions['debt_growth'] = st.slider("Debt Growth %", -5.0, 20.0, 5.0, 0.5)
    with col5:
        assumptions['dividend_payout'] = st.slider("Dividend Payout %", 0.0, 50.0, 20.0, 1.0)
        assumptions['terminal_growth'] = st.slider("Terminal Growth %", 0.0, 5.0, 2.0, 0.25)
        assumptions['wacc'] = st.slider("WACC %", 5.0, 15.0, 9.0, 0.25)

    forecast = build_forecast_model(df_clean, assumptions)
    no_scandal = build_no_scenario_forecast(df_clean, assumptions)

    tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement", "Actual vs No-Scandal"])

    with tabs[0]:
        st.dataframe(forecast[['Year', 'Revenue', 'EBITDA', 'EBIT', 'Net_Income', 'EBITDA_Margin', 'Profit_Margin']].style.format({"Revenue": "{:,.0f}", "EBITDA": "{:,.0f}", "EBIT": "{:,.0f}", "Net_Income": "{:,.0f}", "EBITDA_Margin": "{:.1f}%", "Profit_Margin": "{:.1f}%"}), use_container_width=True)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Revenue & EBITDA Forecast", "Net Income Forecast"))
        fig.add_trace(go.Scatter(x=forecast['Year'], y=forecast['Revenue'], name='Revenue', line=dict(color=COLORS['light_blue'], width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast['Year'], y=forecast['EBITDA'], name='EBITDA', line=dict(color=COLORS['blue'], width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast['Year'], y=forecast['Net_Income'], name='Net Income', line=dict(color=COLORS['success'], width=3), fill='tozeroy'), row=1, col=2)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text_muted']), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.dataframe(forecast[['Year', 'Total_Assets', 'Total_Liabilities', 'Total_Equity', 'Cash', 'Long_Term_Debt']].style.format("{:,.0f}"), use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=forecast['Year'], y=forecast['Total_Assets'], name='Total Assets', marker_color=COLORS['light_blue']))
        fig.add_trace(go.Bar(x=forecast['Year'], y=forecast['Total_Equity'], name='Equity', marker_color=COLORS['success']))
        fig = apply_chart_style(fig, "Forecast Balance Sheet")
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.dataframe(forecast[['Year', 'Operating_Cash_Flow', 'Capex', 'Free_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow']].style.format("{:,.0f}"), use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Waterfall(x=forecast['Year'], y=forecast['Free_Cash_Flow'], name='FCF', measure=['relative']*5, connector={"line":{"color":COLORS['text_muted']}}, increasing={"marker":{"color":COLORS['success']}}, decreasing={"marker":{"color":COLORS['danger']}}))
        fig = apply_chart_style(fig, "Free Cash Flow Waterfall")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.warning("This scenario is hypothetical and does not represent actual historical performance.")
        comp_metric = st.selectbox("Comparison Metric", ["Revenue", "EBITDA", "EBIT", "Net_Income", "Operating_Cash_Flow", "Free_Cash_Flow", "Total_Assets", "Total_Equity", "Total_Liabilities", "Cash"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['Year'], y=forecast[comp_metric], mode='lines+markers', name='Actual/Post-Scandal', line=dict(color=COLORS['danger'], width=3)))
        fig.add_trace(go.Scatter(x=no_scandal['Year'], y=no_scandal[comp_metric], mode='lines+markers', name='Hypothetical No-Scandal', line=dict(color=COLORS['success'], width=3, dash='dash')))
        fig = apply_chart_style(fig, f"Actual vs No-Scandal: {comp_metric}")
        st.plotly_chart(fig, use_container_width=True)

        diff = no_scandal[comp_metric].values - forecast[comp_metric].values
        fig2 = go.Figure(go.Bar(x=forecast['Year'], y=diff, marker_color=[COLORS['success'] if d > 0 else COLORS['danger'] for d in diff]))
        fig2 = apply_chart_style(fig2, f"Value Destruction: {comp_metric}")
        st.plotly_chart(fig2, use_container_width=True)

# ==================== MACHINE LEARNING PAGE ====================
def render_ml():
    st.markdown("<h1>Wirecard Share Price — Machine Learning Forecast</h1>", unsafe_allow_html=True)

    with st.spinner("Training models... This may take a moment."):
        results = train_models(share_prices)

    best_name, best_metrics = get_best_model(results)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Random Forest</div>
            <div style="margin-top: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">MAE</span><span style="color: #FFFFFF; font-weight: 600;">€{:.2f}</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">RMSE</span><span style="color: #FFFFFF; font-weight: 600;">€{:.2f}</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">R²</span><span style="color: #FFFFFF; font-weight: 600;">{:.3f}</span></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #A0B4C8; font-size: 0.85rem;">Directional Accuracy</span><span style="color: #FFFFFF; font-weight: 600;">{:.1f}%</span></div>
            </div>
        </div>
        """.format(results['RandomForest']['metrics']['MAE'], results['RandomForest']['metrics']['RMSE'], 
                   results['RandomForest']['metrics']['R2'], results['RandomForest']['metrics']['Directional_Accuracy']), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Gradient Boosting</div>
            <div style="margin-top: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">MAE</span><span style="color: #FFFFFF; font-weight: 600;">€{:.2f}</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">RMSE</span><span style="color: #FFFFFF; font-weight: 600;">€{:.2f}</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">R²</span><span style="color: #FFFFFF; font-weight: 600;">{:.3f}</span></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #A0B4C8; font-size: 0.85rem;">Directional Accuracy</span><span style="color: #FFFFFF; font-weight: 600;">{:.1f}%</span></div>
            </div>
        </div>
        """.format(results['GradientBoosting']['metrics']['MAE'], results['GradientBoosting']['metrics']['RMSE'],
                   results['GradientBoosting']['metrics']['R2'], results['GradientBoosting']['metrics']['Directional_Accuracy']), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.success(f"**Best Performing Model: {best_name}** — This model achieved the highest R² score of {best_metrics['R2']:.3f}, indicating it explains {best_metrics['R2']*100:.1f}% of share price variance. Directional accuracy of {best_metrics['Directional_Accuracy']:.1f}% suggests reliable trend prediction capability.")

    st.subheader("Actual vs Predicted Share Price")
    test_dates = results['test_data']['dates']
    actual = results['test_data']['actual']
    rf_pred = results['RandomForest']['predictions']
    gb_pred = results['GradientBoosting']['predictions']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=actual, mode='lines+markers', name='Actual Price', line=dict(color=COLORS['white'], width=3)))
    fig.add_trace(go.Scatter(x=test_dates, y=rf_pred, mode='lines', name='Random Forest', line=dict(color=COLORS['light_blue'], width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=test_dates, y=gb_pred, mode='lines', name='Gradient Boosting', line=dict(color=COLORS['success'], width=2, dash='dot')))
    fig = apply_chart_style(fig, "Actual vs Predicted Share Price")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Comparison")
    comp_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'R²', 'Directional Accuracy'],
        'Random Forest': [results['RandomForest']['metrics'][m] for m in ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']],
        'Gradient Boosting': [results['GradientBoosting']['metrics'][m] for m in ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]
    })
    st.dataframe(comp_df.style.format({"Random Forest": "{:.2f}", "Gradient Boosting": "{:.2f}"}).background_gradient(subset=['Random Forest', 'Gradient Boosting'], cmap='RdYlGn'), use_container_width=True)

    st.subheader("Feature Importance — Random Forest")
    fig = px.bar(results['RandomForest']['importance'].head(10), x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
    fig = apply_chart_style(fig, "Top 10 Feature Importances")
    st.plotly_chart(fig, use_container_width=True)


    # Linear Regression Baseline
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Linear Regression Baseline Model")
    from modules.machine_learning import train_linear_regression_baseline
    lr_results = train_linear_regression_baseline(share_prices)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Linear Regression (Baseline)</div>
            <div style="margin-top: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">MAE</span><span style="color: #FFFFFF; font-weight: 600;">€{:.2f}</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">RMSE</span><span style="color: #FFFFFF; font-weight: 600;">€{:.2f}</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="color: #A0B4C8; font-size: 0.85rem;">R²</span><span style="color: #FFFFFF; font-weight: 600;">{:.3f}</span></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #A0B4C8; font-size: 0.85rem;">Directional Accuracy</span><span style="color: #FFFFFF; font-weight: 600;">{:.1f}%</span></div>
            </div>
        </div>
        """.format(lr_results['metrics']['MAE'], lr_results['metrics']['RMSE'], 
                   lr_results['metrics']['R2'], lr_results['metrics']['Directional_Accuracy']), unsafe_allow_html=True)

        st.info("Linear Regression serves as a simple baseline. Tree-based models (Random Forest, Gradient Boosting) capture non-linear relationships better, but all models trained on pre-fraud data structurally cannot predict a fraud-driven collapse — this is a key insight about the limits of ML in forensic contexts, not a weakness in the models.")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lr_results['test_data']['dates'], y=lr_results['test_data']['actual'], 
                                 mode='lines+markers', name='Actual Price', line=dict(color=COLORS['white'], width=3)))
        fig.add_trace(go.Scatter(x=lr_results['test_data']['dates'], y=lr_results['predictions'], 
                                 mode='lines', name='Linear Regression', line=dict(color='#A855F7', width=2, dash='dash')))
        fig = apply_chart_style(fig, "Linear Regression Baseline vs Actual")
        st.plotly_chart(fig, use_container_width=True)

    # Updated comparison table with LR
    st.subheader("Three-Model Comparison")
    comp_df3 = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'R²', 'Directional Accuracy'],
        'Linear Regression': [lr_results['metrics'][m] for m in ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']],
        'Random Forest': [results['RandomForest']['metrics'][m] for m in ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']],
        'Gradient Boosting': [results['GradientBoosting']['metrics'][m] for m in ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]
    })
    st.dataframe(comp_df3.style.format({"Linear Regression": "{:.2f}", "Random Forest": "{:.2f}", "Gradient Boosting": "{:.2f}"}).background_gradient(subset=['Linear Regression', 'Random Forest', 'Gradient Boosting'], cmap='RdYlGn'), use_container_width=True)

# ==================== VALUATION PAGE ====================
def render_valuation():
    st.markdown("<h1>Wirecard AG — Intrinsic Valuation</h1>", unsafe_allow_html=True)

    assumptions = get_default_assumptions()
    forecast = build_forecast_model(df_clean, assumptions)

    dcf = calculate_dcf(forecast, assumptions, df_clean)
    ddm = calculate_ddm(forecast, assumptions, df_clean)
    relative = calculate_relative_valuation(df_clean)
    wacc = calculate_wacc(assumptions, df_clean)

    st.subheader("Valuation Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("DCF", f"€{dcf['Value_Per_Share']:.2f}", "Intrinsic Value / Share", "positive")
    with col2:
        kpi_card("DDM / Residual Income", f"€{ddm['Value_Per_Share']:.2f}", "Intrinsic Value / Share", "positive")
    with col3:
        kpi_card("Relative Valuation", f"€{relative['Average_Value']:.2f}", "Implied Value / Share", "positive")
    with col4:
        combined = (dcf['Value_Per_Share'] + ddm['Value_Per_Share'] + relative['Average_Value']) / 3
        kpi_card("Combined Valuation", f"€{combined:.2f}", "Estimated Value / Share", "positive")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost of Equity (CAPM)")
        wacc_df = pd.DataFrame({
            'Component': ['Risk-Free Rate', 'Beta', 'Market Risk Premium', 'Cost of Equity'],
            'Value': [wacc['Risk_Free_Rate'], wacc['Beta'], wacc['Market_Risk_Premium'], wacc['Cost_of_Equity']],
            'Unit': ['%', '', '%', '%']
        })
        st.dataframe(wacc_df.style.format({"Value": "{:.2f}"}), use_container_width=True)

    with col2:
        st.subheader("WACC Calculation")
        wacc_calc = pd.DataFrame({
            'Component': ['Cost of Equity', 'After-Tax Cost of Debt', 'Equity Weight', 'Debt Weight', 'WACC'],
            'Value': [wacc['Cost_of_Equity'], wacc['After_Tax_Cost_of_Debt'], wacc['Equity_Weight'], wacc['Debt_Weight'], wacc['WACC']],
            'Unit': ['%', '%', '%', '%', '%']
        })
        st.dataframe(wacc_calc.style.format({"Value": "{:.2f}"}), use_container_width=True)

    st.subheader("DCF Valuation Bridge")
    bridge_df = pd.DataFrame({
        'Step': ['PV of Forecast FCF', 'PV of Terminal Value', 'Enterprise Value', 'Less: Net Debt', 'Equity Value', 'Value Per Share'],
        'Amount (€M)': [dcf['PV_FCF'], dcf['PV_Terminal'], dcf['Enterprise_Value'], -dcf['Net_Debt'], dcf['Equity_Value'], dcf['Value_Per_Share'] * dcf['Shares_Outstanding']]
    })
    fig = go.Figure(go.Waterfall(
        x=bridge_df['Step'],
        y=bridge_df['Amount (€M)'],
        measure=['relative', 'relative', 'total', 'relative', 'total', 'total'],
        text=[f"€{v:,.0f}M" if v >= 1000 else f"€{v:.0f}" for v in bridge_df['Amount (€M)']],
        textposition="outside",
        connector={"line": {"color": COLORS['text_muted']}},
        increasing={"marker": {"color": COLORS['success']}},
        decreasing={"marker": {"color": COLORS['danger']}},
        totals={"marker": {"color": COLORS['light_blue']}}
    ))
    fig = apply_chart_style(fig, "Valuation Bridge: FCF → Enterprise Value → Equity Value")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("DCF Sensitivity Analysis")
    sens = sensitivity_analysis(forecast, assumptions, df_clean)
    fig = px.imshow(sens, text_auto=True, aspect="auto", color_continuous_scale='RdYlGn')
    fig = apply_chart_style(fig, "WACC vs Terminal Growth Sensitivity (€/Share)")
    st.plotly_chart(fig, use_container_width=True)

# ==================== SCENARIO ANALYSIS PAGE ====================
def render_scenarios():
    st.markdown("<h1>Scenario Analysis</h1>", unsafe_allow_html=True)

    assumptions = get_default_assumptions()
    scenarios = build_scenarios(df_clean, assumptions)
    results = run_scenario_analysis(build_forecast_model(df_clean, assumptions), scenarios, df_clean)

    base_vps = results['Base']['value_per_share']

    st.subheader("Valuation Scenarios")
    col1, col2, col3 = st.columns(3)
    with col1:
        bear = results['Bear']
        upside = ((bear['value_per_share'] - base_vps) / base_vps) * 100 if base_vps != 0 else 0
        scenario_card("Worst Case", bear['enterprise_value'], bear['equity_value'], bear['value_per_share'], upside, "bear")
    with col2:
        base = results['Base']
        scenario_card("Base Case", base['enterprise_value'], base['equity_value'], base['value_per_share'], 0, "base")
    with col3:
        bull = results['Bull']
        upside = ((bull['value_per_share'] - base_vps) / base_vps) * 100 if base_vps != 0 else 0
        scenario_card("Best Case", bull['enterprise_value'], bull['equity_value'], bull['value_per_share'], upside, "bull")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Scenario Comparison")
    comp_data = pd.DataFrame({
        'Year': list(range(2021, 2026)),
        'Bear Revenue': results['Bear']['forecast']['Revenue'],
        'Base Revenue': results['Base']['forecast']['Revenue'],
        'Bull Revenue': results['Bull']['forecast']['Revenue']
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp_data['Year'], y=comp_data['Bear Revenue'], mode='lines+markers', name='Bear', line=dict(color=COLORS['danger'], width=3)))
    fig.add_trace(go.Scatter(x=comp_data['Year'], y=comp_data['Base Revenue'], mode='lines+markers', name='Base', line=dict(color=COLORS['light_blue'], width=3)))
    fig.add_trace(go.Scatter(x=comp_data['Year'], y=comp_data['Bull Revenue'], mode='lines+markers', name='Bull', line=dict(color=COLORS['success'], width=3)))
    fig = apply_chart_style(fig, "Revenue Scenario Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Shareholder Value Destruction")
    impact = get_shareholder_impact(df_clean, share_prices)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Peak Share Price", f"€{impact['peak_price']:.2f}", impact['peak_date'], "positive")
    with col2:
        kpi_card("Pre-Scandal Price", f"€{impact['pre_scandal_price']:.2f}", "End of 2018", "neutral")
    with col3:
        kpi_card("Post-Scandal Price", f"€{impact['post_scandal_price']:.2f}", "End of 2020", "negative")
    with col4:
        kpi_card("Value Destroyed", f"€{impact['value_destroyed']:.2f}B", f"{impact['decline_pct']:.1f}% decline", "negative")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=share_prices['Date'], y=share_prices['Share_Price'], mode='lines', name='Share Price', line=dict(color=COLORS['light_blue'], width=2)))

    scandal_events = [
        ('2019-04-25', '€1.9B', COLORS['danger']),
        ('2019-06-18', 'EY', COLORS['warning']),
        ('2020-01-01', 'FRD', COLORS['danger']),
        ('2020-06-25', 'INS', COLORS['danger']),

    ]
    
    for date, label, color in scandal_events:
        fig.add_vline(x=pd.to_datetime(date), line_width=2, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top")

    fig = apply_chart_style(fig, "Share Price with Major Scandal Events")
    st.plotly_chart(fig, use_container_width=True)

# ==================== RECOMMENDATIONS PAGE ====================
def render_recommendations():
    st.markdown("<h1>Recommendations</h1>", unsafe_allow_html=True)

    risk_score, _ = calculate_forensic_risk_score(df_clean)
    indicators = get_forensic_indicators(df_clean)
    gov_risks = get_governance_risks()
    recs = get_recommendations(risk_score, indicators, gov_risks)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {'#2d1515' if risk_score >= 70 else '#2d2315' if risk_score >= 50 else '#152d1f'} 0%, #0D2942 100%); 
                border: 1px solid {'#DC2626' if risk_score >= 70 else '#F59E0B' if risk_score >= 50 else '#16A34A'}; 
                border-radius: 12px; padding: 20px; margin-bottom: 24px; text-align: center;">
        <div style="color: #FFFFFF; font-size: 1.2rem; font-weight: 600;">Overall Risk Assessment: {recs['overall_risk'].upper()}</div>
        <div style="color: #A0B4C8; font-size: 0.9rem; margin-top: 8px;">Based on forensic analysis of financial statements, governance structure, and scandal indicators.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        rec_panel("Investors", recs['investor']['key_lessons'], recs['investor']['recommended_actions'])
    with col2:
        rec_panel("Regulators", recs['regulator']['key_lessons'], recs['regulator']['recommended_actions'])
    with col3:
        rec_panel("Management", recs['management']['key_lessons'], recs['management']['recommended_actions'])

# ==================== DATA QUALITY PAGE ====================
def render_data_quality():
    st.markdown("<h1>Data Quality & Validation</h1>", unsafe_allow_html=True)

    report = validate_data(df_clean)
    summary = get_data_summary(df_clean)

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        kpi_card("Rows", f"{report['rows']:,}", "Records", "neutral")
    with col2:
        kpi_card("Columns", f"{report['columns']}", "Fields", "neutral")
    with col3:
        status = "warning" if report['missing_values'] > 0 else "positive"
        kpi_card("Missing Values", f"{report['missing_values']}", f"{report['missing_pct']}%", status)
    with col4:
        status = "warning" if report['duplicate_records'] > 0 else "positive"
        kpi_card("Duplicates", f"{report['duplicate_records']}", "Records", status)
    with col5:
        kpi_card("Outliers", f"{sum(report['outliers'].values())}", "Detected", "warning")
    with col6:
        kpi_card("Available Years", f"{len(report['years_available'])}", f"{min(report['years_available'])}-{max(report['years_available'])}", "positive")
    with col7:
        kpi_card("Peak Revenue Year", f"{summary['peak_revenue_year']}", f"€{summary['peak_revenue']/1000:.2f}B", "neutral")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Data Preview")
    st.dataframe(df_clean.style.format("{:,.0f}"), use_container_width=True)

    st.subheader("Column Mapping")
    mapping = pd.DataFrame({
        'Column Name': df_clean.columns,
        'Data Type': df_clean.dtypes.values,
        'Non-Null Count': df_clean.count().values,
        'Sample Value': [str(df_clean[c].iloc[0]) for c in df_clean.columns]
    })
    st.dataframe(mapping, use_container_width=True)

# ==================== DOWNLOADS PAGE ====================
def render_downloads():
    st.markdown("<h1>Export Centre</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #A0B4C8; margin-bottom: 24px;'>Download analytical results and reports in various formats.</p>", unsafe_allow_html=True)

    ratios_export = calculate_all_ratios(df_clean)
    forecast = build_forecast_model(df_clean, get_default_assumptions())
    ml_results = train_models(share_prices)
    sens = sensitivity_analysis(forecast, get_default_assumptions(), df_clean)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(get_download_link(ratios_export, "financial_ratios.csv", "Financial Ratios"), unsafe_allow_html=True)
    with col2:
        st.markdown(get_download_link(df_clean, "financial_analysis.csv", "Financial Analysis"), unsafe_allow_html=True)
    with col3:
        st.markdown(get_download_link(forecast, "forecast_model.csv", "Forecast Model"), unsafe_allow_html=True)
    with col4:
        pred_df = pd.DataFrame({
            'Date': ml_results['test_data']['dates'],
            'Actual': ml_results['test_data']['actual'],
            'RF_Predicted': ml_results['RandomForest']['predictions'],
            'GB_Predicted': ml_results['GradientBoosting']['predictions']
        })
        st.markdown(get_download_link(pred_df, "ml_predictions.csv", "ML Predictions"), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(get_download_link(sens.reset_index(), "sensitivity_analysis.csv", "Sensitivity Analysis"), unsafe_allow_html=True)
    with col2:
        forensic_report = pd.DataFrame(get_forensic_indicators(df_clean))
        st.markdown(get_download_link(forensic_report, "forensic_warning_report.csv", "Forensic Warning Report"), unsafe_allow_html=True)
    with col3:
        timeline = get_scandal_timeline()
        st.markdown(get_download_link(timeline, "governance_timeline.csv", "Governance Timeline"), unsafe_allow_html=True)
    with col4:
        complete = pd.concat([df_clean, ratios_export], axis=1)
        st.markdown(get_download_link(complete, "complete_analysis.csv", "Complete Analysis"), unsafe_allow_html=True)

# ==================== MAIN ROUTER ====================
page = st.session_state.current_page

if page == "Dashboard":
    render_dashboard()
elif page == "Financial Analysis":
    render_financial_analysis()
elif page == "Forensic Warning Signs":
    render_forensic()
elif page == "Governance & Scandal":
    render_governance()
elif page == "Forecasting":
    render_forecasting()
elif page == "Machine Learning":
    render_ml()
elif page == "Valuation":
    render_valuation()
elif page == "Scenario Analysis":
    render_scenarios()
elif page == "Recommendations":
    render_recommendations()
elif page == "Data Quality":
    render_data_quality()
elif page == "Downloads":
    render_downloads()
