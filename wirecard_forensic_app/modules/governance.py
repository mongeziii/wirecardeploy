"""Governance and scandal timeline module."""
import pandas as pd

def get_scandal_timeline():
    """Return Wirecard scandal timeline events."""
    events = [
        {
            'date': '2015-01-01',
            'event': 'Strong Growth Period Begins',
            'governance_significance': 'Management focused on aggressive expansion',
            'financial_significance': 'Revenue growing 20%+ annually',
            'risk_level': 'low',
            'category': 'growth'
        },
        {
            'date': '2016-10-15',
            'event': 'First Investigative Reports',
            'governance_significance': 'Journalists question business model sustainability',
            'financial_significance': 'Concerns raised about third-party acquiring business',
            'risk_level': 'medium',
            'category': 'regulatory'
        },
        {
            'date': '2017-02-20',
            'event': 'BaFin Investigation Opens',
            'governance_significance': 'German regulator begins market manipulation probe',
            'financial_significance': 'Short sellers targeted; regulator sides with company',
            'risk_level': 'medium',
            'category': 'regulatory'
        },
        {
            'date': '2018-04-10',
            'event': 'KPMG Special Investigation',
            'governance_significance': 'External audit commissioned due to growing concerns',
            'financial_significance': 'Third-party trust accounts under scrutiny',
            'risk_level': 'high',
            'category': 'audit'
        },
        {
            'date': '2019-04-25',
            'event': 'FT Investigation: Missing €1.9B',
            'governance_significance': 'Financial Times reveals potential fraud',
            'financial_significance': '€1.9 billion cash allegedly missing from trust accounts',
            'risk_level': 'high',
            'category': 'scandal'
        },
        {
            'date': '2019-06-18',
            'event': 'EY Audit Refusal',
            'governance_significance': 'Auditor unable to verify cash balances',
            'financial_significance': 'Annual financial statements cannot be certified',
            'risk_level': 'high',
            'category': 'audit'
        },
        {
            'date': '2020-01-01',
            'event': 'Admission of Fraud',
            'governance_significance': 'Management admits €1.9B likely never existed',
            'financial_significance': 'Previous financial statements materially misstated',
            'risk_level': 'high',
            'category': 'scandal'
        },
        {
            'date': '2020-06-25',
            'event': 'Insolvency Filing',
            'governance_significance': 'Company files for insolvency proceedings',
            'financial_significance': 'Share price collapses to near zero',
            'risk_level': 'high',
            'category': 'collapse'
        },
    ]
    return pd.DataFrame(events)

def get_governance_risks():
    """Return governance risk assessment."""
    risks = [
        {
            'risk': 'Audit Risk',
            'score': 95,
            'level': 'Critical',
            'description': 'EY unable to verify €1.9B cash. Auditors relied on falsified documents.',
            'mitigation': 'Independent verification of all material balances required'
        },
        {
            'risk': 'Internal Control Risk',
            'score': 90,
            'level': 'Critical',
            'description': 'No effective controls over third-party trust accounts. Management override prevalent.',
            'mitigation': 'Segregation of duties and independent reconciliation processes'
        },
        {
            'risk': 'Regulatory Risk',
            'score': 85,
            'level': 'High',
            'description': 'BaFin protected company from short sellers rather than investigating allegations.',
            'mitigation': 'Regulator independence and whistleblower protection mechanisms'
        },
        {
            'risk': 'Management Risk',
            'score': 95,
            'level': 'Critical',
            'description': 'CEO Markus Braun and COO Jan Marsalek implicated in fraud.',
            'mitigation': 'Board oversight, executive rotation, and fraud risk assessments'
        },
        {
            'risk': 'Third-Party Risk',
            'score': 90,
            'level': 'Critical',
            'description': 'Opaque third-party acquiring partners in Philippines and other jurisdictions.',
            'mitigation': 'Enhanced due diligence and direct verification of partner transactions'
        },
        {
            'risk': 'Reporting Risk',
            'score': 85,
            'level': 'High',
            'description': 'Financial statements materially misstated for multiple years.',
            'mitigation': 'Real-time monitoring and automated anomaly detection'
        },
    ]
    return pd.DataFrame(risks)
