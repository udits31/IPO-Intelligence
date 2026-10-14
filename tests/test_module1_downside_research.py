"""Protect date, cohort and earlier-year selection boundaries in research fixes."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ml.collect_module1_public_financials import extract,number
from ml.research_module1_downside_fix import ROOT,choose
from ml.research_module1_financial_fix import choose as financial_choose
from ml.research_module1_blended_fix import select


def financial_table():
    return '''<table><tr><th></th><th>31-Mar-26</th><th>31-Mar-24</th><th>31-Mar-23</th></tr>
    <tr><td>Revenue</td><td>9999</td><td>100</td><td>80</td></tr>
    <tr><td>Profit After Tax</td><td>999</td><td>10</td><td>(5)</td></tr>
    <tr><td>Assets</td><td>99999</td><td>200</td><td>180</td></tr></table>
    <p>Valuations Earnings EPS Pre IPO ₹2/- EPS Post IPO 1.8 P/E Pre IPO 50 Returns ROE 5% Fundamentals Price / Book 3 Financials ( In Crs.)</p>'''


def test_financial_parser_ignores_future_period_and_aligns_columns():
    result=extract(financial_table(),'2025-01-01',100)
    assert result['annual_period']=='2024-03-31'
    assert result['annual_profit_after_tax']==10
    assert result['profit_margin_pct']==10
    assert result['return_on_assets_pct']==5
    assert result['ipo_earnings_yield_pct']==2
    assert number('(5)')==-5
    assert np.isnan(number('--'))


def test_valuation_consistency_gate_and_missing_annual_table():
    wrong=financial_table().replace('P/E Pre IPO 50','P/E Pre IPO 500')
    assert extract(wrong,'2025-01-01',100)['ipo_earnings_yield_pct'] is None
    assert extract(financial_table(),'2022-01-01',100)['financial_status']=='no_eligible_annual_table'


def test_all_selectors_reject_later_year_labels():
    later=pd.DataFrame([dict(ipo_id=1,year=2025,target_binary=1)])
    with pytest.raises(ValueError,match='Later'):
        choose([],later)
    with pytest.raises(ValueError,match='Later'):
        financial_choose([],later)
    with pytest.raises(ValueError,match='Later'):
        select(later,later)


def test_whole_cohort_preserved_and_report_counts_recompute():
    original=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv')
    for name,pname,callname in [('trajectory','p_trajectory_fix','trajectory_call'),('blended','p_blended_fix','blended_call')]:
        prefix=ROOT/f'output/module1_{name}_fix_20261024'
        report=json.loads(prefix.with_suffix('.json').read_text())
        ledger=pd.read_csv(prefix.with_name(prefix.name+'_predictions.csv'))
        assert set(ledger.ipo_id)==set(original.ipo_id)
        assert not set(report['training_ids'])&set(ledger.ipo_id)
        threshold=report['selected']['threshold'] if name=='trajectory' else report['choice']['threshold']
        assert ledger[callname].eq(ledger[pname].ge(threshold).astype(int)).all()
        assert int(ledger[callname].eq(ledger.target_binary).sum())==report['results']['tn']+report['results']['tp']
        for relative,expected in report['input_sha256'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
