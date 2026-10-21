"""Actual extraction regressions: identity, dates, annual periods and amounts."""
import json
import numpy as np
import pandas as pd

from ml.extract_all_ipo_fundamentals import annual_eps,offer_components,OUT
from services.dated_ipo_fundamentals import at_decision
from ml.train_stage_fundamental_challenger import attach,ROOT


def test_fiscal_range_and_single_combined_eps_column():
    text='''Basic and diluted earnings per share (EPS)
    Fiscal Basic and Diluted EPS
    FY 2020-21 7.74
    FY 2019-20* (3.59)*
    FY 2018-19 3.08
    Weighted Average 3.19
    Six-month period ended September 30, 2021 (not annualized) 2.22
    '''
    result=annual_eps(text)
    assert [row['year'] for row in result]==[2021,2020,2019]
    assert result[0]['diluted_eps']==7.74
    assert result[1]['diluted_eps']==-3.59


def test_year_footnote_markers_do_not_hide_annual_earnings():
    text='''Basic and diluted earnings per share
    Period ended Basic EPS Diluted EPS Weight
    March 31, 2021(3) 16.92 16.92 3
    March 31, 2020(3) 10.03 10.03 2
    March 31, 2019(3) 5.39 5.39 1
    September 30, 2021(3)(4) 9.01 9.01
    '''
    rows=annual_eps(text)
    assert len(rows)==3 and rows[0]['diluted_eps']==16.92


def test_fiscal_ranges_with_separate_basic_and_diluted_columns():
    text='''Basic and Diluted Earnings per Share (EPS)
    Year ended Basic EPS Diluted EPS Weight
    FY 2018-19 8.25 8.25 1
    FY 2019-20 8.81 8.81 2
    FY 2020-21 13.13 13.13 3
    Quarter ended June 30, 2021 (not annualised) 3.90 3.90
    '''
    result=annual_eps(text)
    assert len(result)==3
    assert result[0]['year']==2021 and result[0]['diluted_eps']==13.13


def test_mixed_offer_keeps_rupees_and_does_not_use_preipo_placement_as_ofs():
    # Native text from Data Patterns' offer-summary table, including footnotes.
    text='''(i) Fresh Issue (1) 4,102,564* Equity Shares aggregating to ₹ 2,400.00* million
    (ii) Offer for Sale(2) 5,952,550* Equity Shares aggregating to ₹ 3,482.24* million
    Our Company has undertaken a Pre-IPO Placement aggregating to ₹ 600 million.
    The size of the fresh issue as disclosed in the Draft Red Herring Prospectus,
    aggregating up to ₹ 3,000 million, has been reduced by ₹ 600 million.'''
    result=offer_components(text)
    assert result['fresh_amount_rs']==2400000000.
    assert result['ofs_amount_rs']==3482240000.
    assert result['fresh_shares']==4102564
    assert result['ofs_shares']==5952550


def test_fresh_issuance_synonym_is_not_lost():
    text='''A FRESH ISSUANCE OF UP TO 41,40,000 EQUITY SHARES AGGREGATING UP TO
    ₹ 11,343.60 LAKHS AND AN OFFER FOR SALE OF UP TO 4,57,200 EQUITY SHARES
    AGGREGATING UP TO ₹ 1,252.73 LAKHS.'''
    result=offer_components(text)
    assert result['fresh_amount_rs']==1134360000.
    assert result['ofs_amount_rs']==125273000.


def test_undisclosed_ofs_amount_does_not_take_employee_reservation_money():
    text='''Offer for Sale(2) Up to 14,174,840 Equity Shares of face value of Rs 5 each
    aggregating up to Rs [bullet] million
    Employee Reservation Portion Up to [bullet] Equity Shares aggregating up to Rs 20.00 million'''
    result=offer_components(text)
    assert result['ofs_shares']==14174840
    assert 'ofs_amount_rs' not in result


def test_conflicting_amount_and_share_count_cannot_become_an_ofs_ratio():
    record=dict(document_date='2024-01-01',facts={
        'fresh_amount_rs':6842520000.,'ofs_amount_rs':20000000.,'ofs_shares':14174840.})
    result=at_decision(record,'2024-01-10',401.)
    assert result['ofs_ratio'] is None
    assert not result['offer_observed']
    assert result['offer_evidence_status']=='conflicting_amount_and_share_count'


def test_mixed_ratio_uses_dated_components_not_zero_or_default():
    record=dict(document_date='2024-01-01',facts={'fresh_amount_rs':2400000000.,'ofs_shares':5952550})
    observed=at_decision(record,'2024-01-10',585.)
    expected=5952550*585/(2400000000.+5952550*585)
    assert np.isclose(observed['ofs_ratio'],expected)
    assert observed['offer_observed']
    assert at_decision(record,'2023-12-31',585.)['ofs_ratio'] is None


def test_all_issuers_accounted_for_including_unavailable_sources():
    result=json.loads(OUT.read_text())
    assert result['processed_count']==result['expected_issuer_count']==len(result['records'])==441
    assert len({row['ipo_id'] for row in result['records']})==441
    available=[row for row in result['records'] if row.get('source_path')]
    assert len(available)==252
    assert sum('annual_diluted_eps' in row.get('facts',{}) for row in available)>149


def test_final_book_features_do_not_use_canonical_fiscal_year_placeholder_dates():
    payload=json.loads(OUT.read_text())
    record=next(row for row in payload['records'] if row['ipo_id']==484)
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    dates=pd.read_csv(ROOT/'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    assert canonical.loc[484].issue_close_date=='2023-03-31'
    assert dates.loc[484].issue_close_date=='2024-03-06'
    frame=pd.DataFrame([{'ipo_id':484}])
    joined=attach(frame,{484:record},canonical,'final_book',dates)
    assert joined.iloc[0].financial_observed
    assert joined.iloc[0].fundamental_decision_date=='2024-03-06'
