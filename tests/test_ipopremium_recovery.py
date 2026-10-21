import json
from pathlib import Path
from ml.recover_ipopremium import parse,period,component,number
from ml.recover_ipopremium_documents import extract
from ml.recover_wayback_preclose import valid_captures


def example():
    return dict(text='Example Ltd Listed on Oct 21, 2026 PRICE BAND ₹258–271 LOT SIZE 55 shares Open Oct 13 (completed) Close Oct 15 (completed) Allotment Oct 16 (completed) Last updated on 15-Oct-2026 18:42:03',
      expected=dict(issuer_key='recent_example',company_name='Example Ltd',issue_price=271,listing_date='2026-10-21'),
      source_url='https://www.ipopremium.in/view/ipo/123/example',fetched_at='2026-10-26T00:00:00Z',_path='example.json',_sha256='abc',links=[],
      tables=[dict(heading='Day-Wise (Times)',rows=[['Category','Day 1','Day 2','Day 3'],['QIBs','1','2','20'],['HNIs','3','4','40'],['Retail','5','6','60'],['Total','3','4','40']]),
       dict(heading='IPO Details',rows=[['Total Issue Size','10,000 shares (₹10 Cr)'],['Offer for Sale','10,000 shares (₹10 Cr)']]),
       dict(heading='Key Performance Indicators (KPI)',rows=[[],['KPI','JUN-26','MAR-26','MAR-25'],['Basic\nEPS','3','6','5']]),
       dict(heading='Peer Comparison (Financial Performance)',rows=[['Company','NAV/Share','RoNW','EPS'],['Competitor Ltd','100','20','9'],['Example Limited','20','20','6']]),
       dict(heading='Subscription Status',rows=[['Category','Reserved','Applied','Times'],['Retail','100','1000','10']])])


def test_daywise_history_is_retained_without_backdating_final_update():
    r=parse(example(),dict(cutoff_at_ist='2026-10-15T17:00:00+05:30'))
    assert r['identity_verified'] and len(r['daywise'])==3
    assert r['last_table_update_ist']=='2026-10-15T18:42:03+05:30'
    assert all(d['source_timestamp'] is None for d in r['daywise'])
    assert r['application_counts']['Retail']['quality'].startswith('publisher_approximate')


def test_offer_annual_eps_and_own_company_book_are_not_peer_or_quarter_values():
    r=parse(example(),dict(cutoff_at_ist='2026-10-15T17:00:00+05:30'))
    assert r['offer']['ofs_ratio']==1
    assert r['financial']['ipo_pre_offer_eps']==6
    assert r['financial']['eps_annual_period']=='2026-03-31'
    assert r['financial']['nav_per_share']==20
    assert abs(r['financial']['ipo_price_book_raw']-271/20)<1e-8


def test_future_financial_periods_do_not_enter_historical_valuation():
    r=parse(example(),dict(cutoff_at_ist='2025-10-15T17:00:00+05:30'))
    assert r['financial']['ipo_pre_offer_eps']==5


def test_malformed_publisher_amount_is_not_a_number_or_zero():
    assert component('1,000 shares aggregating ₹1092.26.00 Cr')['cr'] is None
    assert component(', shares')['shares'] is None
    assert number('0')==0 and number('-') is None


def test_wrong_cap_or_listing_date_rejects_identity():
    page=example();page['expected']['listing_date']='2026-10-22'
    assert not parse(page,dict(cutoff_at_ist='2026-10-15T17:00:00+05:30'))['identity_verified']


def test_native_anchor_mf_percentage_requires_matching_issuer_and_summary():
    text='Example Limited. Out of the total allocation of 10,000 Equity Shares to Anchor Investors, 7,188 Equity Shares (i.e. 71.88% of total allocation to Anchor Investors) were allocated to 9 domestic Mutual Funds applying through 17 schemes.'
    assert extract(text,'Example Ltd','Anchor List')['anchor_domestic_mf_pct']==71.88
    assert extract(text,'Other Industries Ltd','Anchor List')['anchor_domestic_mf_pct'] is None
    assert extract('Example Ltd reserves 30% for domestic mutual funds','Example Ltd','Anchor List')['anchor_domestic_mf_pct'] is None


def test_cdx_named_columns_and_seconds_boundary():
    header=['original','statuscode','timestamp','mimetype','digest']
    rows=[header,['https://example.org','200','20261015113001','text/html','x'],['https://example.org','200','20261015112959','text/html','y'],['https://example.org','200','20261015113000','text/html','z'],['https://example.org','200','20261015082959','text/html','q']]
    valid=valid_captures(rows,'2026-10-15')
    assert [r['timestamp'] for r in valid]==['20261015112959']
    assert valid_captures([],'2026-10-15')==[]
