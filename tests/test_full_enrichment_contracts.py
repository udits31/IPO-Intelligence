import json
import math
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytest
from services.point_in_time import eligible
from services.retail_hni_economics import assess,category_lots
from services.enriched_monitor import assemble,evaluate
from ml.extract_cached_deal_structure import parse,brlm_prior_statistics
from ml.research_module1_enriched import enrich
from scrapers.exchange_enrichment_scraper import flags,normalize_filing,normalize_deals


def test_unknown_clock_and_future_publication_are_excluded():
    assert not eligible(dict(source_timestamp=None,source_timestamp_verified=True),'2026-10-19T15:30:00+05:30')
    assert not eligible(dict(source_timestamp='2026-10-19T15:31:00+05:30',source_timestamp_verified=True),'2026-10-19T15:30:00+05:30')
    assert eligible(dict(source_timestamp='2026-10-19T15:30:00+05:30',source_timestamp_verified=True),'2026-10-19T15:30:00+05:30')


def test_old_undated_features_cannot_bypass_new_gate():
    x=enrich(dict(ofs_ratio=.8,gmp_pct=2,log_qib=1,log_hni=2,log_retail=3),{},'2026-10-19T17:00:00+05:30')
    assert 'ofs_ratio' not in x
    assert x['log_min_category_mult']==1


def test_offer_ratio_and_brlm_row_do_not_capture_registrar():
    html='''<table><tr><td>Fresh Issue</td><td>₹80 Cr</td></tr><tr><td>Offer for Sale</td><td>₹20 Cr</td></tr>
    <tr><td>Book Running Lead Manager</td><td><a>Lead One Ltd</a></td></tr>
    <tr><td>Registrar</td><td><a>Registrar Ltd</a></td></tr><tr><td>Lot Size</td><td>50 shares</td></tr></table>'''
    x=parse(html,300)
    assert x['ofs_ratio']==.2 and x['min_retail_application_rupees']==15000
    assert x['brlm_names']==['Lead One Ltd']


def test_brlm_prior_does_not_see_current_or_future_listing():
    rows=[dict(ipo_id='1',open_date='2025-01-01',brlm_names='["Lead"]'),dict(ipo_id='2',open_date='2025-02-01',brlm_names='["Lead"]'),dict(ipo_id='3',open_date='2025-02-15',brlm_names='["Lead"]')]
    outcomes=pd.DataFrame([dict(ipo_id=1,status='verified_daily_exchange_archive',listing_date='2025-01-10',exchange_open=110,issue_price=100),dict(ipo_id=2,status='verified_daily_exchange_archive',listing_date='2025-02-20',exchange_open=50,issue_price=100)])
    result=brlm_prior_statistics(rows,outcomes)
    assert result[2]['brlm_prior_deals']==1
    assert result[2]['brlm_prior_listing_win_rate']==1


def test_nni_application_sizes_are_price_specific_not_hardcoded_14_lots():
    x=category_lots(271,55)
    assert x['snii_min_lots']*x['lot_cost']>200000
    assert (x['snii_min_lots']-1)*x['lot_cost']<=200000
    assert x['bnii_min_lots']*x['lot_cost']>1000000


def test_multiple_proxy_is_not_exact_allotment_or_automatic_apply():
    x=assess(category='retail',cap_price=300,lot_size=50,expected_return_pct=10,subscription_mult=10,
             probability_positive=.9,precision_hurdle=.8,allow_multiple_approximation=True)
    assert x['allotment_probability']==.1 and x['EU_rupees']==150
    assert x['policy']=='WATCH' and 'not_exact' in x['allotment_method']


def test_financing_on_blocked_capital_can_make_positive_gross_return_unprofitable():
    x=assess(category='bnii',cap_price=300,lot_size=50,expected_return_pct=1,subscription_mult=100,
             application_lots=70,financing_apr_pct=15,blocked_days=7,funded_fraction=1,
             probability_positive=.99,precision_hurdle=.8,allow_multiple_approximation=True)
    assert x['financing_cost_rupees']>0 and x['EU_rupees']<0 and x['policy']=='SKIP'
    assert x['bnii_funded_breakeven_return_pct']>1


def test_missing_flow_and_unlock_status_never_imply_accumulation():
    x=evaluate(dict(close=120,issue_price=100,day1_vwap=101,sma_10=115,sma_20=110,filing_sentiment_score=.5))
    assert x['action']=='HOLD'
    x=evaluate(dict(close=120,issue_price=100,day1_vwap=101,sma_10=115,sma_20=110,filing_sentiment_score=.5,delivery_accumulation_ratio=1.3,in_unlock_danger_zone=False))
    assert x['action']=='ACCUMULATE'


def test_after_close_news_does_not_enter_same_close_decision():
    prices=pd.DataFrame(dict(date=pd.date_range('2026-09-01',periods=31),close=[100]*31))
    filing=dict(source_timestamp='2026-10-01T15:31:00+05:30',source_timestamp_verified=True,filing_sentiment_score=-1,hard_governance_red_flag=True)
    x=assemble(prices,decision_timestamp='2026-10-01T15:30:00+05:30',issue_price=100,filings=[filing])
    assert x['coverage']['filings_admitted']==0 and 'hard_governance_red_flag' not in x


def test_announcement_iso_month_day_never_swapped():
    row=normalize_filing(dict(disseminated_at_ist='2026-03-04T15:29:00',subject='Auditor resignation'),dict(ipo_id=1,symbol='TEST',retrieved_at_utc='2026-10-25T00:00:00Z'))
    assert row['source_timestamp']=='2026-03-04T15:29:00+05:30'
    assert row['hard_governance_red_flag'] is True
    assert flags('Routine tax notice')['hard_governance_red_flag'] is False


def test_bulk_trade_timestamp_not_treated_as_publication_or_institutional_identity():
    row=normalize_deals(dict(data=[dict(BD_DT_DATE='30-DEC-2025',BD_SYMBOL='TEST',BD_CLIENT_NAME='Example Capital',BD_QTY_TRD=100,BD_BUY_SELL='BUY')]),dict(ipo_id=1,symbol='TEST'))[0]
    assert row['trade_date']=='2025-12-30' and row['source_timestamp'] is None
    assert row['institutional_identity_verified'] is False


def test_first_observed_time_can_support_future_decision_but_not_backdating():
    row=dict(source_timestamp=None,available_by_timestamp='2026-10-25T16:00:00+05:30',observation_verified=True)
    assert eligible(row,'2026-10-25T16:01:00+05:30')
    assert not eligible(row,'2026-10-24T16:01:00+05:30')


def test_bounded_quotas_are_not_exact_reservation_weights():
    x=parse('<table><tr><td>QIB</td><td>Not more than 50%</td></tr><tr><td>NII</td><td>15%</td></tr><tr><td>Retail</td><td>35%</td></tr></table>')
    assert x['qib_quota_pct']==50 and x['quota_weights_exact'] is False


def test_bse_outcome_has_exchange_neutral_price_names_and_is_immutable(tmp_path,monkeypatch):
    from services.bse_listing_outcomes import record_bse_outcomes
    from ml import audit_module1_listing_opens
    from datetime import date
    monkeypatch.setattr(audit_module1_listing_opens,'bse_row',lambda *args:dict(exchange_open=105,exchange_close=106,archive_sha256='abc',source_url='https://www.bseindia.com/archive'))
    db=tmp_path/'snapshots.db';cache={'https://www.chittorgarh.com/ipo/nse/123/':dict(company_name='National Stock Exchange of India',listing_date='2026-09-24',bse_code=544937,issue_price=100)}
    result=record_bse_outcomes(cache,db_path=db,as_of=date(2026,10,25))
    assert result['saved']==['chitto_123']
    with sqlite3.connect(db) as conn:
        conn.row_factory=sqlite3.Row;row=dict(conn.execute('SELECT * FROM listing_outcome_bse_v1').fetchone())
        assert row['exchange']=='BSE' and row['exchange_open_price']==105 and 'nse_open_price' not in row
        with pytest.raises(sqlite3.IntegrityError):conn.execute('DELETE FROM listing_outcome_bse_v1')


def test_blocked_bse_archive_is_not_retried_on_every_page_refresh(tmp_path,monkeypatch):
    from services.bse_listing_outcomes import record_bse_outcomes
    from ml import audit_module1_listing_opens
    from datetime import date
    calls=[]
    def blocked(*args):calls.append(1);raise ValueError('HTTP 403')
    monkeypatch.setattr(audit_module1_listing_opens,'bse_row',blocked)
    db=tmp_path/'snapshots.db';cache={'https://www.chittorgarh.com/ipo/nse/123/':dict(company_name='National Stock Exchange of India',listing_date='2026-09-24',bse_code=544937,issue_price=100)}
    for _ in range(2):record_bse_outcomes(cache,db_path=db,as_of=date(2026,10,25))
    assert len(calls)==1


def test_gmp_comparison_uses_its_frozen_threshold():
    from ml.run_full_enrichment_audit import metrics
    x=metrics([0,1],[.60,.70],decision_threshold=.625)
    assert x['correct']==2 and x['decision_threshold']==.625


def test_missing_delivery_percentage_does_not_crash_monitor():
    prices=pd.DataFrame(dict(date=pd.date_range('2026-09-01',periods=31),close=[100]*31))
    rows=[dict(session_date='2026-09-30',source_timestamp='2026-09-30T15:00:00+05:30',source_timestamp_verified=True,delivery_pct=None,deliverable_qty=100)]*5
    x=assemble(prices,decision_timestamp='2026-10-01T15:30:00+05:30',issue_price=100,delivery=rows)
    assert x['coverage']['delivery_admitted']==0
