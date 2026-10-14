"""Exercise the actual analysis route before, between sessions and after cutoff."""

from datetime import datetime
from functools import partial
import gzip
import hashlib
import sqlite3
from unittest.mock import patch
from zoneinfo import ZoneInfo

import api.main as api
from scrapers import live_ipo_scraper as live
from services import snapshot_service as snapshots


IST = ZoneInfo('Asia/Kolkata')
URL = 'https://www.chittorgarh.com/ipo/application-test/987654/'


def analyze_at(monkeypatch, db_path, clock, observed, *, category='OPEN', qib=4.2, save_forecast=True):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.astimezone(tz) if tz else clock.replace(tzinfo=None)

    data = {
        'url': URL, 'company_name': 'Application Test', 'category': category,
        'date_range': '13 - 15 Oct', 'listing_date': '2026-10-21',
        'issue_price': 100, 'issue_size_cr': 150,
        'subscription_source': live.NSE_SUBSCRIPTION_SOURCE,
        'subscription_source_family': 'nse_exchange',
        'subscription_observed': True, 'subscription_observed_at': observed.isoformat(),
        'subscription_source_displayed_at': observed.isoformat(),
        'qib_subscription': qib, 'hni_subscription': 2.1, 'retail_subscription': 1.2,
    }
    monkeypatch.setattr(api, 'datetime', Clock)
    monkeypatch.setattr(snapshots, 'datetime', Clock)
    monkeypatch.setattr(api.resolver, 'resolve', lambda _: {
        'success': True, 'source': 'Dated NSE observation', 'data': data})
    monkeypatch.setattr(api, 'fetch_live_macro_regime', lambda: {})
    monkeypatch.setattr(api, 'generate_ai_investment_memo', lambda **_: {})
    monkeypatch.setattr(live, 'load_scraped_cache', lambda: {URL: data})
    monkeypatch.setattr(api, 'record_automatic_decision', partial(
        snapshots.record_automatic_decision, db_path=str(db_path)))
    monkeypatch.setattr(api, 'get_latest_application_forecast', partial(
        snapshots.get_latest_application_forecast, db_path=str(db_path)))
    monkeypatch.setattr(api, 'get_last_complete_bid_capture', partial(
        snapshots.get_last_complete_bid_capture, db_path=str(db_path)))
    return api.scrape_and_evaluate_ipo(api.ScrapeAndEvaluateRequest(
        url_or_name='Application Test', fetch_news=False, save_forecast=save_forecast))


def test_background_page_refresh_does_not_create_manual_forecasts(tmp_path,monkeypatch):
    from services import prospective_module1
    monkeypatch.setattr(prospective_module1,'analysis_card',lambda *a,**k:None)
    db=tmp_path/'snapshots.db';snapshots.init_db(str(db))
    clock=datetime(2026,10,15,16,45,tzinfo=IST)
    result=analyze_at(monkeypatch,db,clock,clock.replace(minute=40),save_forecast=False)
    assert result['success']
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(*) FROM automatic_decision_v1').fetchone()[0]==0


def test_run_analysis_saves_before_close_and_locks_without_later_rewrite(tmp_path, monkeypatch):
    db = tmp_path / 'snapshots.db'
    snapshots.init_db(str(db))
    clock = datetime(2026, 10, 15, 16, 45, tzinfo=IST)
    before = analyze_at(monkeypatch, db, clock, clock.replace(minute=40))
    forecast = before['application_forecast']
    assert forecast is not None and not forecast['is_locked']
    assert forecast['recorded_at_ist'] == clock.isoformat(timespec='seconds')
    assert 0 < forecast['positive_listing_probability'] < 1
    assert forecast['inputs']['qib_subscription'] == 4.2
    assert before['application_forecast_window']['is_open']

    after = analyze_at(monkeypatch, db, clock.replace(hour=17, minute=0),
                       clock.replace(hour=17, minute=0), category='CLOSED', qib=100)
    locked = after['application_forecast']
    assert locked['is_locked']
    assert locked['id'] == forecast['id']
    assert locked['positive_listing_probability'] == forecast['positive_listing_probability']
    assert locked['inputs']['qib_subscription'] == 4.2
    assert not after['application_forecast_window']['is_open']


def test_between_sessions_uses_last_published_bids_and_the_real_run_time(tmp_path, monkeypatch):
    db = tmp_path / 'snapshots.db'
    snapshots.init_db(str(db))
    clock = datetime(2026, 10, 14, 20, 0, tzinfo=IST)
    published = clock.replace(hour=16, minute=30)
    result = analyze_at(monkeypatch, db, clock, published)
    saved = result['application_forecast']
    assert saved is not None and not saved['is_locked']
    assert saved['source_asof_ist'] == published.isoformat()
    assert saved['recorded_at_ist'] == clock.isoformat(timespec='seconds')
    assert saved['inputs']['snapshot_is_current'] is False
    assert saved['inputs']['snapshot_age_seconds'] == 3.5 * 3600


def test_future_bid_table_cannot_create_an_application_forecast(tmp_path, monkeypatch):
    db = tmp_path / 'snapshots.db'
    snapshots.init_db(str(db))
    clock = datetime(2026, 10, 15, 16, 45, tzinfo=IST)
    result = analyze_at(monkeypatch, db, clock, clock.replace(hour=19))
    assert result['application_forecast'] is None


def test_open_offer_refresh_keeps_complete_book_when_nse_is_empty(monkeypatch):
    item = {'company_name': 'Application Test', 'category': 'OPEN',
            'date_range': '13 - 15 Oct', 'subscription_source': live.NSE_SUBSCRIPTION_SOURCE,
            'subscription_observed': True, 'qib_subscription': 4.2,
            'hni_subscription': 2.1, 'retail_subscription': 1.2,
            'subscription_observed_at': '2026-10-14T16:30:00+05:30'}
    monkeypatch.setattr(live, '_subscription_window_open', lambda: True)
    with patch('services.nse_subscription_service.fetch_nse_subscription_evidence',
               return_value={'status': 'no_subscription_table'}) as fetch:
        result = live.refresh_cached_subscriptions(URL, item, force=True)
    fetch.assert_called_once()
    assert result['qib_subscription'] == 4.2
    assert result['subscription_observed_at'] == item['subscription_observed_at']


def test_all_supported_date_formats_share_final_day_cutoff():
    clock = datetime(2026, 10, 15, 16, 59, tzinfo=IST)
    for dates in ['13 - 15 Oct', '2026-10-13 to 2026-10-15', 'Oct 13-15, 2026']:
        item = {'date_range': dates, 'category': 'OPEN'}
        assert snapshots.application_forecast_window(item, clock)['is_open']
        assert not snapshots.application_forecast_window(
            item, clock.replace(hour=17, minute=0))['is_open']


def test_slow_cross_check_keeps_the_forecast_that_was_ready_before_cutoff(tmp_path, monkeypatch):
    db = tmp_path / 'snapshots.db'
    snapshots.init_db(str(db))
    state = {'now': datetime(2026, 10, 15, 16, 55, tzinfo=IST)}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return state['now'].astimezone(tz) if tz else state['now'].replace(tzinfo=None)

    def secondary(_):
        state['now'] = state['now'].replace(minute=56)
        return {'status': 'verified', 'fetched_at_ist': state['now'].isoformat(),
                'raw_html': 'published before cutoff',
                'raw_sha256': hashlib.sha256(b'published before cutoff').hexdigest(),
                'values': {'qib_subscription': 4.2, 'hni_subscription': 2.1,
                           'retail_subscription': 1.2}}

    def exchange(*args, **kwargs):
        state['now'] = state['now'].replace(hour=17, minute=1)
        return {'status': 'verified', 'fetched_at_ist': state['now'].isoformat(),
                'source_displayed_at_ist': state['now'].isoformat(),
                'values': {'qib_subscription': 99, 'hni_subscription': 99,
                           'retail_subscription': 99}}

    monkeypatch.setattr(snapshots, 'datetime', Clock)
    monkeypatch.setattr(live, 'fetch_subscription_evidence', secondary)
    monkeypatch.setattr('services.nse_subscription_service.fetch_nse_subscription_evidence', exchange)
    cache = {URL: {'company_name': 'Application Test', 'category': 'OPEN',
                   'date_range': '13 - 15 Oct', 'source_observed': True}}
    snapshots.capture_active_subscriptions(now_ist=state['now'], cache=cache, db_path=str(db))
    saved = snapshots.get_latest_application_forecast('Application Test', db_path=str(db))
    assert saved is not None
    assert saved['recorded_at_ist'] == '2026-10-15T16:56:00+05:30'
    assert saved['inputs']['qib_subscription'] == 4.2
    assert saved['source_family'] == 'chittorgarh_investorgain'


def test_last_complete_capture_requires_matching_raw_evidence_and_pre_cutoff_time(tmp_path):
    db = tmp_path / 'snapshots.db'
    snapshots.init_db(str(db))
    with sqlite3.connect(db) as conn:
        for clock, raw_hash in [('16:40:00', hashlib.sha256(b'good').hexdigest()),
                                ('16:45:00', 'corrupt'),
                                ('17:01:00', hashlib.sha256(b'good').hexdigest())]:
            stamp = f'2026-10-15T{clock}+05:30'
            conn.execute('''INSERT INTO subscription_capture_v2 (
                ipo_id,company_name,ipo_url,attempted_at_ist,fetched_at_ist,slot_key,
                closing_date,snapshot_stage,status,qib_x,nii_x,retail_x,
                raw_html_gzip,raw_sha256,source_family,capture_phase)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                'chitto_987654', 'Application Test', URL, stamp, stamp, clock,
                '2026-10-15', 'FINAL_CLOSE', 'verified', 4.2, 2.1, 1.2,
                gzip.compress(b'good'), raw_hash, 'nse_exchange', 'pre_deadline'))
    capture = snapshots.get_last_complete_bid_capture('Application Test',
        now_ist=datetime(2026, 10, 15, 17, 5, tzinfo=IST), db_path=str(db))
    assert capture['fetched_at_ist'] == '2026-10-15T16:40:00+05:30'


def test_last_good_book_preserves_publisher_clock_instead_of_refetch_time(tmp_path):
    db = tmp_path / 'snapshots.db'
    snapshots.init_db(str(db))
    raw = b'{"updateTime":"Updated as on 15-Oct-2026 14:00:00"}'
    stamp = '2026-10-15T16:45:00+05:30'
    with sqlite3.connect(db) as conn:
        conn.execute('''INSERT INTO subscription_capture_v2 (
            ipo_id,company_name,ipo_url,attempted_at_ist,fetched_at_ist,slot_key,
            closing_date,snapshot_stage,status,qib_x,nii_x,retail_x,
            raw_html_gzip,raw_sha256,source_family,capture_phase)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            'clock_test', 'Clock Test', URL, stamp, stamp, '2026-10-15_1645',
            '2026-10-15', 'FINAL_CLOSE', 'verified', 4.2, 2.1, 1.2,
            gzip.compress(raw), hashlib.sha256(raw).hexdigest(), 'nse_exchange', 'pre_deadline'))
    capture = snapshots.get_last_complete_bid_capture('Clock Test',
        now_ist=datetime(2026, 10, 15, 16, 50, tzinfo=IST), db_path=str(db))
    assert capture['fetched_at_ist'] == stamp
    assert capture['source_asof_ist'] == '2026-10-15T14:00:00+05:30'
    assert capture['source_published_at_ist'] == capture['source_asof_ist']
