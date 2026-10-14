from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import core.attribution_engine as attribution_module
from core.attribution_engine import AttributionEngine
from core.monitor_engine import HoldSellMonitor


def test_unfinished_and_future_daily_bars_do_not_drive_a_close_rule(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 21, 14, 0, tzinfo=ZoneInfo('Asia/Kolkata'))
    monkeypatch.setattr(attribution_module, 'datetime', Clock)
    from services.trendlyne_service import trendlyne_service
    monkeypatch.setattr(trendlyne_service, 'fetch_live_stock_data', lambda *a, **k: None)
    prices = pd.DataFrame({'ipo_id': [1, 1, 1], 'date': pd.to_datetime(['2026-10-20', '2026-10-21', '2026-10-22']),
                           'close_price_adjusted': [100., 999., 9999.], 'nifty50_close': [100, 101, 102],
                           'sector_index_close': [100, 101, 102]})
    ipos = pd.DataFrame([{'id': 1, 'company_name': 'Test', 'sector': 'Technology', 'nse_symbol': 'TEST',
                          'listing_date': '2026-10-20', 'issue_price': 100.,
                          'anchor_lockin_50pct_date': '2026-11-19', 'anchor_lockin_full_date': '2027-01-18'}])
    engine = AttributionEngine(prices_frame=prices, ipos_frame=ipos, prefer_supplied_prices=True)
    result = engine.compute_attribution(1)
    assert result['evaluation_date'] == '2026-10-20'
    assert result['current_price_adjusted'] == 100.
    assert result['price_age_calendar_days'] == 1


def test_failed_price_refresh_retains_reference_but_cannot_issue_current_advice(monkeypatch):
    monitor = HoldSellMonitor()
    monkeypatch.setattr(monitor.attribution_engine, 'compute_attribution', lambda *a, **k: {
        'price_is_stale': True, 'company_name': 'Test', 'evaluation_date': '2026-09-01',
        'current_price_adjusted': 80., 'price_source': 'Local market history'})
    result = monitor.evaluate_holding(1)
    assert result['status'] == 'DATA_STALE'
    assert result['current_price_adjusted'] == 80.
    assert 'verdict' not in result


def test_news_source_failure_and_last_success_reach_the_monitor_ui(monkeypatch):
    monitor = HoldSellMonitor()
    attr = monitor.attribution_engine.compute_attribution(364, as_of_date='2024-03-15')
    monkeypatch.setattr(monitor.attribution_engine, 'compute_attribution', lambda *a, **k: attr)
    monkeypatch.setattr(monitor.news_service, 'get_company_news', lambda *a, **k: {
        'articles': [], 'sentiment_score': None, 'status': 'news_source_unavailable',
        'freshness': 'stale', 'error': 'source_error', 'checked_at': '2026-10-21T10:00:00+00:00',
        'last_success_at': '2026-10-21T09:00:00+00:00'})
    result = monitor.evaluate_holding(364)
    assert result['news_sentiment']['score'] is None
    assert result['news_sentiment']['freshness'] == 'stale'
    assert result['news_sentiment']['last_success_at'] == '2026-10-21T09:00:00+00:00'
    assert result['news_changed_decision'] is False


def test_unverified_anchor_calendar_does_not_claim_a_legal_selling_ban():
    monitor = HoldSellMonitor()
    result = monitor.evaluate_holding(364, as_of_date='2023-12-05', fetch_news=False)
    assert result['lockin_status']['dates_verified'] is False
    text = ' '.join(result['key_reasons']) + ' ' + result['thesis_breakdown'][2]['explanation']
    assert 'unverified' in text.lower() or 'not been matched' in text.lower()
    assert 'zero institutional selling' not in text.lower()
    assert result['verdict'] != 'ACCUMULATE / STRONG BUY'
