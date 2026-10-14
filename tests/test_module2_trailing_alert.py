"""Actual archived HD Fire prices reproduce the disputed trailing alert."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from core.attribution_engine import parse_yahoo_chart
from core.monitor_engine import HoldSellMonitor


def test_actual_hd_fire_pullback_is_not_an_automatic_sell():
    root = Path(__file__).resolve().parents[1]
    sha = 'fc3a3c05098d6f58616130f43bd9d0b69629f7bb5fd75d64d40043516936bc85'
    raw = gzip.decompress((root/'data/archive/yahoo_chart'/f'{sha}.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == sha
    prices = parse_yahoo_chart(json.loads(raw), 'HDFIRE.NS').rename(columns={'close': 'close_price_adjusted'})
    cache = json.loads((root/'data/live_scraped_cache.json').read_text(encoding='utf-8'))
    item = cache['https://www.chittorgarh.com/ipo/hd-fire-protect-ipo/2628/']
    listed = pd.Timestamp(item['listing_date'])
    ipo = pd.DataFrame([dict(id=2628, company_name=item['company_name'], sector=item['sector'],
        nse_symbol=item['nse_symbol'], issue_price=item['issue_price'], listing_date=item['listing_date'],
        anchor_lockin_50pct_date=(listed+pd.Timedelta(days=30)).date().isoformat(),
        anchor_lockin_full_date=(listed+pd.Timedelta(days=90)).date().isoformat(),
        anchor_lockin_dates_verified=False)])
    prices['ipo_id'] = 2628
    prices['price_basis'] = 'Unadjusted daily close'
    prices['price_source'] = 'Yahoo Finance (NSE)'
    prices['sector_index_close'] = float('nan')
    prices['nifty50_close'] = float('nan')
    monitor = HoldSellMonitor(prices_frame=prices, ipos_frame=ipo, prefer_supplied_prices=True)
    result = monitor.evaluate_holding(2628, as_of_date='2026-10-22', fetch_news=False)
    assert result['current_price_adjusted'] == pytest.approx(276.7, abs=.001)
    assert result['verdict'] == 'WATCH / PRICE PULLBACK'
    assert result['decision_reason']['code'] == 'unvalidated_trailing_alert'
    assert result['decision_reason']['trailing_reference_price'] == 282.21
    assert result['decision_reason']['automatic_exit_supported'] is False
    assert 'manually chosen' in result['decision_reason']['summary']
    assert 'not an automatic sell' in result['decision_reason']['summary']
    assert result['decision_reason']['history_note']
    assert 'NO AUTOMATIC EXIT' in result['actionable_playbook']['action']
    assert 'Exit position immediately' not in result['thesis_breakdown'][3]['explanation']


@pytest.mark.parametrize('ipo_id', [364, 659, 661])
def test_existing_real_monitors_expose_primary_explanation(ipo_id):
    monitor = HoldSellMonitor()
    date = '2024-03-15' if ipo_id == 364 else '2026-10-20'
    result = monitor.evaluate_holding(ipo_id, as_of_date=date, fetch_news=False)
    assert 'verdict' in result
    assert result['decision_reason']['summary']
    assert result['decision_reason']['code']
    assert 'Fixed risk rules' in result['decision_reason']['method']
