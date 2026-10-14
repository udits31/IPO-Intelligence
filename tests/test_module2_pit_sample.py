"""Actual NSE PIT responses, no manufactured institutional trades."""
import gzip
import hashlib
import json

import pandas as pd

from ml.collect_module2_pit_sample import ROOT, REPORT, normalize


def records():
    return json.loads(REPORT.read_text(encoding='utf-8'))['records']


def test_pilot_raw_responses_are_hash_bound_and_wrong_modes_are_not_sales():
    rows = records()
    assert len(rows) == 5
    for record in rows:
        raw = gzip.decompress((ROOT/record['raw_path']).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record['raw_sha256']
        assert len(json.loads(raw)['data']) == record['response_rows']
    cms = next(r for r in rows if r['symbol'] == 'CMSINFO')
    assert cms['response_rows'] == 5
    assert all(t['status'] == 'not_explicit_open_market_buy_or_sell' for t in cms['transactions'])


def test_actual_medplus_purchase_is_not_invented_promoter_or_anchor_evidence():
    med = next(r for r in records() if r['symbol'] == 'MEDPLUS')
    item = med['transactions'][0]
    assert item['quantity'] == 5000
    assert item['transaction_type'] == 'Buy'
    assert item['status'] == 'dated_open_market_transaction_role_unverified'
    assert item['promoter_explicit'] is None


def test_actual_transaction_is_unavailable_before_its_publication():
    med = next(r for r in records() if r['symbol'] == 'MEDPLUS')
    raw = json.loads(gzip.decompress((ROOT/med['raw_path']).read_bytes()))['data'][0]
    listing = pd.Timestamp('2021-12-23')
    # Actual transaction date was 23 December; actual publication was after
    # close on 24 December. Trade date cannot be used as the public clock.
    result = normalize(raw, 'MEDPLUS', listing, listing + pd.Timedelta(hours=15, minutes=30))
    assert result['status'] == 'not_available_at_decision'
