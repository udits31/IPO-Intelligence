"""Recover real NSE BE-series closes omitted by the EQ-only delivery collector.

Delivery fields may be absent in an otherwise valid exchange price row. This
price-only repair never invents delivery quantities, prices, or trading dates.
Existing datasets and live rules are preserved; outputs are separate artifacts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ml.build_module2_nse_development_prices import reconcile
from ml.fetch_module2_nse_delivery import PAGE, ENDPOINT, HEADERS, cohort_requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/audit/module2_development_prices_2021_2024.csv'
EXISTING = ROOT/'data/audit/module2_nse_delivery_development_full_window.csv'
RAW = ROOT/'data/archive/module2_nse_be_closes'
REPAIR = ROOT/'data/audit/module2_nse_development_be_closes.csv'
OUTPUT = ROOT/'data/audit/module2_development_prices_2021_2024_nse_with_be.csv'
REPORT = ROOT/'output/module2_nse_be_recovery_20261021.json'


def parse_prices(payload, *, ipo_id, symbol, first, last, source_url, digest, series='BE'):
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        raise ValueError('Missing NSE data array')
    rows = []
    for record in payload['data']:
        if record.get('CH_SERIES') != series or str(record.get('CH_SYMBOL', '')).upper() != symbol.upper():
            continue
        date = pd.to_datetime(record.get('mTIMESTAMP'), format='%d-%b-%Y', errors='coerce')
        price = pd.to_numeric(record.get('CH_CLOSING_PRICE'), errors='coerce')
        volume = pd.to_numeric(record.get('CH_TOT_TRADED_QTY'), errors='coerce')
        if (pd.isna(date) or not first <= date <= last or pd.isna(price) or not np.isfinite(price)
                or price <= 0 or pd.isna(volume) or not np.isfinite(volume) or volume <= 0):
            continue
        rows.append({'ipo_id': ipo_id, 'symbol': symbol, 'session_date': date.strftime('%Y-%m-%d'),
                     'nse_close': float(price), 'series': series, 'traded_qty': int(volume),
                     'source_url': source_url, 'source_sha256': digest,
                     'delivery_observation_status': 'not_used_in_price_only_repair'})
    result = pd.DataFrame(rows)
    if not result.empty and result.duplicated(['ipo_id', 'session_date']).any():
        raise ValueError('Ambiguous duplicate NSE BE close')
    return rows


def main():
    base = pd.read_csv(BASE, parse_dates=['date'])
    original = pd.read_csv(EXISTING)
    canonical = pd.read_csv(ROOT/'data/canonical_listed_ipos.csv')
    planned = cohort_requests(base, canonical, through_index=61)
    first = base.sort_values(['ipo_id', 'date']).groupby('ipo_id').head(62)
    available = set(zip(original.ipo_id.astype(int), original.session_date.astype(str)))
    missing = {(int(r.ipo_id), r.date.strftime('%Y-%m-%d')) for r in first.itertuples()
               if (int(r.ipo_id), r.date.strftime('%Y-%m-%d')) not in available}
    wanted = [p for p in planned if any(key[0] == p['ipo_id'] for key in missing)]
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    landing = session.get(PAGE, headers=HEADERS, timeout=20)
    landing.raise_for_status()
    observations, attempts = [], []
    for i, request in enumerate(wanted, 1):
        path = RAW/f"{request['ipo_id']}_{request['from']:%Y%m%d}_{request['to']:%Y%m%d}.json.gz"
        params = {'from': request['from'].strftime('%d-%m-%Y'), 'to': request['to'].strftime('%d-%m-%Y'),
                  'symbol': request['symbol'], 'type': 'priceVolumeDeliverable', 'series': 'BE'}
        url = requests.Request('GET', ENDPOINT, params=params).prepare().url
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            if path.exists():
                with gzip.open(path, 'rb') as stream:
                    raw = stream.read()
            else:
                response = session.get(ENDPOINT, params=params, headers=HEADERS, timeout=25)
                if response.status_code in (403, 429):
                    attempts.append({'ipo_id': request['ipo_id'], 'status': 'access_or_rate_limit',
                                     'http_status': response.status_code})
                    break
                response.raise_for_status()
                raw = response.content
                json.loads(raw)  # Archive only actual JSON responses.
                with gzip.open(path, 'wb') as stream:
                    stream.write(raw)
            digest = hashlib.sha256(raw).hexdigest()
            rows = parse_prices(json.loads(raw), ipo_id=request['ipo_id'], symbol=request['symbol'],
                                first=request['from'], last=request['to'], source_url=url, digest=digest)
            recovered = [r for r in rows if (r['ipo_id'], r['session_date']) in missing]
            observations.extend(recovered)
            attempts.append({'ipo_id': request['ipo_id'], 'symbol': request['symbol'],
                             'status': 'verified_response', 'source_url': url, 'raw_sha256': digest,
                             'retrieved_at_utc': stamp, 'be_price_rows': len(rows),
                             'missing_sessions_recovered': len(recovered)})
            print(f"{i}/{len(wanted)} {request['symbol']}: {len(recovered)} missing sessions recovered", flush=True)
        except (requests.RequestException, ValueError, TypeError) as exc:
            attempts.append({'ipo_id': request['ipo_id'], 'status': 'unavailable', 'error': str(exc)[:200]})
            print(f"{i}/{len(wanted)} {request['symbol']}: unavailable ({type(exc).__name__})", flush=True)
        time.sleep(.45)
    columns = ['ipo_id', 'symbol', 'session_date', 'nse_close', 'series', 'traded_qty',
               'source_url', 'source_sha256', 'delivery_observation_status']
    repair = pd.DataFrame(observations, columns=columns)
    repair.to_csv(REPAIR, index=False)
    # Only price columns are joined. No missing delivery field becomes 0.
    combined = pd.concat([original[['ipo_id','session_date','symbol','nse_close']],
                          repair[['ipo_id','session_date','symbol','nse_close']]], ignore_index=True)
    if combined.duplicated(['ipo_id', 'session_date']).any():
        raise ValueError('Repair would overwrite an existing exchange observation')
    fixed, audit = reconcile(base, combined)
    fixed['price_basis'] = 'Official NSE security-wise archive unadjusted Close (EQ/BE)'
    fixed.to_csv(OUTPUT, index=False)
    old, old_audit = reconcile(base, original)
    report = {'source_page': PAGE, 'raw_price_rows_recovered': len(repair),
              'ipo_histories_before': old_audit['eligible_ipo_count'],
              'ipo_histories_after': audit['eligible_ipo_count'],
              'recovered_complete_ipo_ids': sorted(set(fixed.ipo_id)-set(old.ipo_id)),
              'attempts': attempts, 'reconciliation': audit,
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (BASE, EXISTING, REPAIR)},
              'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              'limitations': ['BE delivery fields can be unavailable; this repair recovers stock closes only.',
                              'Benchmark histories retain the original dated secondary provenance.',
                              'Incomplete late-2024 base histories and historical symbol changes remain separate coverage gaps.',
                              'Neither model, paper nor previously saved forecasts is modified.']}
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('raw_price_rows_recovered','ipo_histories_before',
        'ipo_histories_after','recovered_complete_ipo_ids')}, indent=2))


if __name__ == '__main__':
    main()
