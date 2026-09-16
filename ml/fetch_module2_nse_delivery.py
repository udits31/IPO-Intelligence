"""Collect actual NSE security-wise price, volume and delivery observations.

Source: NSE Security-wise Archives (Equities), via the endpoint called by
NSE's own security-wise-archives.js. This is research input only. A model
decision made at the close of session t must use delivery data through t-1:
the publication time of session-t delivery is not verified here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PAGE = 'https://www.nseindia.com/report-detail/eq_security'
ENDPOINT = 'https://www.nseindia.com/api/historicalOR/generateSecurityWiseHistoricalData'
COHORTS = {
    'development': ROOT / 'data/audit/module2_development_prices_2021_2024.csv',
    'holdout': ROOT / 'data/audit/module2_holdout_prices_2025_2026_nse_audited.csv',
}
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'application/json,text/plain,*/*',
    'Referer': PAGE,
}
FIELDS = [
    'ipo_id', 'symbol', 'session_date', 'nse_close', 'traded_qty', 'traded_value',
    'total_trades', 'deliverable_qty', 'deliverable_pct', 'source_url', 'retrieved_at_utc',
]


def cohort_requests(prices: pd.DataFrame, canonical: pd.DataFrame,
                    through_index: int = 30) -> list[dict]:
    symbols = canonical.set_index('id')['nse_symbol'].to_dict()
    requests_to_make = []
    for ipo_id, rows in prices.groupby('ipo_id', sort=True):
        symbol = symbols.get(int(ipo_id))
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        ordered = rows.sort_values('date')
        # Default: through the 31st observed close. The separate full-window
        # mode extends through the future evaluation session. Features may
        # only use delivery through the 30th, guarding publication-time leakage.
        if len(ordered) <= through_index:
            continue
        requests_to_make.append({
            'ipo_id': int(ipo_id), 'symbol': symbol.strip(),
            'from': ordered.iloc[0]['date'], 'to': ordered.iloc[through_index]['date'],
            'checkpoint': ordered.iloc[30]['date'],
        })
    return requests_to_make


def parse_response(ipo_id: int, symbol: str, payload: dict, source_url: str,
                   retrieved_at_utc: str) -> list[dict]:
    raw = payload.get('data')
    if not isinstance(raw, list):
        raise ValueError('NSE response has no data array')
    parsed = []
    for item in raw:
        if str(item.get('CH_SYMBOL', '')).upper() != symbol.upper() or item.get('CH_SERIES') != 'EQ':
            continue
        session = pd.to_datetime(item.get('mTIMESTAMP'), format='%d-%b-%Y', errors='coerce')
        if pd.isna(session):
            continue
        close = pd.to_numeric(item.get('CH_CLOSING_PRICE'), errors='coerce')
        traded = pd.to_numeric(item.get('CH_TOT_TRADED_QTY'), errors='coerce')
        delivered = pd.to_numeric(item.get('COP_DELIV_QTY'), errors='coerce')
        percent = pd.to_numeric(item.get('COP_DELIV_PERC'), errors='coerce')
        if (pd.isna(close) or close <= 0 or pd.isna(traded) or traded <= 0
                or pd.isna(delivered) or delivered < 0 or delivered > traded
                or pd.isna(percent) or percent < 0 or percent > 100):
            continue
        parsed.append({
            'ipo_id': ipo_id, 'symbol': symbol, 'session_date': session.date().isoformat(),
            'nse_close': float(close), 'traded_qty': int(traded),
            'traded_value': float(item['CH_TOT_TRADED_VAL']) if item.get('CH_TOT_TRADED_VAL') is not None else None,
            'total_trades': int(item['CH_TOTAL_TRADES']) if item.get('CH_TOTAL_TRADES') is not None else None,
            'deliverable_qty': int(delivered), 'deliverable_pct': float(percent),
            'source_url': source_url, 'retrieved_at_utc': retrieved_at_utc,
        })
    return parsed


def audit_alignment(observed: pd.DataFrame, prices: pd.DataFrame,
                    planned: list[dict]) -> dict:
    """Reject wrong-symbol/partial histories without altering the raw NSE archive."""
    if observed.empty:
        return {'eligible_ipo_ids': [], 'rejected': []}
    reference = prices.sort_values('date').groupby('ipo_id', sort=False).head(31).copy()
    reference['session_date'] = reference['date'].dt.strftime('%Y-%m-%d')
    paired = observed.merge(reference[['ipo_id', 'session_date', 'close_price_adjusted']],
                            on=['ipo_id', 'session_date'], how='inner')
    eligible, rejected = [], []
    for item in planned:
        ipo_id = item['ipo_id']
        matched = paired[paired.ipo_id == ipo_id]
        expected_checkpoint = item['checkpoint'].strftime('%Y-%m-%d')
        if len(matched) < 25:
            rejected.append({'ipo_id': ipo_id, 'symbol': item['symbol'],
                             'reason': 'fewer than 25 of 31 checkpoint sessions matched'})
            continue
        relative_gap = ((matched.nse_close - matched.close_price_adjusted).abs()
                        / matched.close_price_adjusted)
        checkpoint = matched[matched.session_date == expected_checkpoint]
        if checkpoint.empty:
            rejected.append({'ipo_id': ipo_id, 'symbol': item['symbol'],
                             'reason': 'checkpoint close missing from NSE archive'})
            continue
        checkpoint_gap = abs(float(checkpoint.iloc[0].nse_close) /
                             float(checkpoint.iloc[0].close_price_adjusted) - 1)
        if relative_gap.median() > 0.02 or checkpoint_gap > 0.02:
            rejected.append({'ipo_id': ipo_id, 'symbol': item['symbol'],
                             'reason': 'NSE and existing checkpoint price histories disagree by over 2%',
                             'median_close_gap_pct': round(float(relative_gap.median() * 100), 2),
                             'checkpoint_close_gap_pct': round(checkpoint_gap * 100, 2)})
            continue
        eligible.append(ipo_id)
    return {'eligible_ipo_ids': eligible, 'rejected': rejected}


def collect(*, cohort: str, limit: int | None = None, delay_seconds: float = 0.4,
            through_index: int = 30) -> dict:
    prices = pd.read_csv(COHORTS[cohort], parse_dates=['date'])
    canonical = pd.read_csv(ROOT / 'data/canonical_listed_ipos.csv')
    planned = cohort_requests(prices, canonical, through_index=through_index)
    if limit is not None:
        planned = planned[:limit]
    suffix = '_full_window' if through_index == 61 else ''
    destination = ROOT / f'data/audit/module2_nse_delivery_{cohort}{suffix}.csv'
    manifest_path = destination.with_suffix('.json')
    existing = pd.read_csv(destination) if destination.exists() else pd.DataFrame(columns=FIELDS)
    completed = set(existing.ipo_id.astype(int)) if not existing.empty else set()
    session = requests.Session()
    landing = session.get(PAGE, headers=HEADERS, timeout=20)
    landing.raise_for_status()
    if 'Security-wise' not in landing.text:
        raise RuntimeError('NSE report landing page did not return expected content')
    fetched, failures = [], []
    for index, item in enumerate(planned, 1):
        if item['ipo_id'] in completed:
            continue
        params = {
            'from': item['from'].strftime('%d-%m-%Y'),
            'to': item['to'].strftime('%d-%m-%Y'),
            'symbol': item['symbol'], 'type': 'priceVolumeDeliverable', 'series': 'EQ',
        }
        try:
            response = session.get(ENDPOINT, params=params, headers=HEADERS, timeout=25)
            response.raise_for_status()
            found_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
            rows = parse_response(item['ipo_id'], item['symbol'], response.json(),
                                  response.url, found_at)
            if not rows:
                raise ValueError('No valid NSE EQ delivery rows')
            fetched.extend(rows)
            completed.add(item['ipo_id'])
            print(f'{index}/{len(planned)} {item["symbol"]}: {len(rows)} NSE rows', flush=True)
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures.append({'ipo_id': item['ipo_id'], 'symbol': item['symbol'],
                             'error': str(exc)[:250]})
            print(f'{index}/{len(planned)} {item["symbol"]}: unavailable ({exc})', flush=True)
            if isinstance(exc, requests.HTTPError) and response.status_code in (403, 429):
                break  # Do not work around an exchange rate limit or access denial.
        time.sleep(delay_seconds)
    combined = pd.concat([existing, pd.DataFrame(fetched, columns=FIELDS)], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(['ipo_id', 'session_date']).sort_values(['ipo_id', 'session_date'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(destination, index=False)
    alignment = audit_alignment(combined, prices, planned)
    manifest = {
        'source_page': PAGE, 'endpoint': ENDPOINT, 'cohort': cohort,
        'last_requested_session_index': through_index,
        'source_price_snapshot': str(COHORTS[cohort].relative_to(ROOT)),
        'source_price_snapshot_sha256': hashlib.sha256(COHORTS[cohort].read_bytes()).hexdigest(),
        'ipo_count_planned': len(planned), 'ipo_count_with_nse_rows': len(completed),
        'nse_rows': len(combined), 'failures': failures,
        'alignment_audit': alignment,
        'publication_rule': 'Use delivery through the previous completed session at a close-of-day decision; same-session publication time unverified.',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=COHORTS, required=True)
    parser.add_argument('--limit', type=int, help='Pilot first N IPOs; omit for all eligible IPOs')
    parser.add_argument('--delay-seconds', type=float, default=0.4)
    parser.add_argument('--through-outcome', action='store_true',
                        help='Collect through session index 61 into a separate file')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.delay_seconds < 0.4:
        parser.error('Keep at least 0.4 seconds between NSE requests')
    print(json.dumps(collect(cohort=args.cohort, limit=args.limit,
                             delay_seconds=args.delay_seconds,
                             through_index=61 if args.through_outcome else 30), indent=2))
