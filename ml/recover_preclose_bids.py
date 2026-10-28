"""Audit/reconstruct closed IPOs without creating backdated live predictions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from services.historical_bid_service import recover_historical_bid_forecast, resolve_official_issue, name_key
from scrapers.live_ipo_scraper import load_scraped_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--company', action='append', help='Exact issuer name; repeatable. Default: all closed IPOs.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/preclose_bid_recovery_coverage.json')
    args = parser.parse_args()
    response = httpx.get(args.api + '/api/v1/live-ipo-calendar', timeout=60)
    response.raise_for_status()
    # Keep the original 15-IPO audit cohort even when a former CLOSED record
    # moves to LISTED. Do not silently drop companies as outcomes arrive.
    original = json.loads((ROOT / 'output/preclose_bid_recovery_coverage.json').read_text())
    wanted = {name_key(row['company_name']): row for row in original['ipos']}
    current = {name_key(row['company_name']): row for row in response.json()['calendar']}
    for url, row in load_scraped_cache().items():
        if name_key(row['company_name']) in wanted:
            current[name_key(row['company_name'])] = {**row, 'url': url}
    items = [resolve_official_issue(current.get(key, row)) for key, row in wanted.items()]
    rows = []
    for item in items:
        company = item['company_name']
        if args.company and company not in args.company:
            continue
        try:
            result = recover_historical_bid_forecast(item, force=args.force)
            row = {'company_name': company, 'date_range': item.get('date_range'),
                   'status': 'reconstructed' if result else 'no_eligible_evidence',
                   'listing_date': item.get('listing_date'), 'issue_price': item.get('issue_price'),
                   'nse_symbol': item.get('nse_symbol'), 'url': item.get('url'),
                   'official_issuer_name': item.get('official_issuer_name'),
                   'official_offer_evidence': item.get('official_offer_evidence')}
            if result:
                row.update({key: result[key] for key in ['observed_on', 'source_reported_at_ist',
                    'computed_at_ist', 'positive_listing_probability', 'inputs', 'sources',
                    'eligible_for_prospective_accuracy', 'model_version', 'model_sha256']})
                row['closing_day_timestamp_found'] = bool(result.get('source_reported_at_ist')
                    and result['observed_on'] == result['inputs']['application_closing_date'])
        except Exception as exc:
            row = {'company_name': company, 'status': 'error', 'error': str(exc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {'computed_at_ist': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
              'scope': 'closed IPOs in current calendar', 'cutoff': 'strictly before 17:00 IST on own closing date',
              'historical_reconstruction_not_live_forecast': True, 'ipos': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Saved coverage: {args.output}', flush=True)


if __name__ == '__main__':
    main()
