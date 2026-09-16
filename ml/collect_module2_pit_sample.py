"""Bounded five-IPO pilot of real NSE insider-trading records, not a forecast.

No PDF processing or full-cohort expansion. All records remain archived;
allotments, off-market transfers and unknown categories are not invented
open-market promoter/anchor sales. No live model or replay is modified.
"""
import gzip
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import requests

from ml.calibrate_module2_rules import ROOT
from ml.replay_module2_news_overlay import SOURCES

ENDPOINT = 'https://www.nseindia.com/api/corporates-pit'
ARCHIVE = ROOT/'data/archive/module2_pit_pilot'
REPORT = ROOT/'output/module2_pit_pilot_20261022.json'


def normalize(item, symbol, listing, cutoff):
    record = dict(exchange_record_id=str(item.get('did')), status='unqualified',
                  person_category=item.get('personCategory'), mode=item.get('acqMode'),
                  transaction_type=item.get('tdpTransactionType'),
                  publication_time=item.get('date'), transaction_end=item.get('acqtoDt'))
    if item.get('symbol') != symbol:
        record['status'] = 'wrong_security'
        return record
    published = pd.to_datetime(item.get('date'), format='%d-%b-%Y %H:%M', errors='coerce')
    transaction_end = pd.to_datetime(item.get('acqtoDt'), format='%d-%b-%Y', errors='coerce')
    if pd.isna(published) or pd.isna(transaction_end):
        record['status'] = 'unknown_publication_or_transaction_clock'
    elif published > cutoff or transaction_end > cutoff.normalize():
        record['status'] = 'not_available_at_decision'
    elif transaction_end < listing or published < listing:
        record['status'] = 'before_listing_window'
    elif str(item.get('secType', '')).casefold() != 'equity shares':
        record['status'] = 'not_explicit_equity_shares'
    else:
        action = str(item.get('tdpTransactionType', '')).casefold()
        mode = str(item.get('acqMode', '')).casefold()
        if (mode, action) not in {('market sale', 'sell'), ('market purchase', 'buy')}:
            record['status'] = 'not_explicit_open_market_buy_or_sell'
        else:
            amount = pd.to_numeric(item.get('secAcq'), errors='coerce')
            if pd.isna(amount) or amount <= 0:
                record['status'] = 'unknown_positive_equity_quantity'
            else:
                category = str(item.get('personCategory', '')).strip().casefold()
                unknown_role = category in {'', '-', 'na', 'n/a', 'none'}
                record.update(status='dated_open_market_insider_transaction', quantity=float(amount),
                              transaction_age_calendar_days=int((cutoff.normalize()-transaction_end).days),
                              promoter_explicit=None if unknown_role else
                                  category in {'promoter', 'promoters', 'promoter group'})
                if unknown_role:
                    record['status'] = 'dated_open_market_transaction_role_unverified'
    return record


def run():
    ledger = pd.read_csv(SOURCES['development'][1], parse_dates=['listing_date', 'decision_date'])
    canonical = pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    # Stable earlier-year sample, independent of future returns or verdicts.
    sample = ledger[ledger.listing_year.between(2021, 2023)].sort_values('ipo_id').head(5)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    records = []
    for case in sample.itertuples():
        symbol = canonical.loc[case.ipo_id, 'nse_symbol']
        query = dict(index='equities', symbol=symbol, from_date=case.listing_date.strftime('%d-%m-%Y'),
                     to_date=case.decision_date.strftime('%d-%m-%Y'))
        path = ARCHIVE/f'{case.ipo_id}_{case.decision_date:%Y%m%d}.json.gz'
        metadata = path.with_suffix('.meta.json')
        if path.exists() and metadata.exists():
            meta = json.loads(metadata.read_text(encoding='utf-8'))
            raw = gzip.decompress(path.read_bytes())
            if meta['query'] != query or hashlib.sha256(raw).hexdigest() != meta['raw_sha256']:
                raise ValueError('PIT archive identity or hash conflict')
        else:
            response = requests.get(ENDPOINT, params=query, headers={
                'User-Agent': 'CapstoneResearch/1.0', 'Accept': 'application/json'}, timeout=20)
            response.raise_for_status()  # No block-page retries or access workaround.
            raw = response.content
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
                raise ValueError('Unexpected official PIT response; stop pilot')
            meta = dict(source_url=response.url, query=query,
                        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                        raw_sha256=hashlib.sha256(raw).hexdigest())
            path.write_bytes(gzip.compress(raw, mtime=0))
            metadata.write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
        payload = json.loads(raw)
        cutoff = case.decision_date + pd.Timedelta(hours=15, minutes=30)
        normalized = [normalize(item, symbol, case.listing_date, cutoff) for item in payload['data']]
        records.append(dict(ipo_id=int(case.ipo_id), symbol=symbol, decision_date=str(case.decision_date.date()),
            raw_path=path.relative_to(ROOT).as_posix(), metadata_path=metadata.relative_to(ROOT).as_posix(),
            raw_sha256=meta['raw_sha256'], response_rows=len(normalized),
            statuses=pd.Series([r['status'] for r in normalized], dtype=str).value_counts().to_dict(),
            transactions=normalized))
    report = dict(sample_size=len(records), selection='First five sorted 2021-2023 development IPO IDs; outcome-independent',
        source_page='https://www.nseindia.com/companies-listing/corporate-filings-insider-trading',
        live_model_changed=False, synthetic_rows=0, records=records,
        limitations=['A returned PIT table is not complete anchor-sale coverage or an anchor-allocation identity register.',
                     'Insider transactions, block deals and anchor ownership are distinct evidence channels.',
                     'Historical exchange dates are retrospective labels, not original local deployment captures.',
                     'Five-case pilot: no accuracy or full-cohort coverage claim.'])
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({r['symbol']:dict(rows=r['response_rows'], statuses=r['statuses']) for r in records}, indent=2))
    return report


if __name__ == '__main__':
    run()
