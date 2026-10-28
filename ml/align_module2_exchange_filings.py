"""Align NSE filing queries to the exchange-price Module 2 decision dates.

The old secondary-price cohort sometimes chose an adjacent session. Reuse a
verified old response only when its exact decision date and symbol match;
otherwise query NSE again. Failures remain unavailable, not zero filings.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.fetch_module2_filings import HEADERS, fetch_one
from ml.research_module2_exchange_delivery import assemble


OLD = ROOT / 'data/audit/module2_dated_nse_filings.jsonl'
OUTPUT = ROOT / 'data/audit/module2_dated_nse_filings_exchange_decisions.jsonl'
MANIFEST = OUTPUT.with_suffix('.json')
CANONICAL = ROOT / 'data/canonical_listed_ipos.csv'


def main() -> None:
    cohort, _ = assemble()
    canonical = pd.read_csv(CANONICAL).set_index('id')
    old = {int(row['ipo_id']): row for row in
           (json.loads(line) for line in OLD.read_text(encoding='utf-8').splitlines())}
    existing = {}
    if OUTPUT.exists():
        existing = {int(row['ipo_id']): row for row in
                    (json.loads(line) for line in OUTPUT.read_text(encoding='utf-8').splitlines())}
    session = requests.Session()
    session.headers.update(HEADERS)
    landing = session.get(HEADERS['Referer'], timeout=20)
    landing.raise_for_status()
    rows = []
    reused, fetched, failed = 0, 0, 0
    for index, case in enumerate(cohort.sort_values('ipo_id').itertuples(index=False), 1):
        ipo_id = int(case.ipo_id)
        record = canonical.loc[ipo_id]
        symbol = str(record.nse_symbol)
        listed = str(record.listing_date)[:10]
        decision = str(case.decision_date)[:10]
        prior = existing.get(ipo_id) or old.get(ipo_id)
        if (prior and prior.get('status') == 'verified_response'
                and prior.get('decision_date') == decision
                and prior.get('symbol') == symbol):
            row = prior
            reused += 1
        else:
            row = fetch_one(session, ipo_id, symbol, listed, decision)
            fetched += 1
            if row['status'] != 'verified_response':
                failed += 1
            time.sleep(.5)
        rows.append(row)
        if index % 20 == 0 or index == len(cohort):
            OUTPUT.write_text('\n'.join(json.dumps(item, ensure_ascii=False)
                                        for item in rows) + '\n', encoding='utf-8')
            print(f'{index}/{len(cohort)} exact-date filings: reused={reused} fetched={fetched} failed={failed}',
                  flush=True)
    manifest = {
        'source_page': HEADERS['Referer'],
        'cohort_count': len(cohort), 'reused_exact_date_responses': reused,
        'newly_fetched': fetched, 'failed': failed,
        'verified_response_count': sum(row['status'] == 'verified_response' for row in rows),
        'filings_before_decision_cutoff': sum(row['eligible_rows'] or 0 for row in rows),
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (OLD, CANONICAL)},
        'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'note': 'Retrospective archive retrieval; source dissemination timestamps, not original API publication snapshots.',
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
