"""Archive dated NSE credit-rating attachments for Module 2 research.

The parent announcement must precede the decision cutoff. PDF bytes are
hashed; extracted text is evidence, not a sentiment score. No outcome data is
read by this collector. This is research-only and does not change the monitor.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import requests


ROOT = Path(__file__).resolve().parents[1]
FILINGS = ROOT / 'data/audit/module2_dated_nse_filings_exchange_decisions.jsonl'
OUTPUT = ROOT / 'data/audit/module2_nse_rating_documents.jsonl'
MANIFEST = ROOT / 'output/module2_nse_rating_document_collection.json'
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; academic IPO evidence audit)'}


def fetch(item: dict) -> dict:
    result = {key: item[key] for key in ('ipo_id', 'decision_date', 'cutoff_ist',
                                        'disseminated_at_ist', 'attachment_url', 'exchange_sequence_id')}
    try:
        response = requests.get(item['attachment_url'], headers=HEADERS, timeout=25)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b'%PDF-') or len(content) > 15_000_000:
            raise ValueError('Attachment is not a bounded PDF response')
        document = pymupdf.open(stream=content, filetype='pdf')
        text = '\n'.join(page.get_text() for page in list(document)[:2])
        result.update({'status': 'extracted', 'http_status': response.status_code,
                       'pdf_sha256': hashlib.sha256(content).hexdigest(),
                       'page_count': len(document), 'first_two_pages_text': text[:12000],
                       'retrieved_at_utc': datetime.now(timezone.utc).isoformat()})
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        result.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:300]})
    return result


def run() -> dict:
    announcements = [json.loads(line) for line in FILINGS.read_text(encoding='utf-8').splitlines()]
    items = []
    for record in announcements:
        if record['status'] != 'verified_response':
            continue
        for filing in record['filings']:
            if (filing.get('subject') or '').strip().casefold() != 'credit rating':
                continue
            if filing['disseminated_at_ist'] > record['cutoff_ist']:
                raise ValueError('Future announcement in pre-decision archive')
            url = filing.get('attachment_url') or ''
            if not url.startswith('https://nsearchives.nseindia.com/corporate/'):
                continue
            items.append({'ipo_id': int(record['ipo_id']),
                          'decision_date': record['decision_date'],
                          'cutoff_ist': record['cutoff_ist'],
                          'disseminated_at_ist': filing['disseminated_at_ist'],
                          'attachment_url': url,
                          'exchange_sequence_id': filing.get('exchange_sequence_id')})
    if len({item['attachment_url'] for item in items}) != len(items):
        raise ValueError('Duplicate rating attachment URL')
    collected = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, item) for item in items]
        for job in as_completed(jobs):
            collected.append(job.result())
    collected.sort(key=lambda item: (item['ipo_id'], item['disseminated_at_ist']))
    OUTPUT.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in collected),
                      encoding='utf-8')
    report = {'source': 'NSE corporate-announcement PDF attachments',
              'filing_archive_sha256': hashlib.sha256(FILINGS.read_bytes()).hexdigest(),
              'attachment_count': len(items),
              'extracted_count': sum(item['status'] == 'extracted' for item in collected),
              'failed': [{'ipo_id': item['ipo_id'], 'error': item.get('error')}
                         for item in collected if item['status'] != 'extracted'],
              'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              'output': str(OUTPUT.relative_to(ROOT))}
    MANIFEST.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
