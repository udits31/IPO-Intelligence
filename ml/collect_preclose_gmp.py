"""Recover dated GMP tables for a paired, retrospective pre-close benchmark.

Daily rows are not immutable intraday quotes. Never infer an availability
clock from the article's publication or modification date.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ml.fetch_daywise_subscriptions import HEADERS, SEARCH_URL, POST_URL, name_tokens

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/archive/preclose_gmp'


def parse_gmp_history(content: str, close_date: str) -> list[dict]:
    """Only explicit numeric GMP cells in a dated GMP-history table qualify."""
    closing = pd.Timestamp(close_date)
    result = []
    soup = BeautifulSoup(content, 'html.parser')
    for table in soup.find_all('table'):
        rows = [[c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
                for tr in table.find_all('tr')]
        head = next((i for i, r in enumerate(rows) if len(r) >= 2
                     and r[0].strip().lower() == 'date'
                     and re.fullmatch(r'(?:ipo\s+)?gmp', r[1].strip(), re.I)), None)
        if head is None:
            continue
        for cells in rows[head + 1:]:
            if len(cells) < 2:
                continue
            amount = re.fullmatch(r'\s*(?:₹|Rs\.?|INR)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*', cells[1], re.I)
            if not amount:  # A dash is missing, not zero; percent cells are not rupees.
                continue
            raw_date = cells[0].strip()
            dates = []
            for year in (closing.year - 1, closing.year, closing.year + 1):
                text = raw_date if re.search(r'\b20\d{2}\b', raw_date) else f'{raw_date} {year}'
                parsed = pd.to_datetime(text, errors='coerce', dayfirst=True)
                if pd.notna(parsed) and abs((parsed.normalize() - closing).days) <= 40:
                    dates.append(parsed.normalize())
            if not dates:
                continue
            stamp = min(dates, key=lambda d: abs((d-closing).days))
            result.append({'gmp_date': stamp.strftime('%Y-%m-%d'),
                           'gmp_rs': float(amount[1].replace(',', ''))})
    # Conflicting duplicates are not arbitrarily adjudicated.
    accepted = []
    for day in sorted({r['gmp_date'] for r in result}):
        values = {r['gmp_rs'] for r in result if r['gmp_date'] == day}
        if len(values) == 1:
            accepted.append({'gmp_date': day, 'gmp_rs': values.pop()})
    return accepted


def collect_one(item: dict) -> dict:
    ipo_id = int(item['ipo_id'])
    path = ARCHIVE / f'{ipo_id}.json.gz'
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            cached = json.load(stream)
        if cached.get('identity_close_date') == item['issue_close_date']:
            return cached
    result = {'ipo_id': ipo_id, 'company_name': item['company_name'],
              'identity_close_date': item['issue_close_date'], 'rows': [],
              'retrieved_at_ist': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
              'evidence_class': 'retrospective_daily_table_not_immutable_quote'}
    try:
        tokens = name_tokens(item['company_name'])
        response = requests.get(SEARCH_URL, params={'search': ' '.join(tokens) + ' IPO GMP',
                                                   'per_page': 10}, headers=HEADERS, timeout=20)
        response.raise_for_status()
        for hit in response.json():
            title = BeautifulSoup(html.unescape(hit.get('title', '')), 'html.parser').get_text(' ', strip=True)
            prefix = re.split(r'\bIPO\b', title, flags=re.I)[0]
            if 'gmp' not in title.lower() or set(name_tokens(prefix)) != set(tokens):
                continue  # Exact issuer identity, not a parent/subsidiary fuzzy match.
            raw = requests.get(POST_URL.format(post_id=hit['id']), headers=HEADERS, timeout=20)
            raw.raise_for_status()
            post = raw.json()
            content = post.get('content', {}).get('rendered', '')
            # An explicit closing date in the offer table must agree too.
            soup = BeautifulSoup(content, 'html.parser')
            close_matches = []
            for tr in soup.find_all('tr'):
                cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
                if len(cells) == 2 and re.fullmatch(r'(?:IPO |Bid/Offer )?Close(?: Date)?\s*:', cells[0], re.I):
                    date = pd.to_datetime(cells[1], errors='coerce', dayfirst=True)
                    if pd.notna(date):
                        close_matches.append(date.strftime('%Y-%m-%d'))
            if close_matches:
                if set(close_matches) != {item['issue_close_date']}:
                    continue
                identity_basis = 'exact_name_and_explicit_close_date'
            else:
                # Newer GMP pages omit an offer timetable. Require the exact
                # opening month/day, issue-price upper band and article era,
                # rather than accepting the company name alone.
                text = soup.get_text(' ', strip=True)
                opening = re.search(r'IPO is to open on\s+([A-Za-z]+\s+\d{1,2})(?:\s*[, ]\s*(20\d{2}))?', text, re.I)
                band = re.search(r'price band is fixed at\s*₹?\s*[\d,]+(?:\.\d+)?\s*(?:to|-)\s*₹?\s*([\d,]+(?:\.\d+)?)', text, re.I)
                published = pd.to_datetime(post.get('date'), errors='coerce')
                expected_open = pd.Timestamp(item['issue_open_date'])
                parsed_open = pd.to_datetime(f'{opening[1]} {opening[2] or expected_open.year}', errors='coerce') if opening else pd.NaT
                if (pd.isna(parsed_open) or parsed_open != expected_open or not band
                        or abs(float(band[1].replace(',', ''))-float(item['issue_price'])) > .01
                        or pd.isna(published) or abs((published.normalize()-pd.Timestamp(item['issue_close_date'])).days) > 40):
                    continue
                identity_basis = 'exact_name_open_date_price_band_and_article_era'
            rows = parse_gmp_history(content, item['issue_close_date'])
            if rows:
                result.update(rows=rows, source_url=post['link'], raw_post=post,
                              source_sha256=hashlib.sha256(raw.content).hexdigest(),
                              status='matched_daily_history', identity_basis=identity_basis)
                break
        result.setdefault('status', 'no_identity_and_date_matched_history')
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        result.update(status='fetch_error', error=type(exc).__name__)
    # Raw publisher content is retained, not just extracted numbers.
    if result['rows']:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump(result, stream)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    offers = pd.read_csv(ROOT / 'data/nse_verified_offer_dates.csv')
    ids = pd.read_csv(ROOT / 'data/daywise_subscription_research_cohort.csv').ipo_id
    offers = offers[offers.ipo_id.isin(ids)].sort_values('ipo_id')
    if args.limit:
        offers = offers.head(args.limit)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(collect_one, offers.to_dict('records')))
    rows = []
    for result in results:
        for quote in result['rows']:
            rows.append({k: result[k] for k in ('ipo_id', 'company_name', 'source_url',
                                               'source_sha256', 'retrieved_at_ist', 'evidence_class')} | quote)
    pd.DataFrame(rows, columns=['ipo_id', 'company_name', 'source_url', 'source_sha256',
                                'retrieved_at_ist', 'evidence_class', 'gmp_date', 'gmp_rs']).to_csv(
        ROOT / 'data/preclose_gmp_history.csv', index=False)
    audit = {'attempted': len(results), 'matched': sum(bool(r['rows']) for r in results),
             'quote_rows': len(rows), 'sources': [{k: v for k, v in r.items()
                                                  if k not in ('raw_post', 'rows')} for r in results]}
    (ROOT / 'output/preclose_gmp_collection.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('attempted', 'matched', 'quote_rows')}))


if __name__ == '__main__':
    main()
