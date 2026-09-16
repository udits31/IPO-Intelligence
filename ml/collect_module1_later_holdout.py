"""Recover a separate 2025/26 interim-book holdout; never rewrite live calls.

Public dated historical tables are retrospective evidence, not saved forecasts.
"""
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from ml.recover_module1_other_sources import (ROOT, fetch, norm, ipoji_payload,
    collect_a2z, collect_ipowala, has_eligible_quote, dated_history)

CACHE = ROOT / 'data/archive/module1_later_holdout_20261024'
OUT = ROOT / 'data/audit/module1_later_holdout_20261024.csv'
AUDIT = ROOT / 'output/module1_later_collection_20261024.json'


def bid_history(rows, opened, closed):
    """Only explicitly dated numeric category multiples; parenthesis is display."""
    result = []
    for row in rows:
        key = str(row.get('key', ''))
        match = re.search(r'Day\s*(\d+)\s*-\s*(\d{2}-\d{2}-\d{2,4})\s+(\d{1,2}:\d{2}\s*[AP]M)', key, re.I)
        if not match:
            continue
        date = pd.to_datetime(match[2], format='%d-%m-%y' if len(match[2]) == 8 else '%d-%m-%Y')
        if not pd.Timestamp(opened) <= date <= pd.Timestamp(closed):
            continue
        values = row.get('value', {})
        parsed = {}
        for category, source in [('qib', 'QIB'), ('hni', 'NII'), ('retail', 'RII')]:
            raw = str(values.get(source, '')).strip()
            number = re.fullmatch(r'\(?([0-9,]+(?:\.[0-9]+)?)\)?(?:\s*[xX])?', raw)
            if number:
                parsed[category + '_subscription'] = float(number[1].replace(',', ''))
        if len(parsed) == 3:
            result.append(dict(day=int(match[1]), date=str(date.date()), displayed_time=match[3], **parsed))
    dates = [r['date'] for r in result]
    if len(dates) != len(set(dates)):
        return []
    return sorted(result, key=lambda r: r['date'])


def index_urls(name, url, contains):
    path = CACHE / (name + '.xml.gz')
    old = ROOT / 'data/archive/module1_other_sources_20261024' / (name + '.xml.gz')
    if not path.exists():
        text = gzip.open(old, 'rt', encoding='utf8').read() if old.exists() else fetch(url).text
        with gzip.open(path, 'wt', encoding='utf8') as stream:
            stream.write(text)
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [node.text for node in ET.fromstring(stream.read()).iter()
                if node.tag.endswith('loc') and contains in node.text]


def collect(item, ipoji_urls, a2z_urls):
    path = CACHE / f"{item['ipo_id']}.json.gz"
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            return json.load(stream)
    result = dict(ipo_id=item['ipo_id'], company_name=item['company_name'], sources=[],
                  retrieved_at_utc=datetime.now(timezone.utc).isoformat(), status='no_verified_daywise_book')
    tokens = set(re.findall(r'[a-z0-9]+', item['company_name'].lower())) - {'ltd', 'limited', 'india', 'and'}
    def score(url):
        slug = url.rstrip('/').split('/')[-1]
        return 100 if norm(slug) == norm(item['company_name']) else len(tokens & set(slug.split('-'))) / max(len(tokens), 1)
    for url in sorted(ipoji_urls, key=lambda u: (-score(u), u))[:2]:
        if score(url) < .5:
            continue
        try:
            response = fetch(url)
            payload = ipoji_payload(item, BeautifulSoup(response.text, 'html.parser'))
            source = dict(provider='ipoji', url=url, raw_response_text=response.text,
                          source_sha256=hashlib.sha256(response.content).hexdigest(),
                          status='verified_offer' if payload else 'identity_or_dates_rejected', quotes=[])
            result['sources'].append(source)
            if payload:
                source['quotes'] = dated_history(response.text, item['issue_close_date'])
                history = bid_history(payload['bids'], item['issue_open_date'], item['issue_close_date'])
                # A final dated row establishes which preceding row is penultimate.
                if len(history) >= 2 and history[-1]['date'] == item['issue_close_date']:
                    prior = history[-2]
                    if prior['day'] + 1 == history[-1]['day']:
                        result.update(status='accepted_interim_book', decision_market_date=prior['date'],
                                      bid_source_url=url, bid_source_sha256=source['source_sha256'],
                                      bid_history=history, **{k:v for k,v in prior.items() if k.endswith('_subscription')})
                break
        except Exception as exc:
            result['sources'].append(dict(provider='ipoji', url=url, status='fetch_error', error=str(exc)[:180], quotes=[]))
    if result['status'] == 'accepted_interim_book':
        dated = dict(item, decision_market_date=result['decision_market_date'])
        result['sources'] += collect_a2z(dated, a2z_urls)
        if not has_eligible_quote(dated, result['sources']):
            result['sources'] += collect_ipowala(dated)
    with gzip.open(path, 'wt', encoding='utf8') as stream:
        json.dump(result, stream)
    return result


def run():
    CACHE.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(ROOT / 'data/canonical_listed_ipos.csv').set_index('id')
    outcomes = pd.read_csv(ROOT / 'data/audit/module1_exchange_listing_open_audit.csv')
    outcomes = outcomes[outcomes.year.isin([2025, 2026]) & outcomes.status.eq('verified_daily_exchange_archive') & outcomes.exchange_open.gt(0)]
    offers = pd.read_csv(ROOT / 'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    ipoji = index_urls('ipoji_sitemap', 'https://www.ipoji.com/sitemap-ipo.xml', '/ipo/')
    a2z = index_urls('a2z_sitemap', 'https://www.a2zipo.com/sitemap.xml', '/gmp/')
    items = []
    records = []
    for row in outcomes.itertuples():
        if row.ipo_id not in offers.index:
            records.append(dict(ipo_id=int(row.ipo_id), company_name=row.company_name, status='missing_verified_offer_dates'))
            continue
        offer = offers.loc[row.ipo_id]
        cap = float(canonical.loc[row.ipo_id].price_band_upper)
        if not np.isfinite(cap) or cap <= 0:
            records.append(dict(ipo_id=int(row.ipo_id), company_name=row.company_name, status='missing_offer_cap'))
            continue
        items.append(dict(ipo_id=int(row.ipo_id), company_name=row.company_name,
                          issue_open_date=offer.issue_open_date, issue_close_date=offer.issue_close_date,
                          listing_date=offer.listing_date, issue_price=cap, nse_symbol=str(row.symbol)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = {pool.submit(collect, item, ipoji, a2z):item for item in items}
        for future in as_completed(pending):
            records.append(future.result())
            if len(records) % 10 == 0:
                print(f'checked={len(records)}/{len(outcomes)} interim_books={sum(r["status"]=="accepted_interim_book" for r in records)}', flush=True)
    rows = []
    audit = []
    by_id = outcomes.set_index('ipo_id')
    for record in records:
        item = canonical.loc[record['ipo_id']]
        row = dict(ipo_id=record['ipo_id'], company_name=record['company_name'], status=record['status'])
        source_audit = [{k:v for k,v in s.items() if k != 'raw_response_text'} for s in record.get('sources', [])]
        audit.append(dict(**row, sources=source_audit))
        if record['status'] != 'accepted_interim_book':
            rows.append(row); continue
        outcome = by_id.loc[record['ipo_id']]
        quotes = [dict(q, source_url=s['url']) for s in record['sources'] for q in s.get('quotes', [])]
        day = pd.Timestamp(record['decision_market_date'])
        eligible = [q for q in quotes if day-pd.Timedelta(days=7) <= pd.Timestamp(q['gmp_date']) < day]
        grouped = {}
        for quote in eligible:
            grouped.setdefault(quote['gmp_date'], []).append(quote)
        clean = [q for date, qs in grouped.items() if len({v['gmp_rs'] for v in qs}) == 1 for q in qs]
        selected = sorted(clean, key=lambda q: (q['gmp_date'], q['source_url']))[-1] if clean else None
        row.update(year=int(outcome.year), listing_date=outcome.listing_date, issue_price=float(outcome.issue_price),
                   exchange_open=float(outcome.exchange_open), target_binary=int(outcome.exchange_open > outcome.issue_price),
                   target_return=100*(outcome.exchange_open/outcome.issue_price-1),
                   decision_market_date=record['decision_market_date'], bid_source_url=record['bid_source_url'],
                   bid_source_sha256=record['bid_source_sha256'],
                   issue_open_date=offers.loc[record['ipo_id']].issue_open_date,
                   issue_close_date=offers.loc[record['ipo_id']].issue_close_date,
                   log_offer_cap_price=float(np.log1p(item.price_band_upper)),
                   log_issue_size=float(np.log1p(item.issue_size_cr)) if pd.notna(item.issue_size_cr) and item.issue_size_cr>0 else None,
                   gmp_conflict_days=sum(len({v['gmp_rs'] for v in qs}) > 1 for qs in grouped.values()),
                   exchange_source_url=outcome.source_url, exchange_archive_sha256=outcome.archive_sha256)
        for category in ('qib','hni','retail'):
            row[category+'_subscription'] = record[category+'_subscription']
            row['log_'+category] = float(np.log1p(max(.01, record[category+'_subscription'])))
        if selected:
            row.update(gmp_date=selected['gmp_date'], gmp_rs=selected['gmp_rs'],
                       gmp_pct=100*selected['gmp_rs']/item.price_band_upper,
                       gmp_source_url=selected['source_url'], status='paired_eligible')
        else:
            row['status']='interim_book_missing_prior_gmp'
        rows.append(row)
    pd.DataFrame(rows).sort_values('ipo_id').to_csv(OUT, index=False)
    report = dict(universe_n=len(outcomes), checked=len(rows), status_counts=pd.Series([r['status'] for r in rows]).value_counts().to_dict(),
                  audit=audit, evidence_class='retrospective_public_dated_tables_not_saved_preclose_forecasts')
    AUDIT.write_text(json.dumps(report, indent=2, default=str)+'\n', encoding='utf8')
    print(json.dumps(report['status_counts']), flush=True)


if __name__ == '__main__':
    run()
