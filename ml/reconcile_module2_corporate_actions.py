"""Reconcile real exchange prices with dated NSE bonus/split records.

Research inputs only: never overwrites live prices, canonical IPOs or models.
Prices are expressed as the value of the shares held from the listing day.
Each row uses only share changes effective by that row's date. Cash dividends
are not added; this is not a total-return or executable trading-profit claim.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ml.fetch_module2_nse_delivery import HEADERS, PAGE

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/archive/module2_corporate_actions'
ENDPOINT = 'https://www.nseindia.com/api/corporates-corporateActions'
SOURCES = {
    'development': ROOT/'data/audit/module2_development_prices_2021_2024_nse_extended.csv',
    'holdout': ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full.csv',
}
AUDIT = ROOT/'output/module2_listing_close_resolution_20261021.csv'
REPORT = ROOT/'output/module2_corporate_action_reconciliation_20261021.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def share_change(subject):
    """Only explicit exchange bonus/split ratios; never infer from price drops."""
    text = str(subject).strip()
    bonus = re.fullmatch(r'Bonus\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', text, re.I)
    if bonus:
        new, existing = map(float, bonus.groups())
        if existing <= 0 or new <= 0:
            raise ValueError('Invalid official bonus ratio')
        return 'bonus', 1+new/existing
    split = re.fullmatch(
        r'Face Value Split \(Sub-Division\)\s*-\s*From Rs\s*(\d+(?:\.\d+)?)/- Per Share '
        r'To Re?s?\s*(\d+(?:\.\d+)?)/- Per Share', text, re.I)
    if split:
        old, new = map(float, split.groups())
        if not old > new > 0:
            raise ValueError('Invalid official split ratio')
        return 'split', old/new
    if re.match(r'(?:Interim |Final |Special )?Dividend\b', text, re.I):
        return 'cash_dividend_not_included', 1.
    if text.casefold() in ('annual general meeting', 'extra ordinary general meeting') or re.fullmatch(
            r'Annual General Meeting/Dividend(?:\s*-\s*(?:Rs|Re)\s*\d+(?:\.\d+)? Per Share)?', text, re.I):
        return 'meeting_or_cash_not_included', 1.
    return 'unsupported', None


def action_rows(payload, start, end):
    if not isinstance(payload, list):
        raise ValueError('Corporate-action response is not a list')
    rows, seen = [], set()
    for record in payload:
        stamp = pd.to_datetime(record.get('exDate'), format='%d-%b-%Y', errors='raise')
        if pd.isna(stamp) or not start <= stamp <= end or not record.get('symbol'):
            raise ValueError('Invalid corporate-action date or identity')
        key = (record['symbol'], stamp, str(record.get('subject')), str(record.get('isin')))
        if key in seen:
            continue
        seen.add(key)
        kind, factor = share_change(record.get('subject'))
        rows.append({'symbol': record['symbol'], 'ex_date': stamp,
                     'subject': record.get('subject'), 'isin': record.get('isin'),
                     'kind': kind, 'share_factor': factor})
    return rows


def normalize_prices(prices, canonical, actions):
    result, changes, excluded = [], [], []
    for ipo_id, frame in prices.groupby('ipo_id'):
        # This audit has the same predeclared 31st-close / 30-session horizon.
        # A later rights issue outside that window cannot censor an earlier
        # fully observed outcome, and is not used to adjust an earlier price.
        group = frame.sort_values('date').head(62).copy()
        listing = canonical.loc[ipo_id].listing_date
        symbol = canonical.loc[ipo_id].nse_symbol
        relevant = actions[(actions.symbol == symbol) & actions.ex_date.ge(listing)
                           & actions.ex_date.le(group.date.max())].copy()
        if (relevant.kind == 'unsupported').any() or (
                relevant.ex_date.eq(listing) & relevant.kind.isin(['bonus','split'])).any():
            excluded.append({'ipo_id': int(ipo_id), 'symbol': symbol,
                             'reason': 'Unsupported capital event or listing-day share-basis ambiguity',
                             'subjects': relevant.subject.tolist()})
            continue
        factors = np.ones(len(group))
        for row in relevant.itertuples():
            if row.kind in ('bonus', 'split'):
                factors[group.date.ge(row.ex_date).to_numpy()] *= row.share_factor
                changes.append({'ipo_id': int(ipo_id), 'symbol': symbol,
                                'ex_date': row.ex_date.strftime('%Y-%m-%d'),
                                'subject': row.subject, 'share_factor': row.share_factor})
        group['unadjusted_nse_close'] = group.close_price_adjusted
        group['shares_per_listing_share'] = factors
        group['close_price_adjusted'] = group.unadjusted_nse_close*factors
        group['price_basis'] = 'Official NSE closes in listing-share units; dated NSE bonus/split reconciliation'
        result.append(group)
    return pd.concat(result, ignore_index=True), changes, excluded


def fetch_actions(session, start, end):
    collected, manifests = [], []
    for year in range(start.year, end.year+1):
        first = max(start, pd.Timestamp(year=year,month=1,day=1))
        last = min(end, pd.Timestamp(year=year,month=12,day=31))
        params = {'index':'equities', 'from_date':first.strftime('%d-%m-%Y'),
                  'to_date':last.strftime('%d-%m-%Y')}
        path = RAW/f'actions_{first:%Y%m%d}_{last:%Y%m%d}.json.gz'
        cached = path.exists()
        if cached:
            with gzip.open(path,'rb') as stream:
                raw = stream.read()
        else:
            response = session.get(ENDPOINT,params=params,timeout=30)
            response.raise_for_status()  # Includes 403/429: no bypass or fake empty success.
            raw = response.content
            action_rows(json.loads(raw),first,last)
            with gzip.open(path,'wb') as stream:
                stream.write(raw)
        rows = action_rows(json.loads(raw),first,last)
        collected.extend(rows)
        manifests.append({'source_url':requests.Request('GET',ENDPOINT,params=params).prepare().url,
                          'raw_sha256':digest(raw), 'raw_archive_path':str(path.relative_to(ROOT)),
                          'cached_response':cached, 'parsed_actions':len(rows)})
        print(f'Corporate actions {year}: {len(rows)} actual records',flush=True)
        time.sleep(.5)
    return pd.DataFrame(collected), manifests


def bhavcopy_row(session, date, symbol):
    legacy = date < pd.Timestamp('2024-07-08')
    name = (f'cm{date:%d}{date:%b}'.upper()+f'{date:%Y}bhav.csv.zip') if legacy else (
        f'BhavCopy_NSE_CM_0_0_0_{date:%Y%m%d}_F_0000.csv.zip')
    # Keep cm lower case: the archive URL is case sensitive.
    if legacy:
        name = 'cm'+name[2:]
    url = (f'https://nsearchives.nseindia.com/content/historical/EQUITIES/{date:%Y}/{date:%b}'.upper()
           if legacy else 'https://nsearchives.nseindia.com/content/cm')+'/'+name
    if legacy:
        url = url.replace('HTTPS://NSEARCHIVES.NSEINDIA.COM/CONTENT/HISTORICAL/EQUITIES',
                          'https://nsearchives.nseindia.com/content/historical/EQUITIES')
    path = RAW/name
    if path.exists():
        raw = path.read_bytes()
    else:
        response = session.get(url,timeout=25)
        response.raise_for_status()
        raw = response.content
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if len(archive.namelist()) != 1:
                raise ValueError('Unexpected daily bhavcopy layout')
        path.write_bytes(raw)
    row = parse_bhavcopy(raw,date,symbol)
    return {**row, 'source_url':url,
            'archive_sha256':digest(raw),'raw_archive_path':str(path.relative_to(ROOT))}


def parse_bhavcopy(raw, date, symbol):
    legacy = date < pd.Timestamp('2024-07-08')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.namelist()) != 1:
            raise ValueError('Unexpected daily bhavcopy layout')
        table = pd.read_csv(archive.open(archive.namelist()[0]))
    if legacy:
        dates = pd.to_datetime(table.TIMESTAMP,format='%d-%b-%Y')
        selected = table[table.SYMBOL.eq(symbol) & table.SERIES.isin(['EQ','BE'])]
        close, opening, series = 'CLOSE', 'OPEN', 'SERIES'
    else:
        dates = pd.to_datetime(table.TradDt)
        selected = table[table.TckrSymb.eq(symbol) & table.SctySrs.isin(['EQ','BE'])]
        close, opening, series = 'ClsPric', 'OpnPric', 'SctySrs'
    if not dates.eq(date).all() or len(selected) != 1:
        raise ValueError('Bhavcopy date/stock-series identity not unique')
    row = selected.iloc[0]
    return {'symbol':symbol,'listing_date':date.strftime('%Y-%m-%d'), 'nse_close':float(row[close]),
            'nse_open':float(row[opening]),'series':row[series]}


def load_listing_resolutions(path, canonical):
    """Require the raw daily archive, not merely a CSV asserting agreement."""
    audit = pd.read_csv(path,parse_dates=['listing_date'])
    if audit.ipo_id.duplicated().any():
        raise ValueError('Duplicate listing-close resolutions')
    verified = {}
    for row in audit.itertuples():
        record = canonical.loc[row.ipo_id]
        raw_path = (ROOT/row.raw_archive_path).resolve()
        raw_path.relative_to(RAW.resolve())
        raw = raw_path.read_bytes()
        if digest(raw) != row.archive_sha256:
            raise ValueError('Daily exchange archive hash changed')
        actual = parse_bhavcopy(raw,row.listing_date,row.symbol)
        if (row.symbol != record.nse_symbol or row.listing_date != record.listing_date
                or not np.isclose(actual['nse_close'],row.nse_close,atol=.0001)):
            raise ValueError('Daily exchange resolution disagrees with raw date/identity/close')
        verified[int(row.ipo_id)] = actual
    return verified


def resolve_listing_conflicts(session, frames, canonical):
    evidence = []
    for cohort, prices in frames.items():
        for ipo_id, data in prices.groupby('ipo_id'):
            first = data.sort_values('date').iloc[0]
            canon = canonical.loc[ipo_id]
            if first.date != canon.listing_date or pd.isna(canon.listing_day_close_price):
                continue
            if abs(first.close_price_adjusted/canon.listing_day_close_price-1) <= .02:
                continue
            row = bhavcopy_row(session,canon.listing_date,canon.nse_symbol)
            row.update(ipo_id=int(ipo_id),cohort=cohort, canonical_close=float(canon.listing_day_close_price),
                       replay_close=float(first.close_price_adjusted))
            row['replay_matches_daily_exchange_archive'] = bool(np.isclose(row['replay_close'],row['nse_close'],atol=.0001))
            evidence.append(row)
            time.sleep(.5)
    return pd.DataFrame(evidence)


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    frames = {key:pd.read_csv(path,parse_dates=['date']) for key,path in SOURCES.items()}
    canonical = pd.read_csv(ROOT/'data/canonical_listed_ipos.csv',parse_dates=['listing_date']).set_index('id')
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(PAGE,timeout=20).raise_for_status()
    start = min(frame.date.min() for frame in frames.values())
    end = max(frame.date.max() for frame in frames.values())
    actions, manifest = fetch_actions(session,start,end)
    listing_audit = resolve_listing_conflicts(session,frames,canonical)
    listing_audit.to_csv(AUDIT,index=False)
    outputs = {}
    for cohort,frame in frames.items():
        normalized, changes, excluded = normalize_prices(frame,canonical,actions)
        path = SOURCES[cohort].with_name(SOURCES[cohort].stem+'_ca_reconciled.csv')
        normalized.to_csv(path,index=False)
        outputs[cohort] = {'input_sha256':digest(SOURCES[cohort].read_bytes()),
                           'output_path':str(path.relative_to(ROOT)), 'output_sha256':digest(path.read_bytes()),
                           'input_ipos':int(frame.ipo_id.nunique()), 'output_ipos':int(normalized.ipo_id.nunique()),
                           'share_changes':changes,'excluded':excluded}
    report = {'action_source':'Official NSE equity corporate-action historical responses',
              'source_page':'https://www.nseindia.com/companies-listing/corporate-filings-actions',
              'action_windows':manifest,'cohorts':outputs,'listing_conflicts_checked':len(listing_audit),
              'listing_audit_sha256':digest(AUDIT.read_bytes()),
              'evaluation_window':'First 62 actual stock sessions; future corporate events outside the frozen horizon cannot censor earlier outcomes.',
              'limitations':['Exchange responses retrieved retrospectively, not archived investor-time publications.',
                  'Absence of an event in a successful response is not independent proof of exhaustive historical coverage.',
                  'Only explicit bonus/split ratios are applied, effective on their ex-date; no guessed price adjustment.',
                  'Unsupported capital events exclude the entire affected replay with a recorded reason.',
                  'Cash dividends are not included; no total-return, slippage or execution-profit claim.',
                  'Canonical IPO values and the live model are unchanged.']}
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cohorts':outputs,'listing_conflicts_checked':len(listing_audit)},indent=2))


if __name__ == '__main__':
    main()
