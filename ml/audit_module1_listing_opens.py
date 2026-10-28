"""Audit the unchanged Module 1 cohort's outcomes against real NSE bhavcopies.

No canonical labels, deployed models or paper tables are overwritten. Fetches
one daily final exchange archive per listing date, at two concurrent requests.
The direction/return sensitivity replay must use this same source on both the
training and evaluation sides; a higher score is not a reason to prefer labels.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from ml.model_pipeline import prepare_dataset, EXCLUDED_NON_CORPORATE_IDS
from ml.reconcile_module2_corporate_actions import RAW, bhavcopy_row, parse_bhavcopy
from ml.audit_module2_nse_closes import HEADERS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'data/audit/module1_exchange_listing_open_audit.csv'
REPORT = ROOT/'output/module1_exchange_listing_open_audit_20261021.json'
SYMBOL_URL = 'https://nsearchives.nseindia.com/content/equities/symbolchange.csv'
RUCHI_CIRCULAR = 'https://archives.nseindia.com/content/circulars/CML52890.pdf'


def symbol_history():
    path = RAW/'symbolchange.csv'
    if not path.exists():
        response = requests.get(SYMBOL_URL,headers=HEADERS,timeout=20)
        response.raise_for_status()
        table = pd.read_csv(io.BytesIO(response.content))
        if not {'OLD SYMBOL','NEW SYMBOL','EFFECTIVE DATE'}.issubset(table):
            raise ValueError('Unexpected official NSE symbol-change schema')
        path.write_bytes(response.content)
    table = pd.read_csv(path)
    table['effective_date'] = pd.to_datetime(table['EFFECTIVE DATE'],format='%d-%b-%Y')
    rows = [{'old':r['OLD SYMBOL'].strip(),'new':r['NEW SYMBOL'].strip(),
             'effective':r['effective_date'],'source_url':SYMBOL_URL,
             'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            for _,r in table.iterrows()]
    # This older change is absent from the present symbol-change CSV. Its
    # effective date and old/new symbols are stated in official CML52890.
    circular = RAW/'CML52890.pdf'
    if not circular.exists():
        response = requests.get(RUCHI_CIRCULAR,headers=HEADERS,timeout=20)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF'):
            raise ValueError('Official symbol-change circular is not a PDF')
        circular.write_bytes(response.content)
    rows.append({'old':'RUCHI','new':'PATANJALI','effective':pd.Timestamp('2022-07-13'),
                 'source_url':RUCHI_CIRCULAR,'source_sha256':hashlib.sha256(circular.read_bytes()).hexdigest()})
    return sorted(rows,key=lambda x:x['effective'],reverse=True)


def symbol_at(symbol, date, changes):
    evidence = []
    for row in changes:
        if symbol == row['new'] and date < row['effective']:
            symbol = row['old']
            evidence.append({'old':row['old'],'new':row['new'],
                'effective':row['effective'].strftime('%Y-%m-%d'),
                'source_url':row['source_url'],'source_sha256':row['source_sha256']})
    return symbol,evidence


def bse_row(session,date,code):
    if pd.isna(code) or float(code)!=int(code):
        raise ValueError('No canonical BSE scrip code; cannot guess an exchange identity')
    code = int(code)
    legacy = date < pd.Timestamp('2024-07-08')
    name = f'EQ{date:%d%m%y}_CSV.ZIP' if legacy else f'BhavCopy_BSE_CM_0_0_0_{date:%Y%m%d}_F_0000.CSV'
    url = 'https://www.bseindia.com/download/BhavCopy/Equity/'+name
    path = RAW/name
    if not path.exists():
        response = session.get(url,timeout=25)
        response.raise_for_status()
        raw = response.content
        # A nonexistent download can return the HTML app with status 200.
        if (legacy and not zipfile.is_zipfile(io.BytesIO(raw))) or (
                not legacy and not raw.startswith(b'TradDt,')):
            raise ValueError('BSE did not return the expected actual archive format')
        path.write_bytes(raw)
    raw = path.read_bytes()
    return {**parse_bse(raw,date,code), 'source_url':url,
            'archive_sha256':hashlib.sha256(raw).hexdigest(),'raw_archive_path':str(path.relative_to(ROOT))}


def parse_bse(raw,date,code):
    legacy = date < pd.Timestamp('2024-07-08')
    if legacy:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if len(archive.namelist())!=1 or date.strftime('%d%m%y') not in archive.namelist()[0]:
                raise ValueError('Legacy BSE archive date/layout conflict')
            frame = pd.read_csv(archive.open(archive.namelist()[0]))
        rows = frame[frame.SC_CODE.eq(code)]
        close,opening,series,symbol = 'CLOSE','OPEN','SC_GROUP','SC_NAME'
    else:
        frame = pd.read_csv(io.BytesIO(raw))
        if not pd.to_datetime(frame.TradDt).eq(date).all() or not frame.Src.eq('BSE').all():
            raise ValueError('BSE daily archive date/source conflict')
        rows = frame[pd.to_numeric(frame.FinInstrmId,errors='coerce').eq(code)]
        close,opening,series,symbol = 'ClsPric','OpnPric','SctySrs','TckrSymb'
    if len(rows)!=1 or not float(rows.iloc[0][opening])>0 or not float(rows.iloc[0][close])>0:
        raise ValueError('BSE scrip identity or price unavailable')
    row=rows.iloc[0]
    return {'symbol':str(row[symbol]).strip(),'listing_date':date.strftime('%Y-%m-%d'),
            'exchange_open':float(row[opening]),'exchange_close':float(row[close]),
            'series':row[series],'exchange':'BSE','bse_scrip_code':code}


def verified_outcomes(path=OUTPUT):
    """Recheck archived prices, dates, identity and renaming evidence offline."""
    audit = pd.read_csv(path,parse_dates=['listing_date'])
    if audit.ipo_id.duplicated().any():
        raise ValueError('Duplicate audited outcomes')
    canonical = prepare_dataset().set_index('id')
    verified = audit[audit.status.eq('verified_daily_exchange_archive')].copy()
    archive_cache = {}
    for row in verified.itertuples():
        canon = canonical.loc[row.ipo_id]
        if canon.listing_date != row.listing_date or not abs(canon.issue_price-row.issue_price)<.0001:
            raise ValueError('Canonical IPO date or offer price changed since audit')
        raw_path = (ROOT/row.raw_archive_path).resolve()
        raw_path.relative_to(RAW.resolve())
        if raw_path not in archive_cache:
            raw = raw_path.read_bytes()
            archive_cache[raw_path] = (raw,hashlib.sha256(raw).hexdigest())
        raw,sha = archive_cache[raw_path]
        if sha != row.archive_sha256:
            raise ValueError('Audited exchange archive changed')
        if row.exchange == 'NSE':
            symbol = canon.nse_symbol
            for change in json.loads(row.symbol_change_evidence):
                evidence_path = RAW/('symbolchange.csv' if change['source_url']==SYMBOL_URL else 'CML52890.pdf')
                if (hashlib.sha256(evidence_path.read_bytes()).hexdigest()!=change['source_sha256']
                        or symbol != change['new'] or not row.listing_date < pd.Timestamp(change['effective'])):
                    raise ValueError('Symbol-change provenance or effective date conflict')
                symbol = change['old']
            if symbol != row.symbol:
                raise ValueError('Unaudited historical NSE symbol')
            actual = parse_bhavcopy(raw,row.listing_date,symbol)['nse_open']
        elif row.exchange == 'BSE':
            if pd.isna(canon.bse_scrip_code) or int(canon.bse_scrip_code)!=int(row.bse_scrip_code):
                raise ValueError('BSE scrip identity differs from the canonical IPO')
            actual = parse_bse(raw,row.listing_date,int(row.bse_scrip_code))['exchange_open']
        else:
            raise ValueError('Unknown audited exchange')
        if not abs(actual-row.exchange_open)<.0001:
            raise ValueError('Audited opening differs from the original exchange file')
    return verified


def collect_date(frame, stop, changes):
    if stop.is_set():
        return []
    session = requests.Session()
    session.headers.update(HEADERS)
    results = []
    for row in frame.itertuples():
        item = {'ipo_id':int(row.id), 'company_name':row.company_name, 'year':int(row.year),
                'symbol':row.nse_symbol,'listing_date':row.listing_date.strftime('%Y-%m-%d'),
                'issue_price':float(row.issue_price), 'canonical_open':float(row.listing_open_price),
                'canonical_close':float(row.listing_day_close_price), 'status':'unavailable',
                'canonical_nse_symbol':row.nse_symbol}
        try:
            if stop.is_set():
                break
            symbol,evidence = symbol_at(row.nse_symbol,row.listing_date,changes)
            item['symbol_change_evidence'] = json.dumps(evidence,sort_keys=True)
            try:
                if not isinstance(symbol,str) or not symbol.strip():
                    raise ValueError('No canonical NSE listing symbol')
                actual = bhavcopy_row(session,row.listing_date,symbol)
                actual.update(exchange='NSE',exchange_open=actual['nse_open'],exchange_close=actual['nse_close'])
            except ValueError as exc:
                # BSE-only at the IPO date is not an absent listing result.
                # Exact canonical scrip code is required; no fuzzy match.
                item['nse_unavailable_reason'] = str(exc)
                actual = bse_row(session,row.listing_date,row.bse_scrip_code)
            item.update(actual)
            item['ipo_id'] = int(row.id)
            item['status'] = 'verified_daily_exchange_archive'
            item['canonical_return_pct'] = (item['canonical_open']/item['issue_price']-1)*100
            item['official_return_pct'] = (item['exchange_open']/item['issue_price']-1)*100
            item['return_difference_pp'] = item['official_return_pct']-item['canonical_return_pct']
            item['direction_changed'] = (item['official_return_pct']>0)!=(item['canonical_return_pct']>0)
        except (requests.RequestException,ValueError,KeyError) as exc:
            item['error'] = str(exc)[:220]
            if isinstance(exc,requests.HTTPError) and exc.response.status_code in (403,429):
                stop.set()
        results.append(item)
    time.sleep(.5)
    return results


def main():
    frame = prepare_dataset()
    frame = frame[frame.year.between(2021,2026) & ~frame.id.isin(EXCLUDED_NON_CORPORATE_IDS)].copy()
    if frame.id.duplicated().any():
        raise ValueError('Every requested IPO must have a unique canonical identity')
    RAW.mkdir(parents=True,exist_ok=True)
    changes = symbol_history()
    stop = threading.Event()
    rows = []
    groups = [group for _,group in frame.groupby('listing_date')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(collect_date,group,stop,changes) for group in groups]
        for n,future in enumerate(as_completed(pending),1):
            rows.extend(future.result())
            if n%20==0 or n==len(groups):
                actual = pd.DataFrame(rows).sort_values('ipo_id')
                actual.to_csv(OUTPUT,index=False)
                print(f'Opening-price audit: {n}/{len(groups)} dates; '
                      f'{sum(x["status"]=="verified_daily_exchange_archive" for x in rows)} verified IPOs',flush=True)
    result = pd.DataFrame(rows).sort_values('ipo_id')
    result.to_csv(OUTPUT,index=False)
    matched = result[result.status.eq('verified_daily_exchange_archive')]
    report = {'scope':'Same 2021–2024 corporate training IPOs and 2025–2026 later records; no cohort expansion',
              'source':'Official final daily equity bhavcopies: NSE when listed there on the IPO date, exact-code BSE otherwise; never day close',
              'source_page':'https://www.nseindia.com/all-reports',
              'requested_ipos':len(frame), 'verified_ipos':len(matched),
              'unavailable_ipos':result[~result.status.eq('verified_daily_exchange_archive')].astype(object).where(
                  pd.notna(result[~result.status.eq('verified_daily_exchange_archive')]),None).to_dict('records'),
              'not_attempted_after_access_stop':sorted(set(frame.id.astype(int))-set(result.ipo_id.astype(int))),
              'stopped_for_access_limit':stop.is_set(),
              'by_year':{str(year):{'verified':len(group),
                  'opening_price_disagreements_gt_001':int((group.exchange_open-group.canonical_open).abs().gt(.01).sum()),
                  'return_disagreements_gt_1pp':int(group.return_difference_pp.abs().gt(1).sum()),
                  'direction_label_changes':int(group.direction_changed.sum())}
                  for year,group in matched.groupby('year')},
              'input_sha256':hashlib.sha256((ROOT/'data/canonical_listed_ipos.csv').read_bytes()).hexdigest(),
              'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              'limitations':['Historical exchange archives were retrieved now, not original local captures.',
                  'A verified listing outcome does not verify that historical input bids were available before the application cutoff.',
                  'Disagreements can reflect exchange differences or canonical errors; the audit does not label every discrepancy fabrication.',
                  'No deployed model, canonical CSV or paper has been changed.']}
    REPORT.write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('requested_ipos','verified_ipos','by_year','unavailable_ipos')},indent=2))


if __name__ == '__main__':
    main()
