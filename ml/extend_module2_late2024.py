"""Recover actual 2025 outcome sessions for IPOs listed late in 2024.

The old development download ended on 31 December 2024. That censored 30
IPO histories before their 31-session checkpoint or 30-session outcome. These
are 2024 IPOs, not additional 2025 IPO training records. Outcomes unavailable
at a parameter-freeze date must still be excluded by the calibration protocol.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from ml.fetch_module2_nse_delivery import PAGE, ENDPOINT, HEADERS
from ml.recover_module2_be_closes import parse_prices

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/audit/module2_development_prices_2021_2024.csv'
REPAIRED = ROOT/'data/audit/module2_development_prices_2021_2024_nse_with_be.csv'
OUTPUT = ROOT/'data/audit/module2_development_prices_2021_2024_nse_extended.csv'
REPORT = ROOT/'output/module2_late2024_extension_20261021.json'
RAW = ROOT/'data/archive/module2_late2024_nse'
BENCHMARKS = ROOT/'data/audit/module2_late2024_benchmarks.csv'


def benchmark_closes(tickers):
    first = int(pd.Timestamp('2024-09-01',tz='UTC').timestamp())
    last = int(pd.Timestamp('2025-06-01',tz='UTC').timestamp())
    columns = []
    for ticker in tickers:
        cache = RAW/(quote(ticker,safe='')+'_benchmark.json.gz')
        if cache.exists():
            with gzip.open(cache,'rb') as stream:
                raw = stream.read()
        else:
            response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker,safe="")}',
                params={'period1':first,'period2':last,'interval':'1d'},
                headers={'User-Agent':'Mozilla/5.0'},timeout=20)
            response.raise_for_status()
            raw = response.content
            json.loads(raw)
            with gzip.open(cache,'wb') as stream:
                stream.write(raw)
        results = json.loads(raw).get('chart',{}).get('result') or []
        if len(results)!=1 or results[0].get('meta',{}).get('symbol')!=ticker:
            raise ValueError(f'Benchmark identity missing or conflicting: {ticker}')
        data = results[0]
        times = data.get('timestamp') or []
        values = ((data.get('indicators',{}).get('quote') or [{}])[0].get('close') or [])
        rows = [(pd.Timestamp(ts,unit='s',tz='UTC').tz_convert('Asia/Kolkata').normalize().tz_localize(None),v)
                for ts,v in zip(times,values) if v is not None and v>0]
        series = pd.Series(dict(rows),name=ticker,dtype=float)
        if series.empty:
            raise ValueError(f'No actual benchmark observations: {ticker}')
        columns.append(series)
        time.sleep(.45)
    return pd.concat(columns,axis=1).sort_index()


def main():
    original = pd.read_csv(BASE, parse_dates=['date'])
    fixed = pd.read_csv(REPAIRED, parse_dates=['date'])
    counts = original.groupby('ipo_id').size()
    ids = sorted(counts[counts < 62].index)
    canonical = pd.read_csv(ROOT/'data/canonical_listed_ipos.csv', parse_dates=['listing_date']).set_index('id')
    if not canonical.loc[ids].listing_date.dt.year.eq(2024).all():
        raise ValueError('This extension is restricted to the censored 2024 IPOs')
    stock_meta = original.sort_values('date').groupby('ipo_id').first()
    tickers = sorted(set(stock_meta.loc[ids].sector_ticker) | {'^NSEI'})
    RAW.mkdir(parents=True, exist_ok=True)
    if BENCHMARKS.exists():
        benchmarks = pd.read_csv(BENCHMARKS, index_col=0, parse_dates=True)
    else:
        benchmarks = benchmark_closes(tickers)
        if benchmarks.empty or '^NSEI' not in benchmarks:
            raise RuntimeError('No real Nifty observations; refusing invented benchmark values')
        benchmarks.to_csv(BENCHMARKS, index_label='date')
    session = requests.Session()
    landing = session.get(PAGE, headers=HEADERS, timeout=20)
    landing.raise_for_status()
    added, attempts, excluded = [], [], []
    access_limited = False
    for i, ipo_id in enumerate(ids, 1):
        listing = canonical.loc[ipo_id].listing_date
        last = listing+pd.Timedelta(days=150)
        symbol = str(canonical.loc[ipo_id].nse_symbol)
        records = []
        for series in ('EQ', 'BE'):
            params = {'from': listing.strftime('%d-%m-%Y'), 'to': last.strftime('%d-%m-%Y'),
                      'symbol': symbol, 'type': 'priceVolumeDeliverable', 'series': series}
            url = requests.Request('GET', ENDPOINT, params=params).prepare().url
            path = RAW/f'{ipo_id}_{series}.json.gz'
            try:
                from_cache = path.exists()
                if from_cache:
                    with gzip.open(path, 'rb') as stream:
                        raw = stream.read()
                else:
                    response = session.get(ENDPOINT, params=params, headers=HEADERS, timeout=25)
                    if response.status_code in (403,429):
                        access_limited = True
                        raise RuntimeError(f'NSE access/rate limit {response.status_code}; collection stopped')
                    response.raise_for_status()
                    raw = response.content
                    json.loads(raw)
                    with gzip.open(path, 'wb') as stream:
                        stream.write(raw)
                digest = hashlib.sha256(raw).hexdigest()
                found = parse_prices(json.loads(raw), ipo_id=int(ipo_id), symbol=symbol,
                    first=listing, last=last, source_url=url, digest=digest, series=series)
                # This NSE endpoint returned only the last 70 rows for long
                # windows. Recover the beginning in a shorter, overlapping
                # window instead of treating those missing sessions as absent.
                if len((json.loads(raw).get('data') or [])) >= 70:
                    prefix_params = {**params, 'to': (listing+pd.Timedelta(days=75)).strftime('%d-%m-%Y')}
                    prefix_url = requests.Request('GET',ENDPOINT,params=prefix_params).prepare().url
                    prefix_path = RAW/f'{ipo_id}_{series}_prefix.json.gz'
                    prefix_cached = prefix_path.exists()
                    if prefix_cached:
                        with gzip.open(prefix_path,'rb') as stream:
                            prefix_raw = stream.read()
                    else:
                        time.sleep(.45)
                        prefix_response = session.get(ENDPOINT,params=prefix_params,headers=HEADERS,timeout=25)
                        if prefix_response.status_code in (403,429):
                            access_limited=True
                            raise RuntimeError('NSE access/rate limit while collecting prefix; stopped')
                        prefix_response.raise_for_status()
                        prefix_raw=prefix_response.content
                        json.loads(prefix_raw)
                        with gzip.open(prefix_path,'wb') as stream:
                            stream.write(prefix_raw)
                    prefix_sha=hashlib.sha256(prefix_raw).hexdigest()
                    prefix_rows=parse_prices(json.loads(prefix_raw),ipo_id=int(ipo_id),symbol=symbol,
                        first=listing,last=listing+pd.Timedelta(days=75),source_url=prefix_url,digest=prefix_sha,series=series)
                    overlap={r['session_date']:r['nse_close'] for r in found}
                    if any(r['session_date'] in overlap and abs(r['nse_close']-overlap[r['session_date']])>.0001 for r in prefix_rows):
                        raise ValueError('Overlapping official response prices conflict')
                    found += [r for r in prefix_rows if r['session_date'] not in overlap]
                    attempts.append({'ipo_id':int(ipo_id),'series':series,'source_url':prefix_url,
                                     'raw_sha256':prefix_sha,'rows':len(prefix_rows),'cached_response':prefix_cached,
                                     'purpose':'recover source-truncated beginning'})
                records.extend(found)
                attempts.append({'ipo_id': int(ipo_id), 'series': series, 'source_url': url,
                                 'raw_sha256': digest, 'rows': len(found), 'cached_response': from_cache})
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                attempts.append({'ipo_id': int(ipo_id), 'series': series, 'status':'unavailable', 'error':str(exc)[:200]})
            time.sleep(.45)
            if access_limited:
                break
        if access_limited:
            break
        observed = pd.DataFrame(records)
        reason = None
        if observed.empty or len(observed)<62:
            reason = 'fewer than 62 actual exchange trading observations'
        elif observed.duplicated('session_date').any():
            reason = 'ambiguous same-day multi-series observations'
        else:
            observed['date'] = pd.to_datetime(observed.session_date)
            observed = observed.sort_values('date').head(62)
            if observed.iloc[0].date != listing:
                reason = 'listing-day close missing from exchange response'
            elif list(observed.date) != list(benchmarks['^NSEI'].dropna().loc[listing:].head(62).index):
                reason = 'first 62 stock observations do not match the actual Nifty session calendar'
            else:
                meta = stock_meta.loc[ipo_id]
                nifty = benchmarks['^NSEI'].reindex(observed.date)
                sector = benchmarks[meta.sector_ticker].reindex(observed.date) if meta.sector_ticker in benchmarks else None
                if nifty.isna().any():
                    reason = 'missing real Nifty observation on an exchange stock session'
                else:
                    new = pd.DataFrame({'ipo_id': int(ipo_id), 'date': observed.date.to_numpy(),
                        'close_price_adjusted': observed.nse_close.to_numpy(),
                        'sector_index_close': sector.to_numpy() if sector is not None else float('nan'),
                        'nifty50_close': nifty.to_numpy(), 'ticker': f'{symbol}.NS',
                        'sector_ticker': meta.sector_ticker, 'benchmark_source': meta.benchmark_source,
                        'price_basis': 'Official NSE security-wise archive unadjusted Close (EQ/BE)'})
                    added.append(new)
        if reason:
            excluded.append({'ipo_id': int(ipo_id), 'symbol': symbol, 'reason': reason})
        print(f'{i}/{len(ids)} {symbol}: '+(reason or '62 real sessions verified'), flush=True)
    combined = pd.concat([fixed, *added], ignore_index=True).sort_values(['ipo_id','date'])
    if combined.duplicated(['ipo_id','date']).any():
        raise ValueError('Extension would replace an existing price observation')
    combined.to_csv(OUTPUT,index=False)
    report = {'censored_2024_ipos': len(ids), 'fully_recovered_ipos': len(added),
              'development_ipos_before': int(fixed.ipo_id.nunique()),
              'development_ipos_after': int(combined.ipo_id.nunique()),
              'new_exchange_sessions': sum(len(x) for x in added), 'attempts': attempts,
              'excluded': excluded, 'stopped_for_access_limit': access_limited,
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (BASE,REPAIRED,BENCHMARKS)},
              'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              'limitations':['Real 2025 sessions are outcomes for 2024 IPOs, not permission to fit future labels before their publication.',
                             'Nifty and sector values are dated Yahoo observations, not exchange-certified benchmarks.',
                             'Missing sector cells stay missing; prices and trading sessions are not interpolated.',
                             'Historical symbol-change gaps are not resolved by guessing aliases.']}
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('fully_recovered_ipos','development_ipos_after','new_exchange_sessions','excluded')},indent=2))


if __name__=='__main__':
    main()
