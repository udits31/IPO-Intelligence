"""Dated share-basis reconciliation, without guessing changes from price drops.

NSE raw closes and Yahoo split-adjusted closes have different historical bases.
Both are converted to the share units at the selected observation date. The
latest quote itself is not changed. Cash dividends/rights proceeds are not
invented. An unsupported capital event or a ratio conflict pauses advice.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs,urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
ARCHIVES=ROOT/'data/archive/corporate_actions_live'
INDEX=ROOT/'data/telemetry/corporate_action_sources.json'
BOOTSTRAP=ROOT/'output/module2_corporate_action_reconciliation_20261021.json'
ENDPOINT='https://www.nseindia.com/api/corporates-corporateActions'
PAGE='https://www.nseindia.com/companies-listing/corporate-filings-actions'
HEADERS={'User-Agent':'Mozilla/5.0','Referer':PAGE,'Accept':'application/json,text/plain,*/*'}
_LOCK=threading.RLock()
_SOURCES=None
_ATTEMPTS={}


class PriceBasisError(ValueError):
    pass


def share_change(subject):
    text=str(subject).strip()
    bonus=re.fullmatch(r'Bonus\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',text,re.I)
    if bonus:
        new,old=map(float,bonus.groups())
        if not new>0 or not old>0:
            raise PriceBasisError('Invalid announced bonus ratio')
        return 'bonus',1+new/old
    split=re.fullmatch(r'Face Value Split \(Sub-Division\)\s*-\s*From Rs\s*(\d+(?:\.\d+)?)/- Per Share '
                       r'To Re?s?\s*(\d+(?:\.\d+)?)/- Per Share',text,re.I)
    if split:
        old,new=map(float,split.groups())
        if not old>new>0:
            raise PriceBasisError('Invalid announced split ratio')
        return 'split',old/new
    if re.match(r'(?:Interim |Final |Special )?Dividend\b',text,re.I):
        return 'cash_dividend_not_included',1.
    if text.casefold() in ('annual general meeting','extra ordinary general meeting') or re.fullmatch(
            r'Annual General Meeting/Dividend(?:\s*-\s*(?:Rs|Re)\s*\d+(?:\.\d+)? Per Share)?',text,re.I):
        return 'meeting_or_cash_not_included',1.
    return 'unsupported',None


def parse_actions(payload,first,last):
    if not isinstance(payload,list):
        raise PriceBasisError('NSE corporate actions did not return a list')
    rows=[];seen=set()
    for item in payload:
        date=pd.to_datetime(item.get('exDate'),format='%d-%b-%Y',errors='raise')
        symbol=str(item.get('symbol') or '').strip().upper()
        if pd.isna(date) or not first<=date<=last or not symbol:
            raise PriceBasisError('Corporate-action date or symbol invalid')
        key=(symbol,date,str(item.get('subject')),str(item.get('isin')))
        if key in seen:
            continue
        seen.add(key)
        kind,factor=share_change(item.get('subject'))
        rows.append({'symbol':symbol,'ex_date':date.strftime('%Y-%m-%d'),'subject':item.get('subject'),
                     'isin':item.get('isin'),'kind':kind,'share_factor':factor})
    return rows


def _read_source(meta):
    location=urlparse(meta['source_url'])
    if location.hostname!='www.nseindia.com' or location.path!='/api/corporates-corporateActions':
        raise PriceBasisError('Unexpected corporate-action source authority')
    path=(ROOT/meta['raw_archive_path']).resolve()
    path.relative_to((ROOT/'data/archive').resolve())
    with gzip.open(path,'rb') as stream:
        raw=stream.read()
    if hashlib.sha256(raw).hexdigest()!=meta['raw_sha256']:
        raise PriceBasisError('Corporate-action source archive hash mismatch')
    query=parse_qs(urlparse(meta['source_url']).query)
    first=pd.to_datetime(query['from_date'][0],format='%d-%m-%Y')
    last=pd.to_datetime(query['to_date'][0],format='%d-%m-%Y')
    if first.year!=last.year:
        raise PriceBasisError('Expected one annual action-source window')
    return {**meta,'from_date':first.strftime('%Y-%m-%d'),'to_date':last.strftime('%Y-%m-%d'),
            'year':int(first.year),'events':parse_actions(json.loads(raw),first,last)}


def _sources():
    global _SOURCES
    with _LOCK:
        if _SOURCES is None:
            _SOURCES=[]
            candidates=[]
            for path,key in [(BOOTSTRAP,'action_windows'),(INDEX,'sources')]:
                try:
                    candidates.extend(json.loads(path.read_text(encoding='utf-8')).get(key,[]))
                except (OSError,ValueError):
                    continue
            seen=set()
            for meta in candidates:
                try:
                    if meta.get('raw_sha256') in seen:
                        continue
                    source=_read_source(meta)
                    _SOURCES.append(source)
                    seen.add(meta['raw_sha256'])
                except (OSError,ValueError,KeyError):
                    continue  # A corrupt/absent archive is not verified coverage.
        return _SOURCES


def _save_index():
    metadata=[{key:value for key,value in r.items() if key!='events'} for r in _sources()]
    INDEX.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=INDEX.parent,delete=False) as stream:
            temporary=stream.name
            json.dump({'schema':1,'sources':metadata},stream,indent=2)
        os.replace(temporary,INDEX)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _year_source(year,first,last,allow_network):
    with _LOCK:
        covered=[r for r in _sources() if r['year']==year and r['from_date']<=first.strftime('%Y-%m-%d')
                 and r['to_date']>=last.strftime('%Y-%m-%d')]
        if covered:
            return max(covered,key=lambda r:r['to_date'])
        key=(year,last.strftime('%Y-%m-%d'))
        if not allow_network or time.monotonic()-_ATTEMPTS.get(key,-100000)<900:
            raise PriceBasisError(f'Corporate-action coverage is unavailable through {last:%d %b %Y}; last quote retained without a current call')
        _ATTEMPTS[key]=time.monotonic()
        params={'index':'equities','from_date':f'01-01-{year}','to_date':last.strftime('%d-%m-%Y')}
        session=requests.Session()
        landing=session.get(PAGE,headers=HEADERS,timeout=10)
        landing.raise_for_status()
        response=session.get(ENDPOINT,params=params,headers=HEADERS,timeout=15)
        response.raise_for_status()  # No bypass/retry after an access restriction.
        raw=response.content
        parse_actions(response.json(),pd.Timestamp(year=year,month=1,day=1),last)
        digest=hashlib.sha256(raw).hexdigest()
        ARCHIVES.mkdir(parents=True,exist_ok=True)
        path=ARCHIVES/f'{year}_{last:%Y%m%d}_{digest[:12]}.json.gz'
        with gzip.open(path,'wb') as stream:
            stream.write(raw)
        meta={'source_url':response.url,'raw_sha256':digest,'raw_archive_path':str(path.relative_to(ROOT)),
              'retrieved_at_utc':datetime.now(ZoneInfo('UTC')).isoformat()}
        source=_read_source(meta)
        _sources().append(source)
        _save_index()
        return source


def official_actions(symbol,listed,asof,*,allow_network=True):
    listed,asof=pd.Timestamp(listed).normalize(),pd.Timestamp(asof).normalize()
    events=[];sources=[]
    for year in range(listed.year,asof.year+1):
        first=max(listed,pd.Timestamp(year=year,month=1,day=1))
        last=min(asof,pd.Timestamp(year=year,month=12,day=31))
        source=_year_source(year,first,last,allow_network)
        events.extend({**event,'source_url':source['source_url'],'source_sha256':source['raw_sha256']}
                      for event in source['events'] if event['symbol']==symbol and listed<=pd.Timestamp(event['ex_date'])<=asof)
        sources.append({k:source.get(k) for k in ('source_url','raw_sha256','from_date','to_date','retrieved_at_utc')})
    return events,sources


def reconcile_prices(frame,ipo,*,asof,provider_meta=None,allow_network=False):
    """Return comparable prices, comparable issue price, and auditable basis.

    Research listing-share values are rebased to actual as-of share units.
    Yahoo quote.close history already includes split adjustments; never apply
    those a second time. NSE raw history needs a dated adjustment per row.
    """
    result=frame.copy()
    listed=pd.Timestamp(ipo['listing_date']).normalize()
    asof=pd.Timestamp(asof).normalize()
    original=float(ipo['issue_price'])
    if not math.isfinite(original) or original<=0:
        raise PriceBasisError('A finite positive observed issue price is required')
    basis=str(result.iloc[-1].get('price_basis') or '')
    source=str(result.iloc[-1].get('price_source') or '')
    meta=provider_meta or {}
    source_kind=meta.get('historical_basis')
    if 'listing-share units' in basis:
        if 'shares_per_listing_share' not in result:
            raise PriceBasisError('Listing-share series lacks its dated share factors')
        factor=float(result.iloc[-1].shares_per_listing_share)
        result['close_price_adjusted']=result.close_price_adjusted/factor
        return result,original/factor,{'status':'comparable','kind':'listing_share_rebase','share_factor':factor,
            'original_issue_price':original,'comparable_issue_price':original/factor,
            'events':[],'source_note':'Already reconciled research series; no further action inference.'}
    recognized_raw=('unadjusted' in basis.casefold()) or source=='NSE UDiFF CM'
    if source_kind!='split_adjusted_quote_close' and not recognized_raw:
        return result,original,{'status':'unspecified_legacy_basis','share_factor':None,'original_issue_price':original,
                                'comparable_issue_price':original,'events':[]}
    symbol=str(ipo.get('nse_symbol') or '').strip().upper()
    provider_only=False
    if not symbol or symbol=='NAN':
        code=str(ipo.get('bse_scrip_code') or '').removesuffix('.0')
        from services.bse_identity import archived_bse_identity
        try:
            identity=archived_bse_identity(code,listed,int(ipo['id'])) if code.isdigit() else None
        except (OSError,ValueError,KeyError) as exc:
            raise PriceBasisError(str(exc)) from exc
        if (source_kind!='split_adjusted_quote_close' or not code.isdigit()
                or not identity or meta.get('bse_identity')!=identity
                or meta.get('symbol')!=identity['symbol']+'.BO'):
            raise PriceBasisError('A registered exchange identity is required for share-basis reconciliation')
        events=[{'symbol':code,'ex_date':e['ex_date'],'subject':'Explicit provider split/bonus event',
                 'share_factor':e['share_factor'],'kind':'split','source_url':meta.get('source_url'),
                 'source_sha256':meta.get('raw_sha256')}
                for e in meta.get('splits',[]) if listed<pd.Timestamp(e['ex_date'])<=asof]
        sources=[{'source_url':meta.get('source_url'),'raw_sha256':meta.get('raw_sha256')}]
        provider_only=True
    else:
        try:
            events,sources=official_actions(symbol,listed,asof,allow_network=allow_network)
        except (requests.RequestException,OSError,ValueError,KeyError) as exc:
            raise PriceBasisError(str(exc)) from exc
    unsupported=[e for e in events if e['kind']=='unsupported']
    if unsupported:
        raise PriceBasisError('Unresolved capital event: '+str(unsupported[0]['subject'])+'. Current quote retained; advice paused until its share/entitlement basis is resolved.')
    changes=[e for e in events if e['kind'] in ('bonus','split')]
    if any(pd.Timestamp(e['ex_date'])==listed for e in changes):
        raise PriceBasisError('A listing-day share change needs explicit offer-price basis verification')
    by_date={}
    for event in changes:
        by_date[event['ex_date']]=by_date.get(event['ex_date'],1.)*event['share_factor']
    factor=float(np.prod(list(by_date.values())))
    if not math.isfinite(factor) or factor<=0:
        raise PriceBasisError('Invalid cumulative announced share ratio')
    if source_kind=='split_adjusted_quote_close':
        splits=meta.get('splits')
        if not isinstance(splits,list):
            raise PriceBasisError('Yahoo historical adjustment metadata is unavailable')
        observed={e['ex_date']:float(e['share_factor']) for e in splits
                  if listed<pd.Timestamp(e['ex_date'])<=asof}
        for date,value in by_date.items():
            if date not in observed or not np.isclose(observed[date],value,rtol=1e-8):
                raise PriceBasisError('NSE and Yahoo share-change history conflict on '+date)
        # A provider-only event is not silently declared exchange-verified.
        unmatched=[e for e in splits if listed<pd.Timestamp(e['ex_date'])<=asof and e['ex_date'] not in by_date]
        if unmatched:
            raise PriceBasisError('Yahoo reports a share change absent from the queried NSE action record: '+unmatched[0]['ex_date'])
        future_factor=float(np.prod([e['share_factor'] for e in splits if pd.Timestamp(e['ex_date'])>asof]))
        result['close_price_adjusted']=result.close_price_adjusted*future_factor
    else:
        per_row=np.ones(len(result))
        for date,value in by_date.items():
            per_row[result.date.ge(pd.Timestamp(date)).to_numpy()]*=value
        result['close_price_adjusted']=result.close_price_adjusted*per_row/factor
    return result,original/factor,{'status':'comparable','kind':'dated_split_bonus_asof_share_units',
        'asof_date':asof.strftime('%Y-%m-%d'),'share_factor':factor,'original_issue_price':original,
        'comparable_issue_price':original/factor,'events':changes,'source_windows':sources,
        'cash_dividends_included':False,'current_quote_preserved':True,
        'event_authority':'secondary_provider_event_records' if provider_only else 'NSE_action_records_cross_checked',
        'source_note':'Only explicit effective share ratios; queried source coverage is not independent proof of exhaustive historical records.'}
