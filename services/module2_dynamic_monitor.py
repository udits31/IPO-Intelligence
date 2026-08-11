"""One shared monitor policy for exchange-identified listings outside its CSV.

No separate fallback trading tree, guessed symbol, fake ID, listing-date close
or default offer terms. The same monitor handles dated supplied prices/news.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from core.attribution_engine import AttributionEngine
from core.monitor_engine import HoldSellMonitor
from services.news_identity import tokens


def issuer_matches(expected,observed):
    first,second=tokens(expected),tokens(observed)
    if not first or not second:
        return False
    # A shared one-word brand cannot identify a different longer-name issuer.
    if len(first)==1 or len(second)==1:
        return first==second
    return first==second or (len(first)>=3 and len(second)>=3 and
                             first[:min(len(first),len(second))]==second[:min(len(first),len(second))])


def _number(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (TypeError,ValueError):
        return None


def evaluate_dynamic_listing(data,*,target,canonical_entry=None,as_of_date=None):
    if as_of_date is None:
        from services.market_cache_freshness import invalidate_preclose_market_cache
        invalidate_preclose_market_cache()
    company=str(data.get('company_name') or target).strip()
    pending={'status':'DATA_UNAVAILABLE','company_name':company,
             'message':'Verified listing identity and dated prices are required before holding advice.'}
    issue=_number(data.get('issue_price'))
    listed=pd.to_datetime(data.get('listing_date'),errors='coerce')
    if issue is None or issue<=0 or pd.isna(listed):
        return pending
    cutoff=pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(datetime.now(ZoneInfo('Asia/Kolkata')).date())
    if listed.normalize()>cutoff.normalize():
        return {'status':'PRE_LISTING','company_name':company,'listing_date':listed.date().isoformat(),
                'message':'This issue has not listed at the selected date. Use Module 1 for its pre-close forecast.'}
    if canonical_entry is not None:
        ipo_id=int(canonical_entry['id'])
    else:
        match=re.search(r'/ipo/[^/]+/(\d+)/?',str(data.get('url') or data.get('chittorgarh_url') or target))
        explicit_id=_number(data.get('ipo_id'))
        ipo_id=int(match.group(1)) if match else int(explicit_id) if explicit_id is not None and explicit_id.is_integer() else None
    if ipo_id is None or ipo_id<=0:
        return {**pending,'message':'Listing name was found but no stable source IPO identifier was supplied; no synthetic holding record is created.'}
    nse=str(data.get('nse_symbol') or '').strip().upper()
    nse=nse if nse not in {'NAN','NONE','NULL'} and re.fullmatch(r'[A-Z0-9&\-]{2,24}',nse) else None
    bse_number=_number(data.get('bse_code') or data.get('bse_scrip_code'))
    bse=str(int(bse_number)) if bse_number is not None and bse_number.is_integer() and bse_number>0 else None
    from services.bse_identity import archived_bse_identity
    bse_identity=archived_bse_identity(bse,listed,ipo_id) if bse else None
    candidates=([(nse+'.NS','NSE')] if nse else [])+([(bse_identity['symbol']+'.BO','BSE')] if bse_identity else [])
    stock=None;provider={};exchange=None;failure=None
    for ticker,venue in candidates:
        try:
            observed=AttributionEngine._fetch_yahoo_series(ticker,ttl_seconds=900,start_date=listed)
            meta=observed.attrs.get('provider_meta',{})
            if observed.empty or meta.get('symbol')!=ticker or not issuer_matches(company,meta.get('company_name','')):
                continue
            stock,provider,exchange=observed,dict(meta),venue
            if venue=='BSE':
                provider['bse_identity']=bse_identity
            break
        except Exception as exc:
            failure=type(exc).__name__
    if stock is None:
        return {'status':'PRICE_UNAVAILABLE','company_name':company,'listing_date':listed.date().isoformat(),
                'bse_code':bse,'message':'No dated price history matched the registered exchange code and issuer name. No holding advice is inferred.',
                'source_error_type':failure}
    now_ist=datetime.now(ZoneInfo('Asia/Kolkata'))
    if not as_of_date and listed.date()==now_ist.date() and (now_ist.hour,now_ist.minute)<(15,30):
        return dict(status='FIRST_TRADING_SESSION',company_name=company,ipo_id=ipo_id,
            nse_symbol=nse,bse_code=bse,listing_date=listed.date().isoformat(),issue_price=issue,
            message='First trading session in progress. The live minute chart and intraday signals are available above. The daily hold/sell assessment begins after the first completed market close.')
    try:
        nifty=AttributionEngine._fetch_yahoo_series('^NSEI',ttl_seconds=900).rename(columns={'close':'nifty50_close'})
    except Exception:
        nifty=pd.DataFrame(columns=['date','nifty50_close'])
    sector_name=str(data.get('sector') or 'Unclassified')
    try:
        sector,label=AttributionEngine._get_sector_series(sector_name)
    except Exception:
        sector=pd.DataFrame(columns=['date','close']);label='Nifty 50 broad-market proxy'
    if sector.empty:
        sector=nifty.rename(columns={'nifty50_close':'sector_index_close'})
        label='Nifty 50 broad-market proxy'
    else:
        sector=sector.rename(columns={'close':'sector_index_close'})
    prices=stock.rename(columns={'close':'close_price_adjusted'}).merge(nifty,on='date',how='left').merge(
        sector[['date','sector_index_close']],on='date',how='left')
    prices['ipo_id']=ipo_id
    prices['price_source']=f'Yahoo Finance ({exchange})'
    prices['price_basis']='Unadjusted daily close'
    prices['archive_sha256']=provider.get('raw_sha256')
    prices['benchmark_source']=label
    prices.attrs['provider_meta']=provider
    ipo=pd.DataFrame([{'id':ipo_id,'company_name':company,'sector':sector_name,
        'nse_symbol':nse if exchange=='NSE' else None,'bse_scrip_code':bse,
        'listing_date':listed.date().isoformat(),'issue_price':issue,
        'anchor_lockin_50pct_date':(listed+pd.Timedelta(days=30)).date().isoformat(),
        'anchor_lockin_full_date':(listed+pd.Timedelta(days=90)).date().isoformat(),
        'anchor_lockin_dates_verified':False,'anchor_lockin_date_basis':'listing_date_heuristic_not_allotment_evidence'}])
    monitor=HoldSellMonitor(prices_frame=prices,ipos_frame=ipo,prefer_supplied_prices=True)
    result=monitor.evaluate_holding(ipo_id,as_of_date=as_of_date)
    result['exchange']=exchange
    result['bse_code']=bse
    result['monitor_policy']='shared_core_monitor'
    return result
