"""Read dated IPOJI category/total tables already cached; no downloads/OCR."""
import gzip,html,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from services.historical_bid_service import name_key


def parse_history(content,row):
    match=re.search(r'<script[^>]*id=["\']ipo-subscription-info-for-chart["\'][^>]*>(.*?)</script>',content,re.S)
    if not match:return None
    # Reuse only an issuer/offer identity compatible with the frozen input.
    text=html.unescape(re.sub('<[^>]+>',' ',content))
    if name_key(row['company_name']) not in name_key(text):return None
    try:payload=json.loads(match[1])
    except ValueError:return None
    cutoff=pd.Timestamp(row['cutoff_at_ist']);history=[];reserved=None
    def number(value):
        from services.subscription_numbers import subscription_multiple
        return subscription_multiple(value)
    for item in payload:
        key=item.get('key','');values=item.get('value',{})
        if key.lower()=='shares offered / reserved':reserved={k:number(values.get(k)) for k in ('QIB','NII','RII','Total')};continue
        date=re.search(r'Day\s*(\d+)\s*-\s*(\d{2}-\d{2}-\d{2,4})\s+(\d{1,2}:\d{2}\s*[AP]M)',key,re.I)
        if not date:continue
        fmt='%d-%m-%y %I:%M %p' if len(date[2])==8 else '%d-%m-%Y %I:%M %p'
        observed=pd.to_datetime(date[2]+' '+date[3],format=fmt).tz_localize('Asia/Kolkata')
        if observed>cutoff:continue
        parsed={k:number(values.get(k)) for k in ('QIB','NII','RII','Total','sNII','bNII')}
        history.append(dict(day=int(date[1]),observed=observed,values=parsed))
    current=[]
    for r in history:
        pair=[(r['values'].get(k),np.expm1(row.get(f,np.nan))) for k,f in [('QIB','log_qib'),('NII','log_hni'),('RII','log_retail')]]
        if all(a is not None and np.isfinite(b) and np.isclose(a,b,atol=.015,rtol=1e-4) for a,b in pair):current.append(r)
    if not current:return None
    chosen=max(current,key=lambda r:r['observed']);prior=[r for r in history if r['day']==2 and r['observed']<=chosen['observed'] and r['values']['Total'] is not None]
    result=dict(matched_bid_timestamp=chosen['observed'].isoformat(),history_source_basis='Publisher dated table recovered later; category multiples match frozen replay input.')
    if prior and chosen['values']['Total'] is not None:
        day2=max(prior,key=lambda r:r['observed']);result['day2_to_cutoff_conviction']=float(np.log1p(day2['values']['Total'])/max(.1,np.log1p(chosen['values']['Total'])))
    if reserved and all(v is not None for v in reserved.values()) and reserved['Total']>0 and abs(sum(reserved[k] for k in ('QIB','NII','RII'))-reserved['Total'])<1:
        weights={k:reserved[k]/reserved['Total'] for k in ('QIB','NII','RII')}
        total=sum(weights[k]*chosen['values'][k] for k in weights)
        if chosen['values']['Total'] is not None and abs(total-chosen['values']['Total'])<.03:
            result['quota_weighted_book_mult']=float(total);result['reservation_basis']='Reported category subscription denominators, excluding any separate anchor/employee portion; weighted total reconciles.'
    return result


def cached_pages(value):
    if isinstance(value,dict):
        if isinstance(value.get('raw_response_text'),str):yield value['raw_response_text'],value.get('url') or value.get('source_url')
        for k,v in value.items():
            if isinstance(v,(dict,list)):yield from cached_pages(v)
    elif isinstance(value,list):
        for v in value:yield from cached_pages(v)


def recover(rows):
    by_id={str(r['issuer_key']):r for r in rows};found={}
    paths=[]
    for directory in ('module1_later_holdout_20261024','module1_other_sources_20261024','module1_public_financials_20261024'):
        paths.extend((p,p.name.split('.')[0]) for p in (ROOT/'data/archive'/directory).glob('*.json.gz') if p.name.split('.')[0] in by_id)
    for path,key in paths:
        if key in found and 'day2_to_cutoff_conviction' in found[key] and 'quota_weighted_book_mult' in found[key]:continue
        try:data=json.loads(gzip.decompress(path.read_bytes()))
        except (OSError,ValueError):continue
        for content,url in cached_pages(data):
            result=parse_history(content,by_id[key])
            if result:found[key]={**found.get(key,{}),**result,'cache_path':str(path.relative_to(ROOT)),'source_url':url}
    recent=[r for r in rows if str(r['issuer_key']).startswith('recent_')]
    for path in (ROOT/'data/telemetry/listed_input_recovery').glob('*.html.gz'):
        content=gzip.decompress(path.read_bytes()).decode('utf8',errors='replace')
        if 'ipo-subscription-info-for-chart' not in content:continue
        for row in recent:
            result=parse_history(content,row)
            if result:found[row['issuer_key']]={**found.get(row['issuer_key'],{}),**result,'cache_path':str(path.relative_to(ROOT))}
    return found
