"""Additive, bounded recovery of dated GMP; no change to canonical/live inputs."""
import gzip
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ml.collect_preclose_gmp import parse_gmp_history
from ml.fetch_daywise_subscriptions import HEADERS,SEARCH_URL,POST_URL,name_tokens
from ml.train_stage_fundamental_challenger import ROOT

CACHE=ROOT/'data/archive/module1_gap_recovery_20261024'
OUT=ROOT/'data/audit/module1_gmp_gap_recovery_20261024.csv'
AUDIT=ROOT/'output/module1_gmp_gap_recovery_20261024.json'


def normalized(value):
    words=re.findall(r'[a-z0-9]+',value.lower())
    return ''.join(w for w in words if w not in {'ltd','limited','private','and'})


def history(content,close_date):
    """Normalize explicit GMP/date headers only, never infer missing values."""
    soup=BeautifulSoup(content,'html.parser')
    for tr in soup.find_all('tr'):
        cells=tr.find_all(['td','th'])
        if len(cells)>=2 and cells[0].get_text(' ',strip=True).lower() in ('date','gmp date'):
            title=cells[1].get_text(' ',strip=True)
            if re.fullmatch(r'(?:IPO\s+)?GMP\s*(?:\((?:₹|Rs\.?|INR)\)|₹)?',title,re.I):
                cells[0].string='Date';cells[1].string='GMP'
    return parse_gmp_history(str(soup),close_date)


def verified_post(item,title,post):
    content=post.get('content',{}).get('rendered','')
    soup=BeautifulSoup(content,'html.parser');text=soup.get_text(' ',strip=True)
    prefix=re.split(r'\bIPO\b',title,flags=re.I)[0]
    # Legal-name text or exact normalized title; not fuzzy ticker matching.
    identity=normalized(item['company_name'])
    if identity not in normalized(text) and identity!=normalized(prefix):
        return False
    name_overlap=set(name_tokens(item['company_name']))&set(name_tokens(prefix))
    if normalized(prefix)!=identity and len(name_overlap)<min(2,len(name_tokens(item['company_name']))):
        return False
    close=[]
    for tr in soup.find_all('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])]
        if len(cells)==2 and re.fullmatch(r'(?:IPO |Bid/Offer )?Clos(?:e|ing)(?: Date)?\s*:?',cells[0],re.I):
            parsed=pd.to_datetime(cells[1],dayfirst=True,errors='coerce')
            if pd.notna(parsed):
                close.append(parsed.strftime('%Y-%m-%d'))
    era=pd.to_datetime(post.get('date'),errors='coerce')
    expected=pd.Timestamp(item['issue_close_date'])
    if pd.isna(era) or abs((era.normalize()-expected).days)>45:
        return False
    if close:
        return set(close)=={item['issue_close_date']}
    opening=re.search(r'IPO is to open on\s+([A-Za-z]+\s+\d{1,2})(?:\s*[, ]\s*(20\d{2}))?',text,re.I)
    band=re.search(r'price band is fixed at\s*₹?\s*[\d,]+(?:\.\d+)?\s*(?:to|-)\s*₹?\s*([\d,]+(?:\.\d+)?)',text,re.I)
    if not opening or not band:
        return False
    parsed=pd.to_datetime(f"{opening[1]} {opening[2] or expected.year}",errors='coerce')
    return (pd.notna(parsed) and parsed==pd.Timestamp(item['issue_open_date']) and
        abs(float(band[1].replace(',',''))-float(item['issue_price']))<.01)


def collect(item):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/f"{item['ipo_id']}.json.gz"
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            cached=json.load(stream)
        if not (cached.get('status')=='fetch_error' and '404' in cached.get('error','') and
                cached.get('recovery_version')!='v2-skip-dead-posts'):
            return cached
    result=dict(ipo_id=int(item['ipo_id']),company_name=item['company_name'],rows=[],
        retrieved_at_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
        evidence_class='retrospective_daily_table_not_immutable_quote',attempted_urls=[],
        recovery_version='v2-skip-dead-posts')
    try:
        tokens=name_tokens(item['company_name'])
        while len(tokens)>1 and len(tokens[0])<=2 and len(tokens[1])==1:
            tokens=[tokens[0]+tokens[1]]+tokens[2:]
        query=' '.join(tokens[:2])+' IPO GMP'
        response=requests.get(SEARCH_URL,params={'search':query,'per_page':20},headers=HEADERS,timeout=12)
        response.raise_for_status()
        hits=[]
        for hit in response.json():
            title=BeautifulSoup(html.unescape(hit.get('title','')),'html.parser').get_text(' ',strip=True)
            if 'gmp' in title.lower():
                hits.append((hit,title))
        hits=sorted(hits,key=lambda x:-len(set(tokens)&set(name_tokens(x[1]))))[:3]
        for hit,title in hits:
            url=POST_URL.format(post_id=hit['id']);result['attempted_urls'].append(url)
            raw=requests.get(url,headers=HEADERS,timeout=12)
            if raw.status_code==404:
                continue  # Search can return a page whose ID is not a post.
            raw.raise_for_status();post=raw.json()
            if not verified_post(item,title,post):
                continue
            rows=history(post.get('content',{}).get('rendered',''),item['issue_close_date'])
            if not rows:
                continue
            result.update(rows=rows,source_url=post['link'],source_sha256=hashlib.sha256(raw.content).hexdigest(),
                raw_response_text=raw.text,status='matched_daily_history',identity_basis='legal_name_offer_era_and_verified_offer_dates_or_price')
            break
        result.setdefault('status','no_verified_dated_history')
    except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
        result.update(status='fetch_error',error=str(exc)[:200])
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        json.dump(result,stream)
    return result


def run(early):
    offers=pd.read_csv(ROOT/'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    tasks=[]
    for row in early[early.gmp_pct.isna()].itertuples():
        item=offers.loc[row.ipo_id].to_dict()
        item.update(ipo_id=int(row.ipo_id),company_name=row.company_name,issue_price=float(row.issue_price),
            issue_open_date=str(pd.Timestamp(row.issue_open_date).date()),issue_close_date=str(pd.Timestamp(row.issue_close_date).date()))
        tasks.append(item)
    results=[]
    def save():
        rows=[]
        for result in sorted(results,key=lambda r:r['ipo_id']):
            for quote in result['rows']:
                rows.append({k:result[k] for k in ('ipo_id','company_name','source_url','source_sha256','retrieved_at_ist','evidence_class')}|quote)
        pd.DataFrame(rows,columns=['ipo_id','company_name','source_url','source_sha256','retrieved_at_ist','evidence_class','gmp_date','gmp_rs']).to_csv(OUT,index=False)
        audit=dict(attempted=len(tasks),completed=len(results),matched=sum(bool(r['rows']) for r in results),
            quote_rows=len(rows),sources=[{k:v for k,v in r.items() if k not in ('raw_response_text','rows')} for r in sorted(results,key=lambda r:r['ipo_id'])])
        AUDIT.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(collect,item) for item in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%10==0 or len(results)==len(tasks):
                save();print(json.dumps(dict(gmp_gap_completed=len(results),total=len(tasks),matched=sum(bool(r['rows']) for r in results))),flush=True)
    save()
    return pd.read_csv(OUT)


if __name__=='__main__':
    from ml.research_module1_gmp_fundamentals import digest,PREVIOUS
    from ml.research_module1_stage_transfer import datasets
    from ml.train_stage_fundamental_challenger import attach
    from services.dated_ipo_fundamentals import REGISTRY
    # The preceding complete extraction-validation report binds the cached
    # facts. Reuse those facts; do not reread PDFs for every benchmark run.
    previous=json.loads(PREVIOUS.read_text())
    for relative,expected in previous['input_sha256'].items():
        if digest(ROOT/relative)!=expected:
            raise ValueError(f'Revalidate changed extraction inputs: {relative}')
    payload=json.loads(REGISTRY.read_text(encoding='utf-8'))
    registry={r['ipo_id']:r for r in payload['records'] if r['status']=='dated_document_facts'}
    _,early,_=datasets()
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    early=attach(early,registry,canonical,'penultimate_session')
    frame_path=ROOT/'data/audit/module1_completion_early_20261024.csv'
    early.to_csv(frame_path,index=False)
    manifest=previous['input_sha256'].copy()
    manifest[PREVIOUS.relative_to(ROOT).as_posix()]=digest(PREVIOUS)
    manifest['data/preclose_gmp_history.csv']=digest(ROOT/'data/preclose_gmp_history.csv')
    manifest[frame_path.relative_to(ROOT).as_posix()]=digest(frame_path)
    (ROOT/'output/module1_completion_inputs_20261024.json').write_text(json.dumps(manifest,indent=2)+'\n')
    run(early)
