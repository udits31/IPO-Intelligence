"""Mine local HTML/JSON archives first; never assert an unknown publication time."""
import gzip,hashlib,json,re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from services.point_in_time import evidence,timestamp,eligible
from services.historical_bid_service import name_key
from services.saved_preclose_service import ROOT

OUT=ROOT/'data/deal_structure_enriched.csv'
FIELDS=['fresh_issue_cr','ofs_issue_cr','ofs_ratio','qib_quota_pct','nii_quota_pct','retail_quota_pct',
        'snii_subscription_mult','bnii_subscription_mult','nii_subscription_mult','lot_size_shares',
        'min_retail_application_rupees','anchor_total_cr','anchor_domestic_mf_pct',
        'anchor_30d_unlock_date','anchor_90d_unlock_date','brlm_names','quota_weights_exact','allotment_date']


def amount_cr(text):
    match=re.search(r'(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr\b|crores?\b)',text,re.I)
    return float(match[1].replace(',','')) if match else None


def date_value(text):
    value=pd.to_datetime(text,dayfirst=True,errors='coerce')
    return value.date().isoformat() if pd.notna(value) else None


def parse(content,cap_price=None):
    soup=BeautifulSoup(content,'html.parser');result={};basis={};bounded_quota=False
    for table in soup.find_all('table'):
        rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])] for tr in table.find_all('tr')]
        table_text=table.get_text(' ',strip=True)
        subscription=bool(re.search(r'subscription|subscribed|times\b',table_text,re.I))
        for row_index,row in enumerate(rows):
            if len(row)<2:continue
            label=row[0].lower();value=' '.join(row[1:])
            if re.fullmatch(r'(?:basis (?:of|for) )?allotment(?: date)?|(?:tentative )?allotment date',label.strip()):
                v=date_value(value)
                if v:result['allotment_date']=v
            if re.search(r'fresh\s*(?:issue|offer)',label):
                v=amount_cr(value)
                if v is not None:result['fresh_issue_cr']=v
            if re.search(r'offer\s*for\s*sale|\bofs\b',label):
                v=amount_cr(value)
                if v is not None:result['ofs_issue_cr']=v
            if re.search(r'lot\s*size|bid\s*lot',label):
                m=re.search(r'\b([\d,]+)\s*(?:shares)?',value,re.I)
                if m:result['lot_size_shares']=int(m[1].replace(',',''))
            if 'anchor' in label and re.search(r'30\s*days|50%.*30|first.*50%',label):
                v=date_value(value)
                if v:result['anchor_30d_unlock_date']=v
            if 'anchor' in label and re.search(r'90\s*days|remaining.*50%',label):
                v=date_value(value)
                if v:result['anchor_90d_unlock_date']=v
            if re.search(r'book\s*running\s*lead|\bbrlm',label):
                names=[a.get_text(' ',strip=True) for a in table.find_all('tr')[row_index].find_all('a') if a.get_text(' ',strip=True)]
                if not names:names=[s.strip() for s in re.split(r'\s*;\s*|\s*\|\s*',value) if s.strip()]
                result['brlm_names']=sorted(set(names))
            if 'anchor' in label and re.search(r'amount|raised|book|portion|allocation',label):
                v=amount_cr(value)
                if v is not None:result['anchor_total_cr']=v
            if 'domestic' in label and 'mutual' in label:
                m=re.search(r'(\d+(?:\.\d+)?)\s*%',value)
                if m:result['anchor_domestic_mf_pct']=float(m[1])
            category='qib' if re.match(r'qib\b|qualified institutional',label) else 'nii' if re.match(r'nii\b|non.institutional',label) else 'retail' if re.match(r'retail\b|rii\b',label) else None
            if category and not subscription:
                m=re.search(r'(\d+(?:\.\d+)?)\s*%',value)
                if m:
                    result[category+'_quota_pct']=float(m[1])
                    bounded_quota |= bool(re.search(r'not (?:more|less)|at (?:least|most)|up to|minimum|maximum|[<>≤≥]',value,re.I))
            if subscription:
                key='snii_subscription_mult' if re.match(r'snii\b|shni\b|small.*nii|small.*hni',label) else 'bnii_subscription_mult' if re.match(r'bnii\b|bhni\b|big.*nii|big.*hni',label) else 'nii_subscription_mult' if category=='nii' else None
                if key:
                    # A category row must have an explicit multiple; share counts are not multiples.
                    m=re.search(r'(-?\d+(?:\.\d+)?)\s*[x×]',value,re.I)
                    if m:result[key]=float(m[1])
    # IPO calendar and manager panels often use headings rather than tables.
    for h in soup.find_all(['h2','h3','h4']):
        label=h.get_text(' ',strip=True);text=h.parent.get_text(' ',strip=True).replace(label,'',1).strip()
        if 'anchor' in label.lower() and 'lock' in label.lower():
            key='anchor_90d_unlock_date' if re.search(r'90|remaining',label,re.I) else 'anchor_30d_unlock_date' if re.search(r'30|50%',label,re.I) else None
            if key:
                value=date_value(text)
                if value:result[key]=value
        if label.lower()=='lot size':
            m=re.search(r'([\d,]+)\s*shares',text,re.I)
            if m:result['lot_size_shares']=int(m[1].replace(',',''))
        if re.search(r'book running lead managers?|lead managers? to',label,re.I):
            values=[a.get_text(' ',strip=True) for a in h.parent.find_all('a') if a.get_text(' ',strip=True)]
            if values:result['brlm_names']=sorted(set(values))
    text=soup.get_text(' ',strip=True)
    for term in soup.find_all('dt'):
        if term.get_text(' ',strip=True).lower() in ('allotment date','basis of allotment'):
            value=term.find_next_sibling('dd')
            parsed=date_value(value.get_text(' ',strip=True)) if value else None
            if parsed:result['allotment_date']=parsed
    if re.search(r'100%\s*(?:Offer for Sale|OFS)',text,re.I):result.update(fresh_issue_cr=0.,ofs_ratio=1.)
    if re.search(r'100%\s*Fresh Issue',text,re.I):result.update(ofs_issue_cr=0.,ofs_ratio=0.)
    m=re.search(r'Anchor Investors\s+(?:(?!Financials).){0,450}?raising\s*(₹\s*[\d,.]+\s*crore)',text,re.I)
    if m:result['anchor_total_cr']=amount_cr(m[1])
    m=re.search(r'domestic mutual funds?\s+(?:accounted for|received|allocated)\s*(\d+(?:\.\d+)?)\s*%',text,re.I)
    if m:result['anchor_domestic_mf_pct']=float(m[1])
    m=re.search(r'Investors can bid for a minimum of\s*([\d,]+)\s*shares',text,re.I)
    if m:result['lot_size_shares']=int(m[1].replace(',',''))
    if 'fresh_issue_cr' in result and 'ofs_issue_cr' in result:
        total=result['fresh_issue_cr']+result['ofs_issue_cr']
        if total>0:result['ofs_ratio']=result['ofs_issue_cr']/total
    if cap_price and result.get('lot_size_shares',0)>0:result['min_retail_application_rupees']=result['lot_size_shares']*float(cap_price)
    quotas=[result.get(k+'_quota_pct') for k in ('qib','nii','retail')]
    if all(v is not None for v in quotas):result['quota_weights_exact']=not bounded_quota and abs(sum(quotas)-100)<.5
    return {k:v for k,v in result.items() if k in FIELDS}


def brlm_prior_statistics(rows,outcomes,shrinkage=5.):
    """Earlier listings only; equal treatment across managers, unique issuer outcomes."""
    by_id={str(int(r.ipo_id)):r for r in outcomes.itertuples() if r.status=='verified_daily_exchange_archive'}
    for row in rows:
        opened=pd.to_datetime(row.get('open_date'),errors='coerce');names=json.loads(row.get('brlm_names') or '[]')
        prior=[]
        if pd.notna(opened) and names:
            managers={name_key(n) for n in names}
            for other in rows:
                outcome=by_id.get(other['ipo_id'])
                if outcome is None or pd.Timestamp(outcome.listing_date)>=opened:continue
                if managers.intersection(name_key(n) for n in json.loads(other.get('brlm_names') or '[]')):
                    prior.append((int(outcome.exchange_open>outcome.issue_price),100*(outcome.exchange_open/outcome.issue_price-1)))
            market=[o for o in by_id.values() if pd.Timestamp(o.listing_date)<opened]
            mean=float(np.mean([o.exchange_open>o.issue_price for o in market])) if market else .5
            median=float(np.median([100*(o.exchange_open/o.issue_price-1) for o in market])) if market else 0.
            row['brlm_prior_listing_win_rate']=(sum(p[0] for p in prior)+shrinkage*mean)/(len(prior)+shrinkage)
            row['brlm_prior_median_return']=(len(prior)*float(np.median([p[1] for p in prior]))+shrinkage*median)/(len(prior)+shrinkage) if prior else median
            row['brlm_prior_deals']=len(prior)
        else:row.update(brlm_prior_listing_win_rate=None,brlm_prior_median_return=None,brlm_prior_deals=0)
        row['brlm_prior_pit_verified']=False
    return rows


def html_records(value,context=None):
    context=context or {}
    if isinstance(value,dict):
        ctx={**context,**{k:value[k] for k in ('url','source_url','retrieved_at_utc','retrieved_at_ist','fetched_at_ist','datePublished','source_timestamp') if value.get(k)}}
        for k,v in value.items():
            if k in ('raw_response_text','html','content','raw_html') and isinstance(v,str) and '<' in v:yield v,ctx
            elif isinstance(v,(dict,list)):yield from html_records(v,ctx)
    elif isinstance(value,list):
        for item in value:yield from html_records(item,context)


def run():
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv')
    frame=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv')
    issuers={str(int(r.id)):dict(ipo_id=str(int(r.id)),company_name=r.company_name,cap_price=r.price_band_upper,open_date=r.issue_open_date) for r in canonical.itertuples() if r.id in set(frame.ipo_id) or pd.Timestamp(r.listing_date).year==2026}
    from scrapers.live_ipo_scraper import load_scraped_cache
    for url,item in load_scraped_cache().items():
        if item.get('category')=='LISTED':
            key='recent_'+hashlib.sha256(item['company_name'].encode()).hexdigest()[:10]
            issuers[key]=dict(ipo_id=key,company_name=item['company_name'],cap_price=item.get('issue_price'),open_date=item.get('issue_open_date'))
    names={key:name_key(v['company_name']) for key,v in issuers.items()}
    paths=[p for directory in (ROOT/'data',ROOT/'output') for p in directory.rglob('*') if p.is_file() and any(p.name.endswith(s) for s in ('.html','.html.gz','.json','.json.gz')) and 'deal_structure' not in p.name and 'full_enrichment' not in p.name]
    fields={key:{} for key in issuers};counts=dict(files=len(paths),html_pages=0,matched_pages=0);seen=set();ledger=[]
    def read(path):
        try:
            raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
            text=raw.decode('utf8',errors='replace')
            if path.name.endswith(('.html','.html.gz')):return [(text,{'cache_path':str(path.relative_to(ROOT))})]
            if '<table' not in text.lower() and 'raw_response_text' not in text:return []
            return list(html_records(json.loads(text),{'cache_path':str(path.relative_to(ROOT))}))
        except Exception:return []
    with ThreadPoolExecutor(4) as pool:
        for pages in pool.map(read,paths):
            for html,ctx in pages:
                sha=hashlib.sha256(html.encode()).hexdigest()
                if sha in seen:continue
                seen.add(sha);counts['html_pages']+=1
                soup=BeautifulSoup(html,'html.parser');heading=name_key(' '.join(h.get_text(' ',strip=True) for h in soup.find_all('h1')))
                matching=[key for key,name in names.items() if name and (heading==name or heading.startswith(name+' '))]
                if not matching:continue
                counts['matched_pages']+=1
                pub=ctx.get('source_timestamp') or ctx.get('datePublished')
                if not timestamp(pub):
                    pub=None
                    for meta in soup.find_all('meta'):
                        if meta.get('property') in ('article:published_time','datePublished') and timestamp(meta.get('content')):pub=meta['content'];break
                original_fetch=ctx.get('retrieved_at_utc') or ctx.get('retrieved_at_ist') or ctx.get('fetched_at_ist')
                for key in matching:
                    for field,value in parse(html,issuers[key]['cap_price']).items():
                        # Page metadata is not a versioned publication time for each field.
                        observed=original_fetch if timestamp(original_fetch) else None
                        e=evidence(value,source_url=ctx.get('url') or ctx.get('source_url') or ctx['cache_path'],sha256=sha,
                            source_timestamp=observed or pub,source_timestamp_verified=bool(observed),original_fetch_timestamp=original_fetch)
                        e['publisher_page_timestamp']=pub
                        e.update(ipo_id=key,company_name=issuers[key]['company_name'],field=field,cache_path=ctx['cache_path'])
                        if field in ('snii_subscription_mult','bnii_subscription_mult','nii_subscription_mult'):
                            e['source_timestamp']=None;e['source_timestamp_verified']=False;e['provenance_basis']='subscription_table_time_not_yet_bound'
                        ledger.append(e)
                        old=fields[key].get(field)
                        if old is None or (e['source_timestamp_verified'] and (not old['source_timestamp_verified'] or timestamp(e['source_timestamp'])<timestamp(old['source_timestamp']))):fields[key][field]=e
    rows=[]
    for key,item in issuers.items():
        row={**item}
        for field in FIELDS:
            e=fields[key].get(field);row[field]=json.dumps(e['value']) if e and isinstance(e['value'],list) else e['value'] if e else None
            row[field+'_source_timestamp']=e['source_timestamp'] if e else None
            row[field+'_fetch_timestamp']=e['fetch_timestamp'] if e else None
            row[field+'_pit_verified']=e['source_timestamp_verified'] if e else False
        rows.append(row)
    rows=brlm_prior_statistics(rows,pd.read_csv(ROOT/'data/audit/module1_exchange_listing_open_audit.csv'))
    for row in rows:
        for field in ('brlm_prior_listing_win_rate','brlm_prior_median_return'):
            row[field+'_source_timestamp']=None
            row[field+'_fetch_timestamp']=datetime.now().astimezone().isoformat()
            row[field+'_pit_verified']=False
    pd.DataFrame(rows).to_csv(OUT,index=False)
    provenance=OUT.with_suffix('.jsonl');provenance.write_text(''.join(json.dumps(e,default=str)+'\n' for e in ledger),encoding='utf8')
    cutoffs={str(int(r.ipo_id)):str(r.decision_market_date)+'T17:00:00+05:30' for r in frame.itertuples()}
    recent=ROOT/'output/listed_closing_day_backtest.json'
    if recent.exists():
        for r in json.loads(recent.read_text())['records']:
            key='recent_'+hashlib.sha256(r['company_name'].encode()).hexdigest()[:10]
            cutoffs[key]=r.get('cutoff_at_ist') or r.get('bid',{}).get('observed_at_ist')
    summary={**counts,'issuers':len(rows),'coverage':{f:sum(r[f] is not None for r in rows) for f in FIELDS},
        'verified_observation_clock_coverage':{f:sum(r[f+'_pit_verified'] for r in rows) for f in FIELDS},
        'point_in_time_coverage':{f:sum(r[f] is not None and eligible(dict(source_timestamp=r[f+'_source_timestamp'],source_timestamp_verified=r[f+'_pit_verified']),cutoffs.get(r['ipo_id'])) for r in rows) for f in FIELDS},
        'clock_definition':'A dated cache observation proves available by that observation, not the original publication instant. Only observations no later than each decision cutoff are admitted.'}
    (ROOT/'output/cached_deal_structure_coverage.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
