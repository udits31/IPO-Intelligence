"""Extract IPO-context financial tables from public HTML, without OCR/PDF APIs.

Period-end dates are checked; original publication times are not certified.
"""
import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor,as_completed

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from ml.collect_module1_later_holdout import ROOT,index_urls,CACHE as OLD
from ml.recover_module1_other_sources import fetch,norm,ipoji_payload

CACHE=ROOT/'data/archive/module1_public_financials_20261024'
OUT=ROOT/'data/audit/module1_public_financials_20261024.csv'
FINANCIAL=['profit_margin_pct','return_on_assets_pct','debt_to_networth','revenue_growth_asinh',
           'profit_growth_asinh','annual_loss','ipo_earnings_yield_pct','log_ipo_price_book','financial_sector']


def number(text):
    value=str(text).replace(',','').replace('₹','').replace('−','-').strip()
    if re.fullmatch(r'\(\d+(?:\.\d+)?\)',value):
        return -float(value[1:-1])
    return float(value) if re.fullmatch(r'-?\d+(?:\.\d+)?',value) else np.nan


def extract(content,decision,offer_cap=None):
    soup=BeautifulSoup(content,'html.parser'); decision=pd.Timestamp(decision)
    result={k:None for k in FINANCIAL if k!='financial_sector'}
    annual=[]
    for table in soup.find_all('table'):
        rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])] for tr in table.find_all('tr')]
        if not any(r and r[0].lower()=='profit after tax' for r in rows):
            continue
        header=next((r for r in rows if sum(bool(re.fullmatch(r'\d{1,2}-[A-Za-z]{3}-\d{2,4}',s)) for s in r)>=2),None)
        if not header:
            continue
        # Empty top-left cells must stay present to align period and metric columns.
        for column,value in enumerate(header):
            date=pd.to_datetime(value,format='%d-%b-%y',errors='coerce')
            if pd.isna(date):
                date=pd.to_datetime(value,format='%d-%b-%Y',errors='coerce')
            if pd.isna(date) or date>=decision or (date.month,date.day) not in ((3,31),(12,31)):
                continue
            metrics={re.sub(r'[^a-z]','',r[0].lower()):number(r[column]) for r in rows[1:] if len(r)>column and r}
            # Preserve the basis: total income is not silently called net sales.
            if 'revenue' not in metrics:
                for alias in ('revenuefromoperations','netsales','totalincome','feecommissionincome','feeandcommissionincome'):
                    if np.isfinite(metrics.get(alias,np.nan)):
                        metrics['revenue']=metrics[alias];metrics['revenue_basis']=alias;break
            if np.isfinite(metrics.get('profitaftertax',np.nan)) and np.isfinite(metrics.get('revenue',np.nan)):
                annual.append((date,metrics))
        break
    if not annual:
        return dict(result,financial_status='no_eligible_annual_table')
    annual.sort(key=lambda r:r[0],reverse=True)
    date,latest=annual[0];pat=latest['profitaftertax']; revenue=latest['revenue']
    result.update(annual_period=str(date.date()),annual_profit_after_tax=pat,annual_revenue=revenue,revenue_basis=latest.get('revenue_basis','revenue'),annual_loss=float(pat<0),
                  profit_margin_pct=100*pat/revenue if revenue>0 else None,financial_status='period_checked_public_ipo_table')
    assets=latest.get('assets',np.nan);networth=latest.get('networth',np.nan);debt=latest.get('totalborrowing',latest.get('totalborrowings',np.nan))
    if np.isfinite(assets) and assets>0:
        result['return_on_assets_pct']=100*pat/assets
    if np.isfinite(networth) and networth>0 and np.isfinite(debt) and debt>=0:
        result['debt_to_networth']=debt/networth
    if len(annual)>1 and (date-annual[1][0]).days in (365,366):
        prior=annual[1][1]
        if prior['revenue']>0:
            result['revenue_growth_asinh']=float(np.arcsinh((revenue-prior['revenue'])/prior['revenue']))
        if abs(prior['profitaftertax'])>0:
            result['profit_growth_asinh']=float(np.arcsinh((pat-prior['profitaftertax'])/abs(prior['profitaftertax'])))
    # Scope to the IPO valuation panel, never the later stock-price/listing panel.
    text=soup.get_text(' ',strip=True)
    valuation=re.search(r'Valuations\s+Earnings\s+(.*?)\s+Financials\s*\(',text,re.S)
    if valuation:
        eps=re.search(r'EPS\s+Pre\s+IPO\s+(?:₹\s*)?(-?[\d,]+(?:\.\d+)?)',valuation[1])
        pe=re.search(r'P/E\s+Pre\s+IPO\s+(?:₹\s*)?(-?[\d,]+(?:\.\d+)?)',valuation[1])
        book=re.search(r'Price\s*/\s*Book\s+([\d,]+(?:\.\d+)?)',valuation[1])
        if eps and offer_cap is not None and float(offer_cap)>0:
            earnings=float(eps[1].replace(',',''))
            consistent=True
            if pe and earnings>0 and float(pe[1].replace(',',''))>0:
                consistent=abs(earnings*float(pe[1].replace(',',''))-float(offer_cap))<=max(1.,.05*float(offer_cap))
            if consistent:
                result['ipo_earnings_yield_pct']=100*earnings/float(offer_cap)
                result['ipo_pre_offer_eps']=earnings
                result['valuation_basis']='explicit_ipo_panel_pre_offer_eps'
        if book and float(book[1].replace(',',''))>0:
            result['log_ipo_price_book']=float(np.log1p(float(book[1].replace(',',''))))
    return result


def collect(item,urls):
    path=CACHE/f"{item['ipo_id']}.json.gz"
    if path.exists():
        with gzip.open(path,'rt',encoding='utf8') as stream:
            cached=json.load(stream)
        if cached.get('raw_response_text'):
            cached.update(extract(cached['raw_response_text'],item['decision_market_date'],item['issue_price']))
            with gzip.open(path,'wt',encoding='utf8') as stream:
                json.dump(cached,stream)
        return cached
    result=dict(ipo_id=item['ipo_id'],company_name=item['company_name'],financial_status='no_verified_offer_page')
    old=OLD/f"{item['ipo_id']}.json.gz"
    sources=[]
    if old.exists():
        with gzip.open(old,'rt',encoding='utf8') as stream:
            sources=[s for s in json.load(stream).get('sources',[]) if s['provider']=='ipoji' and s.get('raw_response_text')]
    if not sources:
        tokens=set(re.findall(r'[a-z0-9]+',item['company_name'].lower()))-{'ltd','limited','india','and'}
        def score(url):
            slug=url.rstrip('/').split('/')[-1]
            return 100 if norm(slug)==norm(item['company_name']) else len(tokens&set(slug.split('-')))/max(1,len(tokens))
        for url in sorted(urls,key=lambda u:(-score(u),u))[:2]:
            if score(url)<.5:
                continue
            try:
                page=fetch(url)
                sources.append(dict(url=url,raw_response_text=page.text,source_sha256=hashlib.sha256(page.content).hexdigest()))
                if ipoji_payload(item,BeautifulSoup(page.text,'html.parser')):
                    break
            except Exception as exc:
                result['fetch_error']=str(exc)[:180]
    for source in sources:
        if ipoji_payload(item,BeautifulSoup(source['raw_response_text'],'html.parser')):
            result.update(extract(source['raw_response_text'],item['decision_market_date'],item['issue_price']),
                          source_url=source['url'],source_sha256=source['source_sha256'],raw_response_text=source['raw_response_text'])
            break
    with gzip.open(path,'wt',encoding='utf8') as stream:
        json.dump(result,stream)
    return result


def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv')
    test=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    urls=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
    items=[]
    for row in pd.concat([train,test],ignore_index=True).itertuples():
        items.append(dict(ipo_id=int(row.ipo_id),company_name=row.company_name,issue_open_date=row.issue_open_date,
                          issue_close_date=row.issue_close_date,issue_price=float(canonical.loc[row.ipo_id].price_band_upper),
                          decision_market_date=row.decision_market_date))
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(collect,item,urls) for item in items]):
            result=future.result();result.pop('raw_response_text',None)
            result['financial_sector']=float(canonical.loc[result['ipo_id']].sector=='Financial Services')
            rows.append(result)
            if len(rows)%50==0:
                print(f'financial pages checked={len(rows)}/{len(items)} tables={sum(r["financial_status"]=="period_checked_public_ipo_table" for r in rows)}',flush=True)
    pd.DataFrame(rows).sort_values('ipo_id').to_csv(OUT,index=False)
    print(pd.Series([r['financial_status'] for r in rows]).value_counts().to_dict(),flush=True)


if __name__=='__main__':
    run()
