"""Recover excluded 2025 training issuers; preserve the original test cohort."""
import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor,as_completed

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ml.collect_module1_later_holdout import ROOT,index_urls
from ml.recover_module1_other_sources import (fetch,norm,ipoji_payload,dated_history,
    collect_a2z,collect_ipowala,has_eligible_quote,verify_top10)
from ml.fetch_daywise_subscriptions import find_daywise

CACHE=ROOT/'data/archive/module1_2025_training_gaps_20261024'
OUT=ROOT/'data/audit/module1_2025_training_gap_recovery_20261024.csv'


def history(rows,opened,closed):
    accepted=[]
    for record in rows:
        match=re.search(r'Day\s*(\d+)\s*-\s*(\d{2}-\d{2}-\d{2,4})',str(record.get('key','')),re.I)
        if not match:
            continue
        day=pd.to_datetime(match[2],format='%d-%m-%y' if len(match[2])==8 else '%d-%m-%Y')
        if not pd.Timestamp(opened)<=day<=pd.Timestamp(closed):
            continue
        values={}
        for category,key in [('qib','QIB'),('hni','NII'),('retail','RII')]:
            raw=str(record.get('value',{}).get(key,'')).strip()
            number=re.fullmatch(r'(?:[\d,]+\s+)?\(([\d,]+(?:\.\d+)?)\)',raw)
            direct=re.fullmatch(r'([\d,]+(?:\.\d+)?)\s*(?:x|times)?',raw,re.I)
            if number or direct:
                from services.subscription_numbers import subscription_multiple
                parsed=subscription_multiple((number or direct)[1])
                if parsed is not None:values[category+'_subscription']=parsed
        if len(values)==3:
            accepted.append(dict(day=int(match[1]),date=str(day.date()),**values))
    if len({v['date'] for v in accepted})!=len(accepted):
        return []
    return sorted(accepted,key=lambda r:r['date'])


def source_rank(item,url):
    slug=url.rstrip('/').split('/')[-1]
    if norm(slug)==norm(item['company_name']):
        return 100.
    symbol=str(item.get('nse_symbol','')).lower()
    if len(symbol)>=4 and symbol in slug.split('-'):
        return 95.
    if symbol.startswith('the') and len(symbol)>6 and symbol[3:] in slug.split('-'):
        return 94.
    if len(norm(slug))>=8 and norm(item['company_name']).startswith(norm(slug)):
        return 90.
    initials=''.join(w[0] for w in re.findall(r'[a-z0-9]+',item['company_name'].lower()) if w not in {'and','of'})
    if len(initials)>=4 and initials in slug.split('-'):
        return 85.
    generic={'ltd','limited','co','company','india','and','of','s','services','financial','finance','management','asset','private','dr'}
    tokens=[t for t in re.findall(r'[a-z0-9]+',item['company_name'].lower()) if t not in generic]
    found=set(slug.split('-'))
    common=sum(t in found or (len(t)>=5 and any(f.startswith(t) for f in found)) for t in tokens)
    return common/max(1,len(tokens)) + (1. if tokens and tokens[0] in found else 0.)


def product_dates(content,item):
    soup=BeautifulSoup(content,'html.parser')
    if norm(item['company_name']) not in norm(soup.get_text(' ',strip=True)):
        return None
    for tag in soup.find_all('script',type='application/ld+json'):
        try:
            value=json.loads(tag.get_text())
        except ValueError:
            continue
        if not isinstance(value,dict) or value.get('@type')!='FinancialProduct':
            continue
        props={p.get('name','').lower():p.get('value') for p in value.get('additionalProperty',[])}
        band=re.findall(r'\d+(?:\.\d+)?',str(props.get('price band','')).replace(',',''))
        if not band or abs(float(band[-1])-item['issue_price'])>.01:
            continue
        opened=pd.to_datetime(props.get('ipo open date'),errors='coerce')
        closed=pd.to_datetime(props.get('ipo close date'),errors='coerce')
        listed=pd.to_datetime(props.get('ipo listing date'),errors='coerce')
        if pd.isna(opened) or pd.isna(closed) or pd.isna(listed) or listed!=pd.Timestamp(item['listing_date']):
            continue
        if opened<=closed<listed and (listed-closed).days<=15:
            return dict(issue_open_date=str(opened.date()),issue_close_date=str(closed.date()))
    return None


class Recorded(requests.Session):
    def __init__(self):
        super().__init__();self.responses=[]
    def get(self,*args,**kwargs):
        response=super().get(*args,**kwargs)
        self.responses.append(dict(url=response.url,status=response.status_code,raw_response_text=response.text,
            source_sha256=hashlib.sha256(response.content).hexdigest()))
        return response


def extra_gmp(record,item):
    if record.get('extra_public_gmp_attempted') or record.get('status')!='early_book_recovered':
        return record
    dated=dict(item,issue_open_date=record['issue_open_date'],issue_close_date=record['issue_close_date'],
        decision_market_date=record['decision_market_date'],company_name=record.get('verified_brand_alias') or item['company_name'])
    if has_eligible_quote(dated,record['sources']):
        return record
    query=' '.join(re.findall(r'[a-zA-Z0-9]+',dated['company_name'])[:2])+' GMP'
    try:
        response=fetch('https://top10stockbroker.com/wp-json/wp/v2/search?'+requests.compat.urlencode({'search':query,'per_page':10}))
        hits=[h for h in response.json() if '/ipo-grey-market-premium/' in h.get('url','')][:2]
        for hit in hits:
            page=fetch(hit['url']);accepted=verify_top10(dated,BeautifulSoup(page.text,'html.parser'))
            record['sources'].append(dict(provider='top10stockbroker',url=hit['url'],status='verified_offer' if accepted else 'identity_rejected',
                raw_response_text=page.text,source_sha256=hashlib.sha256(page.content).hexdigest(),
                quotes=dated_history(page.text,record['issue_close_date']) if accepted else []))
    except Exception as exc:
        record['extra_public_gmp_error']=str(exc)[:180]
    record['extra_public_gmp_attempted']=True
    return record


def collect(item,urls,a2z,calendar):
    path=CACHE/f"{item['ipo_id']}.json.gz"
    if path.exists():
        with gzip.open(path,'rt',encoding='utf8') as stream:
            cached=json.load(stream)
        if cached['status']=='early_book_recovered' and has_eligible_quote(dict(item,decision_market_date=cached['decision_market_date']),cached['sources']):
            return cached
        path=CACHE/f"{item['ipo_id']}_v2.json.gz"
        if path.exists():
            with gzip.open(path,'rt',encoding='utf8') as stream:
                cached=json.load(stream)
            cached=extra_gmp(cached,item)
            with gzip.open(path,'wt',encoding='utf8') as stream:
                json.dump(cached,stream)
            return cached
    result=dict(ipo_id=item['ipo_id'],company_name=item['company_name'],status='no_verified_early_book',sources=[])
    for url in sorted(urls,key=lambda u:(-source_rank(item,u),u))[:4]:
        if source_rank(item,url)<.3:
            continue
        try:
            response=fetch(url);dates=product_dates(response.text,item)
            source=dict(provider='ipoji',url=url,status='identity_rejected' if not dates else 'verified_offer',
                raw_response_text=response.text,source_sha256=hashlib.sha256(response.content).hexdigest(),quotes=[])
            result['sources'].append(source)
            if not dates:
                continue
            verified=dict(item,**dates)
            payload=ipoji_payload(verified,BeautifulSoup(response.text,'html.parser'))
            brand=None
            for tag in BeautifulSoup(response.text,'html.parser').find_all('script',type='application/ld+json'):
                try:
                    product=json.loads(tag.get_text())
                except ValueError:
                    continue
                if isinstance(product,dict) and product.get('@type')=='FinancialProduct':
                    brand=re.sub(r'\s+IPO\s*$','',str(product.get('name','')),flags=re.I)
                    break
            days=history(payload['bids'],**{'opened':dates['issue_open_date'],'closed':dates['issue_close_date']}) if payload else []
            source['quotes']=dated_history(response.text,dates['issue_close_date'])
            session=Recorded()
            watch,watch_status=find_daywise(pd.Series(verified),session,calendar)
            if not watch and brand and brand!=item['company_name']:
                watch,watch_status=find_daywise(pd.Series(dict(verified,company_name=brand)),session,calendar)
            result['ipowatch_status']=watch_status
            result['ipowatch_responses']=session.responses
            expected=calendar[(calendar>=pd.Timestamp(dates['issue_open_date']))&(calendar<=pd.Timestamp(dates['issue_close_date']))]
            if len(expected)<2:
                continue
            decision=str(expected[-2].date())
            chart=next((d for d in days if d['date']==decision),None)
            if watch:
                result.update(status='early_book_recovered',decision_market_date=decision,
                    bid_source_url=watch['source_url'],bid_source_sha256=watch['source_content_sha256'],
                    **{k:watch[k] for k in ('qib_subscription','hni_subscription','retail_subscription')},**dates)
            elif chart and days[-1]['date']==dates['issue_close_date']:
                result.update(status='early_book_recovered',decision_market_date=decision,
                    bid_source_url=url,bid_source_sha256=source['source_sha256'],
                    **{k:chart[k] for k in ('qib_subscription','hni_subscription','retail_subscription')},**dates)
            if result['status']=='early_book_recovered':
                result['bid_history']=days
                result['identity_url']=url
                provider_item=dict(verified,decision_market_date=decision,company_name=brand or item['company_name'])
                result['verified_brand_alias']=brand
                result['sources']+=collect_a2z(provider_item,a2z)
                if not has_eligible_quote(dict(verified,decision_market_date=decision),result['sources']):
                    result['sources']+=collect_ipowala(provider_item)
                break
        except Exception as exc:
            result['last_error']=str(exc)[:180]
    with gzip.open(path,'wt',encoding='utf8') as stream:
        json.dump(result,stream)
    return result


def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    prior=pd.read_csv(ROOT/'data/audit/module1_later_holdout_20261024.csv')
    outcomes=pd.read_csv(ROOT/'data/audit/module1_exchange_listing_open_audit.csv').set_index('ipo_id')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    prior['year']=prior.ipo_id.map(outcomes.year)
    missing=prior[prior.year.eq(2025)&~prior.status.eq('paired_eligible')]
    urls=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
    a2z=index_urls('a2z_sitemap','https://www.a2zipo.com/sitemap.xml','/gmp/')
    calendar=pd.read_csv(ROOT/'output/module1_forward_update_20261024_market.csv')
    calendar=pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.NIFTY50.notna(),'date']))
    items=[dict(ipo_id=int(row.ipo_id),company_name=row.company_name,
        issue_price=float(canonical.loc[row.ipo_id].price_band_upper),nse_symbol=str(canonical.loc[row.ipo_id].nse_symbol),
        listing_date=outcomes.loc[row.ipo_id].listing_date) for row in missing.itertuples()]
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(collect,item,urls,a2z,calendar) for item in items]):
            records.append(future.result())
            print(f'2025 recovery checked={len(records)}/{len(items)} books={sum(r["status"]=="early_book_recovered" for r in records)}',flush=True)
    rows=[]
    for record in records:
        row={k:v for k,v in record.items() if k not in ('sources','ipowatch_responses','bid_history','ipowatch_status')}
        row['year']=2025
        if record['status']!='early_book_recovered':
            rows.append(row);continue
        item=canonical.loc[row['ipo_id']];outcome=outcomes.loc[row['ipo_id']]
        quote_dates={}
        date=pd.Timestamp(record['decision_market_date'])
        for source in record['sources']:
            for quote in source.get('quotes',[]):
                stamp=pd.Timestamp(quote['gmp_date'])
                if date-pd.Timedelta(days=7)<=stamp<date:
                    quote_dates.setdefault(quote['gmp_date'],[]).append(dict(quote,source_url=source['url']))
        eligible=[q for group in quote_dates.values() if len({v['gmp_rs'] for v in group})==1 for q in group]
        selected=sorted(eligible,key=lambda q:(q['gmp_date'],q['source_url']))[-1] if eligible else None
        row.update(issue_price=float(outcome.issue_price),listing_date=outcome.listing_date,exchange_open=float(outcome.exchange_open),
            target_binary=int(outcome.exchange_open>outcome.issue_price),target_return=100*(outcome.exchange_open/outcome.issue_price-1),
            log_offer_cap_price=float(np.log1p(item.price_band_upper)),log_issue_size=float(np.log1p(item.issue_size_cr)))
        for category in ('qib','hni','retail'):
            row['log_'+category]=float(np.log1p(max(.01,row[category+'_subscription'])))
        if selected:
            row.update(status='paired_training_recovered',gmp_date=selected['gmp_date'],gmp_rs=selected['gmp_rs'],
                gmp_pct=100*selected['gmp_rs']/item.price_band_upper,gmp_source_url=selected['source_url'])
        else:
            row['status']='early_book_missing_gmp'
        rows.append(row)
    pd.DataFrame(rows).sort_values('ipo_id').to_csv(OUT,index=False)
    print(pd.Series([r['status'] for r in rows]).value_counts().to_dict(),flush=True)


if __name__=='__main__':
    run()
