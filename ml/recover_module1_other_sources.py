"""Other-source public GMP history recovery, plus source-matched early bid growth.

No logins, paid-history bypasses, OCR, fabricated quotes or live forecast edits.
"""
import gzip
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ml.complete_module1_benchmark import prepare,ROOT
from ml.module1_gap_recovery import normalized

CACHE=ROOT/'data/archive/module1_other_sources_20261024'
OUT=ROOT/'data/audit/module1_other_source_gmp_20261024.csv'
AUDIT=ROOT/'output/module1_other_source_recovery_20261024.json'


def norm(text):
    text=re.sub(r'\bcorp\.?\b','corporation',text.lower())
    return normalized(re.sub(r'\bipo\b','',text))


def title_matches(item,title):
    prefix=re.split(r'\bIPO\b',title,flags=re.I)[0]
    if norm(item['company_name']) in norm(prefix):
        return True
    symbol=str(item.get('nse_symbol',''))
    if len(symbol)>=4 and norm(prefix)==norm(symbol):
        return True  # Exchange-identified ticker alias, with date/cap checks below.
    ignored={'and','of','india','services','industries','corporation','corp','limited','ltd','private','co'}
    expected=set(re.findall(r'[a-z0-9]+',item['company_name'].lower()))-ignored
    found=set(re.findall(r'[a-z0-9]+',prefix.lower()))-ignored
    if expected and norm(' '.join(sorted(expected)))==norm(' '.join(sorted(found))):
        return True
    expected_order=[w for w in re.findall(r'[a-z0-9]+',item['company_name'].lower()) if w not in ignored]
    found_order=[w for w in re.findall(r'[a-z0-9]+',prefix.lower()) if w not in ignored]
    return len(found_order)>=2 and found_order==expected_order[:len(found_order)]


def dated_history(content,closing):
    soup=BeautifulSoup(content,'html.parser');result=[];closing=pd.Timestamp(closing)
    for table in soup.find_all('table'):
        rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])] for tr in table.find_all('tr')]
        header=next(((i,j) for i,r in enumerate(rows) if len(r)>1 and re.search(r'(?i)date',r[0])
            for j in range(1,len(r)) if re.search(r'(?i)\bGMP\b',r[j])),None)
        if header is None:
            continue
        header_row,gmp_column=header
        updated_column=next((j for j,value in enumerate(rows[header_row]) if re.search(r'(?i)last\s+updated',value)),None)
        for cells in rows[header_row+1:]:
            if len(cells)<=gmp_column:
                continue
            raw=cells[gmp_column].replace('–','-').replace('−','-')
            number=re.fullmatch(r'\s*(?:₹|Rs\.?|INR)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:Rs\.?|INR)?\s*(?:\([-+\d.]+%\))?\s*[▲▼]?\s*',raw,re.I)
            interval=re.fullmatch(r'\s*(?:₹|Rs\.?|INR)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:-|to)\s*(?:₹|Rs\.?|INR)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:Rs\.?|INR)?\s*',raw,re.I)
            if not number and not interval:
                continue
            text=re.sub(r'(?i)\s+Latest\s*$','',cells[0].replace('\xa0',' '))
            text=re.sub(r'(?i)(?<=\d)(?:st|nd|rd|th)\b','',text).lstrip(' *')
            explicit_year=bool(re.search(r'\b20\d{2}\b|\d{1,2}[-/]\d{1,2}[-/]\d{2}\b|\d{1,2}\s+[A-Za-z]+\s+\d{2}\b',text))
            candidates=[pd.to_datetime(text if explicit_year else f'{text} {year}',errors='coerce',dayfirst=True)
                for year in (closing.year-1,closing.year,closing.year+1)]
            candidates=[d for d in candidates if pd.notna(d) and abs((d.normalize()-closing).days)<=40]
            if not candidates:
                continue
            date=min(candidates,key=lambda d:abs((d.normalize()-closing).days))
            if updated_column is not None and len(cells)>updated_column:
                updated=pd.to_datetime(cells[updated_column],errors='coerce')
                if pd.notna(updated) and updated.date()>date.date():
                    continue  # Do not backdate a later publisher revision.
            if number:
                result.append(dict(gmp_date=str(date.date()),gmp_rs=float(number[1].replace(',',''))))
            else:
                low=float(interval[1].replace(',',''));high=float(interval[2].replace(',',''))
                if low<=high:
                    result.append(dict(gmp_date=str(date.date()),gmp_rs=(low+high)/2,
                        gmp_low_rs=low,gmp_high_rs=high,gmp_basis='published_range_midpoint'))
    accepted=[]
    for date in sorted({r['gmp_date'] for r in result}):
        observations=[r for r in result if r['gmp_date']==date]
        values={(r['gmp_rs'],r.get('gmp_low_rs'),r.get('gmp_high_rs')) for r in observations}
        if len(values)==1:
            accepted.append(observations[0])
    return accepted


def verify_top10(item,soup):
    heading=soup.find('h1')
    if not heading:
        return False
    title=heading.get_text(' ',strip=True);body=soup.get_text(' ',strip=True)
    identity=norm(item['company_name'])
    if identity not in norm(title) and identity not in norm(body) and not title_matches(item,title):
        return False
    prefix=re.split(r'\bIPO\b',title,flags=re.I)[0]
    if identity not in norm(prefix):
        ignored={'and','of','india','services','industries','corporation','corp','limited','ltd','private','co'}
        significant=set(re.findall(r'[a-z0-9]+',item['company_name'].lower()))-ignored
        title_tokens=set(re.findall(r'[a-z0-9]+',prefix.lower()))
        overlap=significant&title_tokens
        if len(overlap)<min(2,len(significant)) or not overlap or (len(significant)==1 and len(next(iter(significant)))<5):
            return False
    dates={}
    for tr in soup.find_all('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])]
        if len(cells)!=2:
            continue
        for field,pattern in [('issue_open_date',r'(?:IPO\s+)?Open(?:ing)?\s+Date'),('issue_close_date',r'(?:IPO\s+)?Clos(?:e|ing)\s+Date')]:
            if re.fullmatch(pattern,cells[0].strip().rstrip(':').strip(),re.I):
                date=pd.to_datetime(cells[1],errors='coerce')
                if pd.notna(date):
                    dates[field]=str(date.date())
    if all(dates.get(k)==str(pd.Timestamp(item[k]).date()) for k in ('issue_open_date','issue_close_date')):
        return True
    # Some older GMP pages expose only the listing timetable and retail lot
    # cost. These can establish issuer/offer identity without using the listing
    # OPEN PRICE or any outcome as a predictor.
    listing=None;cap=None;lot=None;amount=None
    for tr in soup.find_all('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])]
        if len(cells)!=2:
            continue
        label=re.sub(r'[^a-z]','',cells[0].lower());value=cells[1]
        if label in ('listingdate','ipolistingdate'):
            parsed=pd.to_datetime(value,errors='coerce')
            if pd.notna(parsed):
                listing=str(parsed.date())
        if label in ('priceband','ipopriceband','issueprice'):
            numbers=re.findall(r'\d[\d,]*(?:\.\d+)?',value)
            if numbers:
                cap=float(numbers[-1].replace(',',''))
        if label in ('lotsize','ipolotsize'):
            matched=re.search(r'(?i)minimum\s+(\d[\d,]*)\s+shares',value)
            if matched:
                lot=float(matched[1].replace(',',''))
        if label in ('minimumamount','minimuminvestment'):
            matched=re.search(r'(?:₹|Rs\.?)?\s*(\d[\d,]*(?:\.\d+)?)',value)
            if matched:
                amount=float(matched[1].replace(',',''))
    if cap is None and lot and amount:
        cap=amount/lot
    return bool(item.get('listing_date') and listing==str(pd.Timestamp(item['listing_date']).date()) and
        cap is not None and abs(cap-float(item['issue_price']))<.01)


def ipoji_payload(item,soup):
    product=None
    for script in soup.find_all('script'):
        if script.get('type')!='application/ld+json':
            continue
        try:
            value=json.loads(script.get_text())
        except ValueError:
            continue
        if isinstance(value,dict) and value.get('@type')=='FinancialProduct':
            product=value;break
    if not product:
        return None
    properties={p.get('name','').lower():p.get('value') for p in product.get('additionalProperty',[])}
    for field,label in [('issue_open_date','ipo open date'),('issue_close_date','ipo close date')]:
        if properties.get(label)!=str(pd.Timestamp(item[field]).date()):
            return None
    band=re.findall(r'\d+(?:\.\d+)?',str(properties.get('price band','')).replace(',',''))
    if not band or abs(float(band[-1])-float(item['issue_price']))>.01:
        return None
    # Short brand names in the headline are admissible only with both exact
    # offer dates, cap price and legal issuer name in the public page content.
    if norm(item['company_name']) not in norm(soup.get_text(' ',strip=True)):
        return None
    def data(node):
        tag=soup.find(id=node)
        if not tag:
            return []
        try:
            return json.loads(tag.get_text())
        except ValueError:
            return []
    return dict(gmp=data('ipo-gmp-graph-data'),bids=data('ipo-subscription-info-for-chart'))


def fetch(url):
    response=requests.get(url,timeout=15,headers={'User-Agent':'Mozilla/5.0 (compatible; IPOResearchAudit/1.0)'})
    response.raise_for_status()
    return response


def verify_ipowala(item,soup):
    heading=soup.find('h1')
    if not heading:
        return False
    if not title_matches(item,heading.get_text(' ',strip=True)):
        return False
    content=soup.find(class_='entry-content') or soup.find('article') or soup
    body=content.get_text(' ',strip=True).replace('–','-').replace('—','-')
    date=r'(\d{1,2}[- /]+[A-Za-z]+[- ,/]+20\d{2})'
    opened=re.search(r'(?:begun|open(?:s|ing)?)(?:\s+date)?\s*(?:on|:)?\s*'+date,body,re.I)
    closed=re.search(r'clos(?:e|es|ing)(?:\s+date)?\s*(?:on|:)?\s*'+date,body,re.I)
    band=re.search(r'price.{0,65}?([\d,]+(?:\.\d+)?)\s*(?:-|to)\s*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',body,re.I)
    if not opened or not closed or not band:
        return False
    return (str(pd.Timestamp(opened[1]).date())==str(pd.Timestamp(item['issue_open_date']).date()) and
        str(pd.Timestamp(closed[1]).date())==str(pd.Timestamp(item['issue_close_date']).date()) and
        abs(float(band[2].replace(',',''))-float(item['issue_price']))<.01)


def collect_ipowala(item):
    sources=[]
    try:
        words=re.findall(r'[a-zA-Z0-9]+',item['company_name'])
        query=' '.join(words[:2])+' IPO GMP'
        response=fetch('https://ipowala.in/wp-json/wp/v2/search?'+requests.compat.urlencode({'search':query,'per_page':10}))
        hits=[h for h in response.json() if 'gmp' in h.get('url','').lower()]
        expected=set(w.lower() for w in words)
        hits=sorted(hits,key=lambda h:-len(expected&set(re.findall(r'[a-z0-9]+',html.unescape(h['title']).lower()))))[:2]
        for hit in hits:
            page=fetch(hit['url']);soup=BeautifulSoup(page.text,'html.parser');identity=verify_ipowala(item,soup)
            sources.append(dict(provider='ipowala',url=hit['url'],status='verified_offer' if identity else 'identity_or_dates_rejected',
                source_sha256=hashlib.sha256(page.content).hexdigest(),raw_response_text=page.text,
                quotes=dated_history(page.text,item['issue_close_date']) if identity else []))
        if not hits:
            sources.append(dict(provider='ipowala',status='no_matching_history_page',quotes=[]))
    except (requests.RequestException,ValueError) as exc:
        sources.append(dict(provider='ipowala',status='fetch_error',error=str(exc)[:180],quotes=[]))
    return sources


def verify_top10stock(item,soup):
    heading=soup.find('h1')
    if not heading or (norm(item['company_name']) not in norm(heading.get_text(' ',strip=True)) and
            not title_matches(item,heading.get_text(' ',strip=True))):
        return False
    body=soup.get_text(' ',strip=True)
    date=r'(\d{1,2}\s+[A-Za-z]+\s+20\d{2})'
    opened=re.search(r'Issue\s+open\s+date\s*'+date,body,re.I)
    closed=re.search(r'Issue\s+close\s+date\s*'+date,body,re.I)
    band=re.search(r'Price\s+Range\s*₹?\s*([\d,]+(?:\.\d+)?)\s+to\s+₹?\s*([\d,]+(?:\.\d+)?)',body,re.I)
    return bool(opened and closed and band and
        str(pd.Timestamp(opened[1]).date())==str(pd.Timestamp(item['issue_open_date']).date()) and
        str(pd.Timestamp(closed[1]).date())==str(pd.Timestamp(item['issue_close_date']).date()) and
        abs(float(band[2].replace(',',''))-float(item['issue_price']))<.01)


def collect_top10stock(item,urls):
    identity=norm(item['company_name']);tokens=set(re.findall(r'[a-z0-9]+',item['company_name'].lower()))-{'ltd','limited','india','and'}
    def score(url):
        slug=url.rsplit('/',1)[-1]
        symbol=str(item.get('nse_symbol','')).lower()
        return (100. if norm(slug)==identity else 95. if len(symbol)>=4 and symbol in slug.split('-') else
            len(tokens&set(slug.split('-')))/max(1,len(tokens)))
    candidates=[u for u in sorted(urls,key=lambda u:(-score(u),u))[:2] if score(u)>=.5]
    sources=[]
    for url in candidates:
        try:
            page=fetch(url);soup=BeautifulSoup(page.text,'html.parser');identity=verify_top10stock(item,soup)
            sources.append(dict(provider='top10stock',url=url,status='verified_offer' if identity else 'identity_or_dates_rejected',
                source_sha256=hashlib.sha256(page.content).hexdigest(),raw_response_text=page.text,
                quotes=dated_history(page.text,item['issue_close_date']) if identity else []))
            if identity:
                break
        except (requests.RequestException,ValueError) as exc:
            sources.append(dict(provider='top10stock',url=url,status='fetch_error',error=str(exc)[:180],quotes=[]))
    return sources or [dict(provider='top10stock',status='no_identity_candidate',quotes=[])]


def verify_a2z(item,soup):
    heading=soup.find('h1')
    if not heading:
        return False
    prefix=re.split(r'\bIPO\b',heading.get_text(' ',strip=True),flags=re.I)[0]
    if norm(item['company_name'])!=norm(prefix) and not title_matches(item,heading.get_text(' ',strip=True)):
        return False
    body=soup.get_text(' ',strip=True).replace('–','-')
    date=r'(\d{1,2}\s+[A-Za-z]+\s+20\d{2})'
    opened=re.search(r'opens for subscription on\s*'+date,body,re.I)
    closed=re.search(r'closes on\s*'+date,body,re.I)
    band=re.search(r'price band.{0,45}?₹?\s*([\d,]+(?:\.\d+)?)\s*-\s*₹?\s*([\d,]+(?:\.\d+)?)',body,re.I)
    return bool(opened and closed and band and str(pd.Timestamp(opened[1]).date())==str(pd.Timestamp(item['issue_open_date']).date()) and
        str(pd.Timestamp(closed[1]).date())==str(pd.Timestamp(item['issue_close_date']).date()) and
        abs(float(band[2].replace(',',''))-float(item['issue_price']))<.01)


def collect_a2z(item,urls):
    name=norm(item['company_name']);tokens=set(re.findall(r'[a-z0-9]+',item['company_name'].lower()))-{'ltd','limited','india','and'}
    def score(url):
        slug=url.rstrip('/').split('/')[-2]
        return 100. if norm(slug)==name else len(tokens&set(slug.split('-')))/max(1,len(tokens))
    candidates=[u for u in sorted(urls,key=lambda u:(-score(u),u))[:2] if score(u)>=.5]
    result=[]
    for url in candidates:
        try:
            page=fetch(url);soup=BeautifulSoup(page.text,'html.parser')
            detail=next((a.get('href') for a in soup.find_all('a',href=True) if a['href'].startswith('/ipo/') and a['href'].rsplit('/',1)[-1]==url.rsplit('/',1)[-1]),None)
            if not detail:
                continue
            details=fetch('https://www.a2zipo.com'+detail);identity=verify_a2z(item,BeautifulSoup(details.text,'html.parser'))
            result.append(dict(provider='a2zipo',url=url,status='verified_offer' if identity else 'identity_or_dates_rejected',
                source_sha256=hashlib.sha256(page.content).hexdigest(),raw_response_text=page.text,
                identity_url='https://www.a2zipo.com'+detail,identity_response_text=details.text,
                identity_sha256=hashlib.sha256(details.content).hexdigest(),
                quotes=dated_history(page.text,item['issue_close_date']) if identity else []))
            if identity:
                break
        except (requests.RequestException,ValueError) as exc:
            result.append(dict(provider='a2zipo',url=url,status='fetch_error',error=str(exc)[:180],quotes=[]))
    return result or [dict(provider='a2zipo',status='no_identity_candidate',quotes=[])]


def collect_older_archives(item):
    """Last public-source pass for still-unresolved issuers only."""
    result=[]
    words=re.findall(r'[a-zA-Z0-9]+',item['company_name'])
    query=' '.join(words[:2])+' IPO GMP'
    for provider,base in [('investoracademy','https://investoracademy.in'),('adigitalblogger','https://www.adigitalblogger.com')]:
        try:
            if provider=='investoracademy':
                response=fetch(base+'/wp-json/wp/v2/search?'+requests.compat.urlencode({'search':query,'per_page':10}))
                hits=[h['url'] for h in response.json() if 'gmp' in h.get('url','').lower()][:2]
            else:
                # Normal public website search, not a protected API or login.
                response=fetch(base+'/?'+requests.compat.urlencode({'s':query}))
                soup=BeautifulSoup(response.text,'html.parser')
                links=[a['href'] for a in soup.find_all('a',href=True) if a['href'].startswith(base+'/ipo/') and 'gmp' in a['href']]
                hits=list(dict.fromkeys(links))[:2]
            for url in hits:
                page=fetch(url);soup=BeautifulSoup(page.text,'html.parser');accepted=verify_top10(item,soup)
                result.append(dict(provider=provider,url=url,status='verified_offer' if accepted else 'identity_or_dates_rejected',
                    raw_response_text=page.text,source_sha256=hashlib.sha256(page.content).hexdigest(),
                    quotes=dated_history(page.text,item['issue_close_date']) if accepted else []))
            if not hits:
                result.append(dict(provider=provider,status='no_matching_history_page',quotes=[]))
        except (requests.RequestException,ValueError) as exc:
            result.append(dict(provider=provider,status='fetch_error',error=str(exc)[:180],quotes=[]))
    return result


def has_eligible_quote(item,sources):
    date=pd.Timestamp(item['decision_market_date'])
    days={}
    for source in sources:
        for quote in source.get('quotes',[]):
            stamp=pd.Timestamp(quote['gmp_date'])
            if date-pd.Timedelta(days=7)<=stamp<date:
                days.setdefault(stamp,set()).add(quote['gmp_rs'])
    return any(len(values)==1 for values in days.values())


def collect(item,ipoji_urls,stock_urls,a2z_urls):
    path=CACHE/f"{item['ipo_id']}.json.gz"
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            cached=json.load(stream)
        changed=False
        for source in cached['sources']:
            if source.get('raw_response_text') and source['provider'] in ('ipowala','top10stock','top10stockbroker','investoracademy','adigitalblogger'):
                checker={'ipowala':verify_ipowala,'top10stock':verify_top10stock,'top10stockbroker':verify_top10,
                    'investoracademy':verify_top10,'adigitalblogger':verify_top10}[source['provider']]
                accepted=checker(item,BeautifulSoup(source['raw_response_text'],'html.parser'))
                new_quotes=dated_history(source['raw_response_text'],item['issue_close_date']) if accepted else []
                new_status='verified_offer' if accepted else 'identity_or_dates_rejected'
                if source.get('quotes')!=new_quotes or source['status']!=new_status:
                    source.update(quotes=new_quotes,status=new_status);changed=True
            if source.get('identity_response_text') and source['provider']=='a2zipo':
                accepted=verify_a2z(item,BeautifulSoup(source['identity_response_text'],'html.parser'))
                new_quotes=dated_history(source['raw_response_text'],item['issue_close_date']) if accepted else []
                new_status='verified_offer' if accepted else 'identity_or_dates_rejected'
                if source.get('quotes')!=new_quotes or source['status']!=new_status:
                    source.update(quotes=new_quotes,status=new_status);changed=True
            if source.get('quotes') and source['provider']=='top10stockbroker' and not verify_top10(
                    item,BeautifulSoup(source['raw_response_text'],'html.parser')):
                source.update(quotes=[],status='identity_or_dates_rejected');changed=True
            if source.get('quotes') and source['provider']=='ipowala' and not verify_ipowala(
                    item,BeautifulSoup(source['raw_response_text'],'html.parser')):
                source.update(quotes=[],status='identity_or_dates_rejected');changed=True
            if source.get('quotes') and source['provider'] in ('ipowala','top10stockbroker','top10stock','a2zipo'):
                reparsed=dated_history(source['raw_response_text'],item['issue_close_date'])
                if reparsed!=source['quotes']:
                    source['quotes']=reparsed;changed=True
        if not any(s['provider']=='ipowala' for s in cached['sources']):
            cached['sources']+=collect_ipowala(item);changed=True
        if not any(s['provider']=='top10stock' for s in cached['sources']):
            cached['sources']+=collect_top10stock(item,stock_urls);changed=True
        else:
            symbol=str(item.get('nse_symbol','')).lower()
            aliases=[u for u in stock_urls if len(symbol)>=4 and symbol in u.rsplit('/',1)[-1].split('-')]
            attempted={s.get('url') for s in cached['sources'] if s['provider']=='top10stock'}
            if aliases and not any(u in attempted for u in aliases):
                cached['sources']+=collect_top10stock(item,stock_urls);changed=True
        if not any(s['provider']=='a2zipo' for s in cached['sources']):
            cached['sources']+=collect_a2z(item,a2z_urls);changed=True
        if not has_eligible_quote(item,cached['sources']) and not any(s['provider']=='investoracademy' for s in cached['sources']):
            cached['sources']+=collect_older_archives(item);changed=True
        if changed:
            with gzip.open(path,'wt',encoding='utf-8') as stream:
                json.dump(cached,stream)
        return cached
    result=dict(ipo_id=int(item['ipo_id']),company_name=item['company_name'],sources=[],
        retrieved_at_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),bid_rows=[])
    query=re.sub(r'\b(?:Ltd|Limited|Private)\.?\b','',item['company_name'],flags=re.I).strip()
    words=re.findall(r'[a-zA-Z0-9]+',query)
    search=' '.join(words[:2])+' GMP'
    try:
        response=fetch('https://top10stockbroker.com/wp-json/wp/v2/search?'+requests.compat.urlencode({'search':search,'per_page':10}))
        hits=[h for h in response.json() if '/ipo-grey-market-premium/' in h.get('url','')][:2]
        for hit in hits:
            page=fetch(hit['url']);soup=BeautifulSoup(page.text,'html.parser')
            identity=verify_top10(item,soup)
            result['sources'].append(dict(provider='top10stockbroker',url=hit['url'],status='verified_offer' if identity else 'identity_or_dates_rejected',
                raw_response_text=page.text,source_sha256=hashlib.sha256(page.content).hexdigest(),
                quotes=dated_history(page.text,item['issue_close_date']) if identity else []))
        if not hits:
            result['sources'].append(dict(provider='top10stockbroker',status='no_matching_history_page',quotes=[]))
    except (requests.RequestException,ValueError) as exc:
        result['sources'].append(dict(provider='top10stockbroker',status='fetch_error',error=str(exc)[:180],quotes=[]))
    tokens=set(re.findall(r'[a-z0-9]+',query.lower()))-{'india','and','of','services','industries'}
    def score(url):
        slug=url.rstrip('/').rsplit('/',1)[-1]
        if norm(slug)==norm(query):
            return 100.
        found=set(slug.split('-'))
        return len(tokens&found)/max(1,len(tokens))
    urls=sorted(ipoji_urls,key=lambda u:(-score(u),u))[:2]
    urls=[u for u in urls if score(u)>=.5]
    for url in urls:
        try:
            page=fetch(url);soup=BeautifulSoup(page.text,'html.parser');payload=ipoji_payload(item,soup)
            quotes=[]
            if payload:
                # Preserve unknown chart formats instead of guessing field names.
                quotes=dated_history(page.text,item['issue_close_date'])
                result['bid_rows']=payload['bids'];result['bid_source_url']=url
            result['sources'].append(dict(provider='ipoji',url=url,status=('verified_offer_no_public_gmp_rows' if payload and not quotes else
                'verified_offer' if payload else 'identity_or_dates_rejected'),source_sha256=hashlib.sha256(page.content).hexdigest(),
                raw_response_text=page.text,quotes=quotes,
                public_graph_data=payload['gmp'] if payload else []))
            if payload:
                break
        except (requests.RequestException,ValueError) as exc:
            result['sources'].append(dict(provider='ipoji',url=url,status='fetch_error',error=str(exc)[:180],quotes=[]))
    if not urls:
        result['sources'].append(dict(provider='ipoji',status='no_identity_candidate',quotes=[]))
    result['sources']+=collect_ipowala(item)
    result['sources']+=collect_top10stock(item,stock_urls)
    result['sources']+=collect_a2z(item,a2z_urls)
    if not has_eligible_quote(item,result['sources']):
        result['sources']+=collect_older_archives(item)
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        json.dump(result,stream)
    return result


def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    early,canonical,_=prepare();missing=early[early.gmp_pct.isna()]
    index_path=CACHE/'ipoji_sitemap.xml.gz'
    if index_path.exists():
        with gzip.open(index_path,'rt',encoding='utf-8') as stream:
            index=stream.read()
    else:
        index=fetch('https://www.ipoji.com/sitemap-ipo.xml').text
        with gzip.open(index_path,'wt',encoding='utf-8') as stream:
            stream.write(index)
    urls=[node.text for node in ET.fromstring(index).iter() if node.tag.endswith('loc') and '/ipo/' in node.text]
    stock_path=CACHE/'top10stock_sitemap.xml.gz'
    if stock_path.exists():
        with gzip.open(stock_path,'rt',encoding='utf-8') as stream:
            stock_index=stream.read()
    else:
        stock_index=fetch('https://www.top10stock.com/ipo/sitemap.xml').text
        with gzip.open(stock_path,'wt',encoding='utf-8') as stream:
            stream.write(stock_index)
    stock_urls=[node.text for node in ET.fromstring(stock_index).iter() if node.tag.endswith('loc') and '/ipo/main/' in node.text]
    a2z_path=CACHE/'a2z_sitemap.xml.gz'
    if a2z_path.exists():
        with gzip.open(a2z_path,'rt',encoding='utf-8') as stream:
            a2z_index=stream.read()
    else:
        a2z_index=fetch('https://www.a2zipo.com/sitemap.xml').text
        with gzip.open(a2z_path,'wt',encoding='utf-8') as stream:
            stream.write(a2z_index)
    a2z_urls=[node.text for node in ET.fromstring(a2z_index).iter() if node.tag.endswith('loc') and '/gmp/' in node.text]
    results=[]
    def save():
        rows=[]
        for result in sorted(results,key=lambda r:r['ipo_id']):
            for source in result['sources']:
                for quote in source.get('quotes',[]):
                    rows.append(dict(ipo_id=result['ipo_id'],company_name=result['company_name'],provider=source['provider'],
                        source_url=source['url'],source_sha256=source['source_sha256'],retrieved_at_ist=result['retrieved_at_ist'],
                        evidence_class='retrospective_other_publisher_daily_table',**quote))
        pd.DataFrame(rows,columns=['ipo_id','company_name','provider','source_url','source_sha256','retrieved_at_ist','evidence_class','gmp_date','gmp_rs',
            'gmp_low_rs','gmp_high_rs','gmp_basis']).to_csv(OUT,index=False)
        audit=dict(attempted=len(missing),completed=len(results),history_issuers=len({r['ipo_id'] for r in rows}),quote_rows=len(rows),
            access_limitations=['InvestorGain public history preview is limited to latest three rows; no paid-history bypass attempted.',
                'IPOcentral returned HTTP 403 during source probe; no bypass attempted.'],
            sources=[dict(ipo_id=r['ipo_id'],company_name=r['company_name'],bid_rows=len(r['bid_rows']),
                providers=[{k:v for k,v in s.items() if k not in ('quotes','raw_response_text','identity_response_text','public_graph_data')} for s in r['sources']]) for r in sorted(results,key=lambda r:r['ipo_id'])])
        AUDIT.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    tasks=[]
    for row in missing.itertuples():
        tasks.append(dict(ipo_id=int(row.ipo_id),company_name=row.company_name,issue_price=float(canonical.loc[row.ipo_id].price_band_upper),
            nse_symbol=str(canonical.loc[row.ipo_id].nse_symbol),
            decision_market_date=str(row.decision_market_date.date()),
            listing_date=str(row.listing_date.date()),
            issue_open_date=str(row.issue_open_date.date()),issue_close_date=str(row.issue_close_date.date())))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(collect,item,urls,stock_urls,a2z_urls) for item in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%10==0 or len(results)==len(tasks):
                save();print(json.dumps(dict(other_source_done=len(results),total=len(tasks))),flush=True)
    save()


if __name__=='__main__':
    run()
