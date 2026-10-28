"""Recover missing historical bid books from explicitly timed public tables."""
import gzip
import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from services.saved_preclose_service import DB, ROOT, IST, all_listed, clock
from services.listed_backtest_service import select_book
from services.historical_bid_service import name_key
from services.snapshot_service import extract_ipo_id
from scrapers.live_ipo_scraper import load_scraped_cache


def parse(html,company,cutoff,url):
    soup=BeautifulSoup(html,'html.parser');text=soup.get_text(' ',strip=True)
    heading=' '.join(h.get_text(' ',strip=True) for h in soup.find_all('h1'))
    expected=name_key(company)
    # Brand aliases are accepted only with the full legal issuer in page text.
    if expected not in name_key(heading) and expected not in name_key(text): return []
    host=urlparse(url).hostname.removeprefix('www.');found=[]
    def emit(observed,values,basis):
        if observed.date()!=cutoff.date() or not observed<cutoff or set(values)!={'qib','hni','retail'}: return
        if any(v is None or v<0 for v in values.values()): return
        found.append(dict(observed_on=observed.date().isoformat(),source_reported_at_ist=observed.isoformat(),
            **values,timestamp_basis=basis,evidence_warnings=['Recovered historical page; publisher-reported bid time, not an immutable capture made by this application']))
    def ratio(v):
        m=re.search(r'(\d[\d,]*(?:\.\d+)?)\s*[x×]?',v)
        return float(m[1].replace(',','')) if m else None
    if host=='groww.in':
        m=re.search(r"As of (\d{1,2} [A-Za-z]{3}'\d{2}), (\d{1,2}:\d{2} [AP]M)",text)
        if m:
            observed=datetime.strptime(m[1]+' '+m[2],"%d %b'%y %I:%M %p").replace(tzinfo=IST)
            block=text[max(0,m.start()-700):m.start()];values={}
            for key,label in [('qib',r'Qualified Institutional Buyer[s]?'),('hni',r'Non-Institutional Investor[s]?'),('retail',r'Retail Individual Investor[s]?')]:
                v=re.search(label+r'\s*(\d+(?:\.\d+)?)x',block,re.I)
                if v: values[key]=float(v[1])
            emit(observed,values,'groww_subscription_section_asof')
    elif host=='iporise.com':
        m=re.search(r'Live Subscription Data\s+Last updated:\s*([A-Za-z]{3} \d{1,2}, \d{4}, \d{1,2}:\d{2} [AP]M) IST',text)
        if m:
            observed=datetime.strptime(m[1],'%b %d, %Y, %I:%M %p').replace(tzinfo=IST)
            title=next((h for h in soup.find_all(['h2','h3']) if h.get_text(' ',strip=True)=='Live Subscription Data'),None)
            table=title.find_next('table') if title else None;values={}
            for tr in table.find_all('tr') if table else []:
                cells=[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])]
                if len(cells)<4: continue
                key='qib' if cells[0].startswith('QIB') else 'hni' if cells[0].startswith('NII ') else 'retail' if cells[0].startswith('Retail') else None
                if key: values[key]=ratio(cells[-1])
            emit(observed,values,'iporise_live_subscription_table_asof')
    elif host=='5paisa.com':
        # Require an explicit full date and bid time in the same paragraph.
        for p in soup.find_all('p'):
            paragraph=p.get_text(' ',strip=True)
            if not re.search(r'subscrib|subscription',paragraph,re.I): continue
            tm=re.search(r'\b(\d{1,2}:\d{2}(?::\d{2})? [AP]M)\b',paragraph)
            day=re.search(r'([A-Za-z]+ \d{1,2}, \d{4})',paragraph)
            if not tm or not day: continue
            fmt='%B %d, %Y %I:%M:%S %p' if tm[1].count(':')==2 else '%B %d, %Y %I:%M %p'
            try: observed=datetime.strptime(day[1]+' '+tm[1],fmt).replace(tzinfo=IST)
            except ValueError: continue
            # Use the explicitly dated Day-3 row, never a different day's table.
            for table in soup.find_all('table'):
                rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])] for tr in table.find_all('tr')]
                if not rows or not all(k in rows[0] for k in ('QIB','NII','Retail')): continue
                header=rows[0]
                for row in rows[1:]:
                    if len(row)!=len(header) or observed.strftime('%B') not in row[0] or not re.search(r'\b'+str(observed.day)+r'\b',row[0]): continue
                    values={key:ratio(row[header.index(label)]) for key,label in [('qib','QIB'),('hni','NII'),('retail','Retail')]}
                    emit(observed,values,'5paisa_explicit_dated_subscription_paragraph_and_day_row')
    elif host=='ipometrics.in':
        for table in soup.find_all('table'):
            if 'Snapshot Time (UTC)' not in table.get_text(' ',strip=True): continue
            for tr in table.find_all('tr'):
                cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])]
                if len(cells)<4: continue
                try: observed=datetime.strptime(cells[0],'%b %d, %Y %I:%M %p').replace(tzinfo=timezone.utc).astimezone(IST)
                except ValueError: continue
                emit(observed,dict(zip(('qib','hni','retail'),map(ratio,cells[1:4]))),'ipometrics_intraday_table_utc_converted_to_ist')
    return found


def run():
    cache=load_scraped_cache();audit=all_listed(cache)
    missing={r['company_name']:r for r in audit['records'] if not select_book(r)}
    aliases={'A-One Steels India':'a-one-steels','Adroit Industries (India)':'adroit-industries',
        'Elevate Campuses':'elevate-campuses','Swastika Infra':'swastika-infra','ArMee Infotech':'armee-infotech',
        'Varmora Granito':'varmora-granito','Sonaselection India':'sonaselection-india','SS Retail':'ss-retail',
        'National Stock Exchange of India':'national-stock-exchange-of-india'}
    jobs=[]
    for company,row in missing.items():
        slug=aliases[company]
        urls=[f'https://groww.in/ipo/{slug}-ipo',f'https://www.iporise.com/ipo/{slug}-ipo/subscription',
              f'https://www.5paisa.com/blog/{slug}-ipo-subscription-status',f'https://ipometrics.in/ipo/{slug}/subscription']
        for url in urls: jobs.append((company,row,url))
    def fetch(job):
        company,row,url=job
        try:
            response=httpx.get(url,timeout=20,follow_redirects=True,headers={'User-Agent':'Mozilla/5.0'})
            response.raise_for_status();cutoff=clock(row['cutoff_at_ist'])
            evidence=parse(response.text,company,cutoff,str(response.url))
            ipo_url,item=next((u,i) for u,i in cache.items() if i['company_name']==company)
            ipo_id=extract_ipo_id({**item,'url':ipo_url})
            raw=response.content;sha=hashlib.sha256(raw).hexdigest()
            archive=ROOT/'data/telemetry/closing_book_recovery';archive.mkdir(exist_ok=True)
            (archive/(sha+'.html.gz')).write_bytes(gzip.compress(raw))
            with sqlite3.connect(DB,timeout=20) as c:
                for e in evidence:
                    c.execute('INSERT OR IGNORE INTO historical_subscription_evidence_v1 (ipo_id,source_url,fetched_at_ist,raw_sha256,raw_html_gzip,evidence_json) VALUES (?,?,?,?,?,?)',
                        (ipo_id,str(response.url),datetime.now(IST).isoformat(),sha,gzip.compress(raw),json.dumps(e,sort_keys=True)))
            return dict(company=company,url=str(response.url),status='recovered' if evidence else 'no_qualifying_timed_table',rows=len(evidence),raw_sha256=sha)
        except Exception as exc: return dict(company=company,url=url,status='fetch_or_parse_error',error=str(exc)[:240])
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(fetch,jobs):
            results.append(r)
            print(json.dumps(r),flush=True)
    (ROOT/'output/closing_book_recovery_attempts.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    registry_path=ROOT/'data/historical_bid_sources.json'
    registry=json.loads(registry_path.read_text(encoding='utf8'))
    for result in results:
        if result['status']=='recovered':
            registry[result['company']]=list(dict.fromkeys(registry.get(result['company'],[])+[result['url']]))
    registry_path.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf8')
    from services.listed_backtest_service import run as replay
    print(json.dumps(replay()['summary']),flush=True)


if __name__=='__main__': run()
