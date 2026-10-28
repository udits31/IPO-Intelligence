"""Dated GMP and IPO financial inputs for the fixed listed-issuer replay cohort."""
import gzip,hashlib,json,re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import httpx
import pandas as pd
from bs4 import BeautifulSoup
from services.saved_preclose_service import ROOT,IST,all_listed,clock
from services.historical_bid_service import name_key,resolve_official_issue
from scrapers.live_ipo_scraper import load_scraped_cache
from ml.recover_module1_other_sources import dated_history
from ml.collect_module1_later_holdout import index_urls,bid_history
from ml.recover_module1_2025_training_gaps import source_rank,product_dates
from ml.collect_module1_public_financials import extract

OUT=ROOT/'output/listed_recovered_model_inputs.json'
ARCHIVE=ROOT/'data/telemetry/listed_input_recovery'


def timed_gmp(content,cutoff):
    soup=BeautifulSoup(content,'html.parser');quotes=[]
    for table in soup.find_all('table'):
        rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])] for tr in table.find_all('tr')]
        header=next((r for r in rows if len(r)>1 and 'date' in r[0].lower() and 'GMP' in r[1]),None)
        if not header: continue
        for cells in rows:
            if len(cells)<2 or not re.search(r'\d{1,2}:\d{2}',cells[0]):continue
            date_text=cells[0].replace(' IST','').replace(' · ',' ').replace('Sept','Sep')
            observed=None
            for fmt in ('%d %b %Y, %I:%M %p','%d %b %Y %I:%M %p','%b %d, %Y %I:%M %p'):
                try:observed=datetime.strptime(date_text,fmt).replace(tzinfo=IST);break
                except ValueError:pass
            if observed is None:continue
            value=re.fullmatch(r'[+]?\s*(?:₹\s*)?(-?\d[\d,]*(?:\.\d+)?)(?:\s*\([-+\d.]+%\))?',cells[1].replace('−','-'))
            if value and observed<cutoff:
                quotes.append(dict(gmp_date=observed.date().isoformat(),gmp_asof_ist=observed.isoformat(),gmp_rs=float(value[1].replace(',',''))))
    return quotes


def run():
    ARCHIVE.mkdir(exist_ok=True)
    cache=load_scraped_cache();audit=all_listed(cache)
    index=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
    registry=json.loads((ROOT/'data/historical_bid_sources.json').read_text())
    def collect(row):
        item=resolve_official_issue(next(i for i in cache.values() if i['company_name']==row['company_name']))
        cutoff=clock(row['cutoff_at_ist']);name=row['company_name']
        slug=re.sub('[^a-z0-9]+','-',name.lower()).strip('-')
        aliases={'A-One Steels India':'a-one-steels-india','Adroit Industries (India)':'adroit-industries-%28india%29',
            'Sonaselection India':'sonaselection','Zetwerk Manufacturing Business':'zetwerk',"Shah Investor's Home":'shah-investors-home',
            'Orient Cables (India)':'orient-cables','German Green Steel & Power':'german-green-steel-power'}
        urls=[u for u in registry.get(name,[]) if 'theipolist.in' in u]
        urls+=['https://theipolist.in/ipo-details/'+aliases.get(name,slug)]
        urls+=sorted(index,key=lambda u:-source_rank(item,u))[:2]
        gmp_slugs={'German Green Steel & Power':'german-green-steel','Orient Cables (India)':'orient-cables',"Shah Investor's Home":'shah-investors-home','National Stock Exchange of India':'nse'}
        urls+=['https://www.iporise.com/ipo/'+gmp_slugs.get(name,slug)+'-ipo/gmp']
        if name=='Acevector': urls+=['https://www.iposignal.in/ipo/acevector.html']
        if name=='National Stock Exchange of India': urls+=['https://ipobarta.ai/ipo/nse-ipo/gmp']
        nse_issuer=name_key(name) in ('national stock exchange of india','national stock exchange of india ltd','national stock exchange of india limited')
        result=dict(company_name=name,quotes=[],financial={},bid_history=[],sources=[],errors=[],
            listing_exchange='BSE' if nse_issuer else item.get('listing_exchange','NSE'),
            listing_price_policy='BSE exact scrip/date only; never query NSE self-listing prices' if nse_issuer else 'exchange identity/date verified separately')
        for url in dict.fromkeys(urls):
            try:
                key=hashlib.sha256(url.encode()).hexdigest();path=ARCHIVE/(key+'.html.gz')
                if path.exists(): raw=gzip.decompress(path.read_bytes())
                else:
                    response=httpx.get(url,timeout=20,follow_redirects=True,headers={'User-Agent':'Mozilla/5.0'})
                    response.raise_for_status();raw=response.content;path.write_bytes(gzip.compress(raw))
                html=raw.decode('utf8',errors='replace');soup=BeautifulSoup(html,'html.parser')
                heading=' '.join(h.get_text(' ',strip=True) for h in soup.find_all('h1'))
                identity=name_key(name) in name_key(heading) or name_key(name) in name_key(soup.get_text(' ',strip=True))
                if not identity and name in ('Orient Cables (India)','German Green Steel & Power') and 'iporise.com' in url:
                    brand=gmp_slugs[name].replace('-',' ')
                    body=soup.get_text(' ',strip=True)
                    identity=name_key(heading).startswith(brand) and bool(re.search(r'Issue price\s*₹\s*'+str(int(item['issue_price']))+r'\b',body,re.I))
                if not identity: continue
                sha=hashlib.sha256(raw).hexdigest();source=dict(url=url,raw_sha256=sha,recovered_at_ist=datetime.now(IST).isoformat())
                # IPOJI additionally checks legal issuer, cap price, offer and listing dates.
                dates=product_dates(html,item) if 'ipoji.com' in url else None
                if 'ipoji.com' in url and not dates: continue
                if 'theipolist.in' in url:
                    valid=True
                    for label,wanted in [('Issue close date',cutoff.date())]:
                        h=next((h for h in soup.find_all(['h2','h3','h4']) if h.get_text(' ',strip=True)==label),None)
                        value=pd.to_datetime(h.parent.get_text(' ',strip=True).replace(label,'').strip(),dayfirst=True,errors='coerce') if h else pd.NaT
                        if pd.isna(value) or value.date()!=wanted: valid=False
                    if not valid: continue
                result['sources'].append(source)
                timed=timed_gmp(html,cutoff)
                if 'ipobarta.ai' in url:
                    for table in soup.find_all('table'):
                        if 'Projected Listing' not in table.get_text():continue
                        for tr in table.find_all('tr'):
                            cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])]
                            if len(cells)!=8:continue
                            observed=pd.to_datetime(cells[-1].replace('Sept','Sep'),format='%d %b %Y %H:%M',errors='coerce')
                            value=re.match(r'₹\s*(\d+(?:\.\d+)?)',cells[2])
                            if pd.notna(observed) and observed.date()<cutoff.date() and value:
                                timed.append(dict(gmp_date=observed.date().isoformat(),gmp_asof_ist=observed.to_pydatetime().replace(tzinfo=IST).isoformat(),gmp_rs=float(value[1])))
                result['quotes'] += [dict(q,**source) for q in (timed or dated_history(html,cutoff.date())) if q['gmp_date']<cutoff.date().isoformat() or q.get('gmp_asof_ist')]
                if dates:
                    facts=extract(html,cutoff.date(),item['issue_price'])
                    if facts.get('annual_period'): result['financial']=dict(facts,**source,publication_basis='recovered_ipo_panel_period_checked_not_publication_certified')
                    from ml.recover_module1_other_sources import ipoji_payload
                    payload=ipoji_payload({**item,**dates},soup)
                    if payload: result['bid_history']+= [dict(b,**source) for b in bid_history(payload['bids'],pd.Timestamp(dates['issue_open_date']),pd.Timestamp(dates['issue_close_date']))]
            except Exception as exc: result['errors'].append(dict(url=url,error=str(exc)[:180]))
        print(json.dumps(dict(company=name,quotes=len(result['quotes']),financial=bool(result['financial']),bid_days=len(result['bid_history']))),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(collect,audit['records']))
    OUT.write_text(json.dumps(dict(records=records),indent=2,default=str),encoding='utf8')
    print(json.dumps(dict(companies=len(records),with_gmp=sum(bool(r['quotes']) for r in records),with_financial=sum(bool(r['financial']) for r in records))),flush=True)

if __name__=='__main__':run()
