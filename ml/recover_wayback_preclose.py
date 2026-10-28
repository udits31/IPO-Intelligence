"""Bounded CDX lookup; no future/nearest-after-cutoff substitutions."""
import gzip,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import requests
from services.saved_preclose_service import ROOT
from services.point_in_time import IST
from urllib.parse import urlparse

CACHE=ROOT/'data/archive/wayback_preclose_20261026'


def valid_captures(data,closing_date):
    """CDX column order is explicit; never compare minutes without seconds."""
    if not isinstance(data,list) or not data:return []
    header=data[0];date=closing_date.replace('-','');results=[]
    for values in data[1:]:
        item=dict(zip(header,values));stamp=item.get('timestamp','')
        if len(stamp)==14 and date+'083000'<=stamp<date+'113000' and str(item.get('statuscode'))=='200':results.append(item)
    return sorted(results,key=lambda r:r['timestamp'])


def fetch_capture(session,item,closing_date):
    stamp=item['timestamp'];original=item['original']
    if item not in valid_captures([list(item),list(item.values())],closing_date):raise ValueError('Capture outside requested pre-cutoff window')
    url=f'https://web.archive.org/web/{stamp}id_/{original}'
    result=dict(archive_timestamp_utc=stamp,original_url=original,replay_url=url)
    try:
        response=session.get(url,timeout=25,allow_redirects=False);result['http_status']=response.status_code
        if response.is_redirect:return {**result,'status':'replay_redirect_not_followed','redirect':response.headers.get('Location')}
        if response.status_code in (401,403,429):return {**result,'status':'restricted_or_rate_limited'}
        response.raise_for_status();digest=hashlib.sha256(response.content).hexdigest();path=CACHE/(stamp+'_'+digest+'.html.gz')
        if not path.exists():path.write_bytes(gzip.compress(response.content))
        result.update(status='raw_capture_saved',archive_sha256=digest,path=str(path.relative_to(ROOT)),
            available_by_utc=datetime.strptime(stamp,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).isoformat(),
            table_verification='pending_issuer_and_subscription_table_validation')
    except Exception as exc:result.update(status='replay_failed',error=str(exc)[:200])
    return result


def query(session,url,date):
    if urlparse(url).scheme not in ('https','http'):raise ValueError('An actual complete public IPO page URL is required, including its path and numeric ID')
    params={'url':url,'from':date.replace('-','')+'083000','to':date.replace('-','')+'113000','output':'json','filter':['statuscode:200'],'fl':'timestamp,original,statuscode,mimetype,digest','collapse':'digest'}
    path=CACHE/(hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()+'.json')
    if path.exists():return json.loads(path.read_text())
    time.sleep(1)
    result=dict(source_url=url,closing_date=date,requested_ist_window='14:00-17:00',fetched_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        response=session.get('https://web.archive.org/cdx/search/cdx',params=params,timeout=15)
        result.update(http_status=response.status_code,query_url=response.url)
        if response.status_code in (401,403,429):result['status']='restricted_or_rate_limited'
        else:
            response.raise_for_status();data=response.json();captures=valid_captures(data,date)
            result.update(status='captures_found' if captures else 'no_capture_in_window',cdx_rows=data,eligible_captures=captures)
    except Exception as exc:result.update(status='request_failed',error=str(exc)[:200])
    path.write_text(json.dumps(result,indent=2),encoding='utf8');return result


def run():
    CACHE.mkdir(parents=True,exist_ok=True);session=requests.Session()
    records=json.loads((ROOT/'output/listed_closing_day_backtest.json').read_text())['records']
    from scrapers.live_ipo_scraper import load_scraped_cache
    cache=load_scraped_cache();ipo_urls={r['company_name']:url for url,r in cache.items()}
    queue=json.loads((ROOT/'output/ipopremium_recovery_queue_20261026.json').read_text());premium={r['company_name']:r['url'] for r in queue}
    results=[];restricted=False
    for row in records:
        main=ipo_urls.get(row['company_name']);urls=[main,main.replace('/ipo/','/ipo_subscription/') if main else None,row['bid'].get('source'),premium.get(row['company_name'])]
        checks=[]
        if restricted:
            results.append(dict(company_name=row['company_name'],cutoff_at_ist=row['cutoff_at_ist'],checks=[],status='deferred_after_archive_rate_limit'))
            continue
        for url in dict.fromkeys(u for u in urls if u):
            check=query(session,url,row['cutoff_at_ist'][:10]);checks.append(check)
            captures=check.get('eligible_captures') or valid_captures(check.get('cdx_rows',[]),row['cutoff_at_ist'][:10])
            if captures:
                time.sleep(1);check['latest_replay']=fetch_capture(session,captures[-1],row['cutoff_at_ist'][:10])
            if check.get('status')=='restricted_or_rate_limited':restricted=True;break
        results.append(dict(company_name=row['company_name'],cutoff_at_ist=row['cutoff_at_ist'],checks=checks))
        print(json.dumps(dict(company=row['company_name'],captures=sum(max(0,len(c.get('cdx_rows',[]))-1) for c in checks),statuses=[c['status'] for c in checks])),flush=True)
        (ROOT/'output/wayback_preclose_recovery_20261026.json').write_text(json.dumps(dict(records=results,complete=len(results)==len(records) and not restricted),indent=2),encoding='utf8')
    (ROOT/'output/wayback_preclose_recovery_20261026.json').write_text(json.dumps(dict(records=results,complete=not restricted,rate_limited_or_restricted=restricted),indent=2),encoding='utf8')
    return results


if __name__=='__main__':run()
