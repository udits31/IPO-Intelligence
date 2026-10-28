"""Recover pre-cutoff subscription evidence without calculating any forecasts."""
import gzip
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,time
from pathlib import Path

import pandas as pd

from services import historical_bid_service as history
from services.saved_preclose_service import all_listed,clock,DB,IST
from services.listing_outcome_service import record_verified_listing_outcomes
from scrapers.live_ipo_scraper import load_scraped_cache,parse_date_interval

ROOT=Path(__file__).resolve().parents[1]


def collect(url,item):
    item=history.resolve_official_issue(item)
    opening,closing=parse_date_interval(item.get('date_range') or '',datetime.now(IST).year)
    if not opening or not closing:
        return {'company':item['company_name'],'status':'offer_dates_unverified','recovered':0}
    cutoff=datetime.combine(closing,time(17),IST)
    from services.snapshot_service import extract_ipo_id
    ipo_id=extract_ipo_id({**item,'url':url})
    recovered=0;errors=[]
    for source in history.discover_sources(item['company_name']):
        try:
            source,html=history._fetch(source)
            rows=history.parse_timestamped_snapshot(html,item['company_name'],opening,closing,source)+history.parse_article_snapshot(html,item['company_name'],opening,closing,source)
            rows=[r for r in rows if clock(r.get('source_reported_at_ist')) and clock(r['source_reported_at_ist'])<cutoff]
            raw=html.encode();sha=hashlib.sha256(raw).hexdigest();found=datetime.now(IST).isoformat()
            with sqlite3.connect(DB,timeout=15) as c:
                for row in rows:
                    c.execute('INSERT OR IGNORE INTO historical_subscription_evidence_v1 (ipo_id,source_url,fetched_at_ist,raw_sha256,raw_html_gzip,evidence_json) VALUES (?,?,?,?,?,?)',
                        (ipo_id,source,found,sha,gzip.compress(raw),json.dumps(row,sort_keys=True)))
                    recovered+=1
        except Exception as exc:
            errors.append(str(exc)[:160])
    if not recovered:
        from ml.collect_module1_later_holdout import index_urls,bid_history
        from ml.recover_module1_2025_training_gaps import source_rank,product_dates,history as daywise
        from ml.recover_module1_other_sources import fetch,ipoji_payload
        from bs4 import BeautifulSoup
        urls=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
        for source in sorted(urls,key=lambda u:-source_rank(item,u))[:2]:
            try:
                response=fetch(source);dates=product_dates(response.text,item)
                if not dates:
                    continue
                payload=ipoji_payload({**item,**dates},BeautifulSoup(response.text,'html.parser'))
                if not payload:
                    continue
                days=daywise(payload['bids'],opening,closing)
                timed={r['date']:r for r in bid_history(payload['bids'],opening,closing)}
                for day in days:
                    reported=None
                    if day['date'] in timed:
                        reported=datetime.strptime(day['date']+' '+timed[day['date']]['displayed_time'],'%Y-%m-%d %I:%M %p').replace(tzinfo=IST)
                    if (reported and reported>=cutoff) or (not reported and day['date']>=closing.isoformat()):
                        continue
                    evidence=dict(observed_on=day['date'],source_reported_at_ist=reported.isoformat() if reported else None,
                        qib=day['qib_subscription'],hni=day['hni_subscription'],retail=day['retail_subscription'],
                        evidence_warnings=['Historical public table; original publication time is not certified'])
                    with sqlite3.connect(DB,timeout=15) as c:
                        c.execute('INSERT OR IGNORE INTO historical_subscription_evidence_v1 (ipo_id,source_url,fetched_at_ist,raw_sha256,raw_html_gzip,evidence_json) VALUES (?,?,?,?,?,?)',
                            (ipo_id,source,datetime.now(IST).isoformat(),hashlib.sha256(response.content).hexdigest(),gzip.compress(response.content),json.dumps(evidence,sort_keys=True)))
                    recovered+=1
                break
            except Exception as exc:
                errors.append(str(exc)[:160])
    return {'company':item['company_name'],'status':'recovered' if recovered else 'no_timestamp_verified_before_cutoff','recovered':recovered,'errors':errors}


def run():
    cache=load_scraped_cache()
    with sqlite3.connect(DB) as c:
        c.execute('''CREATE TABLE IF NOT EXISTS historical_subscription_evidence_v1 (
          id INTEGER PRIMARY KEY,ipo_id TEXT NOT NULL,source_url TEXT NOT NULL,fetched_at_ist TEXT NOT NULL,
          raw_sha256 TEXT NOT NULL,raw_html_gzip BLOB NOT NULL,evidence_json TEXT NOT NULL,
          UNIQUE(ipo_id,source_url,raw_sha256,evidence_json))''')
    before=all_listed(cache);covered={r['company_name'] for r in before['records'] if r['subscription_history'] and clock(r['subscription_history'][0]['observed_at_ist']) and clock(r['subscription_history'][0]['observed_at_ist']).date()==clock(r['cutoff_at_ist']).date()}
    targets=[(url,item) for url,item in cache.items() if item.get('category')=='LISTED' and item['company_name'] not in covered]
    progress=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(collect,url,item) for url,item in targets]):
            result=future.result();progress.append(result)
            print(result['company']+': '+result['status'],flush=True)
    openings=record_verified_listing_outcomes(cache=cache,include_all_listed=True)
    audit=all_listed(cache);audit['subscription_collection']=progress;audit['opening_collection']=openings
    output=ROOT/'output/listed_saved_preclose_audit_20261025.json'
    output.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    rows=[]
    for row in audit['records']:
        latest=row['subscription_history'][0] if row['subscription_history'] else {}
        outcome=row.get('outcome') or {}
        rows.append(dict(company=row['company_name'],cutoff_ist=row.get('cutoff_at_ist'),
            saved_preclose_found=row['saved_call_found'],saved_probability=row.get('probability'),saved_at_ist=row.get('recorded_at_ist'),
            actual_open=outcome.get('actual_open'),actual_open_return_pct=outcome.get('return_pct'),
            subscription_asof_ist=latest.get('observed_at_ist'),qib=latest.get('qib'),hni=latest.get('hni'),retail=latest.get('retail'),
            source=latest.get('source'),evidence=latest.get('evidence'),recovered_at_ist=latest.get('recovered_at_ist')))
        rows[-1]['closing_day_reading']=bool(latest.get('observed_at_ist') and row.get('cutoff_at_ist') and clock(latest['observed_at_ist']).date()==clock(row['cutoff_at_ist']).date())
    pd.DataFrame(rows).to_csv(output.with_suffix('.csv'),index=False)
    print(json.dumps(audit['summary']),flush=True)


if __name__=='__main__':
    run()
