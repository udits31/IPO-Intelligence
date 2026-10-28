"""Catch up recoverable sources now, without manufacturing missed forecasts."""
import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from scrapers.live_ipo_scraper import (sync_chittorgarh_calendar_delta,load_scraped_cache,
    fetch_live_macro_regime,fetch_subscription_evidence)
from services.snapshot_service import parse_closing_date
from services.nse_subscription_service import fetch_nse_subscription_evidence
from services.news_service import get_news_snapshot
from services.gmp_service import gmp_service
from services.listing_outcome_service import record_verified_listing_outcomes
from services.module2_decision_ledger import capture_recent_listings

ROOT=Path(__file__).resolve().parents[1]
IST=ZoneInfo('Asia/Kolkata')
OUT=ROOT/'output/live_data_catchup_20261022.json'


def run():
    now=datetime.now(IST)
    report={'checked_at_ist':now.isoformat(),'steps':{},'subscriptions':[],
        'timing_note':'Sources downloaded now are late recovery/reconciliation, not evidence that an afternoon prediction was saved. Missed original capture times are not backfilled.'}
    with sqlite3.connect(ROOT/'data/telemetry/snapshots.db') as conn:
        report['last_original_capture_at_ist']=conn.execute('SELECT max(attempted_at_ist) FROM subscription_capture_v2').fetchone()[0]
    def save():
        OUT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    print('Refreshing calendar and offer details',flush=True)
    try:
        items=sync_chittorgarh_calendar_delta(force_refresh=True,refresh_subscriptions=False)
        report['steps']['calendar']={'returned_items':len(items),'categories':dict(Counter(item.get('category') for item in items))}
    except Exception as exc:
        report['steps']['calendar']={'error':str(exc)}
    try:
        macro=fetch_live_macro_regime()
        report['steps']['macro']={key:value for key,value in macro.items() if key not in ('history','raw')}
    except Exception as exc:
        report['steps']['macro']={'error':str(exc)}
    table=gmp_service.fetch_live_table(force_refresh=True)
    report['steps']['gmp']={'status':gmp_service.last_fetch_status,'returned_rows':len(table)}
    save()
    for url,item in load_scraped_cache().items():
        closing=parse_closing_date(item.get('date_range'),now.year)
        if item.get('category')!='OPEN' and not (item.get('category')=='CLOSED' and closing==now.date()):
            continue
        print('Recovering published demand/news:',item.get('company_name'),flush=True)
        sources=[fetch_subscription_evidence(url),fetch_nse_subscription_evidence(item['company_name'],closing,
            nse_symbol=item.get('nse_symbol'),symbol_verified=item.get('subscription_source_family')=='nse_exchange',
            nse_series=item.get('nse_series'))]
        for source in sources:
            raw=source.pop('raw_html',None)
            if raw is not None:
                raw=raw.encode('utf-8')
                digest=hashlib.sha256(raw).hexdigest()
                path=ROOT/'data/archive/live_catchup'/f'{digest}.gz'
                path.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(path,'wb') as stream:
                    stream.write(raw)
                source['recovery_raw_archive_path']=path.relative_to(ROOT).as_posix()
                source['recovery_raw_sha256']=digest
        news=get_news_snapshot(item['company_name'],force=True)
        report['subscriptions'].append({'company_name':item['company_name'],'closing_date':closing,
            'observed_now_ist':datetime.now(IST).isoformat(),'sources':sources,
            'news_status':news.get('status'),'news_freshness':news.get('freshness'),'news_articles':len(news.get('articles') or [])})
        save()
    print('Checking listing opens and current post-listing price/news evidence',flush=True)
    report['steps']['listing_outcomes']=record_verified_listing_outcomes()
    report['steps']['module2']=capture_recent_listings(lookback_days=60,require_current_session=False)
    verified=sum(source.get('status')=='verified' for row in report['subscriptions'] for source in row['sources'])
    report['summary']=(f"Calendar checked; {len(report['subscriptions'])} active/just-closed offers checked, "
        f"{verified} complete demand-source responses recovered. Module 2 checked {report['steps']['module2']['attempted']} recent listings. "
        'Source failures and pending outcomes remain explicit.')
    report['completed_at_ist']=datetime.now(IST).isoformat()
    save()
    print(json.dumps({'summary':report['summary'],'last_original_capture_at_ist':report['last_original_capture_at_ist'],
                      'timing_note':report['timing_note']},indent=2),flush=True)
    return report


if __name__=='__main__':
    run()
