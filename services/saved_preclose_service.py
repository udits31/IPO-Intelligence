"""Read actually issued calls. Historical bid evidence never becomes a forecast."""
import gzip
import hashlib
import json
from db import ledger as sqlite3
from contextlib import closing
from datetime import datetime,time,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST=ZoneInfo('Asia/Kolkata')
ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'data/telemetry/snapshots.db'


def clock(value):
    try:
        result=datetime.fromisoformat(str(value))
        return result.astimezone(IST) if result.tzinfo else None
    except (TypeError,ValueError):
        return None


def subscriptions(item,ipo_id,cutoff,db_path):
    candidates=[]
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        c.row_factory=sqlite3.Row
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'subscription_capture_v2' in tables:
            for raw in c.execute("SELECT * FROM subscription_capture_v2 WHERE ipo_id=? AND status='verified' ORDER BY id DESC",(ipo_id,)):
                row=dict(raw);observed=clock(row.get('fetched_at_ist'))
                if not observed or observed>=cutoff or any(row.get(k) is None for k in ('qib_x','nii_x','retail_x')):
                    continue
                try:
                    if hashlib.sha256(gzip.decompress(row['raw_html_gzip'])).hexdigest()!=row['raw_sha256']:
                        continue
                except (OSError,TypeError):
                    continue
                candidates.append(dict(observed_at_ist=observed.isoformat(),recovered_at_ist=observed.isoformat(),
                    qib=row['qib_x'],hni=row['nii_x'],retail=row['retail_x'],source=row.get('subscription_url'),
                    evidence='captured_before_close',raw_sha256=row['raw_sha256']))
        if 'historical_bid_reconstruction_v1' in tables:
            for raw in c.execute('SELECT * FROM historical_bid_reconstruction_v1 WHERE ipo_id=? OR LOWER(company_name)=LOWER(?)',(ipo_id,item['company_name'])):
                row=dict(raw);inputs=json.loads(row['input_json']);reported=clock(inputs.get('source_reported_at_ist'))
                # Same-day end-of-day cells have no pre-17:00 proof. Do not infer it.
                if not reported or reported>=cutoff:
                    continue
                if any(inputs.get(k) is None for k in ('qib_subscription','hni_subscription','retail_subscription')):
                    continue
                sources=json.loads(row['evidence_json'])
                candidates.append(dict(observed_at_ist=reported.isoformat(),recovered_at_ist=row['computed_at_ist'],
                    qib=inputs['qib_subscription'],hni=inputs['hni_subscription'],retail=inputs['retail_subscription'],
                    source=sources[0].get('url') if sources else None,evidence='publisher_reported_time_recovered_later',
                    warnings=inputs.get('evidence_warnings',[]),raw_sha256=sources[0].get('raw_sha256') if sources else None))
        if 'historical_subscription_evidence_v1' in tables:
            for raw in c.execute('SELECT * FROM historical_subscription_evidence_v1 WHERE ipo_id=?',(ipo_id,)):
                row=dict(raw);inputs=json.loads(row['evidence_json']);reported=clock(inputs.get('source_reported_at_ist'))
                observed_on=inputs.get('observed_on')
                eligible=reported and reported<cutoff
                if not reported and observed_on:
                    eligible=observed_on<cutoff.date().isoformat()
                if eligible:
                    candidates.append(dict(observed_at_ist=reported.isoformat() if reported else None,observed_on=observed_on,recovered_at_ist=row['fetched_at_ist'],
                        qib=inputs['qib'],hni=inputs['hni'],retail=inputs['retail'],source=row['source_url'],
                        evidence='publisher_reported_time_recovered_later',warnings=inputs.get('evidence_warnings',[]),raw_sha256=row['raw_sha256']))
    candidates.sort(key=lambda r:(r['observed_at_ist'] or r.get('observed_on',''),r['evidence']=='captured_before_close'),reverse=True)
    seen=set();unique=[]
    for row in candidates:
        key=(row['observed_at_ist'],row.get('observed_on'),row['qib'],row['hni'],row['retail'],row['source'])
        if key not in seen:
            seen.add(key);unique.append(row)
    return unique


def comparison(item,url,db_path=DB):
    from scrapers.live_ipo_scraper import parse_date_interval
    from services.snapshot_service import extract_ipo_id
    from services.historical_bid_service import resolve_official_issue
    from services.listing_outcome_service import verified_listing_outcomes
    resolved=resolve_official_issue(item)
    _,closed=parse_date_interval(resolved.get('date_range') or '',datetime.now(IST).year)
    cutoff=datetime.combine(closed,time(17),IST) if closed else None
    ipo_id=extract_ipo_id({**item,'url':url})
    actual=verified_listing_outcomes(Path(db_path)).get(ipo_id)
    outcome=None
    if actual:
        price=float(actual['issue_price']);opening=float(actual.get('exchange_open_price') or actual['nse_open_price'])
        outcome={**actual,'actual_open':opening,'return_pct':100*(opening/price-1),'positive':opening>price}
    candidates=[];archives=[]
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        c.row_factory=sqlite3.Row
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if cutoff and 'frozen_module1_forecast_v1' in tables:
            for raw in c.execute('SELECT * FROM frozen_module1_forecast_v1 WHERE ipo_id=?',(ipo_id,)):
                row=dict(raw);issued=clock(row['recorded_at_ist']);stored_cutoff=clock(row['cutoff_at_ist'])
                if issued and stored_cutoff and issued<min(cutoff,stored_cutoff):
                    candidates.append(dict(row,inputs=json.loads(row['input_json']),mode='saved_pre_close',probability=row['probability']))
        if cutoff and 'automatic_decision_v1' in tables:
            for raw in c.execute("SELECT * FROM automatic_decision_v1 WHERE ipo_id=? AND decision_kind='live_bid_forecast' AND positive_listing_probability IS NOT NULL",(ipo_id,)):
                row=dict(raw);inputs=json.loads(row['input_json']);issued=clock(row['recorded_at_ist']);observed=clock(row['source_asof_ist'])
                stored_cutoff=clock(inputs.get('application_cutoff_at_ist'))
                if issued and observed and stored_cutoff and observed<=issued<min(cutoff,stored_cutoff):
                    candidates.append(dict(row,inputs=inputs,mode='legacy_saved_pre_close',legacy_model=True,
                        probability=row['positive_listing_probability'],gmp_probability=None,quality='complete'))
        if 'prelisting_outcome_forecast_v1' in tables:
            archives=[dict(r) for r in c.execute('SELECT * FROM prelisting_outcome_forecast_v1 WHERE ipo_id=?',(ipo_id,))]
    bids=subscriptions(item,ipo_id,cutoff,db_path) if cutoff else []
    if candidates:
        saved=max(candidates,key=lambda r:clock(r['recorded_at_ist']))
        return {**saved,'outcome':outcome,'cutoff_at_ist':cutoff.isoformat(),
            'subscription_history':bids,'saved_call_found':True,'company_name':item['company_name']}
    return dict(mode='no_saved_pre_close',probability=None,gmp_probability=None,quality='not_recorded',
        saved_call_found=False,company_name=item['company_name'],cutoff_at_ist=cutoff.isoformat() if cutoff else None,
        outcome=outcome,subscription_history=bids,after_close_archives=archives,
        explanation='No model probability was saved before the verified application cutoff. Later scores and recovered subscriptions cannot be relabelled as an earlier prediction.')


def all_listed(cache=None,db_path=DB):
    from scrapers.live_ipo_scraper import load_scraped_cache
    cache=load_scraped_cache() if cache is None else cache
    rows=[comparison(item,url,db_path) for url,item in cache.items() if item.get('category')=='LISTED']
    return {'records':rows,'summary':{'listed_ipos':len(rows),'saved_pre_close':sum(r['saved_call_found'] for r in rows),
        'with_timestamped_pre_close_bids':sum(any(s.get('observed_at_ist') for s in r['subscription_history']) for r in rows),
        'with_pre_close_subscription_evidence':sum(bool(r['subscription_history']) for r in rows),
        'with_closing_day_pre17_bids':sum(any(s.get('observed_at_ist') and r.get('cutoff_at_ist') and clock(s['observed_at_ist']).date()==clock(r['cutoff_at_ist']).date() for s in r['subscription_history']) for r in rows),
        'verified_openings':sum(bool(r['outcome']) for r in rows)}}
