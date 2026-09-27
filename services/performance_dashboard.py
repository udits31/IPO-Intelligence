"""Small presentation payload; never retrain or alter saved calls."""
import csv
import hashlib
import json
import math
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build_dashboard():
    lock_path=ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json'
    digest=hashlib.sha256(lock_path.read_bytes()).hexdigest()
    locked=json.loads(lock_path.read_text(encoding='utf8'))
    m1=json.loads((ROOT/'output/module1_two_track_evidence_20261026.json').read_text(encoding='utf8'))
    sequential=json.loads((ROOT/'output/module2_sequential_path_evaluation_20261026.json').read_text(encoding='utf8'))
    if any(r['locked_evidence_sha256']!=digest for r in (m1,sequential)):
        raise ValueError('Performance reports do not match the locked benchmark.')
    # Display primary opening prices only when they agree with the locked return.
    audit={}
    with (ROOT/'data/audit/module1_exchange_listing_open_audit.csv').open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            if r.get('status')=='verified_daily_exchange_archive':audit[str(r['ipo_id'])]=r
    from services.listing_outcome_service import verified_listing_outcomes
    from services.historical_bid_service import name_key
    recent={name_key(r['company_name']):r for r in verified_listing_outcomes().values()}
    for cohort in m1['cohorts'].values():
        model=cohort['model_direction'];screen=cohort['application_screen']
        cohort['always_positive_accuracy_pct']=100*screen['total_actual_winners']/cohort['n']
        cohort['accuracy_minus_baseline_pp']=model['accuracy_pct']-cohort['always_positive_accuracy_pct']
        for r in cohort['records']:
            source=audit.get(r['issuer_key']) or recent.get(name_key(r['company_name'])) or {}
            try:
                issue=float(source['issue_price']);opening=float(source.get('exchange_open') or source.get('exchange_open_price') or source.get('nse_open_price'))
                if min(issue,opening)>0 and math.isfinite(issue+opening) and abs(100*(opening/issue-1)-r['actual_return_pct'])<.001:
                    r.update(issue_price=issue,actual_open=opening,listing_date=source.get('listing_date'))
            except (TypeError,ValueError,KeyError):pass
    from services.prospective_module1 import scorecard
    live=scorecard()
    live_rows=[]
    for r in live['records']:
        plan=r['inputs'].get('capital_plan') or {}
        live_rows.append({k:r.get(k) for k in ('company_name','probability','model_call','model_outcome',
            'recorded_at_ist','cutoff_at_ist','listing_date','quality','application_screen','outcome')})
        live_rows[-1]['missing_inputs']=plan.get('missing_inputs',[])
    path=ROOT/'output/live_data_review_20261026.json'
    review=json.loads(path.read_text(encoding='utf8')) if path.exists() else None
    intraday=None
    if review:
        r=review['module2'];q=r['exploratory_15_minute']
        intraday=dict(review_date=review['as_of_ist'][:10],alerts=r['alerts'],alert_ipos=r['alert_ipos'],
            candles=r['distinct_minute_bars'],instruments=r['unique_instruments_with_candles'],
            evaluated=q['evaluated'],matched=q['direction_matched'],excluded=q['excluded'],
            always_down_matched=q.get('always_down_matched'),by_signal=q['by_signal'],
            first_alert=r['earliest_live_alert_ist'],last_alert=r['last_live_alert_ist'])
    w=sequential['windows']['LOCKED_EXECUTION_ALIGNED']
    return dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),locked_sha256=digest,
        model=dict(name='Frozen five-expert stack',training_years='2021–2025',training_ipos=290,
                   threshold=.5,calibration='none'),
        module1=m1['cohorts'],module2=dict(n=locked['module2']['cohort_n'],
            winners=locked['module2']['winners'],losers=locked['module2']['losers'],
            active=locked['module2']['results']['M2_R1_WITH_DATED_FILINGS'],
            baseline=locked['module2']['results']['ALWAYS_SELL'],
            daily_paths={key:{'A':p['A'],'B':p['B']} for key,p in w['policies'].items()},
            held_fates=w['held_at_31_fates'],timing_note=sequential['index_note']),
        prospective=dict(summary=live['summary'],records=live_rows),intraday=intraday)
