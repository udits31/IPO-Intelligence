"""Append-only observed books at separate Day2/preclose/postclose milestones."""
import gzip,hashlib,json
from db import ledger as sqlite3
from datetime import datetime,time,timedelta
from contextlib import closing
import pandas as pd
from services.point_in_time import IST,timestamp


def milestone(item,clock,opened,cutoff):
    if not opened or not cutoff:return None
    supplied=item.get('bidding_sessions')
    days=sorted(pd.Timestamp(d).date() for d in supplied) if supplied else [d.date() for d in pd.bdate_range(opened.date(),cutoff.date())]
    if len(days)<2:return None
    penultimate=days[-2]
    basis='supplied_bidding_sessions' if supplied else 'weekday_calendar_holidays_not_certified'
    if clock.date()==penultimate and clock.hour>=17:return 'DAY2_EOD_SNAPSHOT',basis
    if clock.date()==cutoff.date() and 14<=clock.hour<17:return 'DAY3_PRECLOSE_DECISION_SNAPSHOT',basis
    if (clock.date()==cutoff.date() and clock.hour>=18) or (clock.date()==cutoff.date()+timedelta(days=1) and 8<=clock.hour<10):return 'DAY3_POSTCLOSE_FINAL_BOOK',basis
    return None


def init_table(db):
    with closing(sqlite3.connect(db,timeout=10)) as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS module1_milestone_capture_v1 (
            id INTEGER PRIMARY KEY,ipo_id TEXT NOT NULL,company_name TEXT NOT NULL,milestone TEXT NOT NULL,
            slot TEXT NOT NULL,recorded_at_ist TEXT NOT NULL,closing_date TEXT NOT NULL,
            status TEXT NOT NULL,evidence_json TEXT NOT NULL,raw_sha256 TEXT,raw_gzip BLOB,
            UNIQUE(ipo_id,milestone,slot));
            CREATE TRIGGER IF NOT EXISTS milestone_no_update BEFORE UPDATE ON module1_milestone_capture_v1 BEGIN SELECT RAISE(ABORT,'Milestones are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS milestone_no_delete BEFORE DELETE ON module1_milestone_capture_v1 BEGIN SELECT RAISE(ABORT,'Milestones are immutable'); END;''')
        c.commit()


def capture(item,url,*,db,clock=None,fetch=None):
    from services.prospective_module1 import calendar
    from services.snapshot_service import extract_ipo_id
    now=clock or datetime.now(IST);opened,cutoff=calendar(item,now);due=milestone(item,now,opened,cutoff)
    if not due:return None
    kind,calendar_basis=due;key=extract_ipo_id({**item,'url':url})
    interval=5 if kind=='DAY3_PRECLOSE_DECISION_SNAPSHOT' else 15
    slot=now.replace(minute=now.minute//interval*interval,second=0,microsecond=0).isoformat()
    init_table(db)
    with closing(sqlite3.connect(db,timeout=10)) as c:
        if c.execute('SELECT 1 FROM module1_milestone_capture_v1 WHERE ipo_id=? AND milestone=? AND slot=?',(key,kind,slot)).fetchone():return None
    if fetch is None:
        from services.nse_subscription_service import fetch_nse_subscription_evidence
        fetch=lambda:fetch_nse_subscription_evidence(item['company_name'],cutoff.date(),nse_symbol=item.get('nse_symbol'),symbol_verified=item.get('nse_symbol_source')=='nse_current_issue',nse_series=item.get('nse_series'))
    try:evidence=fetch()
    except Exception as exc:evidence=dict(status='fetch_failed',error=str(exc)[:250])
    raw=evidence.pop('raw_html',None);raw=raw.encode() if isinstance(raw,str) else raw
    fetched=timestamp(evidence.get('fetched_at_ist'));published=timestamp(evidence.get('source_displayed_at_ist'))
    finished=clock or datetime.now(IST)
    usable=evidence.get('status')=='verified' and raw is not None and fetched is not None and fetched<=finished
    if kind=='DAY3_PRECLOSE_DECISION_SNAPSHOT' and (finished>=cutoff or not fetched or fetched>=cutoff or (published and published>=cutoff)):
        usable=False;evidence['rejection']='Capture completed after application cutoff; not a preclose snapshot.'
    values=evidence.get('values') or {}
    # Preserve raw split-category fields if offered; absent splits are null, not zero.
    payload=dict(source=evidence,calendar_basis=calendar_basis,usable=usable,
        categories={k:values.get(v) for k,v in [('QIB','qib_subscription'),('NII','hni_subscription'),('sNII','snii_subscription'),('bNII','bnii_subscription'),('Retail','retail_subscription')]},
        gmp=dict(value=item.get('gmp_pct'),observed_at=item.get('gmp_observed_at')),
        raw_category_payload_retained=bool(raw),final_book_certified=evidence.get('is_final') is True,
        eligibility='postclose_reconciliation_only' if kind=='DAY3_POSTCLOSE_FINAL_BOOK' else 'day2_eod_observation' if kind=='DAY2_EOD_SNAPSHOT' else 'preclose_observation')
    quote=timestamp(item.get('gmp_observed_at'))
    if not quote or quote>finished:payload['gmp']=dict(value=None,observed_at=None,status='no_eligible_dated_quote')
    sha=hashlib.sha256(raw).hexdigest() if raw else None
    with closing(sqlite3.connect(db,timeout=10)) as c:
        c.execute('INSERT OR IGNORE INTO module1_milestone_capture_v1 (ipo_id,company_name,milestone,slot,recorded_at_ist,closing_date,status,evidence_json,raw_sha256,raw_gzip) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (key,item['company_name'],kind,slot,finished.isoformat(),str(cutoff.date()),'captured' if usable else 'unavailable',json.dumps(payload,default=str),sha,gzip.compress(raw) if raw else None));c.commit()
    return dict(milestone=kind,status='captured' if usable else 'unavailable')
