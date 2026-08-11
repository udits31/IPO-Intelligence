"""Read-only direction checks of genuinely saved Module 2 daily calls."""
import gzip
import hashlib
import json
import math
from db import ledger as sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'data/telemetry/module2_decisions.db'
YAHOO=ROOT/'data/archive/yahoo_chart'
IST=ZoneInfo('Asia/Kolkata')


def action_direction(verdict):
    text=str(verdict).upper()
    if any(v in text for v in ('SELL','REDUCE','EXIT','TRIM','TAKE PROFIT')):return False
    if any(v in text for v in ('HOLD','WATCH','BUY','ACCUMULATE')):return True
    return None


def archived_prices(rows):
    """Use the newest checksum-verified saved source, never an unfinished bar."""
    from core.attribution_engine import parse_yahoo_chart
    from services.module2_recent_exchange import _identity_matches
    for row in sorted(rows,key=lambda r:(r['price_date'],r['observed_at_utc']),reverse=True):
        p=row['payload'];sha=p.get('price_archive_sha256') or ''
        archive=YAHOO/(sha+'.json.gz')
        if sqlite3.cloud_enabled() and not archive.exists():
            from services.cloud_assets import ensure_file
            try:ensure_file(archive)
            except Exception:continue
        symbol=(str(p.get('bse_code'))+'.BO' if p.get('price_source')=='Yahoo Finance (BSE)'
                else str(p.get('nse_symbol') or '')+'.NS')
        try:
            raw=gzip.decompress(archive.read_bytes())
            if hashlib.sha256(raw).hexdigest()!=sha:continue
            content=json.loads(raw);frame=parse_yahoo_chart(content,symbol)
            meta=frame.attrs.get('provider_meta') or {}
            if not _identity_matches(p.get('company_name') or '',meta.get('company_name') or ''):continue
            frame=frame[frame.date.le(pd.Timestamp(row['price_date']))].copy()
            reference=frame[frame.date.eq(pd.Timestamp(row['price_date']))]
            if reference.empty or not math.isclose(float(reference.close.iloc[-1]),float(p['decision_close']),rel_tol=.0001):continue
            return frame,dict(source=p['price_source'],sha256=sha,known_through=row['price_date'],splits=meta.get('splits',[]))
        except (OSError,KeyError,ValueError,TypeError,IndexError):continue
    # A subset of recent IPOs has only official NSE daily archives. Read those
    # already on disk, with exact symbol + issuer-name matching; no network IO.
    nse=[r for r in rows if r['payload'].get('price_source')=='NSE UDiFF CM']
    if nse:
        from services.module2_recent_exchange import ARCHIVES,_archive_rows
        p=max(nse,key=lambda r:r['price_date'])['payload'];points=[]
        start=min(r['price_date'] for r in rows);end=p['price_date'];symbol=p.get('nse_symbol')
        for day in pd.date_range(start,end):
            path=ARCHIVES/(day.strftime('%Y%m%d')+'.csv.zip')
            if not path.exists():continue
            frame=_archive_rows(path,{symbol})
            if len(frame)!=1:continue
            row=frame.iloc[0]
            if not _identity_matches(p.get('company_name') or '',str(row.FinInstrmNm)):continue
            points.append(dict(date=pd.Timestamp(day),close=float(row.ClsPric),sha256=row.archive_sha256))
        if points:return pd.DataFrame(points),dict(source='NSE UDiFF CM',known_through=end,splits=[])
    return pd.DataFrame(columns=['date','close']),dict(source='unavailable',splits=[])


def evaluate_calls(rows,prices,source,*,horizon,now=None):
    """Horizon counts completed session closes strictly after each saved call.

    Grade from the latest completed market close before the call, retaining the
    original (possibly stale) decision input separately. This is not a trade
    simulation: that close was not necessarily executable at the call time.
    """
    now=now or datetime.now(IST)
    if now.tzinfo is None:raise ValueError('Timezone-aware evaluation clock required')
    current=pd.Timestamp(now).tz_convert(IST)
    bars=[]
    for p in prices.sort_values('date').to_dict('records'):
        day=pd.Timestamp(p['date']).normalize()
        closed=day.tz_localize(IST)+pd.Timedelta(hours=15,minutes=30)
        close=float(p['close'])
        if closed<=current and math.isfinite(close) and close>0:
            bars.append(dict(date=day.date().isoformat(),price=close,closed_at=closed))
    calls=[]
    for r in rows:
        p=r['payload'];issued=pd.Timestamp(r['observed_at_utc']).tz_convert(IST)
        if issued>current:continue
        baseline=float(p['decision_close']);direction=action_direction(p.get('final_verdict'))
        reference=next((b for b in bars if b['date']==r['price_date']),None)
        basis_ok=reference is not None and math.isclose(reference['price'],baseline,rel_tol=.0001)
        known=[b for b in bars if b['closed_at']<=issued]
        comparison=known[-1] if known else None
        future=[b for b in bars if b['closed_at']>issued and b['date']>r['price_date']]
        outcome=future[horizon-1] if len(future)>=horizon else None
        split=outcome and any(r['price_date']<str(s['ex_date'])[:10]<=outcome['date'] for s in source.get('splits',[]))
        reference_known=reference is not None and reference['closed_at']<=issued
        status='evaluated' if outcome and basis_ok and reference_known and not split and direction is not None else 'share_basis_unverified' if split or reference and not basis_ok else 'reference_not_available_at_call' if reference is not None and not reference_known else 'unsupported_call' if direction is None else 'source_unavailable' if reference is None else 'pending'
        change=100*(outcome['price']/comparison['price']-1) if status=='evaluated' else None
        calls.append(dict(id=r['id'],call=p.get('final_verdict'),called_at_ist=issued.isoformat(),
            call_date=issued.date().isoformat(),reference_date=comparison['date'] if comparison else r['price_date'],reference_price=comparison['price'] if comparison else baseline,
            decision_reference_date=r['price_date'],decision_reference_price=baseline,
            stale_decision_input=bool(comparison and comparison['date']!=r['price_date']),
            expected_direction='Up' if direction else 'Flat / down' if direction is False else 'Unspecified',
            horizon_sessions=horizon,observed_future_sessions=len(future),outcome_date=outcome['date'] if status=='evaluated' else None,
            actual_price=outcome['price'] if status=='evaluated' else None,actual_return_pct=change,
            actual_direction='Up' if change is not None and change>0 else 'Down' if change is not None and change<0 else 'Flat' if change is not None else None,
            matched=(direction==(change>0)) if change is not None else None,status=status,
            evidence_kind='saved_live_daily_monitor',model_kind='deterministic_price_and_filing_policy',
            payload_sha256=r['payload_sha256'],monitor_code_sha256=r.get('monitor_code_sha256')))
    evaluated=[r for r in calls if r['matched'] is not None]
    return dict(horizon_sessions=horizon,summary=dict(saved=len(calls),evaluated=len(evaluated),
        matched=sum(r['matched'] for r in evaluated),missed=sum(not r['matched'] for r in evaluated),
        pending=sum(r['status']=='pending' for r in calls),
        direction_match_pct=100*sum(r['matched'] for r in evaluated)/len(evaluated) if evaluated else None),
        calls=sorted(calls,key=lambda r:r['called_at_ist'],reverse=True),
        prices=[{k:b[k] for k in ('date','price')} for b in bars],
        price_source={k:v for k,v in source.items() if k!='splits'},
        method='One first saved daily monitor call per IPO and observation day. Compare the latest completed market close before the call with the selected completed-session close strictly after it. The original decision input is preserved separately and flagged when stale; its older price cannot inflate the result. Up supports Buy/Hold/Watch; flat/down supports Sell/Reduce. This is daily-close direction agreement, not an intraday execution quote, realised profit or an ML accuracy claim. Repeated daily calls are correlated. Saved calls are not recalculated.')


def call_outcomes(ipo_id,*,horizon=1,db_path=DB):
    if horizon not in (1,5,30):raise ValueError('Choose 1, 5 or 30 trading sessions')
    rows=[];sources=[];invalid=0
    if sqlite3.exists(db_path):
        with sqlite3.connect('file:'+str(Path(db_path).resolve())+'?mode=ro',uri=True,timeout=5) as c:
            c.row_factory=sqlite3.Row
            if c.execute("SELECT 1 FROM sqlite_master WHERE name='module2_decisions_v1'").fetchone():
                for raw in c.execute('SELECT * FROM module2_decisions_v1 WHERE ipo_id=? ORDER BY observed_at_utc DESC LIMIT 90',(ipo_id,)):
                    r=dict(raw)
                    if hashlib.sha256(r['payload_json'].encode()).hexdigest()!=r['payload_sha256']:
                        invalid+=1;continue
                    try:r['payload']=json.loads(r['payload_json'])
                    except (ValueError,TypeError):invalid+=1;continue
                    if (r['payload'].get('ipo_id')!=ipo_id or r['payload'].get('price_date')!=r['price_date']
                            or r['payload'].get('observed_at_utc')!=r['observed_at_utc']):
                        invalid+=1;continue
                    rows.append(r)
            if c.execute("SELECT 1 FROM sqlite_master WHERE name='module2_events_v1'").fetchone():
                for raw in c.execute('SELECT * FROM module2_events_v1 WHERE ipo_id=? ORDER BY id DESC LIMIT 10',(ipo_id,)):
                    r=dict(raw)
                    if hashlib.sha256(r['payload_json'].encode()).hexdigest()!=r['payload_sha256']:continue
                    try:r['payload']=json.loads(r['payload_json'])
                    except (ValueError,TypeError):continue
                    if (r['payload'].get('ipo_id')!=ipo_id or r['payload'].get('price_date')!=r['price_date']
                            or r['payload'].get('observed_at_utc')!=r['observed_at_utc']):continue
                    sources.append(r)
    prices,source=archived_prices(rows+sources)
    return dict(ipo_id=ipo_id,source_kind='recorded_monitor_calls',excluded_unverified_payloads=invalid,
        **evaluate_calls(rows,prices,source,horizon=horizon))
