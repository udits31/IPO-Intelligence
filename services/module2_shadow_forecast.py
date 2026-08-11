"""Append-only forward test of the research model; never changes live verdicts.

Uses the exact market-only training feature definitions. Missing OHLCV,
identity conflicts, share-unit changes and unsupported ages block forecasting.
"""
import gzip,hashlib,json,math
from db import ledger as sqlite3
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core.attribution_engine import AttributionEngine,YAHOO_ARCHIVES,parse_yahoo_chart
from ml.research_module2_multihorizon import policy_positions

ROOT=Path(__file__).resolve().parents[1];DB=ROOT/'data/telemetry/module2_decisions.db';IST=ZoneInfo('Asia/Kolkata')
ARTIFACT=ROOT/'output/module2_multihorizon_20261029/multihorizon_research.joblib'
DAILY=ROOT/'output/module2_next_session_improvement_20261029/module2_retrained_research.joblib'
MANIFEST=ROOT/'output/module2_multihorizon_20261029/shadow_protocol.json'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def features_at(prices,issue_price):
    """Same formulas as ml/train_module2_next_session.py, no future rows."""
    g=prices.sort_values('date').reset_index(drop=True)
    if not 21<=len(g)<=61:raise ValueError('Model requires 21-61 completed post-listing sessions')
    required=['open','high','low','close','volume','nifty50_close','sector_index_close']
    if g[required].isna().any().any():raise ValueError('Incomplete dated OHLCV/benchmark history')
    if not np.isfinite(g[required].to_numpy(dtype=float)).all():raise ValueError('Nonfinite market input')
    if not g[['open','high','low','close','nifty50_close','sector_index_close']].gt(0).all().all() or g.volume.lt(0).any():raise ValueError('Invalid price/volume')
    if g.high.lt(g[['open','close','low']].max(axis=1)).any() or g.low.gt(g[['open','close']].min(axis=1)).any():raise ValueError('Invalid OHLC bounds')
    c=g.close;v=g.volume;i=len(g)-1
    def change(s,n):return float((s.iloc[i]/s.iloc[i-n]-1)*100)
    band=g.high.iloc[i]-g.low.iloc[i];prior_volume=v.iloc[i-19:i].mean()
    if band<=0 or prior_volume<=0:raise ValueError('Insufficient traded range or volume')
    values=dict(ret1=c.pct_change(fill_method=None).iloc[i]*100,ret5=change(c,5),ret10=change(c,10),ret20=change(c,20),
        ma10_gap=(c.iloc[i]/c.iloc[i-9:i+1].mean()-1)*100,ma20_gap=(c.iloc[i]/c.iloc[i-19:i+1].mean()-1)*100,
        vol20=c.pct_change(fill_method=None).iloc[i-19:i+1].std()*100,drawdown=(c.iloc[i]/c.max()-1)*100,
        age=i+1,close_issue=(c.iloc[i]/float(issue_price)-1)*100,volume_ratio=v.iloc[i]/prior_volume,
        range_pct=100*band/c.iloc[i],close_location=(c.iloc[i]-g.low.iloc[i])/band,
        gap=(g.open.iloc[i]/c.iloc[i-1]-1)*100,nifty1=change(g.nifty50_close,1),nifty5=change(g.nifty50_close,5),
        nifty20=change(g.nifty50_close,20),sector5=change(g.sector_index_close,5),relative5=change(c,5)-change(g.nifty50_close,5))
    if not all(math.isfinite(float(v)) for v in values.values()):raise ValueError('Missing forecast feature')
    return {k:float(v) for k,v in values.items()}


def frame_from_archive(raw,symbol,listing_date,price_date):
    parsed=parse_yahoo_chart(raw,symbol)
    result=raw['chart']['result'][0];quotes=result['indicators']['quote'][0];times=result['timestamp']
    rows=[]
    for i,stamp in enumerate(times):
        day=pd.Timestamp(stamp,unit='s',tz='UTC').tz_convert(IST).normalize().tz_localize(None)
        if not pd.Timestamp(listing_date)<=day<=pd.Timestamp(price_date):continue
        values={k:quotes.get(k,[None]*len(times))[i] for k in ['open','high','low','close','volume']}
        rows.append(dict(date=day,**values))
    g=pd.DataFrame(rows)
    if g.empty or g.date.duplicated().any() or str(g.date.max().date())!=price_date:raise ValueError('Dated source history is incomplete')
    if (g.date.iloc[0]-pd.Timestamp(listing_date)).days not in range(8):raise ValueError('Missing initial listing history')
    if any(pd.Timestamp(s['ex_date'])>=pd.Timestamp(listing_date) for s in parsed.attrs['provider_meta']['splits']):raise ValueError('Share changes require OHLCV-specific reconciliation')
    return g


def init(c):
    c.execute('''CREATE TABLE IF NOT EXISTS module2_ml_shadow_forecasts_v1 (
        forecast_id TEXT PRIMARY KEY,ipo_id INTEGER NOT NULL,company_name TEXT NOT NULL,
        price_date TEXT NOT NULL,observed_at_utc TEXT NOT NULL,model_sha256 TEXT NOT NULL,
        positive_probability REAL NOT NULL,expected_next_close_return_pct REAL NOT NULL,
        expected_five_session_return_pct REAL NOT NULL,position INTEGER NOT NULL,action TEXT NOT NULL,
        payload_json TEXT NOT NULL,payload_sha256 TEXT NOT NULL,UNIQUE(ipo_id,price_date,model_sha256))''')
    if 'payload_sha256' not in {r[1] for r in c.execute('PRAGMA table_info(module2_ml_shadow_forecasts_v1)')}:
        c.execute('ALTER TABLE module2_ml_shadow_forecasts_v1 ADD COLUMN payload_sha256 TEXT')
    c.execute('''CREATE TABLE IF NOT EXISTS module2_ml_shadow_attempts_v1 (
        attempt_id TEXT PRIMARY KEY,ipo_id INTEGER,observed_at_utc TEXT NOT NULL,status TEXT NOT NULL,detail TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS module2_ml_shadow_market_v1 (
        source_id TEXT PRIMARY KEY,ipo_id INTEGER NOT NULL,symbol TEXT NOT NULL,price_date TEXT NOT NULL,
        observed_at_utc TEXT NOT NULL,raw_sha256 TEXT NOT NULL,raw_gzip BLOB NOT NULL)''')


def archive_source(c,id,symbol,price_date,raw,observed):
    sha=hashlib.sha256(raw).hexdigest()
    c.execute('INSERT OR IGNORE INTO module2_ml_shadow_market_v1 VALUES (?,?,?,?,?,?,?)',
              (sha,id,symbol,price_date,observed,sha,gzip.compress(raw)))


def capture_shadow(result,*,db_path=DB,now=None):
    clock=now or datetime.now(IST)
    if clock.tzinfo is None:raise ValueError('Timezone-aware observation required')
    clock=clock.astimezone(IST)
    if clock.weekday()>=5 or clock.hour<18:return 'outside_shadow_window'
    if not ARTIFACT.exists() or not DAILY.exists() or not MANIFEST.exists():return 'research_artifact_unavailable'
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    if (manifest.get('research_only') is not True or digest(ARTIFACT)!=manifest.get('weekly_artifact_sha256')
        or digest(DAILY)!=manifest.get('daily_artifact_sha256')
        or digest(Path(__file__))!=manifest.get('feature_builder_sha256')
        or digest(ROOT/'ml/research_module2_multihorizon.py')!=manifest.get('policy_code_sha256')):
        return 'blocked_artifact_or_policy_change'
    id=int(result['ipo_id']);price_date=str(result['evaluation_date'])[:10]
    if price_date!=str(clock.astimezone(IST).date()):return 'awaiting_current_close'
    model_sha=digest(ARTIFACT);key=hashlib.sha256(f'{id}|{price_date}|{model_sha}'.encode()).hexdigest()
    with sqlite3.connect(db_path,timeout=10) as c:
        init(c)
        if c.execute('SELECT 1 FROM module2_ml_shadow_forecasts_v1 WHERE forecast_id=?',(key,)).fetchone():return 'already_saved'
        previous=c.execute('SELECT payload_json,payload_sha256 FROM module2_ml_shadow_forecasts_v1 WHERE ipo_id=? AND model_sha256=? ORDER BY price_date',
                           (id,model_sha)).fetchall()
    if any(hashlib.sha256(v[0].encode()).hexdigest()!=v[1] for v in previous):return 'blocked_previous_forecast_checksum'
    try:
        if (pd.Timestamp(price_date)-pd.Timestamp(result['listing_date'])).days<20:
            raise ValueError('Not enough post-listing history: model requires 21 completed sessions')
        if (result.get('price_basis_adjustment') or {}).get('status')!='comparable':raise ValueError('Unverified share units')
        if (result.get('price_basis_adjustment') or {}).get('share_factor',1)!=1:raise ValueError('Adjusted share-unit case not yet supported')
        symbol=str(result.get('nse_symbol') or '')+'.NS';sha=str(result.get('price_archive_sha256') or '')
        archive=YAHOO_ARCHIVES/(sha+'.json.gz')
        exchange_reference=sha
        if not archive.exists() and result.get('price_source')=='NSE UDiFF CM':
            # The monitor can legitimately have only an exchange close.
            # Obtain OHLCV separately, retaining both source hashes and then
            # requiring its exact symbol, issuer, date and closing-price match.
            observed=AttributionEngine._fetch_yahoo_series(symbol,start_date=result['listing_date'])
            sha=str(observed.attrs.get('provider_meta',{}).get('raw_sha256') or '')
            archive=YAHOO_ARCHIVES/(sha+'.json.gz')
        raw=gzip.decompress(archive.read_bytes())
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Quote archive checksum mismatch')
        content=json.loads(raw)
        from services.module2_recent_exchange import _identity_matches
        meta=content['chart']['result'][0]['meta']
        if not _identity_matches(result['company_name'],meta.get('longName') or meta.get('shortName') or ''):raise ValueError('Issuer name mismatch in market source')
        g=frame_from_archive(content,symbol,result['listing_date'],price_date)
        if len(g)>56:raise ValueError('Five-session model is trained on post-listing sessions 21-56')
        if not math.isclose(float(g.close.iloc[-1]),float(result['current_price_adjusted']),rel_tol=.0001):raise ValueError('Monitor and OHLCV price basis differ')
        nifty=AttributionEngine._fetch_yahoo_series('^NSEI').rename(columns={'close':'nifty50_close'})
        sector,label=AttributionEngine._get_sector_series(result['sector'])
        if sector.empty:raise ValueError('No dated sector history')
        for name,frame in [('nifty',nifty),('sector',sector)]:
            proof=frame.attrs.get('provider_meta',{}).get('raw_sha256')
            if not proof:raise ValueError('Missing '+name+' source hash')
            original=gzip.decompress((YAHOO_ARCHIVES/(proof+'.json.gz')).read_bytes())
            if hashlib.sha256(original).hexdigest()!=proof:raise ValueError('Benchmark source checksum mismatch')
        stock=g.merge(nifty[['date','nifty50_close']],on='date',how='left').merge(sector[['date','close']].rename(columns={'close':'sector_index_close'}),on='date',how='left')
        issue=result['price_waterfall']['issue_price'];features=features_at(stock,issue)
        bundle=joblib.load(ARTIFACT);daily=joblib.load(DAILY)
        if bundle['weekly_features']!=bundle['daily_features'] or bundle['daily_features']!=daily['features']:raise ValueError('Feature protocol mismatch')
        x=pd.DataFrame([features])
        with threadpool_limits(limits=1):
            expected=float(bundle['daily_model'].predict(x[bundle['daily_features']])[0]);weekly=float(bundle['weekly_model'].predict(x[bundle['weekly_features']])[0])
            forecast=float(daily['models']['forecast'].predict(x[daily['features']])[0]);prob=float(daily['models']['direction'].predict_proba(x[daily['features']])[0,1])
        features.update(ipo_id=id,decision_date=price_date,rule_position=int('SELL' not in result['verdict']))
        history=[json.loads(v[0]) for v in previous]
        inputs=pd.DataFrame([h['features'] for h in history]+[features]);inputs['decision_date']=pd.to_datetime(inputs.decision_date)
        positions=policy_positions(inputs,[h['daily_holding_prediction'] for h in history]+[expected],
                                  [h['weekly_prediction'] for h in history]+[weekly],bundle['policy'])
        pos=int(positions.iloc[-1]);prior=int(history[-1]['position']) if history else 0
        action='BUY' if pos and not prior else 'SELL' if prior and not pos else 'HOLD' if pos else 'CASH'
        payload=dict(features=features,daily_holding_prediction=expected,weekly_prediction=weekly,position=pos,
            decision_close=float(result['current_price_adjusted']),symbol=symbol,listing_date=result['listing_date'],
            expected_next_close_return_pct=forecast,positive_probability=prob,action=action,research_only=True,
            hypothetical_state='Starts in cash on first actual logged forecast; no invented earlier trades.',
            execution_assumption='Desired position for the next observed open; no real order or guaranteed fill.',
            source=dict(stock_sha256=sha,nifty_sha256=nifty.attrs.get('provider_meta',{}).get('raw_sha256'),
                sector_sha256=sector.attrs.get('provider_meta',{}).get('raw_sha256'),sector_label=label,
                monitor_price_reference_sha256=exchange_reference),
            policy=bundle['policy'],daily_artifact_sha256=digest(DAILY),policy_code_sha256=digest(ROOT/'ml/research_module2_multihorizon.py'))
        payload['feature_builder_sha256']=manifest['feature_builder_sha256']
        completed=datetime.now(IST) if now is None else clock
        encoded=json.dumps(payload,sort_keys=True);payload_sha=hashlib.sha256(encoded.encode()).hexdigest()
        with sqlite3.connect(db_path,timeout=10) as c:
            init(c)
            archive_source(c,id,symbol,price_date,raw,completed.astimezone(timezone.utc).isoformat())
            for benchmark in (nifty,sector):
                meta=benchmark.attrs['provider_meta'];proof=meta['raw_sha256']
                original=gzip.decompress((YAHOO_ARCHIVES/(proof+'.json.gz')).read_bytes())
                archive_source(c,0,meta.get('symbol','benchmark'),price_date,original,completed.astimezone(timezone.utc).isoformat())
            c.execute('''INSERT OR IGNORE INTO module2_ml_shadow_forecasts_v1
                (forecast_id,ipo_id,company_name,price_date,observed_at_utc,model_sha256,positive_probability,
                 expected_next_close_return_pct,expected_five_session_return_pct,position,action,payload_json,payload_sha256)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (key,id,result['company_name'],price_date,completed.astimezone(timezone.utc).isoformat(),model_sha,prob,forecast,weekly,pos,action,encoded,payload_sha))
        return 'saved'
    except (ValueError,KeyError,TypeError,OSError,IndexError) as exc:
        with sqlite3.connect(db_path,timeout=10) as c:
            init(c);c.execute('INSERT OR IGNORE INTO module2_ml_shadow_attempts_v1 VALUES (?,?,?,?,?)',
                (key,id,clock.astimezone(timezone.utc).isoformat(),'ineligible',str(exc)[:300]))
        return 'ineligible: '+str(exc)[:150]


def refresh_outcome_sources(*,db_path=DB):
    """Continue collecting outcome bars after an IPO leaves forecast support.
    Saved predictions are never recomputed. Source bytes mirror to Supabase.
    """
    now=datetime.now(IST)
    if now.weekday()>=5 or now.hour<18:return {'status':'outside_window'}
    with sqlite3.connect(db_path,timeout=10) as c:
        init(c);c.row_factory=sqlite3.Row
        all_rows=[dict(r) for r in c.execute('SELECT * FROM module2_ml_shadow_forecasts_v1 ORDER BY price_date')]
        all_rows=[r for r in all_rows if hashlib.sha256(r['payload_json'].encode()).hexdigest()==r['payload_sha256']]
        latest={r['ipo_id']:r for r in all_rows}
        # Bind already saved source hashes to their original observed clocks.
        for r in all_rows:
            if hashlib.sha256(r['payload_json'].encode()).hexdigest()!=r['payload_sha256']:continue
            p=json.loads(r['payload_json']);proof=p['source']['stock_sha256'];path=YAHOO_ARCHIVES/(proof+'.json.gz')
            if path.exists():
                raw=gzip.decompress(path.read_bytes())
                if hashlib.sha256(raw).hexdigest()==proof:archive_source(c,r['ipo_id'],p['symbol'],r['price_date'],raw,r['observed_at_utc'])
        existing={r['ipo_id']:dict(r) for r in c.execute('SELECT * FROM module2_ml_shadow_market_v1 WHERE ipo_id>0 ORDER BY price_date,observed_at_utc')}
    saved=0;failures=[]
    for id,r in latest.items():
        p=json.loads(r['payload_json']);prior=existing.get(id)
        if prior:
            if prior['price_date']==now.date().isoformat():continue
            data=json.loads(gzip.decompress(prior['raw_gzip']))
            series=parse_yahoo_chart(data,p['symbol'])
            if series.date.gt(pd.Timestamp(r['price_date'])).sum()>=6:continue
        try:
            series=AttributionEngine._fetch_yahoo_series(p['symbol'],start_date=p['listing_date'])
            proof=series.attrs['provider_meta']['raw_sha256'];raw=gzip.decompress((YAHOO_ARCHIVES/(proof+'.json.gz')).read_bytes())
            if hashlib.sha256(raw).hexdigest()!=proof:raise ValueError('Outcome source checksum mismatch')
            series=series[series.date.dt.date<=now.date()]
            if series.empty:continue
            with sqlite3.connect(db_path,timeout=10) as c:
                init(c);archive_source(c,id,p['symbol'],str(series.date.max().date()),raw,datetime.now(timezone.utc).isoformat())
            saved+=1
        except (KeyError,ValueError,TypeError,OSError) as exc:failures.append(dict(ipo_id=id,reason=str(exc)[:180]))
    return dict(status='completed',saved_sources=saved,failures=failures)


def scorecard(*,db_path=DB):
    """Read-only grading from subsequently archived bars, not a calendar guess."""
    if not sqlite3.exists(db_path):return dict(status='awaiting_capture',forecasts=0,issuers=0,records=[])
    with sqlite3.connect(db_path,timeout=10) as c:
        c.row_factory=sqlite3.Row
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name='module2_ml_shadow_forecasts_v1'").fetchone():return dict(status='awaiting_capture',forecasts=0,issuers=0,records=[])
        model=json.loads(MANIFEST.read_text(encoding='utf8'))['weekly_artifact_sha256']
        rows=[dict(r) for r in c.execute('SELECT * FROM module2_ml_shadow_forecasts_v1 WHERE model_sha256=? ORDER BY price_date,ipo_id',(model,))]
        attempts=[dict(r) for r in c.execute('SELECT ipo_id,status,detail,observed_at_utc FROM module2_ml_shadow_attempts_v1 ORDER BY observed_at_utc DESC LIMIT 100')]
        sources={}
        if c.execute("SELECT 1 FROM sqlite_master WHERE name='module2_ml_shadow_market_v1'").fetchone():
            sources={r['ipo_id']:dict(r) for r in c.execute('SELECT * FROM module2_ml_shadow_market_v1 WHERE ipo_id>0 ORDER BY price_date,observed_at_utc')}
    from ml.improve_module2_next_session import realised,summary
    forecasts=[];paper=[];weekly_errors=[];groups={};invalid=0
    for r in rows:
        if hashlib.sha256(r['payload_json'].encode()).hexdigest()!=r.get('payload_sha256'):
            invalid+=1;continue
        r['payload']=json.loads(r['payload_json']);groups.setdefault(r['ipo_id'],[]).append(r)
    for id,entries in groups.items():
        last=entries[-1];payload=last['payload'];bars=[]
        try:
            source=sources.get(id)
            sha=source['raw_sha256'] if source else payload['source']['stock_sha256']
            raw=gzip.decompress(source['raw_gzip']) if source else gzip.decompress((YAHOO_ARCHIVES/(sha+'.json.gz')).read_bytes())
            if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Outcome source checksum mismatch')
            chart=json.loads(raw);parsed=parse_yahoo_chart(chart,payload['symbol'])
            if any(pd.Timestamp(e['ex_date'])>pd.Timestamp(entries[0]['price_date']) for e in parsed.attrs['provider_meta']['splits']):raise ValueError('Outcome share units changed')
            observed=chart['chart']['result'][0];quote=observed['indicators']['quote'][0]
            for i,ts in enumerate(observed['timestamp']):
                stamp=pd.Timestamp(ts,unit='s',tz='UTC');day=stamp.tz_convert(IST).date().isoformat()
                if day>(source['price_date'] if source else last['price_date']):continue
                o=quote.get('open',[])[i];close=quote.get('close',[])[i]
                if o is None or close is None or min(o,close)<=0:continue
                bars.append(dict(date=day,open=float(o),close=float(close),open_at=stamp))
        except (KeyError,ValueError,TypeError,OSError,IndexError):bars=[]
        valid_basis=set()
        for r in entries:
            reference=next((b['close'] for b in bars if b['date']==r['price_date']),None)
            future=[b for b in bars if b['date']>r['price_date'] and b['open_at']>pd.Timestamp(r['observed_at_utc'])]
            basis_ok=reference is not None and math.isclose(reference,r['payload']['decision_close'],rel_tol=.0001)
            if basis_ok:valid_basis.add(r['forecast_id'])
            evaluated=bool(future) and basis_ok;actual=None;correct=None
            if evaluated:
                actual=100*(future[0]['close']/r['payload']['decision_close']-1)
                correct=(r['positive_probability']>=.5)==(actual>0)
            if len(future)>=6 and basis_ok:
                realised_weekly=100*(future[5]['open']/future[0]['open']-1)
                weekly_errors.append(abs(realised_weekly-r['expected_five_session_return_pct']))
            forecasts.append(dict(ipo_id=id,company_name=r['company_name'],price_date=r['price_date'],observed_at_utc=r['observed_at_utc'],
                positive_probability=r['positive_probability'],expected_next_close_return_pct=r['expected_next_close_return_pct'],
                expected_five_session_return_pct=r['expected_five_session_return_pct'],action=r['action'],
                outcome_status='evaluated' if evaluated else 'pending_next_session',actual_next_close_return_pct=actual,direction_correct=correct))
        for b in bars:
            known=[r for r in entries if r['forecast_id'] in valid_basis and r['price_date']<b['date'] and pd.Timestamp(r['observed_at_utc'])<b['open_at']]
            if not known:continue
            prior=known[-1]
            paper.append(dict(ipo_id=id,company_name=prior['company_name'],listing_year=int(prior['payload']['listing_date'][:4]),
                outcome_date=pd.Timestamp(b['date']),next_open=b['open'],next_close=b['close'],position=prior['position'],
                rule_position=prior['payload']['features']['rule_position']))
    resolved=[r for r in forecasts if r['direction_correct'] is not None]
    papers={}
    if paper:
        f=pd.DataFrame(paper)
        papers={name:summary(realised(f,p)) for name,p in [('shadow_ml',f.position),('current_rules',f.rule_position),('always_hold',np.ones(len(f)))]}
    return dict(status='forward_research',forecasts=len(rows),issuers=len(groups),evaluated_next_session=len(resolved),
        direction_accuracy_pct=100*sum(r['direction_correct'] for r in resolved)/len(resolved) if resolved else None,
        always_positive_accuracy_pct=100*sum(r['actual_next_close_return_pct']>0 for r in resolved)/len(resolved) if resolved else None,
        weekly_evaluated=len(weekly_errors),weekly_mae_pct=float(np.mean(weekly_errors)) if weekly_errors else None,
        paper_strategies=papers,unverified_payloads=invalid,records=sorted(forecasts,key=lambda r:r['observed_at_utc'],reverse=True),recent_ineligible=attempts,
        note='Prospectively saved research forecasts. Daily rows are correlated, not independent IPOs. Paper fills assume next observed open and 17.5bps each side; circuit/queue execution is not guaranteed. Existing live verdicts are unchanged.')
