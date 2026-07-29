"""Append-only frozen-model forecasts and independent listing outcomes."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
from db import ledger as sqlite3
import threading
from contextlib import closing
from datetime import datetime,time,timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'data/telemetry/snapshots.db'
PROTOCOL=ROOT/'output/prospective_module1_protocol.json'
IST=ZoneInfo('Asia/Kolkata')
logger=logging.getLogger(__name__)
status={'state':'not_started','last_cycle_at':None,'error':None}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(value):
    try:
        parsed=datetime.fromisoformat(str(value))
        return parsed.astimezone(IST) if parsed.tzinfo else None
    except (ValueError,TypeError):
        return None


def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError):
        return None


def calendar(item,clock):
    from scrapers.live_ipo_scraper import parse_date_interval
    opened,closed=parse_date_interval(item.get('date_range') or '',clock.year)
    if not opened or not closed or opened>closed:
        return None,None
    return datetime.combine(opened,time(10),IST),datetime.combine(closed,time(17),IST)


def resolve_calendar(item,url):
    """Fill missing offer dates only from an issuer/cap/listing-matched page."""
    if calendar(item,datetime.now(IST))[1] or not number(item.get('issue_price')):
        return item
    from services.snapshot_service import extract_ipo_id
    directory=ROOT/'data/telemetry/prospective_calendars';directory.mkdir(parents=True,exist_ok=True)
    name=extract_ipo_id({**item,'url':url})+'_'+datetime.now(IST).strftime('%Y%m%d')
    path=directory/(name+'.json')
    if path.exists():
        evidence=json.loads(path.read_text())
    else:
        from ml.collect_module1_later_holdout import index_urls
        from ml.recover_module1_2025_training_gaps import source_rank,product_dates
        from ml.recover_module1_other_sources import fetch
        urls=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
        evidence={}
        for candidate in sorted(urls,key=lambda u:-source_rank(item,u))[:2]:
            response=fetch(candidate);dates=product_dates(response.text,item)
            if dates:
                evidence={**dates,'url':candidate,'raw_sha256':hashlib.sha256(response.content).hexdigest(),
                    'captured_at_ist':datetime.now(IST).isoformat()}
                (directory/(name+'.html.gz')).write_bytes(gzip.compress(response.content))
                break
        path.write_text(json.dumps(evidence),encoding='utf8')
    if evidence.get('issue_close_date'):
        resolved={**item,'date_range':evidence['issue_open_date']+' to '+evidence['issue_close_date'],'calendar_evidence':evidence}
        overlay_path=directory.parent/'prospective_calendar_overlay.json'
        overlay=json.loads(overlay_path.read_text()) if overlay_path.exists() else {}
        overlay[url]={'date_range':resolved['date_range'],'issue_price':item['issue_price'],
            'listing_date':item.get('listing_date'),'evidence':evidence,
            'raw_archive':str(directory/(name+'.html.gz'))}
        temporary=overlay_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(overlay),encoding='utf8');temporary.replace(overlay_path)
        return resolved
    return item


def init_db(db_path=DB):
    Path(db_path).parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        c.execute('PRAGMA busy_timeout=10000')
        c.executescript('''CREATE TABLE IF NOT EXISTS frozen_module1_forecast_v1 (
          id INTEGER PRIMARY KEY,ipo_id TEXT NOT NULL,company_name TEXT NOT NULL,
          ipo_url TEXT NOT NULL,protocol_sha TEXT NOT NULL,model_sha TEXT NOT NULL,
          recorded_at_ist TEXT NOT NULL,cutoff_at_ist TEXT NOT NULL,slot TEXT NOT NULL,
          kind TEXT NOT NULL,quality TEXT NOT NULL,probability REAL NOT NULL,
          predicted_positive INTEGER NOT NULL,gmp_probability REAL NOT NULL,gmp_positive INTEGER NOT NULL,
          issue_price REAL,listing_date TEXT,input_json TEXT NOT NULL,input_sha TEXT NOT NULL,
          UNIQUE(ipo_id,protocol_sha,slot,kind,input_sha));
          CREATE TABLE IF NOT EXISTS frozen_module1_attempt_v1 (
          id INTEGER PRIMARY KEY,ipo_id TEXT,company_name TEXT,recorded_at_ist TEXT NOT NULL,status TEXT NOT NULL,detail TEXT);
          CREATE TABLE IF NOT EXISTS frozen_module1_outcome_v1 (
          ipo_id TEXT PRIMARY KEY,recorded_at_ist TEXT NOT NULL,outcome_json TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS frozen_module1_eligible_v1 (
          ipo_id TEXT NOT NULL,company_name TEXT NOT NULL,protocol_sha TEXT NOT NULL,
          first_seen_at_ist TEXT NOT NULL,cutoff_at_ist TEXT NOT NULL,
          PRIMARY KEY(ipo_id,protocol_sha));
          CREATE INDEX IF NOT EXISTS frozen_module1_name ON frozen_module1_forecast_v1(company_name,recorded_at_ist);
          CREATE TRIGGER IF NOT EXISTS frozen_forecast_no_update BEFORE UPDATE ON frozen_module1_forecast_v1 BEGIN SELECT RAISE(ABORT,'Forecasts are immutable'); END;
          CREATE TRIGGER IF NOT EXISTS frozen_forecast_no_delete BEFORE DELETE ON frozen_module1_forecast_v1 BEGIN SELECT RAISE(ABORT,'Forecasts are immutable'); END;
          CREATE TRIGGER IF NOT EXISTS frozen_outcome_no_update BEFORE UPDATE ON frozen_module1_outcome_v1 BEGIN SELECT RAISE(ABORT,'Outcomes are immutable'); END;
          CREATE TRIGGER IF NOT EXISTS frozen_outcome_no_delete BEFORE DELETE ON frozen_module1_outcome_v1 BEGIN SELECT RAISE(ABORT,'Outcomes are immutable'); END;''')
        c.commit()

    from services.module1_milestones import init_table
    init_table(db_path)


def validate_direction_policy(protocol,model):
    """This release uses one fixed rule; retraining cannot silently change it."""
    if protocol.get('model',{}).get('cutoff')!=.5 or model.get('threshold')!=.5:
        raise ValueError('Frozen Module 1 direction cutoff must remain 0.50; a new policy requires an explicit versioned release')


@lru_cache(maxsize=1)
def bundles():
    protocol=json.loads(PROTOCOL.read_text())
    protocol['_sha256']=digest(PROTOCOL)
    for key in ('model','gmp_baseline'):
        if digest(ROOT/Path(protocol[key]['path'].replace('\\','/')))!=protocol[key]['sha256']:
            raise ValueError('Frozen '+key+' fingerprint changed')
    model=joblib.load(ROOT/Path(protocol['model']['path'].replace('\\','/')))
    validate_direction_policy(protocol,model)
    baseline=joblib.load(ROOT/Path(protocol['gmp_baseline']['path'].replace('\\','/')))
    return protocol,model,baseline


def predict(features):
    protocol,model,baseline=bundles()
    probabilities=[]
    for name in model['experts']:
        config=model['expert_configs'][name]
        from ml.research_module1_forward_update import SETS
        columns=SETS[config['feature_set']] if config['feature_set']!='gmp_only' else ['gmp_pct']
        x=pd.DataFrame([{key:features.get(key,np.nan) for key in columns}])
        if config['transform']=='signed_log':
            x['gmp_pct']=np.sign(x.gmp_pct)*np.log1p(np.abs(x.gmp_pct))
        probabilities.append(model['models'][name].predict_proba(x)[0,1])
    p=np.clip(np.asarray(probabilities),1e-6,1-1e-6)
    probability=float(model['meta_model'].predict_proba(np.log(p/(1-p)).reshape(1,-1))[0,1])
    gmp=float(baseline['gmp_control'].predict_proba(pd.DataFrame([{'gmp_pct':features.get('gmp_pct',np.nan)}]))[0,1])
    return probability,gmp


_macro_lock=threading.Lock()
def market_frame():
    """Daily cache; never use a close from the forecast's own or later date."""
    path=ROOT/'data/telemetry/prospective_macro.csv'
    with _macro_lock:
        if not path.exists() or datetime.fromtimestamp(path.stat().st_mtime,IST).date()<datetime.now(IST).date():
            import yfinance as yf
            raw=yf.download(['^NSEI','^INDIAVIX'],period='6mo',auto_adjust=False,actions=False,progress=False,threads=False,timeout=15)
            if not raw.empty:
                frame=pd.DataFrame({'nifty':raw[('Close','^NSEI')],'vix':raw[('Close','^INDIAVIX')]})
                frame.index=pd.to_datetime(frame.index).tz_localize(None)
                frame.to_csv(path,index_label='date')
        return pd.read_csv(path,parse_dates=['date']).set_index('date') if path.exists() else pd.DataFrame()


def build_features(item,url,asof,db_path=DB):
    """Actual observations only; absent fields stay NaN for frozen imputation."""
    from services.snapshot_service import extract_ipo_id
    features={};sources={};ipo_id=extract_ipo_id({**item,'url':url})
    opened,cutoff=calendar(item,asof)
    # Read the capture ledger directly: never trust cached final-book values.
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        c.row_factory=sqlite3.Row
        try:
            captures=[dict(r) for r in c.execute('''SELECT * FROM subscription_capture_v2
              WHERE (ipo_id=? OR LOWER(company_name)=LOWER(?)) AND capture_phase='pre_deadline'
              AND status='verified' ORDER BY id''',(ipo_id,item['company_name']))]
        except sqlite3.OperationalError:
            captures=[]
    eligible=[]
    from services.module1_input_contract import VERSION as INPUT_CONTRACT,valid_categories,growth_reference,financial_cache_eligible,stage_contract
    sources['input_contract_version']=INPUT_CONTRACT
    for capture in captures:
        fetched=stamp(capture.get('fetched_at_ist') or capture.get('attempted_at_ist'))
        if fetched and fetched<=asof and (not opened or fetched>=opened) and (not cutoff or fetched<cutoff):
            try:
                raw=gzip.decompress(capture['raw_html_gzip'])
                if hashlib.sha256(raw).hexdigest()!=capture['raw_sha256']:
                    continue
                published=None
                if capture.get('source_family')=='nse_exchange':
                    try:
                        published=datetime.strptime(json.loads(raw).get('updateTime',''),'Updated as on %d-%b-%Y %H:%M:%S').replace(tzinfo=IST)
                    except (ValueError,TypeError):
                        pass
                else:
                    from scrapers.live_ipo_scraper import parse_subscription_published_at
                    published=stamp(parse_subscription_published_at(raw.decode('utf8',errors='replace')))
                if published and (published>fetched or published>asof):
                    continue
            except (TypeError,KeyError,OSError):
                continue
            if valid_categories(capture):
                capture['_publisher_at_ist']=published.isoformat() if published else None
                eligible.append((published or fetched,capture))
    eligible.sort(key=lambda v:v[0])
    if eligible:
        book_time,book=eligible[-1];sources['bids']={k:book.get(k) for k in ('id','fetched_at_ist','raw_sha256','source_family')}
        sources['bids']['published_at_ist']=book.get('_publisher_at_ist')
        sources['bids']['time_basis']='publisher_timestamp' if book.get('_publisher_at_ist') else 'capture_time_publisher_unverified'
        sources['stage_alignment']=stage_contract(book_time,asof,cutoff)
        from services.live_offer_facts import total_book
        try:
            total=total_book(gzip.decompress(book['raw_html_gzip']),book.get('source_family'))
            if total:
                features['context_total_mult']=total['multiple']
                sources['total_subscription']={**total,**sources['bids']}
        except (ValueError,TypeError,OSError) as exc:
            sources['total_subscription_error']=str(exc)[:180]
        reference=growth_reference(eligible,(book_time,book))
        if book.get('source_family')=='nse_exchange':
            from services.nse_subscription_service import _category_values
            try:split=_category_values(json.loads(gzip.decompress(book['raw_html_gzip'])).get('dataList',[]))
            except (ValueError,AttributeError,OSError):split={}
            for category in ('snii','bnii'):
                if split.get(category+'_subscription') is not None:features['context_'+category+'_mult']=split[category+'_subscription']
        for cat,key in [('qib','qib_x'),('hni','nii_x'),('retail','retail_x')]:
            features['log_'+cat]=float(np.log1p(float(book[key])))
            # Use the previous completed bidding day's latest same-source book.
            # Earliest-ever capture is not a Day2 trajectory reference.
            if reference:
                reference_time,previous=reference
                features['log_'+cat+'_growth']=float(np.log1p(book[key])-np.log1p(previous[key]))
                sources['growth_reference']={k:previous.get(k) for k in ('id','fetched_at_ist','raw_sha256','source_family')}
                sources['growth_reference']['effective_at_ist']=reference_time.isoformat()
                sources['growth_reference']['published_at_ist']=previous.get('_publisher_at_ist')
    sources.setdefault('stage_alignment',stage_contract(None,asof,cutoff))
    for field,key in [('log_offer_cap_price','issue_price'),('log_issue_size','issue_size_cr')]:
        value=number(item.get(key))
        if value is not None and value>0:
            features[field]=float(np.log1p(value))
    quote_time=stamp(item.get('gmp_observed_at'))
    quote=number(item.get('gmp_pct'))
    if quote is not None and quote_time and quote_time<=asof:
        features['gmp_pct']=quote
        features['gmp_quote_age_days']=(asof.date()-quote_time.date()).days
        sources['gmp']={'fetched_at':quote_time.isoformat(),'url':item.get('gmp_source_url'),
            'source':item.get('gmp_source'),'family':item.get('gmp_source_family'),'publisher_time_verified':False}
        with closing(sqlite3.connect(db_path,timeout=10)) as c:
            previous=c.execute('SELECT input_json FROM frozen_module1_forecast_v1 WHERE ipo_id=? ORDER BY id DESC LIMIT 500',(ipo_id,)).fetchall()
        for (encoded,) in previous:
            past=json.loads(encoded);observation=past.get('evidence',{}).get('sources',{}).get('gmp',{})
            observed=stamp(observation.get('fetched_at'));value=number(past.get('features',{}).get('gmp_pct'))
            same_source=observation.get('source')==item.get('gmp_source') and bool(item.get('gmp_source'))
            if (observed and value is not None and same_source and observed.date()<quote_time.date()
                    and 0<(quote_time.date()-observed.date()).days<=7):
                features['gmp_slope_pp_per_day']=(quote-value)/(quote_time.date()-observed.date()).days
                sources['gmp']['previous_quote_at']=observed.isoformat()
                break
    try:
        market=market_frame()
        prior=market[market.index<pd.Timestamp(asof.date())]
        if len(prior)>=21 and (asof.date()-prior.index[-1].date()).days<=5:
            n=prior.nifty.dropna();v=prior.vix.dropna()
            features.update(macro_nifty_5d=float((n.iloc[-1]/n.iloc[-6]-1)*100),
                macro_nifty_20d=float((n.iloc[-1]/n.iloc[-21]-1)*100),macro_nifty_vol20=float((n.pct_change()*100).iloc[-20:].std()))
            if len(v)>=6 and (asof.date()-v.index[-1].date()).days<=5:
                features.update(macro_vix=float(v.iloc[-1]),macro_vix_change5=float((v.iloc[-1]/v.iloc[-6]-1)*100))
            sources['market']={'date':str(prior.index[-1].date()),'source':'Yahoo Finance historical close'}
    except Exception as exc:
        sources['market_error']=str(exc)[:180]
    # IPO-context financial panel, captured now. Never claim it existed at a past cutoff.
    if asof.date()>=datetime.now(IST).date():
        try:
            from ml.collect_module1_public_financials import collect,FINANCIAL
            from ml.collect_module1_later_holdout import index_urls
            from ml.recover_module1_other_sources import norm
            directory=ROOT/'data/telemetry/prospective_financials';directory.mkdir(exist_ok=True)
            cache=directory/(ipo_id+'_'+asof.strftime('%Y%m%d')+'.json')
            if (cache.exists() and cache.with_suffix('.html.gz').exists()
                    and json.loads(cache.read_text()).get('raw_sha256')):
                fact=json.loads(cache.read_text())
            elif opened and cutoff and number(item.get('issue_price')):
                # Reuse the verified public HTML parser, with a separate live cache.
                import ml.collect_module1_public_financials as parser
                from bs4 import BeautifulSoup
                from ml.recover_module1_other_sources import fetch,ipoji_payload
                urls=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
                tokens=set(norm(item['company_name']).split())
                from ml.recover_module1_2025_training_gaps import source_rank
                identity=dict(company_name=item['company_name'],nse_symbol=item.get('nse_symbol',''))
                fact={}
                for candidate in sorted(urls,key=lambda u:-source_rank(identity,u))[:2]:
                    response=fetch(candidate);soup=BeautifulSoup(response.text,'html.parser')
                    check=dict(company_name=item['company_name'],issue_open_date=opened.date(),issue_close_date=cutoff.date(),issue_price=item['issue_price'])
                    if ipoji_payload(check,soup):
                        fact=parser.extract(response.text,asof.date(),item['issue_price'])
                        fact.update(source_url=candidate,raw_sha256=hashlib.sha256(response.content).hexdigest(),captured_at_ist=datetime.now(IST).isoformat())
                        (directory/(ipo_id+'_'+asof.strftime('%Y%m%d')+'.html.gz')).write_bytes(gzip.compress(response.content))
                        break
                cache.write_text(json.dumps(fact),encoding='utf8')
            else:
                fact={}
            archive=directory/(ipo_id+'_'+asof.strftime('%Y%m%d')+'.html.gz')
            if fact and not financial_cache_eligible(fact,archive,asof):
                sources['financial_rejected']='Source hash, captured-at time, or annual period failed the point-in-time check'
                fact={}
            if fact and any(number(fact.get(k)) is None for k in ('ipo_earnings_yield_pct','log_ipo_price_book')):
                from services.live_valuation_recovery import recover
                supplement=recover(item,fact,opened.date() if opened else None,cutoff.date() if cutoff else None,asof)
                if supplement:
                    fact={**fact,**{k:supplement[k] for k in ('ipo_earnings_yield_pct','log_ipo_price_book') if number(fact.get(k)) is None}}
                    sources['financial_valuation']={k:supplement.get(k) for k in ('source_url','raw_sha256','captured_at_ist','annual_period','valuation_basis')}
            for field in ('profit_margin_pct','return_on_assets_pct','revenue_growth_asinh','annual_loss','ipo_earnings_yield_pct','log_ipo_price_book'):
                if number(fact.get(field)) is not None:
                    features['pub_'+field]=float(fact[field])
            if item.get('sector') and item['sector'] not in ('Unclassified','Unknown'):
                features['pub_financial_sector']=float(item['sector']=='Financial Services')
            sources['financial']={k:fact.get(k) for k in ('source_url','raw_sha256','captured_at_ist','annual_period')}
        except Exception as exc:
            sources['financial_error']=str(exc)[:180]
    from services.locked_model_configuration import apply_live_recovered,VERSION
    from services.locked_model_configuration import cache_live_offer
    try:
        offer=cache_live_offer(item,ipo_id,asof)
        if offer:sources['offer']=offer
    except (OSError,ValueError,TypeError) as exc:sources['offer_extraction_error']=str(exc)[:180]
    apply_live_recovered(features,sources,item,asof)
    sources['active_configuration']=VERSION
    all_features=set()
    from ml.research_module1_forward_update import SETS
    for config in bundles()[1]['expert_configs'].values():
        all_features.update(['gmp_pct'] if config['feature_set']=='gmp_only' else SETS[config['feature_set']])
    missing=sorted(all_features-set(features))
    return features,{'sources':sources,'missing_features':missing,'imputation':'frozen training medians and indicators',
        'quality':'provisional_missing_inputs' if missing else 'complete','asof_ist':asof.isoformat()}


def record(item,url,*,db_path=DB,clock=None,features=None,evidence=None):
    """clock/features injection is for isolated tests; production uses real completion time."""
    from services.snapshot_service import extract_ipo_id
    init_db(db_path);started=clock or datetime.now(IST);opened,cutoff=calendar(item,started)
    if not cutoff or started>=cutoff or item.get('category')=='LISTED' or item.get('source_observed') is not True:
        return None
    if item.get('listing_date'):
        listed=pd.to_datetime(item['listing_date'],errors='coerce')
        if pd.notna(listed) and (listed.date()<=started.date() or listed.date()<=cutoff.date()):
            return None
    protocol,model,_=bundles()
    if features is None:
        features,evidence=build_features(item,url,started,db_path)
    probability,gmp=predict(features)
    completed=clock or datetime.now(IST)
    if sqlite3.cloud_enabled():
        from services.cloud_assets import persist_file
        identity=extract_ipo_id({**item,'url':url})
        directory=ROOT/'data/telemetry/prospective_financials'
        for extension in ('.json','.html.gz'):
            source=directory/(identity+'_'+completed.strftime('%Y%m%d')+extension)
            if source.exists():persist_file(source)
        if clock is None:completed=datetime.now(IST)
    if completed>=cutoff:
        return None
    kind='pre_bidding_estimate' if completed<opened else 'pre_close_forecast'
    period=5 if completed.hour>=14 else 15
    slot=completed.replace(minute=completed.minute//period*period,second=0,microsecond=0).isoformat()
    from services.locked_model_configuration import layer2,VERSION
    data={'features':features,'evidence':evidence or {},'cutoff_at_ist':cutoff.isoformat(),'stage':kind,
          'configuration':VERSION,'capital_plan':layer2(item['company_name'],features,probability,asof=completed),
          'gmp_missing':number(features.get('gmp_pct')) is None,'calendar_evidence':item.get('calendar_evidence')}
    encoded=json.dumps(data,sort_keys=True,allow_nan=False)
    if digest(PROTOCOL)!=protocol['_sha256']:
        raise ValueError('Prospective protocol changed; restart after explicit version review')
    values=(extract_ipo_id({**item,'url':url}),item['company_name'],url,protocol['_sha256'],protocol['model']['sha256'],
        completed.isoformat(),cutoff.isoformat(),slot,kind,(evidence or {}).get('quality','provisional_missing_inputs'),
        probability,int(probability>=.5),gmp,int(gmp>=protocol['gmp_baseline']['cutoff']),
        number(item.get('issue_price')),item.get('listing_date'),encoded,hashlib.sha256(encoded.encode()).hexdigest())
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        c.execute('INSERT OR IGNORE INTO frozen_module1_forecast_v1 (ipo_id,company_name,ipo_url,protocol_sha,model_sha,recorded_at_ist,cutoff_at_ist,slot,kind,quality,probability,predicted_positive,gmp_probability,gmp_positive,issue_price,listing_date,input_json,input_sha) VALUES ('+','.join('?' for _ in values)+')',values)
        c.commit()
    return {'probability':probability,'recorded_at_ist':completed.isoformat(),'kind':kind}


def scorecard(company=None,db_path=DB):
    init_db(db_path)
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        c.row_factory=sqlite3.Row
        forecasts=[dict(r) for r in c.execute('SELECT * FROM frozen_module1_forecast_v1 ORDER BY recorded_at_ist DESC,id DESC')]
        outcomes={r['ipo_id']:json.loads(r['outcome_json']) for r in c.execute('SELECT * FROM frozen_module1_outcome_v1')}
    latest={}
    for row in forecasts:
        if row['protocol_sha']!=digest(PROTOCOL):
            continue
        if company and row['company_name'].casefold()!=company.casefold():
            continue
        if row['ipo_id'] in latest:
            continue
        outcome=outcomes.get(row['ipo_id'])
        row['inputs']=json.loads(row.pop('input_json'))
        row['outcome']=outcome
        row['direction_correct']=None if not outcome else bool(row['predicted_positive']==outcome['positive'])
        row['gmp_correct']=None if not outcome else bool(row['gmp_positive']==outcome['positive'])
        from services.two_track_evidence import present
        row.update(present(row['probability'],outcome['positive'] if outcome else None,(row['inputs'].get('capital_plan') or {}).get('tier')))
        latest[row['ipo_id']]=row
    rows=list(latest.values());resolved=[r for r in rows if r['outcome']]
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        seen={r[0] for r in c.execute('SELECT company_name FROM frozen_module1_eligible_v1 WHERE protocol_sha=?',(digest(PROTOCOL),))}
        seen.update(r[0] for r in c.execute('SELECT DISTINCT company_name FROM frozen_module1_attempt_v1'))
    brier=sum((r['probability']-int(r['outcome']['positive']))**2 for r in resolved)/len(resolved) if resolved else None
    return {'records':rows,'summary':{'ipos_forecast':len(rows),'openings_verified':len(resolved),
        'ipos_seen':len(seen),'without_saved_forecast':len(seen-{r['company_name'] for r in rows}),
        'brier_score':brier,
        'direction_accuracy_pct':100*sum(r['direction_correct'] for r in resolved)/len(resolved) if resolved else None,
        'gmp_accuracy_pct':100*sum(r['gmp_correct'] for r in resolved)/len(resolved) if resolved else None},'collector':status}


def reconcile(db_path=DB):
    from services.listing_outcome_service import record_verified_listing_outcomes,verified_listing_outcomes
    record_verified_listing_outcomes(db_path=Path(db_path),include_all_listed=True)
    official=verified_listing_outcomes(Path(db_path))
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        for ipo_id,entry in official.items():
            if not c.execute('SELECT 1 FROM frozen_module1_forecast_v1 WHERE ipo_id=?',(ipo_id,)).fetchone():
                continue
            price=float(entry['issue_price']);opening=float(entry.get('exchange_open_price') or entry['nse_open_price'])
            original=c.execute('SELECT issue_price FROM frozen_module1_forecast_v1 WHERE ipo_id=? ORDER BY id LIMIT 1',(ipo_id,)).fetchone()
            if original and original[0] is not None and abs(float(original[0])-price)>.01:
                logger.error('Issue-price conflict for %s; outcome not scored',ipo_id)
                continue
            result={**entry,'actual_open':opening,'issue_price':price,'return_pct':100*(opening/price-1),'positive':opening>price}
            c.execute('INSERT OR IGNORE INTO frozen_module1_outcome_v1 VALUES (?,?,?)',(ipo_id,datetime.now(IST).isoformat(),json.dumps(result)))
        c.commit()


def _cycle():
    from scrapers.live_ipo_scraper import load_scraped_cache
    init_db();bundles();saved=0
    from services.snapshot_service import extract_ipo_id
    for url,item in load_scraped_cache().items():
        if item.get('source_observed') is not True:
            continue
        if item.get('category') not in ('OPEN','UPCOMING'):
            _,known_cutoff=calendar(item,datetime.now(IST))
            if not known_cutoff or known_cutoff.date().isoformat()<bundles()[0]['created_on_ist_date']:
                continue
        try:
            item=resolve_calendar(item,url)
            _,cutoff=calendar(item,datetime.now(IST))
            if cutoff and cutoff.date().isoformat()>=bundles()[0]['created_on_ist_date']:
                from services.module1_milestones import capture as capture_milestone
                capture_milestone(item,url,db=DB)
            if cutoff and cutoff.date().isoformat()>=bundles()[0]['created_on_ist_date']:
                with closing(sqlite3.connect(DB,timeout=10)) as c:
                    c.execute('INSERT OR IGNORE INTO frozen_module1_eligible_v1 VALUES (?,?,?,?,?)',(
                        extract_ipo_id({**item,'url':url}),item['company_name'],digest(PROTOCOL),datetime.now(IST).isoformat(),cutoff.isoformat()))
                    c.commit()
            if item.get('category') not in ('OPEN','UPCOMING'):
                continue
            from services.gmp_service import gmp_service
            item={**item,**gmp_service.resolve_gmp(item['company_name'],chittorgarh_url=url,
                issue_price=item.get('issue_price'),is_listed=False)}
            result=record(item,url)
            saved+=bool(result)
            with closing(sqlite3.connect(DB,timeout=10)) as c:
                c.execute('INSERT INTO frozen_module1_attempt_v1 (company_name,recorded_at_ist,status,detail) VALUES (?,?,?,?)',
                    (item['company_name'],datetime.now(IST).isoformat(),'saved' if result else 'calendar_missing_or_cutoff_passed',json.dumps(result)))
                c.commit()
        except Exception as exc:
            logger.exception('Frozen forecast failed for %s',item['company_name'])
            with closing(sqlite3.connect(DB,timeout=10)) as c:
                c.execute('INSERT INTO frozen_module1_attempt_v1 (company_name,recorded_at_ist,status,detail) VALUES (?,?,?,?)',
                    (item['company_name'],datetime.now(IST).isoformat(),'error',str(exc)[:500]));c.commit()
    try:
        reconcile()
        status['outcome_error']=None
    except Exception as exc:
        status['outcome_error']=str(exc)
        logger.exception('Opening verification will retry; saved forecasts are intact')
    status.update(state='running',last_cycle_at=datetime.now(IST).isoformat(),saved_last_cycle=saved,error=None)
    try:
        from services.sqlite_supabase_sync import request_sync
        request_sync('Module 1 forecast and outcome capture')
    except Exception:
        logger.exception('Could not queue Supabase mirror after Module 1 capture')


_cycle_lock=threading.Lock()


def cycle():
    # The subscription harvester and timer can become due simultaneously.
    if not _cycle_lock.acquire(blocking=False):return
    try:return _cycle()
    finally:_cycle_lock.release()


class Collector(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True,name='FrozenModule1Collector');self.halt=threading.Event()
    def run(self):
        previous=None;retry_at=None
        while not self.halt.is_set():
            now=datetime.now(IST);minutes=5 if now.hour>=14 and now.hour<17 else 15
            slot=now.strftime('%Y-%m-%d_%H')+str(now.minute//minutes)
            if previous!=slot and (retry_at is None or now>=retry_at):
                try:
                    cycle();previous=slot;retry_at=None
                except Exception as exc:
                    retry_at=now+timedelta(minutes=2)
                    status.update(state='error',error=str(exc),last_cycle_at=datetime.now(IST).isoformat())
                    logger.exception('Frozen-model collector cycle failed')
            self.halt.wait(15)
    def stop(self):
        self.halt.set()


collector=Collector()


def analysis_card(item,url,db_path=DB):
    """Saved evidence preferred; retrospective estimates never enter the ledger."""
    from services.snapshot_service import extract_ipo_id
    if not url:
        from scrapers.live_ipo_scraper import load_scraped_cache
        matches=[key for key,value in load_scraped_cache().items() if value.get('company_name','').casefold()==item['company_name'].casefold()]
        url=matches[0] if len(matches)==1 else ''
    item=resolve_calendar(item,url)
    from services.preclose_context import restore_preclose_gmp
    item=restore_preclose_gmp({**item,'url':url},db_path)
    if item.get('category')=='LISTED':
        from services.saved_preclose_service import comparison
        card=comparison(item,url,db_path)
        from services.module1_benchmark_lookup import lookup as benchmark_lookup
        benchmark=benchmark_lookup(item)
        from services.listed_backtest_service import cached
        if card.get('saved_call_found'):
            card['benchmark_comparison']=benchmark
        else:
            card['historical_backtest']=benchmark or cached(item['company_name'])
        return card
    ipo_id=extract_ipo_id({**item,'url':url})
    now=datetime.now(IST);_,current_cutoff=calendar(item,now)
    if current_cutoff and now<current_cutoff and item.get('category') in ('OPEN','UPCOMING'):
        record(item,url,db_path=db_path)
    rows=[r for r in scorecard(item['company_name'],db_path)['records'] if r['ipo_id']==ipo_id]
    if rows and item.get('category')=='LISTED' and not rows[0]['outcome']:
        reconcile(db_path)
        rows=[r for r in scorecard(item['company_name'],db_path)['records'] if r['ipo_id']==ipo_id]
    if rows:
        row=rows[0];return {**row,'mode':'saved_pre_close','probability':row['probability']}
    now=datetime.now(IST);_,cutoff=calendar(item,now)
    if cutoff and now>=cutoff:
        from services.snapshot_service import get_latest_application_forecast
        legacy=get_latest_application_forecast(item['company_name'],db_path=str(db_path))
        if legacy and str(legacy.get('ipo_id'))==ipo_id:
            issued=stamp(legacy.get('recorded_at_ist'));source=stamp(legacy.get('source_asof_ist'))
            if issued and source and source<=issued<cutoff:
                from services.listing_outcome_service import verified_listing_outcomes
                actual=verified_listing_outcomes(Path(db_path)).get(ipo_id)
                outcome=None
                if actual:
                    price=float(actual['issue_price']);opening=float(actual.get('exchange_open_price') or actual['nse_open_price'])
                    outcome={**actual,'actual_open':opening,'return_pct':100*(opening/price-1),'positive':opening>price}
                return {'mode':'legacy_saved_pre_close','legacy_model':True,'quality':'complete',
                    'probability':legacy['positive_listing_probability'],'gmp_probability':None,
                    'recorded_at_ist':legacy['recorded_at_ist'],'cutoff_at_ist':cutoff.isoformat(),
                    'model_version':legacy.get('model_version'),'inputs':legacy.get('inputs',{}),'outcome':outcome}
    if cutoff and now<cutoff and item.get('category') in ('OPEN','UPCOMING'):
        record(item,url,db_path=db_path)
        rows=[r for r in scorecard(item['company_name'],db_path)['records'] if r['ipo_id']==ipo_id]
        if rows:
            return {**rows[0],'mode':'saved_pre_close'}
    asof=min(now,cutoff) if cutoff else now
    features,evidence=build_features(item,url,asof,db_path)
    probability,gmp=predict(features)
    from services.listing_outcome_service import verified_listing_outcomes
    actual=verified_listing_outcomes(Path(db_path)).get(ipo_id)
    outcome=None
    if actual:
        price=float(actual['issue_price']);opening=float(actual.get('exchange_open_price') or actual['nse_open_price'])
        outcome={**actual,'actual_open':opening,'return_pct':100*(opening/price-1),'positive':opening>price}
    return {'mode':'current_estimate' if item.get('category') in ('OPEN','UPCOMING') else 'retrospective_estimate','probability':probability,'gmp_probability':gmp,
        'quality':evidence['quality'],'inputs':{'features':features,'evidence':evidence},
        'recorded_at_ist':now.isoformat(),'outcome':outcome}
