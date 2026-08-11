"""Prospective one-minute observations, separate from the frozen daily policy.

Public Yahoo candles are indicative, not an execution-grade exchange feed.
Every revision and alert is appended with its actual observation time.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re
from db import ledger as sqlite3
from statistics import median
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/telemetry/module2_intraday.db'
ARCHIVES = ROOT / 'data/archive/module2_intraday'
IST = ZoneInfo('Asia/Kolkata')
POLICY = 'intraday_structure_v1_research'
log = logging.getLogger(__name__)


def session_open(now):
    now = now.astimezone(IST)
    # Exchange holidays are additionally guarded by the actual candle date/age.
    return now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) < (15, 30)


def connection(path=DB):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=15000')
    con.executescript('''
      CREATE TABLE IF NOT EXISTS candles_v1 (
        id INTEGER PRIMARY KEY, ipo_id INTEGER NOT NULL, symbol TEXT NOT NULL,
        bar_start_utc TEXT NOT NULL, observed_at_utc TEXT NOT NULL,
        content_sha256 TEXT NOT NULL, raw_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL,
        UNIQUE(symbol,bar_start_utc,content_sha256));
      CREATE TABLE IF NOT EXISTS observations_v1 (
        id INTEGER PRIMARY KEY, ipo_id INTEGER NOT NULL, observed_at_utc TEXT NOT NULL,
        status TEXT NOT NULL, payload_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS alerts_v1 (
        id INTEGER PRIMARY KEY, ipo_id INTEGER NOT NULL, observed_at_utc TEXT NOT NULL,
        bar_start_utc TEXT NOT NULL, policy TEXT NOT NULL, signal TEXT NOT NULL,
        payload_json TEXT NOT NULL, UNIQUE(ipo_id,bar_start_utc,policy,signal));
      CREATE INDEX IF NOT EXISTS intraday_observation_ipo ON observations_v1(ipo_id,id);
      CREATE INDEX IF NOT EXISTS intraday_candle_ipo_time ON candles_v1(ipo_id,bar_start_utc,id);
    ''')
    return con


def tracked_issuers(now=None):
    """Canonical identities plus identities already bound by the daily ledger."""
    today = (now or datetime.now(IST)).astimezone(IST).date()
    frame = pd.read_csv(ROOT / 'data/canonical_listed_ipos.csv')
    rows = {}
    for r in frame.to_dict('records'):
        symbol = str(r.get('nse_symbol') or '')
        try:
            day = datetime.fromisoformat(str(r['listing_date'])[:10]).date()
            if not 0 <= (today-day).days <= 60 or r.get('listing_verified') is not True:
                continue
        except (ValueError, TypeError):
            continue
        if re.fullmatch(r'[A-Z0-9&-]{2,25}', symbol):
            rows[int(r['id'])] = dict(ipo_id=int(r['id']), company_name=r['company_name'],
                symbol=symbol+'.NS', listing_date=day.isoformat(), issue_price=r.get('issue_price'))
    ledger = ROOT / 'data/telemetry/module2_decisions.db'
    if sqlite3.exists(ledger):
        with closing(sqlite3.connect(f'file:{ledger.as_posix()}?mode=ro', uri=True)) as con:
            records = con.execute('SELECT ipo_id,payload_json FROM module2_decisions_v1 '
                'WHERE observation_day_ist >= ? ORDER BY id', ((today-timedelta(days=7)).isoformat(),)).fetchall()
        for ipo_id, payload in records:
            p=json.loads(payload); symbol=str(p.get('nse_symbol') or '')
            if re.fullmatch(r'[A-Z0-9&-]{2,25}',symbol):
                previous=rows.get(ipo_id,{})
                rows[ipo_id]=dict(ipo_id=ipo_id, company_name=p['company_name'],symbol=symbol+'.NS',
                    listing_date=previous.get('listing_date'),
                    issue_price=(p.get('price_waterfall') or {}).get('issue_price'))
            elif str(p.get('bse_code') or '').isdigit():
                rows[ipo_id]=dict(ipo_id=ipo_id,company_name=p['company_name'],
                    symbol=str(p['bse_code'])+'.BO',listing_date=None,
                    issue_price=(p.get('price_waterfall') or {}).get('issue_price'))
    # A listing-day issuer may have no completed daily close yet. Require the
    # explicit calendar ticker AND matching provider issuer name before saving.
    from scrapers.live_ipo_scraper import load_scraped_cache
    for url,item in load_scraped_cache().items():
        if item.get('category')!='LISTED' or item.get('source_observed') is not True:
            continue
        match=re.search(r'/ipo/[^/]+/(\d+)/?',url)
        symbol=str(item.get('nse_symbol') or '')
        try:day=datetime.fromisoformat(str(item['listing_date'])[:10]).date()
        except (ValueError,TypeError,KeyError):continue
        if not match or not 0<=(today-day).days<=60 or not re.fullmatch(r'[A-Z0-9&-]{2,25}',symbol):
            continue
        ipo_id=int(match.group(1))
        if ipo_id in rows:
            if rows[ipo_id]['symbol']==symbol+'.NS':rows[ipo_id]['listing_date']=day.isoformat()
            continue
        rows[ipo_id]=dict(ipo_id=ipo_id,company_name=item['company_name'],symbol=symbol+'.NS',
            listing_date=day.isoformat(),issue_price=item.get('issue_price'),
            require_provider_name_match=True,identity_source_url=url)
    return rows


def parse_candles(payload, symbol, observed_at):
    results=payload.get('chart',{}).get('result') or []
    if len(results)!=1:
        raise ValueError('Provider did not return a single instrument')
    result=results[0];meta=result.get('meta') or {}
    if meta.get('symbol')!=symbol or meta.get('currency')!='INR':
        raise ValueError('Instrument or currency mismatch')
    quotes=result.get('indicators',{}).get('quote') or []
    if len(quotes)!=1:
        raise ValueError('Missing minute candle arrays')
    timestamps=result.get('timestamp') or []; q=quotes[0]
    if any(len(q.get(k,[]))!=len(timestamps) for k in ('open','high','low','close','volume')):
        raise ValueError('Misaligned minute candle arrays')
    rows=[];seen=set()
    for i,stamp in enumerate(timestamps):
        start=datetime.fromtimestamp(stamp,timezone.utc)
        local=start.astimezone(IST)
        if start+timedelta(minutes=1)>observed_at or not (9,15)<=(local.hour,local.minute)<(15,30):
            continue  # Never let an unfinished candle drive an alert.
        if local.second or stamp in seen:
            raise ValueError('Invalid or duplicate minute timestamp')
        seen.add(stamp)
        values={k:q[k][i] for k in ('open','high','low','close','volume')}
        if any(v is None for v in values.values()):
            continue
        values={k:float(v) for k,v in values.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Non-finite minute candle')
        o,h,l,c,v=[values[k] for k in ('open','high','low','close','volume')]
        if min(o,h,l,c)<=0 or v<0 or l>min(o,c) or h<max(o,c) or h<l:
            raise ValueError('Invalid candle prices or volume')
        rows.append(dict(bar_start_utc=start.isoformat(),session_date=local.date().isoformat(),**values))
    rows.sort(key=lambda r:r['bar_start_utc'])
    return rows,dict(provider='Yahoo Finance',symbol=symbol,
        exchange=meta.get('exchangeName'),currency='INR',interval='1m',
        execution_grade=False,declared_delay_seconds=meta.get('exchangeDataDelayedBy'))


def analyse(candles, now, listing_date=None):
    today=now.astimezone(IST).date().isoformat()
    hour,minute=(10,0) if listing_date==today else (9,15)
    session_start=now.astimezone(IST).replace(hour=hour,minute=minute,second=0,microsecond=0)
    rows=[r for r in candles if r['session_date']==today
          and datetime.fromisoformat(r['bar_start_utc'])>=session_start]
    result=dict(status='UNAVAILABLE',signal='DATA_PENDING',alert_eligible=False,
        reason='No completed candle from today.',policy=POLICY,validated_accuracy=None,
        trading_session_open=session_open(now),bar_count=len(rows))
    if not rows:
        return result
    last=rows[-1];end=datetime.fromisoformat(last['bar_start_utc'])+timedelta(minutes=1)
    age=(now-end).total_seconds()
    result.update(status='FRESH' if age<=180 else 'DELAYED',
        latest_price=last['close'],bar_start_utc=last['bar_start_utc'],
        bar_end_utc=end.isoformat(),age_seconds=round(age,1),session_open_price=rows[0]['open'],
        session_high=max(r['high'] for r in rows),session_low=min(r['low'] for r in rows),
        session_volume=sum(r['volume'] for r in rows))
    baseline=median([r['volume'] for r in rows[-21:-1]]) if len(rows)>=21 else None
    result.update(latest_minute_volume=last['volume'],
        volume_vs_previous_20_minute_median=last['volume']/baseline if baseline and baseline>0 else None,
        circuit_band_status='Not supplied by this feed')
    if not session_open(now):
        result.update(status='MARKET_CLOSED',reason='Market session ended; retained candles are historical context.')
        return result
    if age<0 or age>180:
        result.update(reason='Latest completed minute is too old for an active alert.')
        return result
    # VWAP is a typical-price candle approximation, not transaction-level VWAP.
    volume=0.;notional=0.;vwaps=[]
    for r in rows:
        volume+=r['volume'];notional+=(r['high']+r['low']+r['close'])/3*r['volume']
        vwaps.append(notional/volume if volume>0 else None)
    result['bar_based_vwap']=vwaps[-1]
    if len(rows)<2 or not vwaps[-1] or not vwaps[-2]:
        result.update(reason='Waiting for two completed candles with traded volume.')
        return result
    opening=[r for r in rows if datetime.fromisoformat(r['bar_start_utc'])<session_start+timedelta(minutes=15)]
    complete_opening=(len(opening)==15 and
        all((datetime.fromisoformat(r['bar_start_utc'])-session_start).total_seconds()==60*i
            for i,r in enumerate(opening)))
    consecutive=(datetime.fromisoformat(rows[-1]['bar_start_utc'])-
                 datetime.fromisoformat(rows[-2]['bar_start_utc'])).total_seconds()==60
    result.update(opening_range_complete=complete_opening,
        opening_range_low=min(r['low'] for r in opening) if complete_opening else None,
        opening_range_high=max(r['high'] for r in opening) if complete_opening else None,
        signal='ABOVE_VWAP' if last['close']>=vwaps[-1] else 'BELOW_VWAP',
        reason='Latest completed candle is above bar-based VWAP.' if last['close']>=vwaps[-1]
            else 'Latest completed candle is below bar-based VWAP; watch price support.')
    if not consecutive:
        result['reason']+=' Consecutive minute confirmation is unavailable.'
        return result
    if (complete_opening and len(rows)>=17 and
        all(r['close']<result['opening_range_low'] and r['close']<v
            for r,v in zip(rows[-2:],vwaps[-2:]))):
        result.update(signal='REVIEW_EXIT',alert_eligible=True,
            reason='Two consecutive completed candles closed below the first 15-minute low and bar-based VWAP.')
    elif rows[-2]['close']>=vwaps[-2] and last['close']<vwaps[-1]:
        result.update(signal='VWAP_LOST',alert_eligible=True,
            reason='Price crossed below bar-based VWAP on a completed minute candle.')
    elif rows[-2]['close']<=vwaps[-2] and last['close']>vwaps[-1]:
        result.update(signal='VWAP_RECLAIMED',alert_eligible=True,
            reason='Price recovered above bar-based VWAP on a completed minute candle; review the earlier warning.')
    result['rule_basis']='Structural price observations; research rules, not optimised or validated trading forecasts.'
    result['execution_note']='No automatic orders. Alert price is not a guaranteed fill; circuit limits and lack of buyers can prevent exit.'
    return result


def capture(issuer, *, now=None, db_path=DB, fetch=requests.get):
    now=now or datetime.now(timezone.utc); ipo_id=int(issuer['ipo_id']); candles=[]
    report=dict(ipo_id=ipo_id,company_name=issuer['company_name'],symbol=issuer['symbol'],
        listing_date=issuer.get('listing_date'),
        observed_at_utc=now.isoformat(),source='Yahoo Finance · indicative one-minute candles',
        execution_grade=False,policy=POLICY)
    try:
        response=fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+issuer['symbol'],
            params={'interval':'1m','range':'1d'},headers={'User-Agent':'Mozilla/5.0'},timeout=6)
        response.raise_for_status()
        payload=response.json()
        if issuer.get('require_provider_name_match'):
            from services.module2_dynamic_monitor import issuer_matches
            observed=((payload.get('chart',{}).get('result') or [{}])[0].get('meta') or {})
            if (not issuer_matches(issuer['company_name'],observed.get('longName') or observed.get('shortName') or '')
                    or observed.get('exchangeName') not in ('NSI','NSE')):
                raise ValueError('Calendar ticker is not confirmed by the provider issuer identity')
        candles,meta=parse_candles(payload,issuer['symbol'],now)
        digest=hashlib.sha256(response.content).hexdigest()
        ARCHIVES.mkdir(parents=True,exist_ok=True)
        archive=ARCHIVES/(digest+'.json.gz')
        if not archive.exists():
            with gzip.open(archive,'wb') as stream:stream.write(response.content)
        report.update(analyse(candles,now,issuer.get('listing_date')),provider_metadata=meta,
            identity_basis='Calendar ticker plus matching provider issuer' if issuer.get('require_provider_name_match') else 'Existing canonical or saved daily-ledger identity',
            raw_sha256=digest,raw_archive_path=str(archive),source_url=response.url)
    except Exception as exc:
        report.update(status='SOURCE_UNAVAILABLE',signal='DATA_PENDING',alert_eligible=False,
            reason=f'Minute feed unavailable ({type(exc).__name__}); last saved candles are retained.',
            trading_session_open=session_open(now))
    with closing(connection(db_path)) as con:
        previous=con.execute('SELECT payload_json FROM observations_v1 WHERE ipo_id=? ORDER BY id DESC LIMIT 1',
            (ipo_id,)).fetchone()
        previous=json.loads(previous[0]) if previous else {}
        for row in candles:
            payload=json.dumps(row,sort_keys=True,allow_nan=False)
            con.execute('INSERT OR IGNORE INTO candles_v1 '
                '(ipo_id,symbol,bar_start_utc,observed_at_utc,content_sha256,raw_sha256,payload_json) VALUES (?,?,?,?,?,?,?)',
                (ipo_id,issuer['symbol'],row['bar_start_utc'],now.isoformat(),
                 hashlib.sha256(payload.encode()).hexdigest(),report['raw_sha256'],payload))
        con.execute('INSERT INTO observations_v1 (ipo_id,observed_at_utc,status,payload_json) VALUES (?,?,?,?)',
            (ipo_id,now.isoformat(),report['status'],json.dumps(report,allow_nan=False)))
        changed=(previous.get('signal')!=report.get('signal') or previous.get('status')!='FRESH'
            or str(previous.get('bar_start_utc',''))[:10]!=str(report.get('bar_start_utc',''))[:10])
        if report.get('alert_eligible') and changed:
            con.execute('INSERT OR IGNORE INTO alerts_v1 '
                '(ipo_id,observed_at_utc,bar_start_utc,policy,signal,payload_json) VALUES (?,?,?,?,?,?)',
                (ipo_id,now.isoformat(),report['bar_start_utc'],POLICY,report['signal'],json.dumps(report,allow_nan=False)))
        con.commit()
    return report


def read(ipo_id, *, now=None, db_path=DB):
    now=now or datetime.now(timezone.utc)
    with closing(connection(db_path)) as con:
        raw=con.execute('SELECT payload_json FROM observations_v1 WHERE ipo_id=? ORDER BY id DESC LIMIT 1',
            (ipo_id,)).fetchone()
        if not raw:return dict(ipo_id=ipo_id,status='NOT_COLLECTED',signal='DATA_PENDING',alerts=[])
        report=json.loads(raw[0])
        last=con.execute('SELECT payload_json FROM observations_v1 WHERE ipo_id=? '
            'AND status IN (\'FRESH\',\'DELAYED\',\'MARKET_CLOSED\') ORDER BY id DESC LIMIT 1',(ipo_id,)).fetchone()
        if report['status']=='SOURCE_UNAVAILABLE' and last:
            report['last_successful_observation']=json.loads(last[0])
        if report.get('bar_end_utc'):
            age=(now-datetime.fromisoformat(report['bar_end_utc'])).total_seconds()
            report['age_seconds']=round(age,1)
            if not session_open(now):report['status']='MARKET_CLOSED'
            elif age>180:report.update(status='DELAYED',alert_eligible=False)
        report['alerts']=[json.loads(r[0]) for r in con.execute(
            'SELECT payload_json FROM alerts_v1 WHERE ipo_id=? ORDER BY id DESC LIMIT 12',(ipo_id,))]
        # An old event remains in history, but is never marked an active alert.
        report['alert_eligible']=bool(report.get('alert_eligible') and report['status']=='FRESH')
        local=now.astimezone(IST)
        reference=report if report.get('bar_end_utc') else report.get('last_successful_observation') or {}
        if not session_open(now) and reference.get('bar_end_utc'):
            local=datetime.fromisoformat(reference['bar_end_utc']).astimezone(IST)
            report['status']='MARKET_CLOSED'
            report['alert_eligible']=False
        report['chart_session_date']=local.date().isoformat()
        report['chart_is_historical']=local.date()!=now.astimezone(IST).date() or not session_open(now)
        hour,minute=(10,0) if report.get('listing_date')==local.date().isoformat() else (9,15)
        start=local.replace(hour=hour,minute=minute,second=0,microsecond=0).astimezone(timezone.utc)
        finish=local.replace(hour=15,minute=30,second=0,microsecond=0).astimezone(timezone.utc)
        rows=con.execute('''SELECT payload_json FROM (
            SELECT payload_json,bar_start_utc,ROW_NUMBER() OVER (
                PARTITION BY symbol,bar_start_utc ORDER BY julianday(observed_at_utc) DESC,id DESC) AS revision
            FROM candles_v1 WHERE ipo_id=? AND bar_start_utc>=? AND bar_start_utc<?
            AND julianday(observed_at_utc)<=julianday(?))
            WHERE revision=1 ORDER BY bar_start_utc''',
            (ipo_id,start.isoformat(),finish.isoformat(),now.isoformat())).fetchall()
        report['candles']=[r for (encoded,) in rows if
            datetime.fromisoformat((r:=json.loads(encoded))['bar_start_utc'])+timedelta(minutes=1)<=now]
        return report


class IntradayCollector(threading.Thread):
    def __init__(self):
        super().__init__(name='module2-minute-collector',daemon=True)
        self.stopping=threading.Event();self.last_cycle=None;self.running_cycle=False
    def stop(self):self.stopping.set()
    def run(self):
        while not self.stopping.is_set():
            start=datetime.now(timezone.utc)
            local=start.astimezone(IST)
            final_candle_window=local.weekday()<5 and (15,30)<=(local.hour,local.minute)<(15,35)
            if session_open(start) or final_candle_window:
                self.running_cycle=True
                try:
                    issuers=list(tracked_issuers(start).values())
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        results=list(pool.map(capture,issuers))
                    self.last_cycle=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        attempted=len(results),fresh=sum(r['status']=='FRESH' for r in results),
                        unavailable=sum(r['status']=='SOURCE_UNAVAILABLE' for r in results),
                        alerts=sum(bool(r.get('alert_eligible')) for r in results))
                    if results:
                        from services.sqlite_supabase_sync import request_sync
                        request_sync('Module 2 intraday capture')
                except Exception as exc:
                    self.last_cycle=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        status='failed',error_type=type(exc).__name__)
                    log.exception('Module 2 minute collection failed')
                finally:self.running_cycle=False
            self.stopping.wait(max(1,60-(datetime.now(timezone.utc)-start).total_seconds()))


collector=IntradayCollector()
