"""Backend-owned data recovery with durable retries and source-level status."""
from __future__ import annotations

import json
import logging
import math
import re
from db import ledger as sqlite3
import threading
import gzip
import hashlib
from urllib.parse import urlparse
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from services.point_in_time import IST, timestamp
from services.preclose_context import DB, archive_offer_context, attach_offer_context, refresh_cached_offer_extractions
from services.live_offer_facts import OFFER_PARSER_VERSION

log = logging.getLogger(__name__)
VERSION = 'offer-maintenance-v1:' + OFFER_PARSER_VERSION


def offer_key(url, item):
    try:
        price=float(item.get('issue_price'))
        token=f'{price:g}' if math.isfinite(price) else 'unpublished'
    except (ValueError,TypeError):token='unpublished'
    return f"{VERSION}:{url.rstrip('/')}:{token}"


def connection(db_path=DB):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path, timeout=15)
    c.execute('PRAGMA busy_timeout=15000')
    c.executescript('''
      CREATE TABLE IF NOT EXISTS data_maintenance_jobs_v1 (
        job_key TEXT PRIMARY KEY, task TEXT NOT NULL, company_name TEXT,
        state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        last_attempt_at_ist TEXT, next_attempt_at_ist TEXT NOT NULL, detail_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS data_maintenance_events_v1 (
        id INTEGER PRIMARY KEY, job_key TEXT NOT NULL, recorded_at_ist TEXT NOT NULL,
        state TEXT NOT NULL, detail_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS data_source_rejection_v1 (
        id INTEGER PRIMARY KEY,job_key TEXT NOT NULL,source_url TEXT NOT NULL,
        attempted_at_ist TEXT NOT NULL,source_sha256 TEXT NOT NULL,
        raw_html_gzip BLOB NOT NULL,error TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS offer_source_state_v1 (
        job_key TEXT NOT NULL,source_url TEXT NOT NULL,state TEXT NOT NULL,
        next_attempt_at_ist TEXT NOT NULL,detail_json TEXT NOT NULL,
        PRIMARY KEY(job_key,source_url));
    ''')
    return c


def claim(key, task, company, now, db_path=DB):
    """A persisted lease prevents repeats after a restart or overlapping workers."""
    with closing(connection(db_path)) as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT next_attempt_at_ist FROM data_maintenance_jobs_v1 WHERE job_key=?', (key,)).fetchone()
        if row and timestamp(row[0]) and timestamp(row[0]) > now:
            c.rollback()
            return False
        if sqlite3.cloud_enabled():
            if row:
                c.execute("UPDATE data_maintenance_jobs_v1 SET state='running',attempts=attempts+1,last_attempt_at_ist=?,next_attempt_at_ist=? WHERE job_key=?",
                    (now.isoformat(),(now+timedelta(minutes=5)).isoformat(),key))
            else:
                c.execute("INSERT OR IGNORE INTO data_maintenance_jobs_v1 VALUES(?,?,?,'running',1,?,?,'{}')",
                    (key,task,company,now.isoformat(),(now+timedelta(minutes=5)).isoformat()))
            c.commit()
            return True
        c.execute('''INSERT INTO data_maintenance_jobs_v1 VALUES(?,?,?,'running',1,?,?,'{}')
          ON CONFLICT(job_key) DO UPDATE SET state='running', attempts=attempts+1,
          last_attempt_at_ist=excluded.last_attempt_at_ist,next_attempt_at_ist=excluded.next_attempt_at_ist''',
          (key, task, company, now.isoformat(), (now+timedelta(minutes=5)).isoformat()))
        c.commit()
    return True


def finish(key, state, detail, now, delay_minutes, db_path=DB):
    encoded = json.dumps(detail, allow_nan=False)
    with closing(connection(db_path)) as c:
        c.execute('UPDATE data_maintenance_jobs_v1 SET state=?,detail_json=?,next_attempt_at_ist=? WHERE job_key=?',
                  (state, encoded, (now+timedelta(minutes=delay_minutes)).isoformat(), key))
        c.execute('INSERT INTO data_maintenance_events_v1(job_key,recorded_at_ist,state,detail_json) VALUES(?,?,?,?)',
                  (key, now.isoformat(), state, encoded))
        c.commit()


def missing_offer_fields(item):
    fields = ('promoter_ofs_ratio', 'retail_quota_pct', 'original_lot_size')
    missing = [key for key in fields if item.get(key) is None]
    if not item.get('offer_facts'):
        missing.append('verified_offer_source')
    if item.get('promoter_ofs_ratio') is not None and item['promoter_ofs_ratio'] < 1 and item.get('objects_capex_ratio') is None:
        missing.append('objects_capex_ratio')
    return missing


def offer_source_candidates(url, item):
    """Bounded public-source discovery; every response still needs issuer/price checks."""
    from services.historical_bid_service import name_key
    slug=re.search(r'/ipo/([^/]+)/',url)[1]
    named=name_key(item.get('company_name','')).replace(' ','-')+'-ipo'
    return list(dict.fromkeys([url,*[f'https://www.ipoji.com/ipo/{s}' for s in (named,slug)]]))


def source_due(key, url, now, db_path):
    with closing(connection(db_path)) as c:
        row=c.execute('SELECT next_attempt_at_ist FROM offer_source_state_v1 WHERE job_key=? AND source_url=?',(key,url)).fetchone()
    return not row or timestamp(row[0])<=now


def remember_source(key,url,state,detail,now,minutes,db_path):
    with closing(connection(db_path)) as c:
        if sqlite3.cloud_enabled():
            row=c.execute('SELECT state FROM offer_source_state_v1 WHERE job_key=? AND source_url=?',(key,url)).fetchone()
            when=(now+timedelta(minutes=minutes)).isoformat();encoded=json.dumps(detail)
            if row:c.execute('UPDATE offer_source_state_v1 SET state=?,next_attempt_at_ist=?,detail_json=? WHERE job_key=? AND source_url=?',(state,when,encoded,key,url))
            else:c.execute('INSERT OR IGNORE INTO offer_source_state_v1 VALUES(?,?,?,?,?)',(key,url,state,when,encoded))
            c.commit()
            return
        c.execute('INSERT INTO offer_source_state_v1 VALUES(?,?,?,?,?) ON CONFLICT(job_key,source_url) DO UPDATE SET state=excluded.state,next_attempt_at_ist=excluded.next_attempt_at_ist,detail_json=excluded.detail_json',
            (key,url,state,(now+timedelta(minutes=minutes)).isoformat(),json.dumps(detail)))
        c.commit()


def known_allocation_limitation(enriched,missing):
    hosts={urlparse(s.get('source_url') or '').hostname for s in enriched.get('proceeds_provenance',[])}
    return bool(missing==['objects_capex_ratio'] and enriched.get('proceeds_uses')
        and enriched.get('objects_capex_status')=='NOT_SEPARATELY_ITEMISED'
        and hosts.intersection({'www.ipoji.com','ipoji.com'}))


def maintain_offer(url, item, *, now=None, fetch=None, db_path=DB):
    now = now or datetime.now(IST)
    item = {**item, 'url': url}
    key = offer_key(url, item)
    if not claim(key, 'offer_facts', item.get('company_name'), now, db_path):
        return None
    response=None
    source_url=url
    try:
        price = float(item.get('issue_price') or 0)
        if not math.isfinite(price) or price <= 0:
            state, detail, delay = 'awaiting_terms', {'missing_fields': ['published_offer_price']}, 60
        else:
            refresh_cached_offer_extractions(item,db_path)
            enriched = attach_offer_context(item, db_path)
            missing = missing_offer_fields(enriched)
            if missing and not known_allocation_limitation(enriched,missing):
                candidates=offer_source_candidates(url,item)
                source_url=next((u for u in candidates if source_due(key,u,now,db_path)),None)
                if source_url is None:
                    detail={'missing_fields':missing,'error':'Sources checked; awaiting a source change or more detailed offer document',
                            'sources_checked':candidates,'proceeds_uses':enriched.get('proceeds_uses'),
                            'objects_capex_status':enriched.get('objects_capex_status')}
                    with closing(connection(db_path)) as c:
                        transient=c.execute("SELECT 1 FROM offer_source_state_v1 WHERE job_key=? AND state='retrying'",(key,)).fetchone()
                    state='retrying' if transient else 'needs_document_review'
                    finish(key,state,detail,now,30 if transient else 1440,db_path)
                    return {'company_name':item.get('company_name'),'state':state,**detail}
                response = (fetch or httpx.get)(source_url, headers={'User-Agent': 'Mozilla/5.0'}, follow_redirects=True, timeout=12)
                response.raise_for_status()
                record = archive_offer_context(item, response.content, source_url, db_path)
                enriched = attach_offer_context(item, db_path)
                missing = missing_offer_fields(enriched)
                remember_source(key,source_url,'partial' if missing else 'ready',{'missing_fields':missing,'sha256':record['sha256']},now,1440 if missing else 10080,db_path)
            else:
                record = enriched['offer_facts']
            allocation_known=known_allocation_limitation(enriched,missing)
            state = 'ready' if not missing else 'ready_with_limitations' if allocation_known else 'partial'
            detail = {'missing_fields': missing, 'source_url': record.get('source_url'),
                      'source_sha256': record.get('sha256'), 'source_fetched_at': record.get('fetch_timestamp')}
            if allocation_known:
                detail.update(objects_capex_status='NOT_SEPARATELY_ITEMISED',
                    limitation='Published uses recovered; separate CAPEX accounting percentage is not itemised in the checked sources',
                    proceeds_uses=enriched['proceeds_uses'],sources=enriched.get('proceeds_provenance',[]))
            delay = (10080 if item.get('category') in ('LISTED', 'CLOSED') else 1440) if not missing else 1
            if allocation_known:delay=10080 if item.get('category') in ('LISTED','CLOSED') else 1440
        finish(key, state, detail, now, delay, db_path)
    except Exception as exc:
        with closing(connection(db_path)) as c:
            attempt = c.execute('SELECT attempts FROM data_maintenance_jobs_v1 WHERE job_key=?', (key,)).fetchone()[0]
            if response is not None:
                raw=response.content
                c.execute('INSERT INTO data_source_rejection_v1(job_key,source_url,attempted_at_ist,source_sha256,raw_html_gzip,error) VALUES(?,?,?,?,?,?)',
                          (key,source_url or url,now.isoformat(),hashlib.sha256(raw).hexdigest(),gzip.compress(raw),f'{type(exc).__name__}: {exc}'[:350]))
                c.commit()
        delay = min(60, 5 * 2**min(attempt-1, 4))
        state, detail = 'retrying', {'error': f'{type(exc).__name__}: {exc}'[:350]}
        # Deterministic parse/identity failures cannot improve by hammering the
        # same page. Try another source on the next pass; preserve rejected raw.
        permanent=isinstance(exc,ValueError) or isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code in (404,410)
        remember_source(key,source_url or url,'rejected' if permanent else 'retrying',detail,now,1440 if permanent else max(30,delay*2),db_path)
        if permanent:delay=1
        detail['source_url']=source_url or url
        finish(key, state, detail, now, delay, db_path)
        log.warning('Automatic offer recovery will retry for %s: %s', item.get('company_name'), detail['error'])
    return {'company_name': item.get('company_name'), 'state': state, **detail}


class DataMaintenance(threading.Thread):
    def __init__(self, db_path=DB):
        super().__init__(name='ipo-data-maintenance', daemon=True)
        self.db_path = Path(db_path)
        self.stopping = threading.Event()
        self.cycle_lock = threading.Lock()
        self.last_cycle = None

    def stop(self):
        self.stopping.set()

    def _periodic(self, key, task, now, minutes, action):
        if not claim(key, task, None, now, self.db_path):
            return
        try:
            finish(key, 'ready', action() or {}, now, minutes, self.db_path)
        except Exception as exc:
            finish(key, 'retrying', {'error': f'{type(exc).__name__}: {exc}'[:350]}, now, 5, self.db_path)
            log.exception('Automatic %s refresh failed; retained sources stay available', task)

    def run_once(self, *, cache=None, now=None, fetch=None, max_offers=6):
        if not self.cycle_lock.acquire(blocking=False):
            return None
        now = now or datetime.now(IST)
        try:
            if cache is None:
                from scrapers.live_ipo_scraper import load_scraped_cache, sync_chittorgarh_calendar_delta, fetch_live_macro_regime, CALENDAR_FETCH_STATUS
                def discover():
                    items=sync_chittorgarh_calendar_delta(refresh_subscriptions=False)
                    if CALENDAR_FETCH_STATUS.get('status')!='fresh':
                        raise RuntimeError(CALENDAR_FETCH_STATUS.get('error') or 'Calendar is serving its retained cache')
                    return {'offers_discovered':len(items)}
                def market():
                    result=fetch_live_macro_regime()
                    if result.get('is_fallback'):
                        raise RuntimeError('Market source unavailable; last-known context retained')
                    return {'source':result.get('source'),'timestamp':result.get('timestamp')}
                self._periodic('ipo-calendar-v1', 'calendar', now, 30, discover)
                self._periodic('market-context-v1', 'market_context', now, 5 if 9 <= now.hour < 16 else 30, market)
                cache = load_scraped_cache()
            candidates = []
            for url, item in cache.items():
                if item.get('source_observed') is not True or item.get('category') not in ('OPEN', 'UPCOMING', 'CLOSED', 'LISTED'):
                    continue
                if not re.fullmatch(r'https://(?:www\.)?chittorgarh\.com/ipo/[^/]+/\d+/?', url):
                    continue
                candidates.append((url, item))
            candidates.sort(key=lambda r: ({'OPEN': 0, 'UPCOMING': 1, 'CLOSED': 2, 'LISTED': 3}[r[1]['category']],
                                            str(r[1].get('listing_date') or ''), r[0]))
            with closing(connection(self.db_path)) as c:
                keys=[offer_key(url,item) for url,item in candidates]
                for (url,item),key in zip(candidates,keys):
                    c.execute("INSERT OR IGNORE INTO data_maintenance_jobs_v1 VALUES(?,?,?,'queued',0,NULL,?,'{}')",
                              (key,'offer_facts',item.get('company_name'),now.isoformat()))
                if keys:
                    c.execute("UPDATE data_maintenance_jobs_v1 SET state='inactive' WHERE task='offer_facts' AND job_key NOT IN ("+','.join('?' for _ in keys)+')',keys)
                c.commit()
            processed = []
            for url, item in candidates:
                result = maintain_offer(url, item, now=now, fetch=fetch, db_path=self.db_path)
                if result:
                    processed.append(result)
                if len(processed) >= max_offers:
                    break
            if self.db_path.resolve() == DB.resolve():
                from services.news_service import get_news_snapshot
                def warm_news():
                    names = [r['company_name'] for _, r in candidates if r.get('category') in ('OPEN', 'UPCOMING')]
                    failures = []
                    for name in names:
                        if get_news_snapshot(name).get('error'):
                            failures.append(name)
                    if failures:
                        raise RuntimeError('News source unavailable for: ' + ', '.join(failures))
                    return {'issuers_checked': len(names)}
                self._periodic('active-ipo-news-v1', 'news', now, 10, warm_news)
                from services.sqlite_supabase_sync import request_sync
                request_sync('Automatic IPO data maintenance')
            self.last_cycle = {'completed_at_ist': datetime.now(IST).isoformat(), 'issuers_known': len(candidates),
                               'offers_checked': len(processed), 'records': processed}
            return self.last_cycle
        finally:
            self.cycle_lock.release()

    def run(self):
        while not self.stopping.is_set():
            try:
                self.run_once()
            except Exception:
                log.exception('Data maintenance cycle failed; retrying without stopping other collectors')
            self.stopping.wait(60)

    def status(self):
        with closing(connection(self.db_path)) as c:
            c.row_factory = sqlite3.Row
            rows = [dict(r) for r in c.execute('SELECT * FROM data_maintenance_jobs_v1 ORDER BY last_attempt_at_ist DESC')]
        for row in rows:
            row['detail'] = json.loads(row.pop('detail_json'))
        failures = [r for r in rows if r['state'] in ('retrying', 'partial','needs_document_review')]
        return {'running': self.is_alive(), 'last_cycle': self.last_cycle,
                'status': 'updates_delayed' if failures else 'running' if self.is_alive() else 'stopped',
                'retrying_jobs': len(failures), 'ready_jobs': sum(r['state'] == 'ready' for r in rows),
                'queued_jobs': sum(r['state'] == 'queued' for r in rows),
                'ready_with_limitations_jobs':sum(r['state']=='ready_with_limitations' for r in rows),
                'jobs': rows}


worker = DataMaintenance()
