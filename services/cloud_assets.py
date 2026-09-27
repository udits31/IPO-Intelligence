"""Private, checksum-addressed runtime evidence for stateless cloud instances."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUCKET='openbell-runtime'
TABLE='capstone_sqlite_mirror'
_lock=threading.Lock()
_last_refresh={}
log=logging.getLogger(__name__)


def safe_path(relative):
    value=Path(relative)
    if value.is_absolute() or '..' in value.parts or value.parts[0] not in ('data','ml','output'):
        raise ValueError('Invalid runtime asset path')
    if value.suffix.lower() not in ('.json','.jsonl','.csv','.gz','.zip','.joblib'):
        raise ValueError('Unsupported runtime asset')
    return ROOT/value


def local_assets():
    paths=set()
    reports=[ROOT/'output/performance_evidence.json',*ROOT.glob('output/*/report.json')]
    def dependencies(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('input_hashes','input_sha256') and isinstance(item,dict):
                    for name,sha in item.items():
                        path=ROOT/str(name).replace('\\','/')
                        if path.is_file() and path.suffix in ('.csv','.json','.jsonl','.joblib'):
                            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                                continue # no obsolete evidence silently repaired
                            paths.add(path)
                dependencies(item)
        elif isinstance(value,list):
            for item in value:dependencies(item)
    for report in reports:
        if report.exists():dependencies(json.loads(report.read_text(encoding='utf8')))
    for pattern in ('data/live_*cache.json','data/telemetry/prospective_*/**/*.json',
                    'data/telemetry/prospective_*/**/*.gz','data/telemetry/prospective_*.json',
                    'data/telemetry/prospective_macro.csv','data/telemetry/progressive_engine_state.json'):
        paths.update(ROOT.glob(pattern))
    # The static recovered facts are identity/hash checked against these bytes.
    recovery=ROOT/'data/audit/ipopremium_recovered_20261026.jsonl'
    if recovery.exists():
        for line in recovery.read_text(encoding='utf8').splitlines():
            name=json.loads(line).get('source_file')
            if name:paths.add(ROOT/name)
    # Only the currently presented benchmark's primary opening archives.
    import csv
    lock=json.loads((ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json').read_text())
    ids={str(r['issuer_key']) for r in lock['module1']['records']}
    with (ROOT/'data/audit/module1_exchange_listing_open_audit.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['ipo_id'] in ids and row.get('raw_archive_path'):paths.add(ROOT/row['raw_archive_path'])
    # The newest history per issuer contains earlier daily closes too. Avoid
    # uploading every repeated Yahoo download or the whole 3GB research archive.
    ledger=ROOT/'data/telemetry/module2_decisions.db'
    from db.ledger import cloud_enabled,connect
    if cloud_enabled() or ledger.exists():
        opened=connect(ledger) if cloud_enabled() else sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True)
        with opened as connection:
            rows=connection.execute('SELECT payload_json FROM module2_events_v1 WHERE id IN (SELECT MAX(id) FROM module2_events_v1 GROUP BY ipo_id)').fetchall()
        for (text,) in rows:
            data=json.loads(text);sha=data.get('price_archive_sha256')
            if sha:paths.add(ROOT/'data/archive/yahoo_chart'/(sha+'.json.gz'))
    if cloud_enabled():
        # Files created by the production collectors since their last publish.
        paths.update((ROOT/'data/archive/yahoo_chart').glob('*.json.gz'))
    found=[]
    for path in sorted(paths):
        if path.is_file():
            relative=path.relative_to(ROOT).as_posix();safe_path(relative)
            found.append((relative,path))
    return found


def remote_manifest(client):
    rows=[];offset=0
    while True:
        page=(client.table(TABLE).select('row_key,payload').eq('db_name','runtime_assets')
              .eq('table_name','files').order('row_key').range(offset,offset+499).execute().data or [])
        rows.extend(page)
        if len(page)<500:break
        offset+=500
    return {r['payload']['path']:r['payload'] for r in rows}


def persist_file(path):
    """Durably archive a new source before its forecast is committed."""
    from db.supabase_client import get_supabase
    name=Path(path).resolve().relative_to(ROOT.resolve()).as_posix();safe_path(name)
    raw=Path(path).read_bytes();sha=hashlib.sha256(raw).hexdigest();key='sha256/'+sha
    client=get_supabase()
    try:client.storage.from_(BUCKET).upload(key,raw,{'content-type':'application/octet-stream','upsert':'false'})
    except Exception as exc:
        if 'already exists' not in str(exc).lower() and 'duplicate' not in str(exc).lower():raise
    value=dict(path=name,sha256=sha,size=len(raw),object=key)
    canonical=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    client.table(TABLE).upsert(dict(db_name='runtime_assets',table_name='files',
        row_key=json.dumps({'path':name},separators=(',',':')),payload=value,
        row_sha256=hashlib.sha256(canonical.encode()).hexdigest()),on_conflict='db_name,table_name,row_key').execute()


def ensure_file(path):
    path=Path(path)
    if path.exists():return True
    from db.ledger import cloud_enabled
    if not cloud_enabled():return False
    from db.supabase_client import get_supabase
    name=path.resolve().relative_to(ROOT.resolve()).as_posix();safe_path(name);client=get_supabase()
    rows=(client.table(TABLE).select('payload').eq('db_name','runtime_assets').eq('table_name','files')
          .eq('row_key',json.dumps({'path':name},separators=(',',':'))).limit(1).execute().data or [])
    if not rows:return False
    record=rows[0]['payload'];raw=client.storage.from_(BUCKET).download(record['object'])
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise RuntimeError('Source asset checksum mismatch')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_bytes(raw);temporary.replace(path)
    return True


def publish(client=None,*,max_bytes=200_000_000):
    from db.supabase_client import get_supabase
    client=client or get_supabase();assets=local_assets()
    total=sum(p.stat().st_size for _,p in assets)
    if total>max_bytes:raise RuntimeError('Selected runtime assets exceed the 200MB publish guard')
    existing=remote_manifest(client)
    if BUCKET not in [b.name for b in client.storage.list_buckets()]:
        client.storage.create_bucket(BUCKET,options={'public':False})
    clients=threading.local()
    def save(entry):
        if not hasattr(clients,'value'):clients.value=get_supabase()
        writer=clients.value
        name,path=entry;raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if existing.get(name,{}).get('sha256')==sha:return existing[name],0
        key='sha256/'+sha
        try:writer.storage.from_(BUCKET).upload(key,raw,{'content-type':'application/octet-stream','upsert':'false'})
        except Exception as exc:
            if 'already exists' not in str(exc).lower() and 'duplicate' not in str(exc).lower():raise
        value=dict(path=name,sha256=sha,size=len(raw),object=key)
        canonical=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        row=dict(db_name='runtime_assets',table_name='files',row_key=json.dumps({'path':name},separators=(',',':')),
                 payload=value,row_sha256=hashlib.sha256(canonical.encode()).hexdigest())
        writer.table(TABLE).upsert(row,on_conflict='db_name,table_name,row_key').execute()
        return value,1
    with ThreadPoolExecutor(max_workers=6) as pool:published=list(pool.map(save,assets))
    # Keep earlier immutable source assets addressable even if a stateless
    # publisher has hydrated only the files needed for its current workload.
    release={'files':{**existing,**{r['path']:r for r,_ in published}}}
    canonical=json.dumps(release,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    client.table(TABLE).upsert(dict(db_name='runtime_assets',table_name='release',row_key='{"release":"current"}',payload=release,
        row_sha256=hashlib.sha256(canonical.encode()).hexdigest()),on_conflict='db_name,table_name,row_key').execute()
    uploaded=sum(count for _,count in published)
    return dict(status='ok',selected_files=len(assets),selected_bytes=total,uploaded_files=uploaded)


def restore(*,force=False,client=None,scope='all'):
    global _last_refresh
    from db.ledger import cloud_enabled
    if not cloud_enabled():return {'status':'local'}
    if not force and time.monotonic()-_last_refresh.get(scope,0)<60:return {'status':'cached'}
    if not _lock.acquire(blocking=False):return {'status':'refresh_in_progress'}
    try:
        from db.supabase_client import get_supabase
        client=client or get_supabase()
        release=(client.table(TABLE).select('payload').eq('db_name','runtime_assets').eq('table_name','release')
                 .eq('row_key','{"release":"current"}').limit(1).execute().data or [])
        manifest=release[0]['payload']['files'] if release else {}
        if not manifest:raise RuntimeError('Runtime source assets have not been published to Supabase')
        if scope!='all':
            financial={}
            for name in manifest:
                if '/prospective_financials/' in name:
                    issuer=Path(name).name.split('_')[0]
                    financial[issuer]=max(financial.get(issuer,''),Path(name).name.split('.')[0])
            recovered=set()
            if scope in ('research','worker'):
                calendar=ROOT/'data/live_scraped_cache.json'
                facts=ROOT/'data/audit/ipopremium_recovered_20261026.jsonl'
                if calendar.exists() and facts.exists():
                    names={r.get('company_name') for r in json.loads(calendar.read_text()).values() if isinstance(r,dict) and r.get('category') in ('OPEN','UPCOMING')}
                    recovered={str(r.get('source_file')).replace('\\','/') for line in facts.read_text(encoding='utf8').splitlines() if (r:=json.loads(line)).get('company_name') in names}
            def needed(name):
                calendar=name.startswith('data/live_') or name=='data/telemetry/prospective_calendar_overlay.json'
                if scope=='calendar':return calendar
                if scope=='monitor':return calendar or '/yahoo_chart/' in name or 'bhavcopy' in name
                if '/prospective_financials/' in name:
                    issuer=Path(name).name.split('_')[0]
                    return Path(name).name.split('.')[0]==financial[issuer]
                if '/data/archive/' in '/'+name and '/yahoo_chart/' not in name and 'bhavcopy' not in name:
                    return name in recovered
                return True
            manifest={name:record for name,record in manifest.items() if needed(name)}
        clients=threading.local()
        def load(item):
            if not hasattr(clients,'value'):clients.value=get_supabase()
            name,record=item;path=safe_path(name)
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']:return 0
            raw=clients.value.storage.from_(BUCKET).download(record['object'])
            if len(raw)!=record['size'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
                raise RuntimeError('Supabase source-asset checksum mismatch')
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            temporary.write_bytes(raw);temporary.replace(path)
            return 1
        with ThreadPoolExecutor(max_workers=8) as pool:downloaded=sum(pool.map(load,manifest.items()))
        _last_refresh[scope]=time.monotonic()
        return dict(status='ok',files=len(manifest),downloaded=downloaded)
    finally:_lock.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    if args.publish:print(json.dumps(publish()))
    else:
        assets=local_assets();print(json.dumps(dict(files=len(assets),bytes=sum(p.stat().st_size for _,p in assets))))
