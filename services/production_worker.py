"""A leased, continuously running collector for Vercel Daemons."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

log=logging.getLogger(__name__)
STATE={'ready':False,'lease':False,'last_renewed':None}
OWNER=uuid.uuid4().hex


def lease(client,*,release=False):
    # A single atomic API transaction claims/renews a database-backed lease.
    # Rolling deploys cannot run old and new collectors simultaneously.
    statements=["CREATE TABLE IF NOT EXISTS _openbell_collector_lease (name text PRIMARY KEY,owner text NOT NULL,expires_at timestamptz NOT NULL)"]
    if release:
        statements.append(f"UPDATE _openbell_collector_lease SET expires_at=clock_timestamp() WHERE name='collector' AND owner='{OWNER}'")
    else:
        statements.append(f"INSERT INTO _openbell_collector_lease(name,owner,expires_at) VALUES('collector','{OWNER}',clock_timestamp()+interval '90 seconds') ON CONFLICT(name) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at WHERE _openbell_collector_lease.expires_at<clock_timestamp() OR _openbell_collector_lease.owner=excluded.owner")
    statements.append("SELECT owner,expires_at>clock_timestamp() AS active FROM _openbell_collector_lease WHERE name='collector'")
    result=client.rpc('openbell_ledger_transaction_v1',dict(p_database='snapshots.db',p_statements=statements,p_preview=False)).execute().data
    row=result['results'][-1]['rows'][0]
    return row['owner']==OWNER and row['active']


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        healthy=self.path=='/health' and STATE['ready'] and STATE['lease']
        self.send_response(200 if healthy else 503);self.send_header('Content-Type','application/json');self.end_headers()
        self.wfile.write(json.dumps(STATE).encode())
    def log_message(self,*args):pass


async def run():
    os.environ['OPENBELL_STORAGE_BACKEND']='supabase'
    os.environ['OPENBELL_COLLECTORS_ENABLED']='1'
    from db.supabase_client import get_supabase
    from services.cloud_assets import publish,restore
    client=get_supabase();stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    server=ThreadingHTTPServer(('0.0.0.0',8080),Health)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    renewal=None
    async def renew():
        while not stop.is_set():
            try:
                STATE['lease']=await asyncio.to_thread(lease,client)
                STATE['last_renewed']=time.time()
                if not STATE['lease']:stop.set();return
            except Exception:
                STATE['lease']=False;stop.set();log.exception('Collector lease renewal failed');return
            await asyncio.sleep(20)
    try:
        while not stop.is_set():
            STATE['lease']=await asyncio.to_thread(lease,client)
            if STATE['lease']:break
            await asyncio.sleep(10)
        if stop.is_set():return
        renewal=asyncio.create_task(renew())
        await asyncio.to_thread(restore,force=True,scope='worker')
        from api.main import app,lifespan
        async with lifespan(app):
            STATE['ready']=True;last_publish=0.
            while not stop.is_set():
                if not STATE['lease']:raise RuntimeError('Collector lease was lost')
                if time.monotonic()-last_publish>=60:
                    try:await asyncio.to_thread(publish,client)
                    except Exception:log.exception('Runtime source publish failed; retrying next cycle')
                    last_publish=time.monotonic()
                await asyncio.sleep(20)
    finally:
        if renewal:
            renewal.cancel()
            try:await renewal
            except asyncio.CancelledError:pass
        STATE['ready']=False;server.shutdown()
        if STATE['lease']:
            try:await asyncio.to_thread(lease,client,release=True)
            except Exception:log.exception('Collector lease release failed; it expires automatically')


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
