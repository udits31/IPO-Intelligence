"""BSE-only listing verification. Never stores BSE opens as NSE prices."""
import hashlib,json,re
from db import ledger as sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from services.point_in_time import IST


def record_bse_outcomes(cache,*,db_path,as_of):
    from ml.audit_module1_listing_opens import bse_row
    saved=[];pending=[]
    status_path=Path(db_path).parent/'bse_outcome_fetch_status.json'
    try:statuses=json.loads(status_path.read_text(encoding='utf8'))
    except (OSError,ValueError):statuses={}
    with sqlite3.connect(db_path,timeout=15) as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS listing_outcome_bse_v1 (
            ipo_id TEXT PRIMARY KEY,company_name TEXT NOT NULL,listing_date TEXT NOT NULL,
            issue_price REAL NOT NULL,exchange_open_price REAL NOT NULL,exchange_close_price REAL NOT NULL,
            exchange TEXT NOT NULL,bse_scrip_code INTEGER NOT NULL,archive_sha256 TEXT NOT NULL,
            source_url TEXT NOT NULL,source_timestamp TEXT,fetch_timestamp TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS bse_outcome_no_update BEFORE UPDATE ON listing_outcome_bse_v1 BEGIN SELECT RAISE(ABORT,'Outcomes are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS bse_outcome_no_delete BEFORE DELETE ON listing_outcome_bse_v1 BEGIN SELECT RAISE(ABORT,'Outcomes are immutable'); END;''')
        session=requests.Session()
        for url,item in cache.items():
            m=re.search(r'/ipo/[^/]+/(\d+)/?',url)
            if not m:continue
            key='chitto_'+m[1]
            if c.execute('SELECT 1 FROM listing_outcome_bse_v1 WHERE ipo_id=?',(key,)).fetchone():continue
            prior=statuses.get(key,{})
            if prior.get('attempt_date')==str(as_of) and prior.get('status')=='pending':
                pending.append(dict(ipo_id=key,exchange='BSE',reason=prior['reason'],cached_failure=True));continue
            try:
                date=pd.Timestamp(item['listing_date']);code=int(item['bse_code'])
                if date.date()>as_of:continue
                data=bse_row(session,date,code)
                # Exact exchange code and session are checked by the archive parser.
                c.execute('INSERT OR IGNORE INTO listing_outcome_bse_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (key,item['company_name'],str(date.date()),float(item['issue_price']),data['exchange_open'],data['exchange_close'],'BSE',code,data['archive_sha256'],data['source_url'],None,datetime.now(IST).isoformat()))
                saved.append(key)
                statuses[key]=dict(status='verified',attempt_date=str(as_of))
            except Exception as exc:
                reason=str(exc)[:220];pending.append(dict(ipo_id=key,exchange='BSE',reason=reason))
                statuses[key]=dict(status='pending',attempt_date=str(as_of),reason=reason,fetch_timestamp=datetime.now(IST).isoformat())
    status_path.write_text(json.dumps(statuses,indent=2),encoding='utf8')
    return dict(saved=saved,pending=pending)
