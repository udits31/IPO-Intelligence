"""Recover annual IPO valuations when the primary HTML panel is incomplete.

Exact issuer/date/price identity and matching annual accounts are required.
Sources are append-only in SQLite and mirror through the existing Supabase sync.
Fetched-now facts never acquire an earlier availability clock.
"""
import gzip,hashlib,json,math,re
from db import ledger as sqlite3
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import requests
import pandas as pd
from bs4 import BeautifulSoup
from services.nse_subscription_service import _name

ROOT=Path(__file__).resolve().parents[1];DB=ROOT/'data/telemetry/snapshots.db';IST=ZoneInfo('Asia/Kolkata')


def extract(content,item,base,opened,closed):
    soup=BeautifulSoup(content,'html.parser')
    if not any(_name(h.get_text(' ',strip=True))==_name(item['company_name']) for h in soup.find_all('h1')):raise ValueError('Valuation source issuer mismatch')
    tables=[[[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])] for tr in t.find_all('tr')] for t in soup.find_all('table')]
    terms={r[0].lower():r[1] for t in tables for r in t if len(r)==2}
    dates=re.findall(r'[A-Z][a-z]+\s+\d{1,2},\s*20\d{2}',terms.get('ipo date',''))
    if len(dates)!=2 or [pd.Timestamp(v).date() for v in dates]!=[opened,closed]:raise ValueError('Valuation source offer-date mismatch')
    nums=re.findall(r'\d+(?:\.\d+)?',terms.get('price band','').replace(',',''))
    if not nums or abs(max(map(float,nums))-float(item['issue_price']))>.01:raise ValueError('Valuation source cap mismatch')
    annual=pd.Timestamp(base['annual_period'])
    if (annual.month,annual.day)!=(3,31):raise ValueError('Unsupported annual fiscal period')
    result=None
    for table in tables:
        header=next((r for r in table if any(re.fullmatch(r'FY\s*'+str(annual.year),v,re.I) for v in r)),None)
        if not header:continue
        col=next(i for i,v in enumerate(header) if re.fullmatch(r'FY\s*'+str(annual.year),v,re.I))
        metrics={re.sub(r'[^a-z]','',r[0].lower()):r[col].replace(',','') for r in table if r and len(r)>col}
        def num(key):
            s=metrics.get(key,'').strip()
            return float(s) if re.fullmatch(r'-?\d+(?:\.\d+)?',s) else None
        eps=num('dilutedearningspershare');nav=num('netassetvaluepershare')
        if eps is None or nav is None or nav<=0:continue
        for field,key in [('annual_profit_after_tax','profitaftertaxpat'),('annual_revenue','totalincome')]:
            expected=base.get(field);actual=num(key)
            if expected is None or actual is None or not math.isclose(float(expected),actual,rel_tol=.005,abs_tol=.02):raise ValueError('Annual account basis conflicts with original financial table')
        price=float(item['issue_price'])
        result=dict(annual_period=annual.date().isoformat(),ipo_pre_offer_eps=eps,ipo_pre_offer_nav=nav,
            ipo_earnings_yield_pct=100*eps/price,log_ipo_price_book=math.log1p(price/nav),
            valuation_basis='reported_annual_diluted_eps_and_annual_nav_before_offer',identity_verified=True,
            identity_checks=dict(company_name=item['company_name'],opened=str(opened),closed=str(closed),offer_cap=price,
                                 annual_pat=base['annual_profit_after_tax'],annual_total_income=base['annual_revenue']))
    if not result:raise ValueError('No matching annual EPS/NAV table')
    return result


def init(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS financial_valuation_sources_v1 (
        source_id TEXT PRIMARY KEY,company_name TEXT NOT NULL,source_url TEXT NOT NULL,
        captured_at_ist TEXT NOT NULL,raw_sha256 TEXT NOT NULL,raw_html_gzip BLOB NOT NULL,
        status TEXT NOT NULL,facts_json TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS financial_valuation_attempts_v1 (
        attempt_id TEXT PRIMARY KEY,company_name TEXT NOT NULL,source_url TEXT NOT NULL,
        attempted_at_ist TEXT NOT NULL,status TEXT NOT NULL,detail TEXT)''')


def recover(item,base,opened,closed,asof,*,db_path=DB):
    if not base.get('annual_period') or not opened or not closed:return None
    cutoff=pd.Timestamp(asof)
    if cutoff.tzinfo is None:raise ValueError('Timezone-aware valuation cutoff required')
    slug=re.sub(r'\b(?:limited|ltd|ipo)\b','',item['company_name'].lower()).replace('&',' and ')
    slug=re.sub(r'[^a-z0-9]+','-',slug).strip('-')
    url='https://www.samco.in/ipo/'+slug+'-ipo'
    with sqlite3.connect(db_path,timeout=10) as c:
        init(c);c.row_factory=sqlite3.Row
        for r in c.execute("SELECT * FROM financial_valuation_sources_v1 WHERE company_name=? AND status='verified' ORDER BY captured_at_ist DESC",(item['company_name'],)):
            f=json.loads(r['facts_json'])
            if pd.Timestamp(r['captured_at_ist'])>cutoff or f.get('annual_period')!=base['annual_period']:continue
            if abs(float(f['identity_checks']['offer_cap'])-float(item['issue_price']))>.01:continue
            if not all(math.isclose(float(f['identity_checks'][key]),float(base[field]),rel_tol=.005,abs_tol=.02)
                       for key,field in [('annual_pat','annual_profit_after_tax'),('annual_total_income','annual_revenue')]):continue
            if f['identity_checks']['opened']!=str(opened) or f['identity_checks']['closed']!=str(closed):continue
            if hashlib.sha256(gzip.decompress(r['raw_html_gzip'])).hexdigest()!=r['raw_sha256']:continue
            return {**f,'source_url':r['source_url'],'raw_sha256':r['raw_sha256'],'captured_at_ist':r['captured_at_ist']}
        prior=c.execute('SELECT max(attempted_at_ist) FROM financial_valuation_attempts_v1 WHERE company_name=? AND source_url=?',(item['company_name'],url)).fetchone()[0]
        if prior and datetime.now(IST)-datetime.fromisoformat(prior)<timedelta(hours=24):return None
    # No network recovery for a past-asof reconstruction.
    if cutoff.date()<datetime.now(IST).date():return None
    captured=datetime.now(IST).isoformat();status='source_unavailable';detail='';raw=None;facts={}
    try:
        response=requests.get(url,timeout=12,headers={'User-Agent':'Mozilla/5.0'});response.raise_for_status();raw=response.content
        facts=extract(response.text,item,base,opened,closed);status='verified'
    except (requests.RequestException,ValueError,KeyError,TypeError) as exc:detail=str(exc)[:300]
    captured=datetime.now(IST).isoformat();key=hashlib.sha256((url+'\n'+captured).encode()).hexdigest()
    with sqlite3.connect(db_path,timeout=10) as c:
        init(c)
        c.execute('INSERT INTO financial_valuation_attempts_v1 VALUES (?,?,?,?,?,?)',(key,item['company_name'],url,captured,status,detail))
        if raw:
            digest=hashlib.sha256(raw).hexdigest();source=hashlib.sha256((url+'\n'+digest).encode()).hexdigest()
            c.execute('INSERT OR IGNORE INTO financial_valuation_sources_v1 VALUES (?,?,?,?,?,?,?,?)',
                (source,item['company_name'],url,captured,digest,gzip.compress(raw),status,json.dumps(facts)))
    try:
        from services.sqlite_supabase_sync import request_sync
        request_sync('Dated annual IPO valuation recovery')
    except Exception:pass
    # A fetch completes after the caller's decision clock: admit it only on
    # a later decision, never by silently moving the earlier asof forward.
    return None
