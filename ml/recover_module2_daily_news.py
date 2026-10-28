"""Resumable, dated headline recovery across the actual daily decision windows.

Never searches by outcome, rewrites telemetry, treats an empty feed as clean
news, or assigns a retrieved-now headline to an earlier publication clock.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
import gzip,hashlib,json,re,threading,time,sqlite3
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from services.news_service import _quote_directory_title,_relevant_to_company,NewsService
from ml.collect_module2_dated_news import search_name

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/archive/module2_daily_news_recovery'
OUT=ROOT/'output/module2_news_recovery_20261029'
EXAMPLES=ROOT/'output/module2_next_session_20261029/daily_examples.csv'
DB=ROOT/'data/telemetry/module2_historical_news.db'
LOCK=threading.Lock();last_request=0.;blocked=threading.Event()


def digest(raw):return hashlib.sha256(raw).hexdigest()


def parse_feed(raw,company,start,end):
    root=ET.fromstring(raw)
    if root.tag.lower()!='rss' or root.find('channel') is None:raise ValueError('Not an RSS response')
    scorer=NewsService();rows=[];reject={};seen=set()
    for item in root.findall('./channel/item'):
        title=(item.findtext('title') or '').rsplit(' - ',1)[0].strip();url=item.findtext('link') or ''
        try:
            t=parsedate_to_datetime(item.findtext('pubDate') or '')
            if t.tzinfo is None:raise ValueError('Missing timezone')
            stamp=pd.Timestamp(t).tz_convert('Asia/Kolkata')
        except (ValueError,TypeError,OverflowError):reject['undated']=reject.get('undated',0)+1;continue
        reason=None
        if not start<=stamp<end:reason='outside_requested_dates'
        elif not url.startswith('https://') or not title:reason='missing_link_title'
        elif _quote_directory_title(title) or not _relevant_to_company(title,company):reason='identity_or_directory'
        elif title.casefold() in seen:reason='duplicate_title'
        if reason:reject[reason]=reject.get(reason,0)+1;continue
        seen.add(title.casefold());lower=title.casefold()
        categories={
            'earnings':r'profit|earnings|quarter|results|revenue|ebitda|margin',
            'order':r'order|contract|wins|expan|capacity',
            'governance':r'resign|fraud|auditor|default|probe|investigat',
            'regulatory':r'tax|sebi|regulat|penalty|notice|litigation',
            'capital':r'dividend|bonus|split|buyback|fundrais|placement|qip',
            'price_commentary':r'shares? (rise|fall|gain|drop|surge|slip)|stock (rise|fall|gain|drop|surge|slip)|share price'}
        rows.append(dict(title=title,published_at=stamp.isoformat(),url=url,publisher=item.findtext('source') or '',
            score=scorer.analyze_sentiment(title),event_tags=[k for k,pattern in categories.items() if re.search(pattern,lower)],
            availability_policy='preceding_calendar_dates_only; original intraday clock unverified'))
    return rows,reject,len(root.findall('./channel/item'))


def fetch(job):
    global last_request
    id,company,start,end=job
    query=f'"{search_name(company)}" after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}'
    url='https://news.google.com/rss/search?'+urlencode({'q':query,'hl':'en-IN','gl':'IN','ceid':'IN:en'})
    key=digest(url.encode())[:20];path=RAW/f'{id}_{key}.xml.gz';meta=path.with_suffix('.json')
    record=dict(ipo_id=id,company_name=company,start=start.isoformat(),end=end.isoformat(),source_url=url,
        source_family='Google News historical RSS index',archive_path=str(path.relative_to(ROOT)),articles=[])
    try:
        saved=json.loads(meta.read_text(encoding='utf8')) if meta.exists() else None
        if saved and saved.get('source_url')!=url:raise ValueError('Cache identity conflict')
        if path.exists() and saved and saved.get('raw_sha256'):
            with gzip.open(path,'rb') as f:raw=f.read()
            if digest(raw)!=saved['raw_sha256']:raise ValueError('Raw archive hash mismatch')
            retrieved=saved['retrieved_at_utc']
        else:
            if saved and time.time()-saved.get('attempt_epoch',0)<86400:
                return {**record,**saved,'status':'cached_failure_cooldown'}
            if blocked.is_set():return {**record,'status':'source_rate_limit_pause'}
            with LOCK:
                time.sleep(max(0,.65-(time.monotonic()-last_request)));last_request=time.monotonic()
            if blocked.is_set():return {**record,'status':'source_rate_limit_pause'}
            response=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=18)
            if response.status_code in (403,429):blocked.set()
            response.raise_for_status();raw=response.content
            parse_feed(raw,company,start,end)
            retrieved=datetime.now(timezone.utc).isoformat()
            with gzip.open(path,'wb') as f:f.write(raw)
        articles,reject,count=parse_feed(raw,company,start,end)
        record.update(status='verified_rss_response',retrieved_at_utc=retrieved,raw_sha256=digest(raw),
            articles=articles,rejected=reject,raw_items=count,possibly_truncated=count>=100)
    except (requests.RequestException,ValueError,ET.ParseError,OSError) as exc:
        record.update(status='source_unavailable',error=str(exc)[:240],attempt_epoch=time.time())
    meta.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return record


def persist_sources(rows,db_path=None):
    """Append immutable source versions; the existing Supabase mirror syncs
    every application .db, including the original compressed RSS bytes."""
    path=Path(db_path or DB);path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path,timeout=30) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''CREATE TABLE IF NOT EXISTS historical_news_sources_v1 (
            source_id TEXT PRIMARY KEY, ipo_id INTEGER NOT NULL, source_url TEXT NOT NULL,
            raw_sha256 TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL,
            source_payload_json TEXT NOT NULL, raw_rss_gzip BLOB NOT NULL)''')
        for r in rows:
            if r.get('status')!='verified_rss_response':continue
            compressed=(ROOT/r['archive_path']).read_bytes();raw=gzip.decompress(compressed)
            if digest(raw)!=r['raw_sha256']:raise ValueError('Cannot persist changed RSS bytes')
            id=digest((r['source_url']+'\n'+r['raw_sha256']).encode())
            conn.execute('INSERT OR IGNORE INTO historical_news_sources_v1 VALUES (?,?,?,?,?,?,?)',
                (id,int(r['ipo_id']),r['source_url'],r['raw_sha256'],r['retrieved_at_utc'],json.dumps(r,ensure_ascii=False,sort_keys=True),compressed))
        return conn.execute('SELECT COUNT(*) FROM historical_news_sources_v1').fetchone()[0]


def archived_sources():
    """Read the durable database first, with an existing-file fallback."""
    sources={}
    if DB.exists():
        with sqlite3.connect(DB) as conn:
            for id,payload,compressed in conn.execute('SELECT source_id,source_payload_json,raw_rss_gzip FROM historical_news_sources_v1'):
                r=json.loads(payload);sources[(r['source_url'],r['raw_sha256'])]=(r,gzip.decompress(compressed),'sqlite:'+id)
    for path in RAW.glob('*.json'):
        r=json.loads(path.read_text(encoding='utf8'))
        if r.get('status')=='verified_rss_response' and (r['source_url'],r['raw_sha256']) not in sources:
            raw=gzip.decompress((ROOT/r['archive_path']).read_bytes())
            sources[(r['source_url'],r['raw_sha256'])]=(r,raw,r['archive_path'])
    for r,raw,reference in sorted(sources.values(),key=lambda x:x[0]['ipo_id']):
        if digest(raw)!=r['raw_sha256']:raise ValueError('Stored source checksum mismatch')
        yield r,raw,reference


def run():
    RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    examples=pd.read_csv(EXAMPLES,parse_dates=['decision_date'])
    jobs=[]
    for id,g in examples.groupby('ipo_id'):
        start=(g.decision_date.min()-pd.Timedelta(days=8)).tz_localize('Asia/Kolkata')
        end=(g.decision_date.max()+pd.Timedelta(days=1)).tz_localize('Asia/Kolkata')
        jobs.append((int(id),g.company_name.iloc[0],start,end))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,future in enumerate(as_completed([pool.submit(fetch,j) for j in jobs]),1):
            rows.append(future.result())
            if n%25==0 or n==len(jobs):
                print(f'Recovered {n}/{len(jobs)} issuer windows; {sum(bool(r.get("articles")) for r in rows)} nonempty; {sum(r.get("status")=="source_unavailable" for r in rows)} failed',flush=True)
                (OUT/'sources.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in sorted(rows,key=lambda r:r['ipo_id']))+'\n',encoding='utf8')
    summary=dict(planned_issuers=len(jobs),attempted=len(rows),verified=sum(r['status']=='verified_rss_response' for r in rows),
        nonempty=sum(bool(r.get('articles')) for r in rows),articles=sum(len(r.get('articles',[])) for r in rows),
        rate_limit_paused=blocked.is_set(),empty_is_not_no_news=True)
    summary['database_source_versions']=persist_sources(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    pd.DataFrame([dict(ipo_id=r['ipo_id'],company=r['company_name'],status=r['status'],articles=len(r.get('articles',[])),
                       missing='No eligible archived headlines recovered' if not r.get('articles') else '') for r in rows]).to_csv(OUT/'remaining_news_gaps.csv',index=False)
    print(json.dumps(summary),flush=True)
    return rows


if __name__=='__main__':run()
