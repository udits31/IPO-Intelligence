"""Canonical training projection over append-only subscription capture history."""
import gzip,hashlib,json,math
from db import ledger as sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'data/telemetry/snapshots.db'
IST=ZoneInfo('Asia/Kolkata')


def install_view(conn):
    conn.execute('''CREATE VIEW IF NOT EXISTS subscription_training_capture_v1 AS
        SELECT * FROM (SELECT c.*,ROW_NUMBER() OVER (
            PARTITION BY ipo_id,source_family,slot_key,capture_phase
            ORDER BY COALESCE(fetched_at_ist,attempted_at_ist) DESC,id DESC) AS training_rank
        FROM subscription_capture_v2 c WHERE status='verified' AND capture_phase='pre_deadline'
        AND qib_x>=0 AND qib_x<1000000 AND nii_x>=0 AND nii_x<1000000
        AND retail_x>=0 AND retail_x<1000000 AND LENGTH(raw_sha256)=64 AND raw_html_gzip IS NOT NULL)
        WHERE training_rank=1''')


def publisher(raw,family):
    if family=='nse_exchange':
        try:return datetime.strptime(json.loads(raw).get('updateTime',''),'Updated as on %d-%b-%Y %H:%M:%S').replace(tzinfo=IST)
        except (ValueError,TypeError):return None
    from scrapers.live_ipo_scraper import parse_subscription_published_at
    value=parse_subscription_published_at(raw.decode('utf8',errors='replace'))
    return datetime.fromisoformat(value) if value else None


def training_captures(day=None,*,asof=None,db_path=DB):
    """Apply the time cutoff BEFORE deduplication; verify actual source bytes.
    No attempt or source failure becomes a synthetic zero-valued training row.
    """
    cutoff=None
    if asof:
        cutoff=datetime.fromisoformat(str(asof))
        if cutoff.tzinfo is None:raise ValueError('Timezone-aware training cutoff required')
        cutoff=cutoff.astimezone(IST)
    with sqlite3.connect('file:'+Path(db_path).as_posix()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        records=[dict(r) for r in c.execute("SELECT * FROM subscription_capture_v2 WHERE status='verified' AND capture_phase='pre_deadline'"+
            (' AND substr(slot_key,1,10)=?' if day else '')+' ORDER BY COALESCE(fetched_at_ist,attempted_at_ist) DESC,id DESC', (day,) if day else ())]
    kept={};rejected=[]
    for r in records:
        key=(r['ipo_id'],r['source_family'],r['slot_key'],r['capture_phase'])
        if key in kept:continue
        try:
            fetched=datetime.fromisoformat(r['fetched_at_ist'] or r['attempted_at_ist'])
            if fetched.tzinfo is None or (cutoff and fetched>cutoff):continue
            if not all(math.isfinite(float(r[k])) and 0<=float(r[k])<1000000 for k in ('qib_x','nii_x','retail_x')):raise ValueError('Invalid category values')
            raw=gzip.decompress(r['raw_html_gzip'])
            if hashlib.sha256(raw).hexdigest()!=r['raw_sha256']:raise ValueError('Source checksum mismatch')
            published=publisher(raw,r['source_family'])
            if published and (published>fetched or (cutoff and published>cutoff)):raise ValueError('Future source clock')
            if key not in kept:
                kept[key]={k:v for k,v in r.items() if k!='raw_html_gzip'}
                kept[key]['source_published_at_ist']=published.isoformat() if published else None
        except (ValueError,TypeError,OSError,KeyError) as exc:rejected.append(dict(capture_id=r['id'],reason=str(exc)))
    return dict(records=sorted(kept.values(),key=lambda r:(r['slot_key'],r['ipo_id'],r['source_family'])),
        verified_raw_rows=len(records),training_rows=len(kept),rejected=rejected,
        protocol='One latest eligible complete reading per IPO/source/slot/phase; time and source checks precede selection. Raw rows retained.')


def existing_evidence(ipo_id,slot,source,db_path=DB):
    """Resume a partially completed slot without fetching its completed source."""
    with sqlite3.connect(db_path,timeout=5) as c:
        c.row_factory=sqlite3.Row
        row=c.execute('SELECT * FROM subscription_capture_v2 WHERE ipo_id=? AND slot_key=? AND source_family=? ORDER BY id DESC LIMIT 1',
                      (ipo_id,slot,source)).fetchone()
    if not row:return None
    r=dict(row);raw=gzip.decompress(r['raw_html_gzip']) if r.get('raw_html_gzip') else None
    if raw and hashlib.sha256(raw).hexdigest()!=r.get('raw_sha256'):raise ValueError('Stored capture checksum mismatch')
    p=publisher(raw,source) if raw else None
    result={k:r.get(k) for k in ('status','http_status','fetched_at_ist','raw_sha256','error','subscription_url','parser_version')}
    result.update(values=dict(zip(('qib_subscription','hni_subscription','retail_subscription'),(r.get('qib_x'),r.get('nii_x'),r.get('retail_x')))),
        raw_html=raw.decode('utf8') if raw else None,source_displayed_at_ist=p.isoformat() if p else None)
    if source=='nse_exchange' and raw:result.update(nse_symbol=json.loads(raw).get('symbol'),series='EQ')
    return result
