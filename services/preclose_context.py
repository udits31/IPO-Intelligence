"""Read preserved pre-close context without changing issued model calls."""
import hashlib
import json
import math
import re
from db import ledger as sqlite3
import gzip
from contextlib import closing
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from services.point_in_time import timestamp

IST = ZoneInfo('Asia/Kolkata')
DB = Path(__file__).resolve().parents[1] / 'data/telemetry/snapshots.db'


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def restore_preclose_gmp(item, db_path=DB):
    """Use the last eligible saved quote for closed/listed IPO page context.

    A mutable live quote must not erase an earlier ledger observation. This
    lookup is display context only: it never edits a saved forecast or its inputs.
    """
    if not (item.get('category') in ('CLOSED', 'LISTED') or item.get('is_listed')
            or item.get('bidding_ended')):
        return item
    if not sqlite3.exists(db_path):
        return item
    company = item.get('company_name') or ''
    match = re.search(r'/ipo/[^/]+/(\d+)', item.get('url') or item.get('chittorgarh_url') or '')
    ipo_id = 'chitto_' + match[1] if match else ''
    records = []
    closing_dates = []
    with closing(sqlite3.connect(f'{Path(db_path).resolve().as_uri()}?mode=ro', uri=True, timeout=10)) as c:
        c.row_factory = sqlite3.Row
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        where = 'ipo_id=?' if ipo_id else 'LOWER(company_name)=LOWER(?)'
        identity = ipo_id or company
        if 'subscription_capture_v2' in tables:
            closing_dates.extend(r[0] for r in c.execute(
                f'SELECT DISTINCT closing_date FROM subscription_capture_v2 WHERE {where}', (identity,)) if r[0])
        for table in ('automatic_decision_v1', 'frozen_module1_forecast_v1'):
            if table not in tables:
                continue
            for raw in c.execute(f'SELECT * FROM {table} WHERE {where}', (identity,)):
                row = dict(raw)
                try:
                    encoded = row['input_json']
                    data = json.loads(encoded)
                except (TypeError, ValueError):
                    continue
                checksum = hashlib.sha256(encoded.encode()).hexdigest()
                if row.get('input_sha') and checksum != row['input_sha']:
                    continue
                stored_cutoff = timestamp(row.get('cutoff_at_ist') or data.get('application_cutoff_at_ist'))
                if stored_cutoff:
                    closing_dates.append(stored_cutoff.date().isoformat())
                quote = (data.get('evidence') or {}).get('sources', {}).get('gmp') or {}
                value = _number((data.get('features') or {}).get('gmp_pct'))
                if value is None:
                    value = _number(data.get('gmp_pct'))
                observed = timestamp(quote.get('fetched_at') or data.get('gmp_observed_at'))
                recorded = timestamp(row.get('recorded_at_ist'))
                if value is None or not observed or not recorded or observed > recorded:
                    continue
                records.append(dict(value=value, observed=observed, recorded=recorded,
                    source=quote.get('source'), family=quote.get('family'), url=quote.get('url'),
                    provenance=dict(table=table, row_id=row['id'], input_sha256=checksum,
                                    recorded_at_ist=recorded.isoformat())))
    calendar = item.get('calendar_evidence') or {}
    explicit_close = calendar.get('issue_close_date') or item.get('issue_close_date')
    if explicit_close:
        closing_dates = [str(explicit_close)[:10]]
    if not closing_dates:
        from scrapers.live_ipo_scraper import parse_date_interval
        _, closed = parse_date_interval(item.get('date_range') or '', datetime.now(IST).year)
        if closed and closed.isoformat() != str(item.get('listing_date'))[:10]:
            closing_dates.append(closed.isoformat())
    dates = []
    for value in closing_dates:
        try:
            dates.append(datetime.fromisoformat(value).date())
        except (TypeError, ValueError):
            pass
    if not dates:
        return item
    cutoff = datetime.combine(max(dates), time(17), IST)
    candidates = [r for r in records if r['observed'] < cutoff and r['recorded'] < cutoff]
    current_time = timestamp(item.get('gmp_observed_at'))
    current_value = _number(item.get('gmp_pct'))
    if current_time and current_time < cutoff and current_value is not None:
        candidates.append(dict(value=current_value, observed=current_time, recorded=current_time,
            source=item.get('gmp_source'), family=item.get('gmp_source_family'),
            url=item.get('gmp_source_url'), rupees=item.get('gmp_rs'),
            provenance=dict(table='retained_pre_close_cache', recorded_at_ist=current_time.isoformat())))
    if not candidates:
        return item
    quote = max(candidates, key=lambda r: (r['observed'], r['recorded']))
    result = dict(item)
    if current_time and current_time >= cutoff:
        result['later_gmp_context'] = {k: item.get(k) for k in
            ('gmp_pct', 'gmp_rs', 'gmp_observed_at', 'gmp_source', 'gmp_source_url')}
    result.update(gmp_pct=quote['value'], gmp_rs=quote.get('rupees'),
        gmp_source='saved_pre_close_observation', gmp_basis='saved_pre_close',
        gmp_observed_at=quote['observed'].isoformat(), gmp_source_url=quote.get('url'),
        gmp_source_family=quote.get('family'), gmp_is_stale=False,
        gmp_preclose_provenance={**quote['provenance'], 'cutoff_at_ist': cutoff.isoformat(),
            'original_source': quote.get('source'), 'publisher_time_verified': False})
    return result


def archive_offer_context(item, raw, source_url, db_path=DB, *, source_fetched_at=None):
    """Append an identity-checked static offer source; keep it in the DB mirror."""
    from bs4 import BeautifulSoup
    from services.historical_bid_service import name_key
    from services.live_offer_facts import parse_offer, OFFER_PARSER_VERSION
    soup=BeautifulSoup(raw,'html.parser');title=soup.title
    expected=name_key(item['company_name'])
    title_name=name_key(title.get_text()) if title else ''
    legal_name_in_body=expected in name_key(soup.get_text(' ',strip=True))
    title_prefix=' '.join(expected.split()[:2])
    if not title or not (expected in title_name or (title_prefix in title_name and legal_name_in_body)):
        raise ValueError('Offer source company identity does not match')
    facts=parse_offer(raw.decode('utf8') if isinstance(raw,bytes) else raw,item['issue_price'])
    if facts.get('offer_cap_price')!=float(item['issue_price']):
        raise ValueError('Offer source has no matching cap price')
    content=raw.encode() if isinstance(raw,str) else raw
    sha=hashlib.sha256(content).hexdigest()
    computed=datetime.now(IST).isoformat()
    record=dict(company_name=item['company_name'],fetch_timestamp=source_fetched_at or computed,
        source_url=source_url,sha256=sha,identity_verified=True,
        identity_checks=dict(cap_reference=float(item['issue_price'])),
        offer={k:facts.get(k) for k in ('ofs_ratio','fresh_issue_cr','ofs_issue_cr','total_issue_cr','ofs_derivation')},
        reservation={'Retail':{'pct':facts.get('retail_quota_pct')}},
        retail_quota_basis=facts.get('retail_quota_basis','net_offer'),
        retail_quota_total_pct=facts.get('retail_quota_total_pct'),
        lot_size_shares=facts.get('lot_size_shares'),objects_capex_ratio=facts.get('objects_capex_ratio'),
        proceeds_uses=facts.get('proceeds_uses'),objects_capex_cr=facts.get('objects_capex_cr'),
        objects_capex_status=facts.get('objects_capex_status'),
        parser_version=OFFER_PARSER_VERSION,evidence_class='recovered_static_offer_terms')
    with closing(sqlite3.connect(db_path,timeout=10)) as c:
        c.execute('''CREATE TABLE IF NOT EXISTS offer_context_archive_v1 (
            id INTEGER PRIMARY KEY, company_name TEXT NOT NULL, source_url TEXT NOT NULL,
            fetched_at_ist TEXT NOT NULL, source_sha256 TEXT NOT NULL, raw_html_gzip BLOB NOT NULL,
            facts_json TEXT NOT NULL, UNIQUE(company_name,source_sha256))''')
        c.execute('''CREATE TABLE IF NOT EXISTS offer_context_parse_v1 (
            id INTEGER PRIMARY KEY,company_name TEXT NOT NULL,source_sha256 TEXT NOT NULL,
            parser_version TEXT NOT NULL,computed_at_ist TEXT NOT NULL,facts_json TEXT NOT NULL,
            UNIQUE(company_name,source_sha256,parser_version))''')
        c.execute('INSERT OR IGNORE INTO offer_context_archive_v1(company_name,source_url,fetched_at_ist,source_sha256,raw_html_gzip,facts_json) VALUES(?,?,?,?,?,?)',
                  (item['company_name'],source_url,record['fetch_timestamp'],sha,gzip.compress(content),json.dumps(record)))
        c.execute('INSERT OR IGNORE INTO offer_context_parse_v1(company_name,source_sha256,parser_version,computed_at_ist,facts_json) VALUES(?,?,?,?,?)',
                  (item['company_name'],sha,OFFER_PARSER_VERSION,computed,json.dumps(record)))
        c.commit()
    if Path(db_path).resolve()==DB.resolve():
        from services.sqlite_supabase_sync import request_sync
        request_sync('recovered_static_offer_terms')
    return record


def refresh_cached_offer_extractions(item, db_path=DB):
    """Upgrade extraction from saved HTML without a new fetch or a fake clock."""
    if not sqlite3.exists(db_path):return
    from services.live_offer_facts import OFFER_PARSER_VERSION
    with closing(sqlite3.connect(f'{Path(db_path).resolve().as_uri()}?mode=ro',uri=True,timeout=10)) as c:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name='offer_context_archive_v1'").fetchone():return
        rows=c.execute('SELECT source_url,fetched_at_ist,source_sha256,raw_html_gzip FROM offer_context_archive_v1 WHERE company_name=? ORDER BY id DESC',
                       (item.get('company_name'),)).fetchall()
        has_parses=c.execute("SELECT 1 FROM sqlite_master WHERE name='offer_context_parse_v1'").fetchone()
        existing={r[0] for r in c.execute('SELECT source_sha256 FROM offer_context_parse_v1 WHERE company_name=? AND parser_version=?',
                  (item.get('company_name'),OFFER_PARSER_VERSION))} if has_parses else set()
    for url,fetched,sha,compressed in rows:
        if sha in existing:continue
        raw=gzip.decompress(compressed)
        if hashlib.sha256(raw).hexdigest()!=sha or not raw.lstrip().startswith(b'<'):continue
        try:archive_offer_context(item,raw,url,db_path,source_fetched_at=fetched)
        except ValueError:continue


def attach_offer_context(item, db_path=DB):
    """Join verified recovered offer terms to the page, including after listing."""
    from services.locked_model_configuration import ROOT, recovered_record
    from services.historical_bid_service import name_key
    path = ROOT / 'data/telemetry/prospective_offer_facts' / (name_key(item.get('company_name', '')) + '.json')
    record = None
    verified_records=[]
    stored_verified = False
    if sqlite3.exists(db_path):
        with closing(sqlite3.connect(f'{Path(db_path).resolve().as_uri()}?mode=ro',uri=True,timeout=10)) as c:
            if c.execute("SELECT 1 FROM sqlite_master WHERE name='offer_context_archive_v1'").fetchone():
                has_parses=c.execute("SELECT 1 FROM sqlite_master WHERE name='offer_context_parse_v1'").fetchone()
                from services.live_offer_facts import OFFER_PARSER_VERSION
                for raw,sha,encoded in c.execute('SELECT raw_html_gzip,source_sha256,facts_json FROM offer_context_archive_v1 WHERE company_name=? ORDER BY id DESC',(item.get('company_name'),)):
                    if hashlib.sha256(gzip.decompress(raw)).hexdigest()==sha:
                        if has_parses:
                            parsed=c.execute('SELECT facts_json FROM offer_context_parse_v1 WHERE company_name=? AND source_sha256=? AND parser_version=?',
                                (item.get('company_name'),sha,OFFER_PARSER_VERSION)).fetchone()
                            if parsed:encoded=parsed[0]
                        candidate=json.loads(encoded)
                        if candidate.get('sha256')==sha and abs((_number((candidate.get('identity_checks') or {}).get('cap_reference')) or 0)-(_number(item.get('issue_price')) or 0))<=.01:
                            verified_records.append(candidate)
                if verified_records:
                    reference_total=_number(item.get('issue_size_cr'))
                    def quality(r):
                        offer=r.get('offer') or {}
                        total=_number(offer.get('total_issue_cr', (offer.get('total') or {}).get('cr')))
                        distance=abs(total-reference_total)/max(1,reference_total) if total is not None and reference_total is not None else 0
                        decimals=len(str(total).split('.')[-1]) if total is not None and total%1 else 0
                        return (distance>.01,-decimals,distance)
                    record=min(verified_records,key=quality);stored_verified=True
    if not stored_verified:
        record = json.loads(path.read_text()) if path.exists() else recovered_record(item.get('company_name', ''))
    if not record or not record.get('identity_verified'):
        return item
    reference = _number((record.get('identity_checks') or {}).get('cap_reference'))
    price = _number(item.get('issue_price'))
    if reference is None or price is None or abs(reference-price)>.01:
        return item
    source = record.get('source_file')
    if stored_verified:
        pass
    elif source:
        archive = ROOT / source
        if not archive.exists():return item
        content = archive.read_bytes()
        digests = [hashlib.sha256(content).hexdigest()]
        if archive.suffix == '.gz':
            digests.append(hashlib.sha256(gzip.decompress(content)).hexdigest())
        if record.get('sha256') not in digests:return item
    else:
        # Existing live records refer to the archived financial-page bytes.
        match = re.search(r'/(\d+)/?$', item.get('url') or '')
        archives = list((ROOT/'data/telemetry/prospective_financials').glob(f'chitto_{match[1]}_*.html.gz')) if match else []
        if not any(hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==record.get('sha256') for p in archives):
            return item
    offer = record.get('offer') or {}
    retail = (record.get('reservation') or {}).get('Retail') or {}
    result = dict(item, offer_facts=record)
    fields = dict(promoter_ofs_ratio=offer.get('ofs_ratio'),
        fresh_issue_cr=offer.get('fresh_issue_cr', (offer.get('fresh') or {}).get('cr')),
        ofs_issue_cr=offer.get('ofs_issue_cr', (offer.get('ofs') or {}).get('cr')),
        retail_quota_pct=retail.get('pct'), original_lot_size=record.get('lot_size_shares'),
        objects_capex_ratio=record.get('objects_capex_ratio'))
    result.update({k:v for k,v in fields.items() if _number(v) is not None})
    result['retail_quota_basis'] = record.get('retail_quota_basis', 'total_offer')
    result['retail_quota_total_pct'] = record.get('retail_quota_total_pct')
    result['proceeds_uses'] = record.get('proceeds_uses')
    result['objects_capex_status'] = record.get('objects_capex_status')
    proceeds_sources=[r for r in verified_records if r.get('proceeds_uses')]
    if proceeds_sources:
        # Prefer the quantified table, retaining the independent corroboration.
        proceeds_source=max(proceeds_sources,key=lambda r:sum(u.get('amount_cr') is not None for u in r['proceeds_uses']))
        result['proceeds_uses']=proceeds_source['proceeds_uses']
        distinct_sources={}
        for r in proceeds_sources:
            distinct_sources.setdefault(r.get('source_url'),{k:r.get(k) for k in ('source_url','sha256','fetch_timestamp')})
        result['proceeds_provenance']=list(distinct_sources.values())
        if result.get('objects_capex_ratio') is None:
            result['objects_capex_status']=proceeds_source.get('objects_capex_status')
    # A newer rounded table may add capex while the earlier verified offer
    # source has more precise total/Fresh/OFS amounts. Preserve that precision.
    capex_source=next((r for r in verified_records if _number(r.get('objects_capex_cr')) is not None),None)
    if capex_source and (_number(result.get('fresh_issue_cr')) or 0)>0:
        result['objects_capex_ratio']=float(capex_source['objects_capex_cr'])/float(result['fresh_issue_cr'])
        result['proceeds_uses']=capex_source.get('proceeds_uses')
        result['capex_provenance']={k:capex_source.get(k) for k in ('source_url','sha256','fetch_timestamp','parser_version')}
    return result


def saved_book_total(source_sha256, db_path=DB):
    """Recover the total from the same immutable source used by the saved call."""
    if not source_sha256 or not sqlite3.exists(db_path):return None
    with closing(sqlite3.connect(f'{Path(db_path).resolve().as_uri()}?mode=ro',uri=True,timeout=10)) as c:
        c.row_factory=sqlite3.Row
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name='subscription_capture_v2'").fetchone():return None
        row=c.execute('SELECT * FROM subscription_capture_v2 WHERE raw_sha256=? AND status=? ORDER BY id DESC LIMIT 1',
                      (source_sha256,'verified')).fetchone()
    if row is None:return None
    raw=gzip.decompress(row['raw_html_gzip'])
    if hashlib.sha256(raw).hexdigest()!=source_sha256:return None
    from services.live_offer_facts import total_book
    total=total_book(raw,row['source_family'])
    if not total:return None
    return {**total,'raw_sha256':source_sha256,'fetched_at_ist':row['fetched_at_ist'],
            'source_family':row['source_family'],'source_url':row['subscription_url']}
