"""
Snapshot Telemetry Service
==========================
The append-only subscription_capture_v2 table records every scheduled fetch
attempt, including unavailable pages and errors, in data/telemetry/snapshots.db.
The older pre_bidding_snapshots table is retained only for audit: historical
rows may contain stale or unsourced numbers and are not evaluation evidence.

Concurrency Model:
- PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
  Accurately documented: SQLite still has a single writer at a time. WAL mode
  allows readers to query concurrently without blocking the writer, and busy_timeout
  makes brief lock contention tolerable without throwing immediate OperationalErrors.

Capture cadence: IST-aligned 15-minute polls from 10:00 to 14:29 and 5-minute
polls from 14:30 through 16:55, weekdays. Collection stops at 17:00.
A delayed response arriving after the closing cutoff is audit evidence only. Each row has source
HTML and SHA-256; fetched_at is our fetch clock, not the publisher's timestamp.
"""

import os
import re
import gzip
import hashlib
import json
from db import ledger as sqlite3
import logging
import threading
from contextlib import closing
from datetime import datetime, timezone, date, time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_DIR = os.path.join('data', 'telemetry')
DB_PATH = os.path.join(DB_DIR, 'snapshots.db')
RECOVERED_OBSERVATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'data', 'recovered_bid_observations.json'
)
IST = ZoneInfo('Asia/Kolkata')


def extract_ipo_id(data_or_name: Any, fallback_name: str = '') -> str:
    """
    Extracts a robust, deterministic unique identifier for an IPO.
    Precedence:
    1. Direct numeric/string id in dictionary (id or ipo_id)
    2. Chittorgarh URL numeric ID or slug (/ipo/.../<id>/)
    3. NSE Symbol (e.g. nse_ORIENT)
    4. BSE Scrip Code (e.g. bse_543412)
    5. Clean alphanumeric slug of company name (co_<slug>)
    """
    if isinstance(data_or_name, dict):
        raw_id = data_or_name.get('id') or data_or_name.get('ipo_id')
        if raw_id is not None and str(raw_id).strip() and str(raw_id).strip() != 'None':
            return str(raw_id).strip()

        url = data_or_name.get('chittorgarh_url') or data_or_name.get('url') or ''
        if url:
            m_num = re.search(r'/(\d+)/?$', url)
            if m_num:
                return f"chitto_{m_num.group(1)}"
            m_slug = re.search(r'/ipo/([^/]+)/', url)
            if m_slug:
                return f"chitto_{m_slug.group(1)}"

        nse = data_or_name.get('nse_symbol')
        if nse and str(nse).strip() and str(nse).strip() != 'None':
            return f"nse_{str(nse).strip().upper()}"

        bse = data_or_name.get('bse_code') or data_or_name.get('bse_scrip_code')
        if bse and str(bse).strip() and str(bse).strip() != 'None':
            return f"bse_{str(bse).strip()}"

        name = data_or_name.get('company_name') or fallback_name or 'unknown_ipo'
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower().strip()).strip('_')
        return f"co_{slug}"

    raw_str = str(data_or_name).strip()
    if '/ipo/' in raw_str:
        m_num = re.search(r'/(\d+)/?$', raw_str)
        if m_num:
            return f"chitto_{m_num.group(1)}"
        m_slug = re.search(r'/ipo/([^/]+)/', raw_str)
        if m_slug:
            return f"chitto_{m_slug.group(1)}"

    slug = re.sub(r'[^a-z0-9]+', '_', raw_str.lower()).strip('_')
    return f"co_{slug}" if slug else 'co_target'


def parse_closing_date(raw_date_range: Optional[str], default_year: int = 2026) -> Optional[date]:
    """Parses closing date from Chittorgarh date range strings."""
    if not raw_date_range:
        return None
    raw = str(raw_date_range).strip()
    m_iso = re.search(r'(?:to|–|—|-)\s*(\d{4}-\d{2}-\d{2})', raw)
    if m_iso:
        try:
            return datetime.strptime(m_iso.group(1), '%Y-%m-%d').date()
        except Exception:
            pass
    m_month = re.search(r'[-–—]\s*(\d{1,2})\s+([A-Za-z]{3})(?:\s+(\d{4}))?', raw)
    if m_month:
        day = int(m_month.group(1))
        month_str = m_month.group(2)
        year = int(m_month.group(3)) if m_month.group(3) else default_year
        try:
            dt = datetime.strptime(f'{day} {month_str} {year}', '%d %b %Y')
            return dt.date()
        except Exception:
            pass
    from scrapers.live_ipo_scraper import parse_date_interval
    return parse_date_interval(raw, default_year)[1]


def application_forecast_window(item: Dict[str, Any], clock: datetime) -> Dict[str, Any]:
    """The offer stays open between daily sessions, until its final-day cutoff."""
    from scrapers.live_ipo_scraper import parse_date_interval
    clock = clock.astimezone(IST)
    opening, closing = parse_date_interval(item.get('date_range') or '', clock.year)
    if opening is None or closing is None or opening > closing:
        return {'is_open': False, 'cutoff_at_ist': None}
    starts = datetime.combine(opening, time(10, 0), tzinfo=IST)
    cutoff = datetime.combine(closing, time(17, 0), tzinfo=IST)
    return {'is_open': (item.get('category') == 'OPEN'
                        and not item.get('is_listed') and starts <= clock < cutoff),
            'opens_at_ist': starts.isoformat(), 'cutoff_at_ist': cutoff.isoformat()}


def determine_snapshot_stage(closing_date: Optional[date], now_ist: datetime) -> Tuple[str, str]:
    """
    Returns (snapshot_stage, hour_bucket).
    On closing day, establishes dedicated late-session checkpoints:
      T_MINUS_120 (14:45 - 15:15 IST) -> YYYY-MM-DD_T120
      T_MINUS_60  (15:45 - 16:15 IST) -> YYYY-MM-DD_T60
      T_MINUS_30  (16:20 - 16:40 IST) -> YYYY-MM-DD_T30
      FINAL_CLOSE (16:50 - 17:15 IST) -> YYYY-MM-DD_CLOSE
    Otherwise, tags standard hourly snapshots:
      HOURLY -> YYYY-MM-DD_HH
    """
    date_str = now_ist.strftime('%Y-%m-%d')
    time_min = now_ist.hour * 60 + now_ist.minute
    is_closing_day = (closing_date is not None and now_ist.date() == closing_date)

    if is_closing_day:
        if 14 * 60 + 45 <= time_min <= 15 * 60 + 15:
            return 'T_MINUS_120', f"{date_str}_T120"
        if 15 * 60 + 45 <= time_min <= 16 * 60 + 15:
            return 'T_MINUS_60', f"{date_str}_T60"
        if 16 * 60 + 20 <= time_min <= 16 * 60 + 40:
            return 'T_MINUS_30', f"{date_str}_T30"
        if 16 * 60 + 50 <= time_min <= 17 * 60 + 15:
            return 'FINAL_CLOSE', f"{date_str}_CLOSE"

    return 'HOURLY', now_ist.strftime('%Y-%m-%d_%H')


def init_db(db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        
        # Check if table needs schema migration (from company_name to ipo_id uniqueness)
        try:
            cols = [c[1] for c in conn.execute("PRAGMA table_info(pre_bidding_snapshots);").fetchall()]
            if cols and 'ipo_id' not in cols:
                backup = 'pre_bidding_snapshots_legacy_' + datetime.now(IST).strftime('%Y%m%d%H%M%S')
                logger.warning('Preserving old snapshot schema as %s', backup)
                conn.execute(f'ALTER TABLE pre_bidding_snapshots RENAME TO {backup}')
        except Exception as mig_err:
            logger.warning("Schema check notice: %s", mig_err)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS pre_bidding_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ipo_id TEXT NOT NULL,
                company_name TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                hour_bucket TEXT NOT NULL,
                snapshot_stage TEXT NOT NULL DEFAULT 'HOURLY',
                days_to_close REAL,
                gmp_pct REAL,
                gmp_rs REAL,
                retail_x REAL,
                nii_x REAL,
                qib_x REAL,
                promoter_ofs_ratio REAL,
                objects_capex_ratio REAL,
                issue_size_cr REAL,
                india_vix REAL,
                source_channel TEXT NOT NULL,
                final_qib_subscription REAL DEFAULT NULL,
                listing_return_pct REAL DEFAULT NULL,
                UNIQUE(ipo_id, hour_bucket)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscription_capture_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ipo_id TEXT NOT NULL,
                company_name TEXT NOT NULL,
                ipo_url TEXT NOT NULL,
                subscription_url TEXT,
                attempted_at_ist TEXT NOT NULL,
                fetched_at_ist TEXT,
                slot_key TEXT NOT NULL,
                closing_date TEXT,
                snapshot_stage TEXT NOT NULL,
                status TEXT NOT NULL,
                http_status INTEGER,
                qib_x REAL,
                nii_x REAL,
                retail_x REAL,
                raw_sha256 TEXT,
                raw_html_gzip BLOB,
                error TEXT,
                parser_version TEXT NOT NULL DEFAULT 'subscription_table_v2'
            ,source_family TEXT NOT NULL DEFAULT 'chittorgarh_investorgain'
            ,capture_phase TEXT NOT NULL DEFAULT 'pre_deadline'
            )
        """)
        columns = {column[1] for column in conn.execute('PRAGMA table_info(subscription_capture_v2)')}
        if 'source_family' not in columns:
            conn.execute("ALTER TABLE subscription_capture_v2 ADD COLUMN source_family TEXT NOT NULL DEFAULT 'chittorgarh_investorgain'")
        if 'capture_phase' not in columns:
            conn.execute("ALTER TABLE subscription_capture_v2 ADD COLUMN capture_phase TEXT NOT NULL DEFAULT 'pre_deadline'")
        conn.execute('CREATE INDEX IF NOT EXISTS idx_capture_ipo_time ON subscription_capture_v2(ipo_id, attempted_at_ist)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_capture_slot ON subscription_capture_v2(slot_key)')
        from services.subscription_training import install_view
        install_view(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS automatic_decision_v1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ipo_id TEXT NOT NULL,
                company_name TEXT NOT NULL,
                slot_key TEXT NOT NULL,
                snapshot_stage TEXT NOT NULL,
                decision_kind TEXT NOT NULL,
                recorded_at_ist TEXT NOT NULL,
                source_asof_ist TEXT,
                source_family TEXT,
                source_status TEXT NOT NULL,
                model_version TEXT,
                decision_engine_version TEXT,
                input_json TEXT NOT NULL,
                verdict TEXT,
                conviction TEXT,
                positive_listing_probability REAL,
                rationale TEXT,
                error TEXT,
                UNIQUE(ipo_id, slot_key, decision_kind)
            )
        """)
        decision_columns = {column[1] for column in conn.execute('PRAGMA table_info(automatic_decision_v1)')}
        if 'decision_engine_version' not in decision_columns:
            conn.execute('ALTER TABLE automatic_decision_v1 ADD COLUMN decision_engine_version TEXT')
        conn.execute("""UPDATE automatic_decision_v1
                        SET decision_engine_version = 'prebid_rules_2026_10_v1', model_version = NULL
                        WHERE decision_kind = 'provisional_scorecard'
                          AND decision_engine_version IS NULL""")
        conn.execute('CREATE INDEX IF NOT EXISTS idx_decision_ipo_time ON automatic_decision_v1(ipo_id, recorded_at_ist)')
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prelisting_replay_v1 (
                ipo_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                recorded_at_ist TEXT NOT NULL,
                listing_date TEXT NOT NULL,
                observed_at_ist TEXT NOT NULL,
                evidence_found_at_ist TEXT NOT NULL,
                evidence_status TEXT NOT NULL,
                source_family TEXT NOT NULL,
                evidence_sha256 TEXT,
                qib_x REAL NOT NULL,
                nii_x REAL NOT NULL,
                retail_x REAL NOT NULL,
                model_version TEXT NOT NULL,
                positive_listing_probability REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prelisting_outcome_forecast_v1 (
                ipo_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                recorded_at_ist TEXT NOT NULL,
                listing_date TEXT NOT NULL,
                capture_id INTEGER NOT NULL,
                capture_phase TEXT NOT NULL,
                captured_at_ist TEXT NOT NULL,
                source_family TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                qib_x REAL NOT NULL,
                nii_x REAL NOT NULL,
                retail_x REAL NOT NULL,
                model_version TEXT NOT NULL,
                positive_listing_probability REAL NOT NULL,
                FOREIGN KEY (capture_id) REFERENCES subscription_capture_v2(id)
            )
        """)
        conn.commit()


class SnapshotService:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def record_snapshot(self,
                        company_name: str,
                        ipo_id: Optional[str] = None,
                        hour_bucket: Optional[str] = None,
                        snapshot_stage: str = 'HOURLY',
                        gmp_pct: Optional[float] = None,
                        gmp_rs: Optional[float] = None,
                        retail_x: Optional[float] = None,
                        nii_x: Optional[float] = None,
                        qib_x: Optional[float] = None,
                        promoter_ofs_ratio: Optional[float] = None,
                        objects_capex_ratio: Optional[float] = None,
                        issue_size_cr: Optional[float] = None,
                        india_vix: Optional[float] = None,
                        days_to_close: Optional[float] = None,
                        source_channel: str = 'query_dedup') -> bool:
        """
        Records a single snapshot if one does not already exist for this (ipo_id, hour_bucket).
        Returns True if a new row was inserted, False if deduplicated.
        """
        if not company_name:
            return False

        clean_id = ipo_id or extract_ipo_id(company_name)
        now_ist = datetime.now(IST)
        observed_at = now_ist.isoformat()
        bucket = hour_bucket or now_ist.strftime('%Y-%m-%d_%H')

        try:
            with closing(sqlite3.connect(self.db_path, timeout=5.0)) as conn:
                conn.execute("PRAGMA busy_timeout=5000;")
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO pre_bidding_snapshots (
                        ipo_id, company_name, observed_at, hour_bucket, snapshot_stage,
                        days_to_close, gmp_pct, gmp_rs, retail_x, nii_x, qib_x,
                        promoter_ofs_ratio, objects_capex_ratio, issue_size_cr,
                        india_vix, source_channel
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    clean_id, company_name.strip(), observed_at, bucket, snapshot_stage,
                    days_to_close, gmp_pct, gmp_rs, retail_x, nii_x, qib_x,
                    promoter_ofs_ratio, objects_capex_ratio, issue_size_cr,
                    india_vix, source_channel
                ))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as exc:
            logger.warning("Snapshot logging failed for %s (%s): %s", clean_id, company_name, exc)
            return False

    def record_scheduled_snapshots(self, active_ipos: List[Dict[str, Any]], macro: Dict[str, Any]) -> int:
        """
        Records snapshots across all offerings that are ACTIVELY BIDDING.
        Skips closed, listed, and non-bidding offerings.
        Determines whether each active IPO is on its final bidding day to record
        fixed closing checkpoints (T-120, T-60, T-30, FINAL_CLOSE).
        """
        inserted = 0
        vix = macro.get('india_vix')
        now_ist = datetime.now(IST)

        for item in active_ipos:
            cat = str(item.get('category', '')).strip().upper()
            is_listed = bool(item.get('is_listed')) or cat == 'LISTED'
            bidding_ended = bool(item.get('bidding_ended')) or cat in ['CLOSED', 'LISTED'] or is_listed

            # Strictly skip non-active deals
            if cat != 'OPEN' or bidding_ended:
                continue

            clean_id = extract_ipo_id(item)
            cname = item.get('company_name', clean_id)
            closing_dt = parse_closing_date(item.get('date_range'))
            stage, bucket = determine_snapshot_stage(closing_dt, now_ist)

            days_to_close = None
            if closing_dt:
                days_to_close = max(0.0, (closing_dt - now_ist.date()).days)

            success = self.record_snapshot(
                company_name=cname,
                ipo_id=clean_id,
                hour_bucket=bucket,
                snapshot_stage=stage,
                gmp_pct=item.get('gmp_pct'),
                gmp_rs=item.get('gmp_rs'),
                retail_x=item.get('retail_subscription'),
                nii_x=item.get('hni_subscription'),
                qib_x=item.get('qib_subscription'),
                promoter_ofs_ratio=item.get('promoter_ofs_ratio'),
                objects_capex_ratio=item.get('objects_capex_ratio'),
                issue_size_cr=item.get('issue_size_cr'),
                india_vix=vix,
                days_to_close=days_to_close,
                source_channel=f"harvester:{stage.lower()}"
            )
            if success:
                inserted += 1

        return inserted

    def get_snapshot_count(self) -> int:
        try:
            with closing(sqlite3.connect(self.db_path, timeout=5.0)) as conn:
                res = conn.execute("SELECT COUNT(*) FROM pre_bidding_snapshots;").fetchone()
                return int(res[0]) if res else 0
        except Exception:
            return 0


snapshot_service = SnapshotService()


def record_snapshot(data_or_name: Any,
                    macro_or_gmp: Any = None,
                    source: str = 'query_dedup',
                    **kwargs) -> bool:
    """Convenience wrapper accepting either a deal dictionary or individual fields."""
    if isinstance(data_or_name, dict):
        data = data_or_name
        macro = macro_or_gmp if isinstance(macro_or_gmp, dict) else {}
        clean_id = extract_ipo_id(data)
        now_ist = datetime.now(IST)
        closing_dt = parse_closing_date(data.get('date_range'))
        stage, bucket = determine_snapshot_stage(closing_dt, now_ist)
        days_to_close = max(0.0, (closing_dt - now_ist.date()).days) if closing_dt else None

        return snapshot_service.record_snapshot(
            company_name=data.get('company_name', clean_id),
            ipo_id=clean_id,
            hour_bucket=bucket,
            snapshot_stage=stage,
            gmp_pct=data.get('gmp_pct'),
            gmp_rs=data.get('gmp_rs'),
            retail_x=data.get('retail_subscription'),
            nii_x=data.get('hni_subscription'),
            qib_x=data.get('qib_subscription'),
            promoter_ofs_ratio=data.get('promoter_ofs_ratio'),
            objects_capex_ratio=data.get('objects_capex_ratio'),
            issue_size_cr=data.get('issue_size_cr'),
            india_vix=macro.get('india_vix') if macro else data.get('india_vix'),
            days_to_close=days_to_close,
            source_channel=source
        )

    clean_id = kwargs.get('ipo_id') or extract_ipo_id(str(data_or_name))
    return snapshot_service.record_snapshot(
        company_name=str(data_or_name),
        ipo_id=clean_id,
        gmp_pct=macro_or_gmp if isinstance(macro_or_gmp, (int, float)) else kwargs.get('gmp_pct'),
        source_channel=source,
        **kwargs
    )


def record_scheduled_snapshots(active_ipos: Optional[List[Dict[str, Any]]] = None,
                               macro: Optional[Dict[str, Any]] = None,
                               **kwargs) -> int:
    """Convenience wrapper that automatically loads active offerings from cache if not provided."""
    if active_ipos is None:
        try:
            from scrapers.live_ipo_scraper import load_scraped_cache, fetch_live_macro_regime
            cache = load_scraped_cache()
            active_ipos = list(cache.values())
            if macro is None:
                macro = fetch_live_macro_regime()
        except Exception:
            active_ipos = []
            macro = {}
    return snapshot_service.record_scheduled_snapshots(active_ipos or [], macro or {})


def get_snapshots_summary() -> Dict[str, Any]:
    """Only the v2 evidence feed is suitable for point-in-time comparison.

    Legacy snapshots remain on disk for audit, but some contain unsourced
    zeroes and cached values and must not be shown as verified observations.
    """
    return {
        'legacy_unverified_snapshot_count': snapshot_service.get_snapshot_count(),
        'legacy_warning': 'Legacy snapshots are not verified and must not be used for model evaluation.',
        'subscription_collector': get_subscription_capture_summary(),
    }


def subscription_slot(now_ist: datetime) -> Optional[str]:
    """IST-aligned bid-window polls; subscription collection stops at 17:00."""
    now_ist = now_ist.astimezone(IST)
    minutes = now_ist.hour * 60 + now_ist.minute
    if now_ist.weekday() >= 5 or not 600 <= minutes < 1020:
        return None
    interval = 5 if minutes >= 870 else 15
    slot_minute = minutes - minutes % interval
    return f'{now_ist.date().isoformat()}_{slot_minute // 60:02d}{slot_minute % 60:02d}'


def _fetched_after_closing_cutoff(closing_date: Optional[date], fetched_at: Optional[str]) -> bool:
    """A poll started before cutoff is not pre-cutoff evidence if it finished later."""
    if closing_date is None or not fetched_at:
        return False
    try:
        observed = datetime.fromisoformat(fetched_at)
        if observed.tzinfo is None:
            return True
        observed = observed.astimezone(IST)
    except (TypeError, ValueError):
        return True
    return observed >= datetime.combine(closing_date, time(17, 0), tzinfo=IST)


@lru_cache(maxsize=1)
def _scheduled_predictor():
    from core.predictor import HybridIPOPredictor
    return HybridIPOPredictor()


def backfill_captured_bid_research(*, cache: Optional[Dict[str, Any]] = None,
                                   now_ist: Optional[datetime] = None,
                                   db_path: str = DB_PATH) -> int:
    """Freeze a retrospective daily replay of real, pre-cutoff app captures.

    This does not create an investor-time forecast or enter prospective accuracy.
    It only reconstructs a path while the listing outcome is still unknown.
    """
    if cache is None:
        from scrapers.live_ipo_scraper import load_scraped_cache
        cache = load_scraped_cache()
    clock = (now_ist or datetime.now(IST)).astimezone(IST)
    listing_dates = {}
    for url, item in cache.items():
        try:
            listing_dates[extract_ipo_id({**item, 'url': url})] = date.fromisoformat(
                str(item['listing_date']))
        except (KeyError, TypeError, ValueError):
            continue
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute("""CREATE TABLE IF NOT EXISTS captured_bid_replay_v1 (
            ipo_id TEXT NOT NULL, capture_day_ist TEXT NOT NULL,
            company_name TEXT NOT NULL, capture_id INTEGER NOT NULL,
            captured_at_ist TEXT NOT NULL, computed_at_ist TEXT NOT NULL,
            source_family TEXT NOT NULL, evidence_sha256 TEXT NOT NULL,
            model_version TEXT NOT NULL, model_bundle_sha256 TEXT NOT NULL,
            qib_x REAL NOT NULL, nii_x REAL NOT NULL, retail_x REAL NOT NULL,
            positive_listing_probability REAL NOT NULL,
            PRIMARY KEY (ipo_id, capture_day_ist)
        )""")
        replay_columns = {column[1] for column in conn.execute(
            'PRAGMA table_info(captured_bid_replay_v1)')}
        if 'model_code_sha256' in replay_columns and 'model_bundle_sha256' not in replay_columns:
            conn.execute('ALTER TABLE captured_bid_replay_v1 RENAME COLUMN '
                         'model_code_sha256 TO model_bundle_sha256')
        captures = conn.execute("""SELECT id, ipo_id, company_name, attempted_at_ist,
            fetched_at_ist, closing_date, source_family, raw_sha256,
            raw_html_gzip, qib_x, nii_x, retail_x
            FROM subscription_capture_v2
            WHERE capture_phase = 'pre_deadline' AND status = 'verified'
              AND raw_sha256 IS NOT NULL AND raw_html_gzip IS NOT NULL
              AND qib_x IS NOT NULL AND nii_x IS NOT NULL AND retail_x IS NOT NULL""").fetchall()
        latest = {}
        for row in captures:
            if listing_dates.get(row['ipo_id'], date.min) <= clock.date():
                continue
            try:
                attempt = datetime.fromisoformat(row['attempted_at_ist'])
                fetched = datetime.fromisoformat(row['fetched_at_ist'])
                close_date = date.fromisoformat(row['closing_date'])
                raw = gzip.decompress(row['raw_html_gzip'])
            except (TypeError, ValueError, OSError, EOFError):
                continue
            if attempt.tzinfo is None or fetched.tzinfo is None:
                continue
            attempt = attempt.astimezone(IST)
            fetched = fetched.astimezone(IST)
            if (attempt.date() != fetched.date() or fetched > clock
                    or attempt.date() > close_date
                    or attempt.hour >= 17 or fetched.hour >= 17
                    or hashlib.sha256(raw).hexdigest() != row['raw_sha256']):
                continue
            key = (row['ipo_id'], fetched.date().isoformat())
            candidate = (row['source_family'] == 'nse_exchange', fetched, row)
            if key not in latest or candidate[:2] > latest[key][:2]:
                latest[key] = candidate
        inserted = 0
        model_sha = hashlib.sha256((Path(__file__).resolve().parents[1] /
                                    'ml/model_bundle.json').read_bytes()).hexdigest()
        for (ipo_id, day), (_, _, row) in latest.items():
            if conn.execute("""SELECT 1 FROM captured_bid_replay_v1
                WHERE ipo_id = ? AND capture_day_ist = ?""", (ipo_id, day)).fetchone():
                continue
            scored = _scheduled_predictor().score_interim_bid_snapshot(
                qib_sub=row['qib_x'], hni_sub=row['nii_x'],
                retail_sub=row['retail_x'])
            cursor = conn.execute("""INSERT OR IGNORE INTO captured_bid_replay_v1
                (ipo_id, capture_day_ist, company_name, capture_id,
                 captured_at_ist, computed_at_ist, source_family,
                 evidence_sha256, model_version, model_bundle_sha256,
                 qib_x, nii_x, retail_x, positive_listing_probability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                ipo_id, day, row['company_name'], row['id'], row['fetched_at_ist'],
                clock.isoformat(timespec='seconds'), row['source_family'],
                row['raw_sha256'], scored['model_version'], model_sha,
                row['qib_x'], row['nii_x'], row['retail_x'],
                scored['positive_listing_probability']))
            inserted += cursor.rowcount
        conn.commit()
    return inserted


def record_prelisting_replays(*, cache: Optional[Dict[str, Any]] = None,
                              now_ist: Optional[datetime] = None,
                              db_path: str = DB_PATH) -> int:
    """Freeze one pre-listing model replay per closed IPO, never an investor-time forecast.

    Prefer this app's dated pre-deadline capture. If absent, a later-recovered
    third-party report can be scored, but stays unverified and is never included
    in prospective *application-time* accuracy. No post-close bid totals enter.
    """
    if cache is None:
        from scrapers.live_ipo_scraper import load_scraped_cache
        cache = load_scraped_cache()
    clock = (now_ist or datetime.now(IST)).astimezone(IST)
    archive = {'observations': []}
    if os.path.exists(RECOVERED_OBSERVATIONS_PATH):
        with open(RECOVERED_OBSERVATIONS_PATH, encoding='utf-8') as evidence_file:
            archive = json.load(evidence_file)
    inserted = 0
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        for ipo_url, item in cache.items():
            if item.get('category') != 'CLOSED':
                continue
            try:
                listing_date = date.fromisoformat(str(item.get('listing_date')))
            except (TypeError, ValueError):
                continue
            if listing_date <= clock.date():
                continue  # The outcome must still be in the future when locked.
            close_date = parse_closing_date(item.get('date_range'), clock.year)
            if close_date is None or close_date >= clock.date():
                continue
            ipo_id = extract_ipo_id({**item, 'url': ipo_url})
            if conn.execute('SELECT 1 FROM prelisting_replay_v1 WHERE ipo_id = ?', (ipo_id,)).fetchone():
                continue
            candidates = []
            captures = conn.execute("""
                SELECT attempted_at_ist, fetched_at_ist, source_family, raw_sha256,
                       qib_x, nii_x, retail_x
                FROM subscription_capture_v2
                WHERE LOWER(company_name) = LOWER(?) AND capture_phase = 'pre_deadline'
                  AND status = 'verified' AND qib_x IS NOT NULL AND nii_x IS NOT NULL
                  AND retail_x IS NOT NULL
                ORDER BY attempted_at_ist DESC, id DESC
            """, (item.get('company_name', ''),)).fetchall()
            for row in captures:
                try:
                    observed = datetime.fromisoformat(row['fetched_at_ist'] or row['attempted_at_ist']).astimezone(IST)
                except (TypeError, ValueError):
                    continue
                if observed.date() > close_date or observed.hour >= 17 or observed > clock:
                    continue
                candidates.append((2, observed, dict(row), 'app_capture_verified'))
            for row in archive.get('observations', []):
                if row.get('company_name', '').casefold() != item.get('company_name', '').casefold():
                    continue
                try:
                    observed = datetime.fromisoformat(row['source_reported_at_ist']).astimezone(IST)
                    found = datetime.fromisoformat(row['discovered_at_ist']).astimezone(IST)
                except (KeyError, TypeError, ValueError):
                    continue
                if observed.date() > close_date or observed.hour >= 17 or found > clock:
                    continue
                candidates.append((1, observed, row, 'external_report_recovered_later_unverified'))
            if not candidates:
                continue
            _, observed, source, evidence_status = max(candidates, key=lambda c: (c[0], c[1]))
            qib, nii, retail = (float(source[key]) for key in ('qib_x', 'nii_x', 'retail_x'))
            result = _scheduled_predictor().score_interim_bid_snapshot(
                qib_sub=qib, hni_sub=nii, retail_sub=retail)
            found_at = (source.get('fetched_at_ist') or source.get('attempted_at_ist')) \
                if evidence_status == 'app_capture_verified' else source['discovered_at_ist']
            cursor = conn.execute("""
                INSERT OR IGNORE INTO prelisting_replay_v1 (
                    ipo_id, company_name, recorded_at_ist, listing_date, observed_at_ist,
                    evidence_found_at_ist, evidence_status, source_family, evidence_sha256,
                    qib_x, nii_x, retail_x, model_version, positive_listing_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ipo_id, item['company_name'], clock.isoformat(timespec='seconds'),
                  listing_date.isoformat(), observed.isoformat(timespec='seconds'), found_at,
                  evidence_status, source['source_family'] if evidence_status == 'app_capture_verified'
                  else source['source_name'], source.get('raw_sha256'), qib, nii, retail,
                  result['model_version'], result['positive_listing_probability']))
            inserted += cursor.rowcount
        conn.commit()
    return inserted


def record_prelisting_outcome_forecasts(*, cache: Optional[Dict[str, Any]] = None,
                                        now_ist: Optional[datetime] = None,
                                        db_path: str = DB_PATH) -> int:
    """Freeze a listing-outcome forecast from a pre-cutoff bid capture.

    This is deliberately separate from an investor-time application forecast:
    the model may run after bidding closed even though its inputs were observed
    earlier. Never backdate the run or use a later bid table or known opening.
    """
    if cache is None:
        from scrapers.live_ipo_scraper import load_scraped_cache
        cache = load_scraped_cache()
    clock = (now_ist or datetime.now(IST)).astimezone(IST)
    inserted = 0
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        for ipo_url, item in cache.items():
            if item.get('category') != 'CLOSED' or item.get('is_listed'):
                continue
            try:
                listing_date = date.fromisoformat(str(item.get('listing_date')))
            except (TypeError, ValueError):
                continue
            if listing_date <= clock.date() or item.get('listing_open_price') is not None:
                continue
            close_date = parse_closing_date(item.get('date_range'), clock.year)
            if close_date is None or close_date >= listing_date:
                continue
            ipo_id = extract_ipo_id({**item, 'url': ipo_url})
            if conn.execute('SELECT 1 FROM prelisting_outcome_forecast_v1 WHERE ipo_id = ?',
                            (ipo_id,)).fetchone():
                continue
            captures = conn.execute("""
                SELECT id, attempted_at_ist, fetched_at_ist, capture_phase,
                       source_family, raw_sha256, raw_html_gzip, qib_x, nii_x, retail_x
                FROM subscription_capture_v2
                WHERE ipo_id = ? AND capture_phase = 'pre_deadline'
                  AND source_family IN ('nse_exchange', 'chittorgarh_investorgain')
                  AND status = 'verified' AND raw_sha256 IS NOT NULL
                  AND raw_html_gzip IS NOT NULL
                  AND qib_x IS NOT NULL AND nii_x IS NOT NULL AND retail_x IS NOT NULL
                ORDER BY CASE source_family WHEN 'nse_exchange' THEN 0 ELSE 1 END,
                         attempted_at_ist DESC, id DESC
            """, (ipo_id,)).fetchall()
            selected = None
            for row in captures:
                try:
                    captured = datetime.fromisoformat(row['fetched_at_ist'] or row['attempted_at_ist'])
                except (TypeError, ValueError):
                    continue
                if captured.tzinfo is None:
                    continue
                captured = captured.astimezone(IST)
                if (captured > clock or captured.date() >= listing_date
                        or captured.date() > close_date
                        or (captured.date() == close_date and captured.hour >= 17)):
                    continue
                try:
                    raw_bytes = gzip.decompress(row['raw_html_gzip'])
                except (OSError, ValueError, TypeError):
                    continue
                if hashlib.sha256(raw_bytes).hexdigest() != row['raw_sha256']:
                    continue
                selected = (row, captured)
                break
            if selected is None:
                continue
            row, captured = selected
            try:
                result = _scheduled_predictor().score_interim_bid_snapshot(
                    qib_sub=row['qib_x'], hni_sub=row['nii_x'],
                    retail_sub=row['retail_x'])
            except Exception:
                logger.exception('Pre-listing outcome forecast failed for %s', ipo_id)
                continue
            cursor = conn.execute("""
                INSERT OR IGNORE INTO prelisting_outcome_forecast_v1 (
                    ipo_id, company_name, recorded_at_ist, listing_date, capture_id,
                    capture_phase, captured_at_ist, source_family, evidence_sha256,
                    qib_x, nii_x, retail_x, model_version,
                    positive_listing_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ipo_id, item['company_name'], clock.isoformat(timespec='seconds'),
                  listing_date.isoformat(), row['id'], row['capture_phase'],
                  captured.isoformat(timespec='seconds'), row['source_family'],
                  row['raw_sha256'], row['qib_x'], row['nii_x'], row['retail_x'],
                  result['model_version'], result['positive_listing_probability']))
            inserted += cursor.rowcount
        conn.commit()
    return inserted


def record_automatic_decision(*, item: Dict[str, Any], ipo_url: str, slot: str,
                              stage: str, now_ist: datetime, values: Optional[Dict[str, Any]],
                              source_family: Optional[str], source_status: str,
                              source_asof: Optional[str], post_close: bool,
                              source_sha256: Optional[str] = None,
                              decision_at_ist: Optional[datetime] = None,
                              forecast_only: bool = False,
                              db_path: str = DB_PATH) -> bool:
    """Lock the response to one scheduled observation; never score post-close bids as pre-deadline."""
    ipo_id = extract_ipo_id({**item, 'url': ipo_url})
    kind = 'reconciliation_only' if post_close else 'provisional_scorecard'
    observed = values or {}
    inputs = {
        'qib_subscription': observed.get('qib_subscription'),
        'hni_subscription': observed.get('hni_subscription'),
        'retail_subscription': observed.get('retail_subscription'),
        'gmp_pct': item.get('gmp_pct'),
        'gmp_observed_at': item.get('gmp_observed_at'),
        'promoter_ofs_ratio': item.get('promoter_ofs_ratio'),
        'source_asof_ist': source_asof,
        'source_family': source_family,
        'source_status': source_status,
    }
    verdict = conviction = rationale = model_version = error = None
    engine_version = 'prebid_rules_2026_10_v1' if not post_close else None
    if not post_close:
        try:
            predictor = _scheduled_predictor()
            complete = source_status == 'verified' and all(
                observed.get(key) is not None for key in
                ('qib_subscription', 'hni_subscription', 'retail_subscription')
            )
            result = predictor.predict(
                company_name=item.get('company_name') or ipo_id,
                sector=item.get('sector') or 'Unclassified',
                qib_sub=observed.get('qib_subscription') if complete else 0.0,
                retail_sub=observed.get('retail_subscription') if complete else 0.0,
                hni_sub=observed.get('hni_subscription') if complete else 0.0,
                issue_size_cr=item.get('issue_size_cr') or 500.0,
                gmp_pct=item.get('gmp_pct'),
                promoter_ofs_ratio=item.get('promoter_ofs_ratio'),
                category='OPEN', bidding_ended=False, subscription_verified=complete,
                fetch_news=False,
            )
            if result.get('hybrid_ml_scores', {}).get('hybrid_positive_listing_prob') is not None:
                raise ValueError('Interim scorecard unexpectedly produced a final-bid probability')
            verdict = result.get('verdict')
            conviction = result.get('conviction')
            rationale = result.get('reasoned_explanation')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'[:500]
            logger.exception('Automatic decision failed for %s at %s', ipo_id, slot)
    experimental_forecast = None
    decision_clock = (decision_at_ist or datetime.now(IST)).astimezone(IST)
    window = application_forecast_window(item, decision_clock)
    in_application_window = not post_close and window['is_open']
    if in_application_window and source_status == 'verified':
        source_dt = None
        try:
            parsed_source = datetime.fromisoformat(source_asof) if source_asof else None
            source_dt = parsed_source.astimezone(IST) if parsed_source and parsed_source.tzinfo else None
        except (TypeError, ValueError):
            pass
        # Last published bids remain useful between sessions and on feed
        # failure. Preserve their age rather than pretending they are fresh.
        # The prediction itself must still be issued before the final cutoff.
        source_age = (decision_clock - source_dt).total_seconds() if source_dt else None
        if (source_dt and source_age is not None and source_age >= 0
                and source_dt >= datetime.fromisoformat(window['opens_at_ist'])
                and all(observed.get(key) is not None for key in
                        ('qib_subscription', 'retail_subscription', 'hni_subscription'))):
            try:
                experimental_forecast = _scheduled_predictor().score_interim_bid_snapshot(
                    qib_sub=observed['qib_subscription'],
                    retail_sub=observed['retail_subscription'],
                    hni_sub=observed['hni_subscription'],
                )
            except Exception:
                logger.exception('Experimental intraday forecast failed for %s at %s', ipo_id, slot)
    if experimental_forecast is not None and decision_at_ist is None:
        # Record completion time, not the slot start. Model computation can
        # itself cross the cutoff when a request arrives just before 17:00.
        decision_clock = datetime.now(IST)
        if not application_forecast_window(item, decision_clock)['is_open']:
            experimental_forecast = None
        elif source_dt:
            source_age = (decision_clock - source_dt).total_seconds()
    return_forecast = None
    if experimental_forecast is not None:
        try:
            from services.return_forecast_service import make_return_forecast
            publisher_asof = item.get('subscription_source_displayed_at')
            return_forecast = make_return_forecast(
                item, observed, source_asof=source_asof, decision_at=decision_clock,
                publisher_asof=publisher_asof)
            # Do not silently promote a challenger into the existing direction
            # scorecard. Its own probability, version and returns are nested
            # and evaluated separately from the unchanged legacy classifier.
        except Exception:
            logger.exception('Return challenger failed for %s; preserving direction-only call', ipo_id)
        if decision_at_ist is None:
            decision_clock = datetime.now(IST)
            if not application_forecast_window(item, decision_clock)['is_open']:
                experimental_forecast = return_forecast = None
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.execute('PRAGMA busy_timeout=5000')
        context_saved = False
        forecast_saved = False
        if not forecast_only:
            cursor = conn.execute("""
            INSERT OR IGNORE INTO automatic_decision_v1 (
                ipo_id, company_name, slot_key, snapshot_stage, decision_kind,
                recorded_at_ist, source_asof_ist, source_family, source_status,
                      model_version, decision_engine_version, input_json, verdict, conviction,
                positive_listing_probability, rationale, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ipo_id, item.get('company_name') or ipo_id, slot, stage, kind,
            datetime.now(IST).isoformat(timespec='seconds'), source_asof,
            source_family, source_status, model_version, engine_version,
            json.dumps(inputs, sort_keys=True, default=str), verdict, conviction,
            None, rationale, error,
            ))
            context_saved = cursor.rowcount > 0
        if experimental_forecast is not None:
            forecast_inputs = {
                'qib_subscription': observed['qib_subscription'],
                'retail_subscription': observed['retail_subscription'],
                'hni_subscription': observed['hni_subscription'],
                'source_asof_ist': source_asof,
                'source_published_at_ist': item.get('subscription_source_displayed_at'),
                'publisher_time_unverified': not bool(item.get('subscription_source_displayed_at')),
                'source_family': source_family,
                'source_sha256': source_sha256,
                'decision_recorded_at_ist': decision_clock.isoformat(timespec='seconds'),
                'forecast_target': 'listing_open_above_issue_price',
                'model_training_stage': 'closing_book',
                'decision_stage': 'before_final_application_cutoff',
                'application_cutoff_at_ist': window['cutoff_at_ist'],
                'snapshot_age_seconds': round(source_age, 1),
                'snapshot_is_current': source_age <= 1200,
                'intraday_validation': 'pending_prospective_outcomes',
                'return_forecast': return_forecast,
                'issue_price': item.get('issue_price'),
                'issue_size_cr': item.get('issue_size_cr'),
            }
            forecast_cursor = conn.execute("""
                INSERT OR IGNORE INTO automatic_decision_v1 (
                    ipo_id, company_name, slot_key, snapshot_stage, decision_kind,
                    recorded_at_ist, source_asof_ist, source_family, source_status,
                    model_version, decision_engine_version, input_json, verdict, conviction,
                    positive_listing_probability, rationale, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ipo_id, item.get('company_name') or ipo_id, f'{slot}_{source_family}', stage,
                'live_bid_forecast', decision_clock.isoformat(timespec='seconds'),
                source_asof, source_family, source_status,
                experimental_forecast['model_version'], 'intraday_snapshot_proxy_v1',
                json.dumps(forecast_inputs, sort_keys=True), 'FORECAST RECORDED',
                'UNVALIDATED', experimental_forecast['positive_listing_probability'],
                'Experimental investor-time positive-open probability. Closing-book coefficients applied to a live bid snapshot; historical final-book accuracy does not transfer. Any return challenger is separately versioned in inputs.',
                None,
            ))
            forecast_saved = forecast_cursor.rowcount > 0
        conn.commit()
        if Path(db_path).resolve().parent == Path(DB_PATH).resolve().parent:
            from services.sqlite_supabase_sync import request_sync
            request_sync('Module 1 automatic decision ledger')
        return context_saved or forecast_saved


def preferred_forecast(rows: list[dict]) -> Optional[dict]:
    """Prefer a recently publisher-dated book over an undated re-fetch.

    Keep every issued source-specific call in the ledger. A newly downloaded
    old aggregator page must not temporarily displace the newer exchange book.
    """
    if not rows:
        return None
    latest = max(datetime.fromisoformat(row['recorded_at_ist']) for row in rows)
    dated = []
    for row in rows:
        inputs = row.get('inputs') or json.loads(row.get('input_json') or '{}')
        stamp = inputs.get('source_published_at_ist') or (
            row.get('source_asof_ist') if row.get('source_family') == 'nse_exchange'
            and not inputs.get('publisher_time_unverified') else None)
        try:
            observed = datetime.fromisoformat(stamp)
            if observed.tzinfo and 0 <= (latest - observed).total_seconds() <= 1200:
                dated.append((observed, row))
        except (ValueError, TypeError):
            continue
    if dated:
        return max(dated, key=lambda pair: (pair[0], pair[1].get('source_family') == 'nse_exchange',
                                           pair[1]['recorded_at_ist']))[1]
    return max(rows, key=lambda row: (row['recorded_at_ist'], row.get('id', 0)))


def get_latest_application_forecast(company: str, *, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Return an actually issued call, independent of later reconciliation rows."""
    try:
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT * FROM automatic_decision_v1
                WHERE LOWER(company_name) = LOWER(?) AND decision_kind = 'live_bid_forecast'
                  AND positive_listing_probability IS NOT NULL
                ORDER BY recorded_at_ist DESC, id DESC LIMIT 100""", (company.strip(),)).fetchall()
        row = preferred_forecast([dict(row) for row in rows])
        if row is None:
            return None
        result = dict(row)
        result['inputs'] = json.loads(result.pop('input_json'))
        return result
    except sqlite3.OperationalError:
        return None


def get_last_complete_bid_capture(company: str, *, now_ist: datetime,
                                  db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Recover a complete last published book when a newer fetch is empty/partial."""
    try:
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT * FROM subscription_capture_v2
                WHERE LOWER(company_name) = LOWER(?) AND status = 'verified'
                  AND source_family IN ('nse_exchange', 'chittorgarh_investorgain')
                  AND capture_phase = 'pre_deadline'
                  AND qib_x IS NOT NULL AND nii_x IS NOT NULL AND retail_x IS NOT NULL
                ORDER BY fetched_at_ist DESC, id DESC LIMIT 100""", (company.strip(),)).fetchall()
        for row in rows:
            try:
                fetched = datetime.fromisoformat(row['fetched_at_ist'])
                closes = date.fromisoformat(row['closing_date'])
                if (fetched.tzinfo is None or fetched > now_ist
                        or _fetched_after_closing_cutoff(closes, row['fetched_at_ist'])):
                    continue
                raw = gzip.decompress(row['raw_html_gzip'])
                if hashlib.sha256(raw).hexdigest() != row['raw_sha256']:
                    continue
                result = dict(row)
                publisher = None
                if row['source_family'] == 'nse_exchange':
                    try:
                        publisher = datetime.strptime(json.loads(raw).get('updateTime', ''),
                            'Updated as on %d-%b-%Y %H:%M:%S').replace(tzinfo=IST).isoformat()
                    except (ValueError, TypeError, AttributeError):
                        pass
                else:
                    from scrapers.live_ipo_scraper import parse_subscription_published_at
                    publisher = parse_subscription_published_at(raw.decode('utf-8', errors='replace'))
                if publisher and datetime.fromisoformat(publisher) > fetched:
                    continue
                result['source_published_at_ist'] = publisher
                result['source_asof_ist'] = publisher or row['fetched_at_ist']
                return result
            except (TypeError, ValueError, OSError):
                continue
    except sqlite3.OperationalError:
        pass
    return None


def get_automatic_decision_rows(*, company: Optional[str] = None, limit: int = 100,
                                db_path: str = DB_PATH) -> Dict[str, Any]:
    """Read timestamped automatic assessments without retroactively recomputing them."""
    query = """SELECT ipo_id, company_name, slot_key, snapshot_stage, decision_kind,
                      recorded_at_ist, source_asof_ist, source_family, source_status,
                      model_version, decision_engine_version, input_json, verdict, conviction,
                      positive_listing_probability, rationale, error
               FROM automatic_decision_v1"""
    params: List[Any] = []
    if company:
        query += ' WHERE LOWER(company_name) = LOWER(?) OR ipo_id = ?'
        params.extend([company.strip(), company.strip()])
    query += ' ORDER BY recorded_at_ist DESC, id DESC LIMIT ?'
    params.append(min(max(int(limit), 1), 500))
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(query, params)]
    for row in rows:
        row['inputs'] = json.loads(row.pop('input_json'))
    first_party_captures = []
    if company:
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.row_factory = sqlite3.Row
            first_party_captures = [dict(row) for row in conn.execute("""
                SELECT slot_key, attempted_at_ist, fetched_at_ist, source_family,
                       qib_x, nii_x, retail_x, raw_sha256
                FROM subscription_capture_v2
                WHERE LOWER(company_name) = LOWER(?)
                  AND capture_phase = 'pre_deadline' AND status = 'verified'
                  AND qib_x IS NOT NULL AND nii_x IS NOT NULL AND retail_x IS NOT NULL
                ORDER BY attempted_at_ist DESC, id DESC LIMIT 10
            """, (company.strip(),))]
    recovered = []
    if company and os.path.exists(RECOVERED_OBSERVATIONS_PATH):
        with open(RECOVERED_OBSERVATIONS_PATH, encoding='utf-8') as evidence_file:
            archive = json.load(evidence_file)
        recovered = [observation for observation in archive.get('observations', [])
                     if observation.get('company_name', '').casefold() == company.strip().casefold()]
    replays = []
    outcome_forecasts = []
    captured_bid_replays = []
    if company:
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.row_factory = sqlite3.Row
            try:
                captured_bid_replays = [dict(row) for row in conn.execute("""
                    SELECT capture_day_ist, captured_at_ist, computed_at_ist,
                           source_family, evidence_sha256, model_version,
                           qib_x, nii_x, retail_x, positive_listing_probability
                    FROM captured_bid_replay_v1 WHERE LOWER(company_name) = LOWER(?)
                    ORDER BY capture_day_ist DESC""", (company.strip(),))]
            except sqlite3.OperationalError:
                pass
            replays = [dict(row) for row in conn.execute("""
                SELECT company_name, recorded_at_ist, listing_date, observed_at_ist,
                       evidence_found_at_ist, evidence_status, source_family, evidence_sha256,
                       qib_x, nii_x, retail_x, model_version, positive_listing_probability
                FROM prelisting_replay_v1 WHERE LOWER(company_name) = LOWER(?)
            """, (company.strip(),))]
            outcome_forecasts = [dict(row) for row in conn.execute("""
                SELECT company_name, recorded_at_ist, listing_date, capture_id,
                       capture_phase, captured_at_ist, source_family, evidence_sha256,
                       qib_x, nii_x, retail_x, model_version,
                       positive_listing_probability
                FROM prelisting_outcome_forecast_v1
                WHERE LOWER(company_name) = LOWER(?)
            """, (company.strip(),))]
    from services.historical_bid_service import get_historical_bid_forecast
    groups = {}
    for row in rows:
        if row['decision_kind'] == 'live_bid_forecast':
            # Scheduled calls share the slot before the source-family suffix.
            key = row['slot_key'].removesuffix('_nse_exchange').removesuffix('_chittorgarh_investorgain')
            groups.setdefault(key, []).append(row)
    primary_history = [preferred_forecast(group) for group in groups.values()]
    return {'count': len(rows), 'decisions': rows, 'primary_forecast_history': primary_history,
            'historical_bid_forecast': get_historical_bid_forecast(company, db_path=db_path) if company else None,
            'application_forecast': get_latest_application_forecast(company, db_path=db_path) if company else None,
            'pre_deadline_captures': first_party_captures,
            'captured_bid_replays': captured_bid_replays,
            'recovered_observations': recovered,
            'prelisting_replay': replays[0] if replays else None,
            'prelisting_outcome_forecast': outcome_forecasts[0] if outcome_forecasts else None}


def get_intraday_forecast_scorecard(*, db_path: str = DB_PATH,
                                    cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compare saved pre-cutoff forecasts with verified exchange openings.

    One latest forecast per IPO is used for aggregate accuracy. Daily records
    remain visible, but repeated forecasts for one IPO do not inflate n.
    Missing opening prices remain pending, never counted as correct or wrong.
    """
    if cache is None:
        from scrapers.live_ipo_scraper import load_scraped_cache
        cache = load_scraped_cache()
    from services.listing_outcome_service import verified_listing_outcomes
    outcomes = verified_listing_outcomes(db_path)
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        saved = [dict(row) for row in conn.execute("""
            SELECT ipo_id, company_name, slot_key, recorded_at_ist,
                   source_asof_ist, source_family, model_version,
                   positive_listing_probability, input_json
            FROM automatic_decision_v1
            WHERE decision_kind = 'live_bid_forecast'
            ORDER BY slot_key DESC, id DESC
        """)]
    grouped = {}
    for row in saved:
        grouped.setdefault((row['ipo_id'], row['slot_key'][:10]), []).append(row)
    saved = [preferred_forecast(group) for _, group in sorted(grouped.items(), key=lambda entry: entry[0][1], reverse=True)]
    daily, seen_days = [], set()
    for row in saved:
        key = (row['ipo_id'], row['slot_key'][:10])
        if key in seen_days:
            continue
        seen_days.add(key)
        outcome = outcomes.get(str(row['ipo_id'])) or {}
        try:
            issue_price = float(outcome.get('issue_price'))
            listing_open = float(outcome.get('nse_open_price'))
            if not (issue_price > 0 and listing_open > 0):
                raise ValueError('Opening price not yet verified')
            actual_return_pct = round((listing_open / issue_price - 1.0) * 100, 3)
            actual_positive = actual_return_pct > 0
            probability = float(row['positive_listing_probability'])
            matched = (probability >= 0.5) == actual_positive
            brier = round((probability - float(actual_positive)) ** 2, 5)
        except (TypeError, ValueError):
            issue_price = listing_open = actual_return_pct = actual_positive = matched = brier = None
        daily.append({
            'ipo_id': row['ipo_id'], 'company_name': row['company_name'],
            'forecast_date_ist': key[1], 'recorded_at_ist': row['recorded_at_ist'],
            'source_asof_ist': row['source_asof_ist'], 'source_family': row['source_family'],
            'model_version': row['model_version'],
            'positive_listing_probability': row['positive_listing_probability'],
            'inputs': json.loads(row['input_json']),
            'outcome_status': 'resolved' if matched is not None else 'pending_exchange_open',
            'issue_price': issue_price, 'listing_open_price': listing_open,
            'actual_open_return_pct': actual_return_pct,
            'actual_positive': actual_positive, 'direction_correct': matched,
            'brier': brier,
        })
    latest_by_ipo = {}
    for record in daily:
        latest_by_ipo.setdefault(record['ipo_id'], record)
    resolved = [row for row in latest_by_ipo.values() if row['direction_correct'] is not None]
    return {
        'evaluation_policy': 'latest saved pre-cutoff forecast per IPO; outcome is exchange listing open versus issue price',
        'scope': 'experimental intraday proxy, not a validated investor-time model',
        'daily_records': daily,
        'summary': {
            'ipos_with_forecast': len(latest_by_ipo),
            'ipos_with_verified_open': len(resolved),
            'pending_open': len(latest_by_ipo) - len(resolved),
            'direction_accuracy_pct': round(100 * sum(row['direction_correct'] for row in resolved) / len(resolved), 1)
            if resolved else None,
            'brier_score': round(sum(row['brier'] for row in resolved) / len(resolved), 4)
            if resolved else None,
        },
    }


def get_prelisting_replay_scorecard(*, db_path: str = DB_PATH,
                                    cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate frozen post-cutoff/pre-listing replays separately from live forecasts."""
    if cache is None:
        from scrapers.live_ipo_scraper import load_scraped_cache
        cache = load_scraped_cache()
    from services.listing_outcome_service import verified_listing_outcomes
    outcomes = verified_listing_outcomes(db_path)
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        saved = [dict(row) for row in conn.execute(
            'SELECT * FROM prelisting_replay_v1 ORDER BY recorded_at_ist DESC, ipo_id')]
    records = []
    for row in saved:
        outcome = outcomes.get(str(row['ipo_id'])) or {}
        actual_return = matched = brier = None
        try:
            issue_price = float(outcome.get('issue_price'))
            opening_price = float(outcome.get('nse_open_price'))
            if issue_price > 0 and opening_price > 0:
                actual_return = round((opening_price / issue_price - 1) * 100, 3)
                actual_positive = actual_return > 0
                matched = (row['positive_listing_probability'] >= 0.5) == actual_positive
                brier = (row['positive_listing_probability'] - float(actual_positive)) ** 2
        except (TypeError, ValueError):
            pass
        records.append({**row, 'actual_open_return_pct': actual_return,
                        'direction_correct': matched, 'brier': brier})
    resolved = [row for row in records if row['direction_correct'] is not None]
    verified = [row for row in resolved if row['evidence_status'] == 'app_capture_verified']
    return {
        'scope': 'Locked after application cutoff but before scheduled listing; not investor-time forecasts',
        'records': records,
        'summary': {
            'replays_locked': len(records), 'openings_verified': len(resolved),
            'pending_open': len(records) - len(resolved),
            'app_captured_inputs': sum(row['evidence_status'] == 'app_capture_verified' for row in records),
            'unverified_recovered_inputs': sum(row['evidence_status'] != 'app_capture_verified' for row in records),
            'direction_accuracy_app_captured_pct': round(
                100 * sum(row['direction_correct'] for row in verified) / len(verified), 1
            ) if verified else None,
            'direction_accuracy_unverified_pct': round(
                100 * sum(row['direction_correct'] for row in resolved if row['evidence_status'] != 'app_capture_verified') /
                sum(row['evidence_status'] != 'app_capture_verified' for row in resolved), 1
            ) if any(row['evidence_status'] != 'app_capture_verified' for row in resolved) else None,
        },
    }


def get_prelisting_outcome_scorecard(*, db_path: str = DB_PATH,
                                     cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score frozen pre-listing calls only after an exchange opening is recorded."""
    if cache is None:
        from scrapers.live_ipo_scraper import load_scraped_cache
        cache = load_scraped_cache()
    from services.listing_outcome_service import verified_listing_outcomes
    outcomes = verified_listing_outcomes(db_path)
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        saved = [dict(row) for row in conn.execute(
            'SELECT * FROM prelisting_outcome_forecast_v1 ORDER BY recorded_at_ist DESC, ipo_id')]
    records = []
    for row in saved:
        outcome = outcomes.get(str(row['ipo_id'])) or {}
        actual_return = matched = brier = None
        try:
            issue_price = float(outcome.get('issue_price'))
            opening_price = float(outcome.get('nse_open_price'))
            if issue_price > 0 and opening_price > 0:
                actual_return = round((opening_price / issue_price - 1) * 100, 3)
                positive = actual_return > 0
                matched = (row['positive_listing_probability'] >= 0.5) == positive
                brier = (row['positive_listing_probability'] - float(positive)) ** 2
        except (TypeError, ValueError):
            pass
        records.append({**row, 'actual_open_return_pct': actual_return,
                        'direction_correct': matched, 'brier': brier})
    resolved = [row for row in records if row['direction_correct'] is not None]
    pre_cutoff_resolved = [row for row in resolved if row['capture_phase'] == 'pre_deadline']
    post_close_resolved = [row for row in resolved if row['capture_phase'] != 'pre_deadline']
    return {
        'scope': 'Frozen before listing; pre-cutoff inputs and legacy post-close inputs are scored separately',
        'records': records,
        'summary': {
            'forecasts_locked': len(records), 'openings_verified': len(resolved),
            'pending_open': len(records) - len(resolved),
            'nse_captures': sum(row['source_family'] == 'nse_exchange' for row in records),
            'third_party_captures': sum(row['source_family'] != 'nse_exchange' for row in records),
            'pre_cutoff_input_count': sum(row['capture_phase'] == 'pre_deadline' for row in records),
            'legacy_post_close_input_count': sum(row['capture_phase'] != 'pre_deadline' for row in records),
            'direction_accuracy_pct': round(
                100 * sum(row['direction_correct'] for row in resolved) / len(resolved), 1
            ) if resolved else None,
            'brier_score': round(sum(row['brier'] for row in resolved) / len(resolved), 4)
            if resolved else None,
            'pre_cutoff_input_accuracy_pct': round(
                100 * sum(row['direction_correct'] for row in pre_cutoff_resolved) / len(pre_cutoff_resolved), 1
            ) if pre_cutoff_resolved else None,
            'legacy_post_close_accuracy_pct': round(
                100 * sum(row['direction_correct'] for row in post_close_resolved) / len(post_close_resolved), 1
            ) if post_close_resolved else None,
        },
    }


def _record_capture_decision(**kwargs):
    try:
        return record_automatic_decision(**kwargs)
    except Exception:
        logger.exception('Decision recording failed for %s; source captures and other IPOs continue', kwargs.get('ipo_url'))
        return False


def capture_active_subscriptions(*, now_ist: Optional[datetime] = None,
                                 cache: Optional[Dict[str, Any]] = None,
                                 db_path: str = DB_PATH,
                                 capture_exchange: bool = True) -> int:
    """Capture each IPO/source once per slot; resume only unfinished sources."""
    from scrapers.live_ipo_scraper import (
        load_scraped_cache, save_scraped_cache, fetch_subscription_evidence,
        SUBSCRIPTION_SOURCE,
    )
    from services.nse_subscription_service import fetch_nse_subscription_evidence, SOURCE_NAME

    use_live_clock = now_ist is None
    now_ist = (now_ist or datetime.now(IST)).astimezone(IST)
    slot = subscription_slot(now_ist)
    if slot is None:
        return 0
    rows = cache if cache is not None else load_scraped_cache()
    inserted = 0
    cache_updated = False
    for ipo_url, item in rows.items():
        if item.get('source_observed') is not True:
            continue
        if not application_forecast_window(item, now_ist)['is_open']:
            continue
        if use_live_clock and subscription_slot(datetime.now(IST)) is None:
            break
        closing_dt = parse_closing_date(item.get('date_range'), now_ist.year)
        post_close = now_ist.hour * 60 + now_ist.minute >= 1020
        if post_close:
            if closing_dt != now_ist.date() or item.get('category') not in ('OPEN', 'CLOSED'):
                continue
        elif item.get('category') != 'OPEN':
            continue
        # Restarting the backend must not repeat an already completed slot.
        # Keep all earlier attempts intact, including unavailable-source rows.
        ipo_id = extract_ipo_id({**item, 'url': ipo_url})
        required_sources = {'chittorgarh_investorgain'}
        if capture_exchange:
            required_sources.add('nse_exchange')
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            completed_sources = {r[0] for r in conn.execute(
                'SELECT DISTINCT source_family FROM subscription_capture_v2 WHERE ipo_id=? AND slot_key=?',
                (ipo_id, slot))}
        if required_sources <= completed_sources:
            continue
        if use_live_clock:
            try:
                from services.gmp_service import gmp_service
                quote = gmp_service.resolve_gmp(item.get('company_name') or '',
                    chittorgarh_url=ipo_url, issue_price=item.get('issue_price'), is_listed=False)
                item = {**item, **quote}
                rows[ipo_url] = item
                cache_updated = True
            except Exception:
                logger.exception('GMP refresh failed for %s; no fresh quote will be invented', item.get('company_name'))
        attempted_at = datetime.now(IST).isoformat(timespec='seconds')
        stage, _ = determine_snapshot_stage(closing_dt, now_ist)
        if post_close:
            stage = 'POST_CLOSE_RECONCILIATION'
        try:
            from services.subscription_training import existing_evidence
            evidence = (existing_evidence(ipo_id,slot,'chittorgarh_investorgain',db_path)
                        if 'chittorgarh_investorgain' in completed_sources else fetch_subscription_evidence(ipo_url))
        except Exception as exc:
            evidence = {'status': 'fetch_error', 'error': f'{type(exc).__name__}: {exc}'[:500]}
        chitto_post_close = post_close or _fetched_after_closing_cutoff(
            closing_dt, evidence.get('fetched_at_ist'))
        values = evidence.get('values') if evidence.get('status') in (
            'verified', 'partial_subscription_table') else None
        raw_html = evidence.get('raw_html')
        raw_bytes = gzip.compress(raw_html.encode('utf-8')) if raw_html is not None else None
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.execute('PRAGMA busy_timeout=5000')
            conn.execute("""
                INSERT INTO subscription_capture_v2 (
                    ipo_id, company_name, ipo_url, subscription_url,
                    attempted_at_ist, fetched_at_ist, slot_key, closing_date,
                    snapshot_stage, status, http_status, qib_x, nii_x, retail_x,
                    raw_sha256, raw_html_gzip, error, source_family, capture_phase
                ) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (SELECT 1 FROM subscription_capture_v2 WHERE ipo_id=? AND slot_key=? AND source_family=?)
            """, (
                extract_ipo_id({**item, 'url': ipo_url}), item.get('company_name') or ipo_url,
                ipo_url, evidence.get('subscription_url'), attempted_at,
                evidence.get('fetched_at_ist'), slot,
                closing_dt.isoformat() if closing_dt else None,
                'POST_CLOSE_RECONCILIATION' if chitto_post_close else stage,
                evidence.get('status', 'fetch_error'), evidence.get('http_status'),
                values.get('qib_subscription') if values else None,
                values.get('hni_subscription') if values else None,
                values.get('retail_subscription') if values else None,
                evidence.get('raw_sha256'), raw_bytes, evidence.get('error'),
                'chittorgarh_investorgain',
                'post_close_reconciliation' if chitto_post_close else 'pre_deadline',
                ipo_id,slot,'chittorgarh_investorgain',
            ))
            added=conn.total_changes
            conn.commit()
        inserted += added
        selected_post_close = chitto_post_close
        selected_values = values
        selected_family = 'chittorgarh_investorgain'
        selected_status = evidence.get('status', 'fetch_error')
        selected_asof = evidence.get('source_displayed_at_ist') or evidence.get('fetched_at_ist')
        item = {**item, 'subscription_source_displayed_at': evidence.get('source_displayed_at_ist'),
                'subscription_source_family': 'chittorgarh_investorgain'}
        selected_sha256 = evidence.get('raw_sha256')
        if values:
            rows[ipo_url] = {
                **item,
                # Replace the whole book: a missing category in this poll must not
                # inherit a value from an earlier (possibly different) table.
                **{key: values.get(key) for key in ('qib_subscription', 'hni_subscription', 'retail_subscription')},
                'subscription_observed': evidence.get('status') == 'verified',
                'subscription_observation_status': evidence.get('status'),
                'subscription_source': SUBSCRIPTION_SOURCE,
                'subscription_observed_at': evidence.get('fetched_at_ist'),
                'subscription_source_displayed_at': evidence.get('source_displayed_at_ist'),
                'subscription_url': evidence.get('subscription_url'),
                'subscription_is_final': False,
                'subscription_capture_phase': 'post_close_reconciliation' if chitto_post_close else 'pre_deadline',
                'subscription_raw_sha256': evidence.get('raw_sha256'),
            }
            cache_updated = True
        else:
            if not values:
                logger.warning('Subscription capture unavailable for %s: %s (%s)',
                               item.get('company_name'), evidence.get('status'), evidence.get('error'))
        if capture_exchange and (not use_live_clock or subscription_slot(datetime.now(IST)) is not None):
            # Record a usable reading immediately. A slow exchange cross-check
            # must not delay an already available call past the final cutoff.
            if selected_status == 'verified' and selected_values:
                _record_capture_decision(
                    item=item, ipo_url=ipo_url, slot=slot, stage=stage, now_ist=now_ist,
                    values=selected_values, source_family=selected_family,
                    source_status=selected_status, source_asof=selected_asof,
                    post_close=selected_post_close, source_sha256=selected_sha256,
                    forecast_only=True, db_path=db_path)
            try:
                nse = (existing_evidence(ipo_id,slot,'nse_exchange',db_path)
                       if 'nse_exchange' in completed_sources else fetch_nse_subscription_evidence(
                    item.get('company_name') or '', closing_dt,
                    nse_symbol=item.get('nse_symbol'),
                    symbol_verified=item.get('nse_symbol_source') == 'nse_current_issue',
                    nse_series=item.get('nse_series'),
                ))
            except Exception as exc:
                nse = {'status':'fetch_error','error':f'{type(exc).__name__}: {exc}'[:500]}
            nse_post_close = post_close or _fetched_after_closing_cutoff(
                closing_dt, nse.get('fetched_at_ist'))
            nse_values = nse.get('values') or {}
            nse_raw = nse.get('raw_html')
            with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
                conn.execute('PRAGMA busy_timeout=5000')
                conn.execute("""
                    INSERT INTO subscription_capture_v2 (
                        ipo_id, company_name, ipo_url, subscription_url,
                        attempted_at_ist, fetched_at_ist, slot_key, closing_date,
                        snapshot_stage, status, http_status, qib_x, nii_x, retail_x,
                        raw_sha256, raw_html_gzip, error, parser_version,
                        source_family, capture_phase
                    ) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (SELECT 1 FROM subscription_capture_v2 WHERE ipo_id=? AND slot_key=? AND source_family=?)
                """, (
                    extract_ipo_id({**item, 'url': ipo_url}), item.get('company_name') or ipo_url,
                    ipo_url, nse.get('subscription_url'), attempted_at,
                    nse.get('fetched_at_ist'), slot,
                    closing_dt.isoformat() if closing_dt else None,
                    'POST_CLOSE_RECONCILIATION' if nse_post_close else stage,
                    nse.get('status', 'fetch_error'), nse.get('http_status'),
                    nse_values.get('qib_subscription'), nse_values.get('hni_subscription'),
                    nse_values.get('retail_subscription'), nse.get('raw_sha256'),
                    gzip.compress(nse_raw.encode('utf-8')) if nse_raw is not None else None,
                    nse.get('error'), nse.get('parser_version', 'nse_category_json_v1'),
                    'nse_exchange', 'post_close_reconciliation' if nse_post_close else 'pre_deadline',
                    ipo_id,slot,'nse_exchange',
                ))
                added=conn.total_changes
                conn.commit()
            inserted += added
            source_time = nse.get('source_displayed_at_ist')
            try:parsed_time = datetime.fromisoformat(source_time) if source_time else None
            except (TypeError,ValueError):parsed_time = None
            if parsed_time is not None and parsed_time.tzinfo is None:parsed_time = None
            source_fresh = parsed_time is not None and -60 <= (now_ist - parsed_time).total_seconds() <= 1200
            if nse_values and source_fresh:
                selected_values = nse_values
                selected_post_close = nse_post_close
                selected_family = 'nse_exchange'
                selected_status = nse.get('status', 'fetch_error')
                selected_asof = source_time
                selected_sha256 = nse.get('raw_sha256')
                item = {**item, 'subscription_source_displayed_at': source_time,
                        'subscription_source_family': 'nse_exchange',
                        'subscription_publisher_time_unverified': False}
            if nse_values and source_fresh:
                previous = rows[ipo_url]
                rows[ipo_url] = {
                    **previous,
                    **{key: nse_values.get(key) for key in ('qib_subscription', 'hni_subscription', 'retail_subscription')},
                    'subscription_observed': nse.get('status') == 'verified',
                    'subscription_observation_status': nse.get('status'),
                    'subscription_source': SOURCE_NAME,
                    'subscription_source_family': 'nse_exchange',
                    'subscription_observed_at': source_time,
                    'subscription_source_displayed_at': source_time,
                    'subscription_fetched_at': nse.get('fetched_at_ist'),
                    'subscription_url': nse.get('subscription_url'),
                    'subscription_is_final': False,
                    'subscription_capture_phase': 'post_close_reconciliation' if nse_post_close else 'pre_deadline',
                    'subscription_raw_sha256': nse.get('raw_sha256'),
                    'nse_symbol': nse.get('nse_symbol'),
                    'nse_series': nse.get('series'),
                    'nse_symbol_source': 'nse_current_issue',
                    'subscription_cross_check': {
                        'source_family': 'chittorgarh_investorgain',
                        'fetched_at_ist': evidence.get('fetched_at_ist'),
                        'status': evidence.get('status'),
                        'values': values,
                    },
                }
                cache_updated = True
        if not selected_values or any(selected_values.get(key) is None for key in
                                      ('qib_subscription', 'hni_subscription', 'retail_subscription')):
            previous_book = get_last_complete_bid_capture(
                item.get('company_name') or '', now_ist=datetime.now(IST), db_path=db_path)
            if previous_book:
                selected_values = {'qib_subscription': previous_book['qib_x'],
                                   'hni_subscription': previous_book['nii_x'],
                                   'retail_subscription': previous_book['retail_x']}
                selected_family = previous_book['source_family']
                selected_status = 'verified'
                selected_asof = previous_book.get('source_asof_ist') or previous_book['fetched_at_ist']
                item = {**item, 'subscription_source_displayed_at': previous_book.get('source_published_at_ist'),
                        'subscription_publisher_time_unverified': not bool(previous_book.get('source_published_at_ist'))}
                selected_sha256 = previous_book['raw_sha256']
                selected_post_close = False
        _record_capture_decision(
            item=item, ipo_url=ipo_url, slot=slot, stage=stage, now_ist=now_ist,
            values=selected_values, source_family=selected_family,
            source_status=selected_status, source_asof=selected_asof,
            post_close=selected_post_close, source_sha256=selected_sha256,
            db_path=db_path,
        )
    if cache_updated and cache is None:
        save_scraped_cache(rows)
    if use_live_clock:
        try:
            from services.return_forecast_service import record_priority_snapshot
            record_priority_snapshot(cache=rows, now=datetime.now(IST), slot=slot, db_path=db_path)
        except Exception:
            logger.exception('Could not freeze the joint IPO priority comparison at %s', slot)
    if inserted and Path(db_path).resolve().parent == Path(DB_PATH).resolve().parent:
        from services.sqlite_supabase_sync import request_sync
        request_sync('Module 1 subscription capture ledger')
    return inserted


def get_subscription_capture_summary(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Expose latest attempt separately from last verified observation."""
    now_ist = datetime.now(IST)
    today = now_ist.date().isoformat()
    expected = 0
    if now_ist.weekday() < 5:
        end = min(now_ist.hour * 60 + now_ist.minute, 1019)
        expected = sum(1 for minute in list(range(600, 870, 15)) + list(range(870, 1020, 5))
                       if minute <= end)
    try:
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT ipo_id, company_name, COUNT(*) AS attempts,
                       COUNT(DISTINCT CASE WHEN capture_phase = 'pre_deadline'
                         AND slot_key GLOB '????-??-??_[0-9][0-9][0-9][0-9]' THEN slot_key END) AS captured_slots,
                       COUNT(DISTINCT CASE WHEN capture_phase = 'post_close_reconciliation' THEN slot_key END) AS post_close_slots,
                       SUM(CASE WHEN status = 'verified' AND capture_phase = 'pre_deadline' THEN 1 ELSE 0 END) AS verified_attempts,
                       MAX(CASE WHEN status = 'verified' AND capture_phase = 'pre_deadline' THEN fetched_at_ist END) AS last_verified_at,
                       SUM(CASE WHEN capture_phase = 'post_close_reconciliation' THEN 1 ELSE 0 END) AS post_close_attempts,
                       MAX(CASE WHEN capture_phase = 'post_close_reconciliation' THEN fetched_at_ist END) AS last_post_close_at
                FROM subscription_capture_v2 WHERE substr(slot_key, 1, 10) = ?
                GROUP BY ipo_id, company_name ORDER BY company_name
            """, (today,)).fetchall()
            result = []
            for row in rows:
                recent = conn.execute("""
                    SELECT slot_key, attempted_at_ist, status, http_status, error,
                           qib_x, nii_x, retail_x, raw_sha256, source_family, capture_phase
                    FROM subscription_capture_v2 WHERE ipo_id = ?
                    ORDER BY id DESC LIMIT 1
                """, (row['ipo_id'],)).fetchone()
                result.append({**dict(row), 'missing_slots_today': max(0, expected - row['captured_slots']),
                               'latest_attempt': dict(recent) if recent else None})
            from scrapers.live_ipo_scraper import load_scraped_cache
            seen = {entry['ipo_id'] for entry in result}
            for ipo_url, item in load_scraped_cache().items():
                if item.get('category') != 'OPEN' or item.get('source_observed') is not True:
                    continue
                ipo_id = extract_ipo_id({**item, 'url': ipo_url})
                if ipo_id not in seen:
                    result.append({'ipo_id': ipo_id, 'company_name': item.get('company_name'),
                                   'attempts': 0, 'captured_slots': 0, 'verified_attempts': 0,
                                   'last_verified_at': None, 'post_close_attempts': 0, 'post_close_slots': 0,
                                   'last_post_close_at': None, 'missing_slots_today': expected,
                                   'latest_attempt': None})
            return {'date_ist': today, 'window_ist': '10:00–17:00',
                    'expected_slots_so_far': expected, 'ipos': result}
    except sqlite3.Error as exc:
        return {'date_ist': today, 'error': str(exc), 'ipos': []}


def get_subscription_capture_rows(*, date_ist: Optional[str] = None, limit: int = 500,
                                  db_path: str = DB_PATH) -> Dict[str, Any]:
    """Read-only comparison feed; raw HTML stays in the evidence database."""
    selected_date = date_ist or datetime.now(IST).date().isoformat()
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, ipo_id, company_name, ipo_url, subscription_url,
                   attempted_at_ist, fetched_at_ist, slot_key, closing_date,
                   snapshot_stage, status, http_status, qib_x, nii_x, retail_x,
                   raw_sha256, error, parser_version, source_family, capture_phase
            FROM subscription_capture_v2 WHERE substr(slot_key, 1, 10) = ?
            ORDER BY id DESC LIMIT ?
        """, (selected_date, limit)).fetchall()
    return {'date_ist': selected_date, 'count': len(rows), 'captures': [dict(row) for row in rows]}


class SnapshotHarvesterThread(threading.Thread):
    """
    In-process background harvester polling IST-aligned 15-minute slots
    and five-minute late-day slots before 17:00.
    """
    def __init__(self, interval_seconds: int = 900):
        super().__init__(daemon=True, name="SnapshotHarvesterThread")
        self.interval_seconds = interval_seconds  # retained for older callers
        self._stop_event = threading.Event()
        self._last_slot = None
        self._last_calendar_hour = None
        self._retry_at = None

    def run(self):
        logger.info('Subscription capture harvester started (IST-aligned slots)')
        while not self._stop_event.is_set():
            try:
                now_ist = datetime.now(IST)
                slot = subscription_slot(now_ist)
                if slot and slot != self._last_slot and (self._retry_at is None or now_ist >= self._retry_at):
                    calendar_hour = now_ist.strftime('%Y-%m-%d_%H')
                    if calendar_hour != self._last_calendar_hour:
                        from scrapers.live_ipo_scraper import sync_chittorgarh_calendar_delta
                        try:
                            sync_chittorgarh_calendar_delta(refresh_subscriptions=False)
                            self._last_calendar_hour = calendar_hour
                        except Exception:
                            logger.exception('Calendar refresh failed; capture continues with known offers')
                    inserted = capture_active_subscriptions()
                    self._last_slot = slot
                    self._retry_at = None
                    logger.info('Recorded %d subscription capture attempts at %s IST', inserted,
                                now_ist.strftime('%H:%M'))
                    try:
                        from services.prospective_module1 import cycle as frozen_cycle
                        frozen_cycle()
                    except Exception:
                        logger.exception('Frozen candidate update failed after subscription poll')
                    try:
                        from services.sqlite_supabase_sync import request_sync
                        request_sync('Module 1 subscription capture')
                    except Exception:
                        logger.exception('Could not queue Supabase mirror after subscription capture')
            except Exception as exc:
                self._retry_at = datetime.now(IST) + timedelta(minutes=2)
                logger.exception('Subscription capture harvester error: %s', exc)
            self._stop_event.wait(15)

    def stop(self):
        self._stop_event.set()


harvester_thread = SnapshotHarvesterThread()
