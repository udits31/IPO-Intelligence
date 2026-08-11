"""Append-only, point-in-time Module 2 decisions for prospective evaluation.

The ledger records what the monitor actually returned, including the price-only
call and dated news evidence. It never backfills a decision from later prices.
Outcomes belong in a separate evaluator after their horizon matures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from db import ledger as sqlite3
import threading
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "telemetry" / "module2_decisions.db"
IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_module2_decision(result: dict[str, Any], db_path: Path = DB_PATH,
                            observed_at: datetime | None = None) -> bool:
    """Freeze a daily decision and append material price/news reactions.

    Returns False for stale, undated, or non-raw prices. Those results may be
    displayed, but they are not evidence for a prospective accuracy claim.
    """
    try:
        now = observed_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        now = now.astimezone(timezone.utc)
        observation_day = now.astimezone(IST).date()
        price_day = date.fromisoformat(str(result["evaluation_date"])[:10])
        age = (observation_day - price_day).days
        if not 0 <= age <= 5:
            return False
        if result.get("price_source") not in ("Yahoo Finance (NSE)", "Yahoo Finance (BSE)", "NSE UDiFF CM"):
            return False
        if result.get("price_source") == "NSE UDiFF CM" and not re.fullmatch(
                r"[0-9a-f]{64}", str(result.get("price_archive_sha256") or "")):
            return False
        if result.get("price_basis") != "Unadjusted daily close":
            return False
        if result.get('price_source')=='Yahoo Finance (BSE)' and (
                not str(result.get('bse_code') or '').isdigit()
                or not re.fullmatch(r'[0-9a-f]{64}',str(result.get('price_archive_sha256') or ''))
                or (result.get('price_basis_adjustment') or {}).get('status')!='comparable'):
            return False
        close = float(result["current_price_adjusted"])
        if not math.isfinite(close) or close <= 0:
            return False
        ipo_id = int(result["ipo_id"])
        if ipo_id <= 0:
            return False
        news = result.get("recent_news") or []
        for article in news:
            published = parsedate_to_datetime(article["published_at"]).astimezone(timezone.utc)
            if published > now:
                return False
        payload = {
            "ipo_id": ipo_id,
            "company_name": result.get("company_name"),
            "nse_symbol": result.get("nse_symbol"),
            "bse_code":result.get('bse_code'),
            "observed_at_utc": now.isoformat(),
            "price_date": price_day.isoformat(),
            "price_source": result["price_source"],
            "price_basis": result["price_basis"],
            "price_archive_sha256": result.get("price_archive_sha256"),
            "decision_close": close,
            "benchmark_source": result.get("benchmark_source"),
            "attribution_10d": result.get("attribution_10d"),
            "attribution_30d": result.get("attribution_30d"),
            "price_waterfall": result.get("price_waterfall"),
            "price_basis_adjustment":result.get('price_basis_adjustment'),
            "history_price_basis":result.get('history_price_basis'),
            "lockin_status": result.get("lockin_status"),
            "price_only_verdict": result["price_only_verdict"],
            "news_sentiment": result.get("news_sentiment"),
            "recent_news": news,
            "final_verdict": result["verdict"],
            "news_changed_decision": bool(result.get("news_changed_decision")),
        }
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if sqlite3.cloud_enabled() and payload.get('price_archive_sha256'):
            from services.cloud_assets import persist_file,ensure_file
            source=(ROOT/'data/audit/nse_bhavcopy_udiff'/(price_day.strftime('%Y%m%d')+'.csv.zip')
                    if result['price_source']=='NSE UDiFF CM' else
                    ROOT/'data/archive/yahoo_chart'/(payload['price_archive_sha256']+'.json.gz'))
            try:
                if not ensure_file(source):raise ValueError('Decision source archive is unavailable')
                raw=source.read_bytes()
                if result['price_source']!='NSE UDiFF CM':
                    import gzip
                    raw=gzip.decompress(raw)
                if hashlib.sha256(raw).hexdigest()!=payload['price_archive_sha256']:
                    raise ValueError('Decision source archive checksum mismatch')
                persist_file(source)
            except Exception:
                logger.exception('Source archival failed; no unbacked daily call saved')
                return False
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""CREATE TABLE IF NOT EXISTS module2_decisions_v1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ipo_id INTEGER NOT NULL,
                observation_day_ist TEXT NOT NULL,
                observed_at_utc TEXT NOT NULL,
                price_date TEXT NOT NULL,
                price_only_verdict TEXT NOT NULL,
                final_verdict TEXT NOT NULL,
                news_changed_decision INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL,
                monitor_code_sha256 TEXT NOT NULL,
                attribution_code_sha256 TEXT NOT NULL,
                news_code_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(ipo_id, observation_day_ist)
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS module2_events_v1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ipo_id INTEGER NOT NULL,
                observed_at_utc TEXT NOT NULL,
                observation_day_ist TEXT NOT NULL,
                price_date TEXT NOT NULL,
                signal_sha256 TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_module2_events_ipo_time "
                         "ON module2_events_v1(ipo_id, observed_at_utc)")
            news_keys = sorted({(str(article.get("url") or article.get("link") or ""),
                                 str(article.get("published_at") or "")) for article in news})
            signal = {
                "price_date": price_day.isoformat(), "decision_close": close,
                "price_only_verdict": payload["price_only_verdict"],
                "final_verdict": payload["final_verdict"],
                "news_articles": news_keys,
            }
            signal_hash = hashlib.sha256(json.dumps(signal, sort_keys=True).encode()).hexdigest()
            previous = conn.execute("""SELECT signal_sha256, price_date, payload_json
                FROM module2_events_v1 WHERE ipo_id = ? ORDER BY id DESC LIMIT 1""", (ipo_id,)).fetchone()
            same_price = previous is not None and previous[1] == price_day.isoformat()
            # A temporarily empty RSS response is not proof that yesterday's
            # article disappeared; avoid creating a false intraday reversal.
            news_disappeared = same_price and not news and bool(
                json.loads(previous[2]).get("recent_news")) if previous else False
            if (previous is None or previous[0] != signal_hash) and not news_disappeared:
                conn.execute("""INSERT INTO module2_events_v1
                    (ipo_id, observed_at_utc, observation_day_ist, price_date,
                     signal_sha256, payload_sha256, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                    ipo_id, now.isoformat(), observation_day.isoformat(),
                    price_day.isoformat(), signal_hash, payload_hash, payload_json))
            cursor = conn.execute("""INSERT OR IGNORE INTO module2_decisions_v1 (
                ipo_id, observation_day_ist, observed_at_utc, price_date,
                price_only_verdict, final_verdict, news_changed_decision,
                payload_sha256, monitor_code_sha256, attribution_code_sha256,
                news_code_sha256, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                ipo_id, observation_day.isoformat(), now.isoformat(), price_day.isoformat(),
                str(payload["price_only_verdict"]), str(payload["final_verdict"]),
                int(payload["news_changed_decision"]), payload_hash,
                _sha256(ROOT / "core" / "monitor_engine.py"),
                _sha256(ROOT / "core" / "attribution_engine.py"),
                _sha256(ROOT / "services" / "news_service.py"), payload_json,
            ))
            conn.commit()
            if db_path.resolve().parent == (ROOT / 'data' / 'telemetry').resolve():
                if observed_at is None:
                    try:
                        from services.module2_shadow_forecast import capture_shadow
                        capture_shadow(result,db_path=db_path)
                    except Exception:
                        logger.exception('Research shadow forecast failed; current monitor decision retained')
                from services.sqlite_supabase_sync import request_sync
                request_sync('Module 2 decision ledger')
            return cursor.rowcount == 1
    except (KeyError, TypeError, ValueError, OverflowError, sqlite3.Error) as exc:
        logger.warning("Module 2 prospective decision was not recorded: %s", exc)
        return False


def get_module2_event_history(ipo_id: int, *, limit: int = 30,
                              db_path: Path = DB_PATH) -> dict[str, Any]:
    """Read observed reaction changes; never derive old signals from new data."""
    path = Path(db_path)
    if not sqlite3.exists(path):
        return {"ipo_id": ipo_id, "events": []}
    try:
        with closing(sqlite3.connect(path, timeout=5.0)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT id, observed_at_utc, price_date,
                signal_sha256, payload_sha256, payload_json
                FROM module2_events_v1 WHERE ipo_id = ?
                ORDER BY id DESC LIMIT ?""", (ipo_id, min(max(limit, 1), 100))).fetchall()
        return {"ipo_id": ipo_id, "events": [
            {"id": row["id"], "observed_at_utc": row["observed_at_utc"],
             "price_date": row["price_date"], "signal_sha256": row["signal_sha256"],
             "payload_sha256": row["payload_sha256"],
             "decision": json.loads(row["payload_json"])} for row in rows]}
    except sqlite3.OperationalError:
        return {"ipo_id": ipo_id, "events": []}


def capture_recent_listings(lookback_days: int = 60,
                            require_current_session: bool = False) -> dict[str, Any]:
    """Attempt every verified recent listing, preserving explicit failures.

    Automatic after-close runs require a close dated the current IST day;
    manual runs may freeze the last observed close (for weekends/holidays).
    """
    if not 1 <= lookback_days <= 365:
        raise ValueError("lookback_days must be between 1 and 365")
    from core.monitor_engine import HoldSellMonitor

    today = datetime.now(IST).date()
    listed = pd.read_csv(ROOT / "data" / "canonical_listed_ipos.csv",
                         parse_dates=["listing_date"])
    candidates = listed[
        listed["listing_verified"].eq(True)
        & listed["listing_date"].between(
            pd.Timestamp(today - timedelta(days=lookback_days)), pd.Timestamp(today))
        & listed["nse_symbol"].notna()
    ].sort_values(["listing_date", "id"])
    monitor = HoldSellMonitor()
    records = []
    for row in candidates.itertuples(index=False):
        status = "unavailable"
        try:
            result = monitor.evaluate_holding(int(row.id))
            if "verdict" not in result:
                status = str(result.get("error", "monitor unavailable"))
            elif require_current_session and result.get("evaluation_date") != today.isoformat():
                status = "current session close unavailable"
            else:
                status = "recorded" if record_module2_decision(result) else "not eligible or already frozen"
        except Exception as exc:
            status = f"unavailable ({type(exc).__name__})"
            logger.warning("Module 2 capture failed for IPO %s: %s", row.id, exc)
        records.append({"ipo_id": int(row.id), "nse_symbol": str(row.nse_symbol), "status": status})
    try:
        from services.module2_recent_exchange import recent_exchange_monitors
        recent, coverage = recent_exchange_monitors(
            lookback_days=lookback_days, as_of=today,
            refresh_today=True)
        for ipo_id, recent_monitor in recent.items():
            status = "unavailable"
            try:
                result = recent_monitor.evaluate_holding(ipo_id)
                if "verdict" not in result:
                    status = str(result.get("error", "monitor unavailable"))
                elif require_current_session and result.get("evaluation_date") != today.isoformat():
                    status = "current session NSE archive unavailable"
                else:
                    status = "recorded" if record_module2_decision(result) else "not eligible or already frozen"
            except Exception as exc:
                status = f"unavailable ({type(exc).__name__})"
                logger.warning("Recent NSE Module 2 capture failed for IPO %s: %s", ipo_id, exc)
            records.append({"ipo_id": ipo_id, "nse_symbol": recent_monitor.df_ipos.iloc[0].nse_symbol,
                            "status": status, "source": "NSE UDiFF CM"})
        for ipo_id, reason in coverage["excluded"].items():
            records.append({"ipo_id": int(ipo_id), "status": f"excluded: {reason}",
                            "source": "NSE UDiFF CM"})
    except Exception:
        logger.exception("Recent exchange-listed Module 2 cohort could not be checked")
    try:
        from services.module2_shadow_forecast import refresh_outcome_sources
        refresh_outcome_sources()
    except Exception:
        logger.exception('Research outcome refresh failed; saved forecasts retained')
    return {"observation_day_ist": today.isoformat(), "attempted": len(records),
            "recorded": sum(item["status"] == "recorded" for item in records),
            "records": records}


class Module2CollectorThread(threading.Thread):
    """Freeze one cohort-wide decision after each weekday market session."""

    def __init__(self) -> None:
        super().__init__(name="module2-prospective-collector", daemon=True)
        self._stop_event = threading.Event()
        self._last_attempt_day: date | None = None
        self._last_attempt_at: datetime | None = None
        self._last_news_attempt_at: datetime | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now(IST)
            day = now.date()
            retry_due = (self._last_attempt_at is None or
                         (now - self._last_attempt_at).total_seconds() >= 1800)
            market_hours = day.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (17, 0)
            news_interval = 3600 if market_hours else 7200
            news_due = (self._last_news_attempt_at is None or
                        (now - self._last_news_attempt_at).total_seconds() >= news_interval)
            if 8 <= now.hour < 23 and news_due:
                self._last_news_attempt_at = now
                try:
                    report = capture_recent_listings(lookback_days=60,
                                                     require_current_session=False)
                    logger.info("Module 2 intraday price/news check: %s IPOs checked",
                                report["attempted"])
                    from services.sqlite_supabase_sync import request_sync
                    request_sync('Module 2 price and news capture')
                except Exception:
                    logger.exception("Module 2 intraday news check failed")
            if (day.weekday() < 5 and (now.hour, now.minute) >= (17, 30)
                    and self._last_attempt_day != day and retry_due):
                self._last_attempt_at = now
                try:
                    from services.listing_outcome_service import record_verified_listing_outcomes
                    outcome_report = record_verified_listing_outcomes()
                    logger.info("NSE listing opens recorded: %s", len(outcome_report["saved"]))
                    report = capture_recent_listings(lookback_days=60,
                                                     require_current_session=True)
                    logger.info("Module 2 prospective capture: %s/%s recorded",
                                report["recorded"], report["attempted"])
                    pending = any(
                        "current session" in entry["status"]
                        or entry["status"].startswith("unavailable")
                        for entry in report["records"])
                    if not pending or now.hour >= 23:
                        self._last_attempt_day = day
                    from services.sqlite_supabase_sync import request_sync
                    request_sync('Module 2 close and listing outcome capture')
                except Exception:
                    logger.exception("Module 2 prospective capture failed")
            self._stop_event.wait(60)


module2_collector_thread = Module2CollectorThread()
