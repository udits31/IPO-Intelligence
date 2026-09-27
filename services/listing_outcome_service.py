"""Freeze official listing opens separately from all pre-listing forecasts.

The NSE archive may appear after the first session. A missing archive leaves
the outcome pending; it never causes a forecast to be recomputed or backdated.
"""

from __future__ import annotations

import re
from db import ledger as sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ml.audit_module2_nse_closes import exchange_rows
from scrapers.live_ipo_scraper import load_scraped_cache
from services.module2_recent_exchange import ARCHIVES, _archive_rows, _identity_matches


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "telemetry" / "snapshots.db"
IST = ZoneInfo("Asia/Kolkata")


def verified_listing_outcomes(db_path: Path = DB) -> dict[str, dict]:
    if not sqlite3.exists(db_path):
        return {}
    with closing(sqlite3.connect(db_path, timeout=5)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM listing_outcome_v1").fetchall()
        except sqlite3.OperationalError:
            return {}
        try:
            bse_rows=conn.execute('SELECT * FROM listing_outcome_bse_v1').fetchall()
        except sqlite3.OperationalError:bse_rows=[]
    return {str(row['ipo_id']):dict(row) for row in [*rows,*bse_rows]}


def record_verified_listing_outcomes(*, cache: dict | None = None,
                                     as_of: date | None = None,
                                     db_path: Path = DB,include_all_listed: bool = False) -> dict:
    """Resolve only currently listed, exchange-identifiable forecasted IPOs."""
    cache = cache if cache is not None else load_scraped_cache()
    today = as_of or datetime.now(IST).date()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    saved, pending, excluded = [], [], []
    bse_candidates=[(url,item) for url,item in cache.items() if re.sub(r'[^a-z]','',str(item.get('company_name','')).lower()) in ('nationalstockexchangeofindia','nationalstockexchangeofindialtd','nationalstockexchangeofindialimited')]
    if bse_candidates:
        from services.bse_listing_outcomes import record_bse_outcomes
        bse_result=record_bse_outcomes(dict(bse_candidates),db_path=db_path,as_of=today)
        saved.extend(bse_result['saved']);pending.extend(bse_result['pending'])
    with closing(sqlite3.connect(db_path, timeout=5)) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("""CREATE TABLE IF NOT EXISTS listing_outcome_v1 (
            ipo_id TEXT PRIMARY KEY, company_name TEXT NOT NULL,
            listing_date TEXT NOT NULL, issue_price REAL NOT NULL,
            nse_open_price REAL NOT NULL, nse_symbol TEXT NOT NULL,
            isin TEXT NOT NULL, archive_sha256 TEXT NOT NULL,
            recorded_at_ist TEXT NOT NULL
        )""")
        candidates = {str(row[0]) for row in conn.execute(
            "SELECT DISTINCT ipo_id FROM automatic_decision_v1 WHERE decision_kind = 'live_bid_forecast'")}
        candidates.update(str(row[0]) for row in conn.execute(
            "SELECT ipo_id FROM prelisting_outcome_forecast_v1"))
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='frozen_module1_forecast_v1'").fetchone():
            candidates.update(str(row[0]) for row in conn.execute('SELECT DISTINCT ipo_id FROM frozen_module1_forecast_v1'))
        for url, item in cache.items():
            if any(url==u for u,_ in bse_candidates):continue
            match = re.search(r"/ipo/[^/]+/(\d+)/?", url)
            if not match or (not include_all_listed and f"chitto_{match.group(1)}" not in candidates):
                continue
            ipo_id = f"chitto_{match.group(1)}"
            if conn.execute("SELECT 1 FROM listing_outcome_v1 WHERE ipo_id = ?", (ipo_id,)).fetchone():
                continue
            if item.get("category") != "LISTED" or item.get("source_observed") is not True:
                pending.append({"ipo_id": ipo_id, "reason": "listing not exchange-identifiable yet"})
                continue
            try:
                listing_date = date.fromisoformat(str(item["listing_date"]))
                issue_price = float(item["issue_price"])
            except (KeyError, TypeError, ValueError):
                excluded.append({"ipo_id": ipo_id, "reason": "listing date or issue price missing"})
                continue
            symbol = str(item.get("nse_symbol") or "").strip().upper()
            if listing_date > today or issue_price <= 0 or not re.fullmatch(r"[A-Z0-9]{2,20}", symbol):
                pending.append({"ipo_id": ipo_id, "reason": "listing date future or NSE symbol unavailable"})
                continue
            archive = ARCHIVES / f"{listing_date:%Y%m%d}.csv.zip"
            if not archive.exists():
                exchange_rows(f"{listing_date:%Y%m%d}", {symbol})
            if not archive.exists():
                pending.append({"ipo_id": ipo_id, "reason": "official listing-day archive unavailable"})
                continue
            rows = _archive_rows(archive, {symbol})
            if len(rows) != 1 or not _identity_matches(str(item.get("company_name") or ""),
                                                        str(rows.iloc[0].FinInstrmNm)):
                excluded.append({"ipo_id": ipo_id, "reason": "NSE symbol or issuer identity conflict"})
                continue
            row = rows.iloc[0]
            try:
                opening = float(row.OpnPric)
            except (TypeError, ValueError):
                opening = 0.0
            if opening <= 0 or not str(row.ISIN).startswith("INE"):
                excluded.append({"ipo_id": ipo_id, "reason": "invalid NSE open or ISIN"})
                continue
            conn.execute("""INSERT OR IGNORE INTO listing_outcome_v1
                (ipo_id, company_name, listing_date, issue_price, nse_open_price,
                 nse_symbol, isin, archive_sha256, recorded_at_ist)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                ipo_id, item["company_name"], listing_date.isoformat(), issue_price,
                opening, symbol, str(row.ISIN), str(row.archive_sha256),
                datetime.now(IST).isoformat(timespec="seconds")))
            saved.append(ipo_id)
        conn.commit()
    return {"saved": saved, "pending": pending, "excluded": excluded}
