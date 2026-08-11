"""Extend the live Module 2 cohort with exchange-identified recent IPOs.

The canonical research file deliberately stops earlier than the live calendar.
This adapter does not add rows to that training file or invent price points: it
reads the actual NSE UDiFF daily archives and keeps only names matching the
calendar's listed symbol. Missing archives or identity conflicts are excluded.
"""

from __future__ import annotations

import hashlib
import re
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from core.attribution_engine import AttributionEngine
from core.monitor_engine import HoldSellMonitor
from scrapers.live_ipo_scraper import load_scraped_cache


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "data" / "audit" / "nse_bhavcopy_udiff"
SOURCE = "NSE UDiFF CM"
_ARCHIVE_ATTEMPTS = {}


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Z0-9]+", value.upper())
            if token not in {"LTD", "LIMITED", "N", "L", "AND", "THE"}]


def _identity_matches(calendar_name: str, exchange_name: str) -> bool:
    expected, observed = _tokens(calendar_name), _tokens(exchange_name)
    # Exact issuer-specific abbreviation in NSE's 23-Oct-2026 UDiFF archive.
    # Keep this bounded: globally treating TRANS as transport/transmissions
    # or accepting arbitrary prefixes would weaken issuer identity checks.
    exchange_aliases = {('BONFIGLIOLI', 'TRANS'): ('BONFIGLIOLI', 'TRANSMISSIONS')}
    observed = list(exchange_aliases.get(tuple(observed), tuple(observed)))
    if not expected or not observed or expected[0] != observed[0]:
        return False
    return len(set(expected) & set(observed)) >= min(2, len(expected))


def _archive_rows(path: Path, wanted_symbols: set[str]) -> pd.DataFrame:
    stamp = path.stem.split(".")[0]
    try:
        expected_date = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 1:
                return pd.DataFrame()
            table = pd.read_csv(archive.open(names[0]), usecols=[
                "TradDt", "TckrSymb", "SctySrs", "FinInstrmNm", "ISIN", "OpnPric", "ClsPric"])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return pd.DataFrame()
    if not table.TradDt.astype(str).eq(expected_date.isoformat()).all():
        return pd.DataFrame()
    table = table[table.TckrSymb.isin(wanted_symbols)
                  & table.SctySrs.isin(("EQ", "BE"))].copy()
    table["archive_sha256"] = digest
    return table


def recent_exchange_monitors(*, lookback_days: int = 60,
                             as_of: date | None = None,
                             refresh_today: bool = False) -> tuple[dict[int, HoldSellMonitor], dict]:
    """Return same-rule monitors for newly listed exchange-confirmed symbols."""
    today = as_of or date.today()
    canonical = pd.read_csv(ROOT / "data" / "canonical_listed_ipos.csv")
    known = set(canonical.id.astype(int))
    candidates = {}
    for url, item in load_scraped_cache().items():
        if item.get("category") != "LISTED" or item.get("source_observed") is not True:
            continue
        match = re.search(r"/ipo/[^/]+/(\d+)/?", url)
        if not match:
            continue
        ipo_id = int(match.group(1))
        if ipo_id in known:
            continue
        try:
            listing = date.fromisoformat(str(item["listing_date"]))
            price = float(item["issue_price"])
        except (KeyError, TypeError, ValueError):
            continue
        symbol = str(item.get("nse_symbol") or "").strip().upper()
        if (not symbol or not re.fullmatch(r"[A-Z0-9]{2,20}", symbol)
                or not 0 <= (today - listing).days <= lookback_days or price <= 0):
            continue
        candidates[ipo_id] = {"item": item, "listing": listing,
                              "symbol": symbol, "issue_price": price}
    if not candidates:
        return {}, {"candidates": 0, "exchange_verified": 0, "excluded": {}}

    if refresh_today:
        current_path = ARCHIVES / f"{today:%Y%m%d}.csv.zip"
        if not current_path.exists() and time.monotonic() - _ARCHIVE_ATTEMPTS.get(today, -10000) >= 900:
            _ARCHIVE_ATTEMPTS[today] = time.monotonic()
            from ml.audit_module2_nse_closes import exchange_rows
            exchange_rows(f"{today:%Y%m%d}", {entry["symbol"] for entry in candidates.values()})

    archives = sorted(path for path in ARCHIVES.glob("*.csv.zip")
                      if path.name[:8].isdigit()
                      and min(entry["listing"] for entry in candidates.values())
                      <= date(int(path.name[:4]), int(path.name[4:6]), int(path.name[6:8])) <= today)
    frames = [_archive_rows(path, {entry["symbol"] for entry in candidates.values()})
              for path in archives]
    observed = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) \
        if any(not frame.empty for frame in frames) else pd.DataFrame()
    if observed.empty:
        return {}, {"candidates": len(candidates), "exchange_verified": 0,
                    "excluded": {str(ipo_id): "no NSE archive rows" for ipo_id in candidates}}

    nifty = AttributionEngine._fetch_yahoo_series("^NSEI", ttl_seconds=900)
    nifty = nifty.rename(columns={"close": "nifty50_close"})
    monitors, excluded = {}, {}
    for ipo_id, entry in candidates.items():
        item, symbol, listing = entry["item"], entry["symbol"], entry["listing"]
        rows = observed[observed.TckrSymb.eq(symbol)].copy()
        rows = rows[rows.TradDt.astype(str).ge(listing.isoformat())]
        if rows.empty or not all(_identity_matches(item["company_name"], name)
                                 for name in rows.FinInstrmNm.astype(str)):
            excluded[str(ipo_id)] = "NSE symbol absent or issuer name conflicts"
            continue
        if rows.groupby("TradDt").size().gt(1).any() or rows.ISIN.nunique() != 1:
            excluded[str(ipo_id)] = "duplicate series or changing ISIN"
            continue
        sector, benchmark_source = AttributionEngine._get_sector_series(
            str(item.get("sector") or ""))
        if sector.empty:
            sector = nifty.rename(columns={"nifty50_close": "sector_index_close"})
            benchmark_source = "Nifty 50 broad-market proxy"
        else:
            sector = sector.rename(columns={"close": "sector_index_close"})
        prices = pd.DataFrame({
            "ipo_id": ipo_id,
            "date": pd.to_datetime(rows.TradDt),
            "close_price_adjusted": pd.to_numeric(rows.ClsPric),
            "archive_sha256": rows.archive_sha256,
            "price_source": SOURCE,
            "price_basis": "Unadjusted daily close",
            "benchmark_source": benchmark_source,
        }).merge(nifty, on="date", how="left").merge(
            sector[["date", "sector_index_close"]], on="date", how="left")
        prices = prices.sort_values("date").reset_index(drop=True)
        if prices.close_price_adjusted.isna().any() or prices.close_price_adjusted.le(0).any():
            excluded[str(ipo_id)] = "invalid NSE close"
            continue
        ipo = pd.DataFrame([{
            "id": ipo_id, "company_name": item["company_name"],
            "sector": item.get("sector") or "Unclassified", "nse_symbol": symbol,
            "listing_date": listing.isoformat(), "issue_price": entry["issue_price"],
            "anchor_lockin_50pct_date": (listing + timedelta(days=30)).isoformat(),
            "anchor_lockin_full_date": (listing + timedelta(days=90)).isoformat(),
            "anchor_lockin_dates_verified": False,
            "anchor_lockin_date_basis": "listing_date_heuristic_not_allotment_evidence",
        }])
        # The official archive can appear later than the exchange close. For
        # the scheduled same-day capture only, permit the already-supported
        # dated Yahoo NSE close while retaining the exchange identity check.
        # If Yahoo is still stale, the collector's current-session gate rejects
        # the decision and retries; it never invents today's price.
        has_current_archive = prices.date.dt.date.eq(today).any()
        monitors[ipo_id] = HoldSellMonitor(
            prices_frame=prices, ipos_frame=ipo,
            prefer_supplied_prices=not (refresh_today and not has_current_archive))
    return monitors, {"candidates": len(candidates), "exchange_verified": len(monitors),
                      "excluded": excluded}
