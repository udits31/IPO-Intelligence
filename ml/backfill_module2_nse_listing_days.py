"""Restore Yahoo-missing initial IPO trading days from official NSE archives.

Only pre-first-Yahoo sessions are considered. Exact official closes and
existing dated benchmark observations are used; no prices are synthesized.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.audit_module2_nse_closes import exchange_rows


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/audit/module2_holdout_prices_2025_2026_nse_calendar.csv"
TARGET = ROOT / "data/audit/module2_holdout_prices_2025_2026_nse_audited.csv"
ROWS = ROOT / "output/module2_nse_listing_backfill.csv"
REPORT = ROOT / "output/module2_nse_listing_backfill.json"


def market_value(prices: pd.DataFrame, date: str, column: str,
                 ticker: str | None = None) -> float:
    rows = prices[prices.date.eq(date)]
    if ticker is not None:
        rows = rows[rows.sector_ticker.eq(ticker)]
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty or values.max() - values.min() > 0.001:
        raise ValueError(f"No consistent {column} for {date}, {ticker}")
    return float(values.iloc[0])


def backfill(prices: pd.DataFrame, ipos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dates = sorted(prices.loc[prices.nifty50_close.notna(), "date"].unique())
    if not dates:
        raise ValueError("No reference market trading calendar")
    by_id = ipos.set_index("id", verify_integrity=True)
    requested: dict[str, set[str]] = {}
    targets = []
    for ipo_id, group in prices.groupby("ipo_id"):
        if ipo_id not in by_id.index:
            continue
        listed = str(by_id.loc[ipo_id, "listing_date"])[:10]
        first = str(group.date.min())
        if listed >= first:
            continue
        symbol = str(by_id.loc[ipo_id, "nse_symbol"])
        sector_tickers = group.sector_ticker.dropna().unique()
        if len(sector_tickers) != 1:
            raise ValueError(f"Inconsistent sector benchmark for IPO {ipo_id}")
        for date in dates:
            if listed <= date < first:
                requested.setdefault(date, set()).add(symbol)
                targets.append((int(ipo_id), date, symbol, sector_tickers[0],
                                str(group.benchmark_source.iloc[0])))
    official = {}
    manifests = []
    for date, symbols in requested.items():
        rows, manifest = exchange_rows(date.replace("-", ""), symbols)
        if manifest["status"] != "verified_archive":
            raise ValueError(f"Official archive unavailable for {date}: {manifest}")
        official.update({(row["date"], row["symbol"]): row for row in rows})
        manifests.append(manifest)
    inserts, provenance = [], []
    for ipo_id, date, symbol, sector_ticker, benchmark_source in targets:
        observed = official.get((date, symbol))
        if observed is None:
            raise ValueError(f"No NSE traded close for {ipo_id} {symbol} on {date}")
        nifty = market_value(prices, date, "nifty50_close")
        sector = market_value(prices, date, "sector_index_close", sector_ticker)
        inserts.append({"ipo_id": ipo_id, "date": date,
                        "close_price_adjusted": observed["nse_close"],
                        "sector_index_close": sector, "nifty50_close": nifty,
                        "ticker": f"{symbol}.NS", "sector_ticker": sector_ticker,
                        "benchmark_source": benchmark_source,
                        "price_basis": "Official NSE UDiFF unadjusted Close"})
        provenance.append({"ipo_id": ipo_id, "date": date, "symbol": symbol,
                           "nse_close": observed["nse_close"], "nse_series": observed["series"],
                           "nse_source_url": observed["source_url"],
                           "nse_archive_sha256": observed["archive_sha256"],
                           "benchmark_source": benchmark_source,
                           "benchmark_basis": "Existing Yahoo market close on the same verified NSE session"})
    added = pd.DataFrame(inserts)
    if added.empty:
        raise ValueError("No missing initial trading days found")
    revised = pd.concat([prices, added], ignore_index=True).sort_values(["ipo_id", "date"])
    if revised.duplicated(["ipo_id", "date"]).any() or revised.nifty50_close.isna().any():
        raise ValueError("Backfill introduced duplicate or non-trading rows")
    report = {"ipo_count": added.ipo_id.nunique(), "inserted_exchange_rows": len(added),
              "inserted_by_ipo": added.groupby("ipo_id").size().to_dict(),
              "source_dates": sorted(requested), "source_archives": manifests,
              "rule": "Only missing sessions between canonical NSE listing date and first Yahoo stock close; exact NSE close; benchmark from existing same-day history."}
    return revised, pd.DataFrame(provenance), report


def main() -> None:
    prices = pd.read_csv(SOURCE)
    ipos = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    revised, provenance, report = backfill(prices, ipos)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    revised.to_csv(TARGET, index=False)
    provenance.to_csv(ROWS, index=False)
    report["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report["audited_sha256"] = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_archives"}, indent=2))


if __name__ == "__main__":
    main()
