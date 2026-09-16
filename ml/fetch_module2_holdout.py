"""Fetch dated, auditable raw-close Module 2 development or holdout prices.

Yahoo's historical chart is a secondary price source, not exchange-verified
evidence. The result is deliberately separate from the production price file.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from core.attribution_engine import SECTOR_BENCHMARKS
COHORTS = {
    "development": ("2021-01-01", "2025-01-01", "module2_development_prices_2021_2024"),
    "holdout": ("2025-01-01", "2026-10-17", "module2_holdout_prices_2025_2026"),
}


def close_frame(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      actions=False, progress=False, threads=False,
                      group_by="column", timeout=20)
    if raw.empty:
        return pd.DataFrame()
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, default="holdout")
    args = parser.parse_args()
    start_date, end_date, stem = COHORTS[args.cohort]
    output = ROOT / "data" / "audit" / f"{stem}.csv"
    manifest = output.with_suffix(".json")
    ipo = pd.read_csv(ROOT / "data" / "canonical_listed_ipos.csv")
    ipo["listing_date"] = pd.to_datetime(ipo["listing_date"], errors="coerce")
    eligible = ipo[ipo["listing_date"].between(start_date, end_date, inclusive="left")].copy()
    eligible = eligible[eligible["nse_symbol"].notna()]
    eligible["ticker"] = eligible["nse_symbol"].str.strip() + ".NS"
    tickers = sorted(eligible["ticker"].unique())
    sector_tickers = sorted({symbol for items in SECTOR_BENCHMARKS.values()
                             for symbol, _ in items})
    benchmarks = close_frame(["^NSEI", *sector_tickers], start_date, end_date)
    if benchmarks.empty or "^NSEI" not in benchmarks or benchmarks["^NSEI"].count() < 31:
        raise RuntimeError("Nifty 50 history is unavailable; refusing an incomplete price snapshot")
    batches = []
    for start in range(0, len(tickers), 20):
        batch = tickers[start:start + 20]
        print(f"Fetching symbols {start + 1}-{start + len(batch)} of {len(tickers)}", flush=True)
        batches.append(close_frame(batch, start_date, end_date))
    stock = pd.concat(batches, axis=1) if batches else pd.DataFrame()
    rows = []
    missing = []
    for record in eligible.itertuples(index=False):
        if record.ticker not in stock or stock[record.ticker].dropna().empty:
            missing.append({"ipo_id": int(record.id), "ticker": record.ticker,
                            "reason": "no Yahoo close observations"})
            continue
        listing = pd.Timestamp(record.listing_date).normalize()
        series = stock[record.ticker].dropna()
        series = series[series.index >= listing]
        # Yahoo can carry forward an individual stock's previous close on an
        # NSE holiday (observed on 2026-01-15). Such a row is not a trading
        # session and must not advance any Module 2 checkpoint or horizon.
        series = series[series.index.isin(benchmarks["^NSEI"].dropna().index)]
        if series.empty:
            missing.append({"ipo_id": int(record.id), "ticker": record.ticker,
                            "reason": "no observation on/after listing date"})
            continue
        sector_candidates = SECTOR_BENCHMARKS.get(str(record.sector), [])
        # Select a benchmark using only sessions visible by the fixed
        # 31st-close decision, never its later-year/future availability.
        relevant_days = series.index[:31]
        expected_days = len(relevant_days)
        options = []
        for ticker, label in sector_candidates:
            observed = int(benchmarks[ticker].reindex(relevant_days).count()) if ticker in benchmarks else 0
            if expected_days and observed / expected_days >= 0.8:
                options.append((observed, ticker, label))
        if options:
            _, sector_symbol, sector_label = max(options)
        else:
            sector_symbol, sector_label = "^NSEI", "Nifty 50 broad-market proxy"
        for date, close in series.items():
            sector = benchmarks[sector_symbol].get(date) if sector_symbol in benchmarks else None
            nifty = benchmarks["^NSEI"].get(date) if "^NSEI" in benchmarks else None
            rows.append({"ipo_id": int(record.id), "date": date.date().isoformat(),
                         "close_price_adjusted": float(close),
                         "sector_index_close": sector, "nifty50_close": nifty,
                         "ticker": record.ticker, "sector_ticker": sector_symbol,
                         "benchmark_source": sector_label,
                         "price_basis": "Yahoo historical unadjusted Close"})
    if not rows:
        raise RuntimeError("No IPO prices were fetched; refusing an empty price snapshot")
    frame = pd.DataFrame(rows).sort_values(["ipo_id", "date"])
    if output.exists():
        previous_count = pd.read_csv(output, usecols=["ipo_id"])["ipo_id"].nunique()
        if frame.ipo_id.nunique() < previous_count:
            raise RuntimeError(f"Refusing to replace {previous_count} matched IPOs with only "
                               f"{frame.ipo_id.nunique()}; retry the upstream fetch")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest.write_text(json.dumps({"source": "Yahoo Finance historical daily OHLC",
                                    "source_url_pattern": "https://finance.yahoo.com/quote/{ticker}/history/",
                                    "cohort": args.cohort, "start": start_date, "end_exclusive": end_date,
                                    "price_basis": "unadjusted Close; splits and dividends not back-adjusted",
                                    "eligible_nse_ipo_count": len(eligible),
                                    "matched_ipo_count": int(frame.ipo_id.nunique()),
                                    "rows": len(frame), "missing": missing}, indent=2), encoding="utf-8")
    print(f"Saved {len(frame)} rows for {frame.ipo_id.nunique()} IPOs; missing {len(missing)}")


if __name__ == "__main__":
    main()
