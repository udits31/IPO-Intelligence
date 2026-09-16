"""Fetch observed IPO OHLCV for the fixed Module 2 checkpoint experiment.

Only the first 62 post-listing sessions per IPO are retained. Features must
use sessions 0..30; later sessions are reserved solely for outcomes. Missing
volume/ranges remain missing, and close conflicts are reported, not repaired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "development": ("2021-01-01", "2025-01-01", "2021_2024"),
    "holdout": ("2025-01-01", "2026-10-17", "2025_2026"),
}
FIELDS = {"Open": "open", "High": "high", "Low": "low",
          "Close": "ohlcv_close", "Volume": "volume"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    args = parser.parse_args()
    start, end, years = COHORTS[args.cohort]
    source = ROOT / "data" / "audit" / f"module2_{args.cohort}_prices_{years}.csv"
    output = ROOT / "data" / "audit" / f"module2_{args.cohort}_ohlcv_{years}.csv"
    manifest = output.with_suffix(".json")
    base = pd.read_csv(source, parse_dates=["date"])
    base = base.sort_values(["ipo_id", "date"]).groupby("ipo_id", sort=False).head(62)
    tickers = sorted(base["ticker"].dropna().unique())
    observations = []
    failed = []
    for offset in range(0, len(tickers), 20):
        batch = tickers[offset:offset + 20]
        print(f"Fetching OHLCV {offset + 1}-{offset + len(batch)} of {len(tickers)}", flush=True)
        raw = yf.download(batch, start=start, end=end, auto_adjust=False,
                          actions=False, progress=False, threads=False, timeout=20)
        if raw.empty:
            failed.extend(batch)
            continue
        for ticker in batch:
            if ("Close", ticker) not in raw.columns:
                failed.append(ticker)
                continue
            frame = pd.DataFrame({
                column: raw[(field, ticker)] if (field, ticker) in raw.columns else pd.NA
                for field, column in FIELDS.items()
            })
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame = frame.dropna(subset=["ohlcv_close"]).reset_index(names="date")
            frame["ticker"] = ticker
            observations.append(frame)
    if not observations:
        raise RuntimeError("No internet OHLCV was returned; refusing an empty snapshot")
    observed = pd.concat(observations, ignore_index=True)
    joined = base.merge(observed, on=["ticker", "date"], how="left", validate="many_to_one")
    joined["close_conflict_pct"] = (
        (joined["ohlcv_close"] / joined["close_price_adjusted"] - 1).abs() * 100
    )
    coverage = joined["ohlcv_close"].notna().mean()
    if coverage < .75:
        raise RuntimeError(f"Only {coverage:.1%} of price rows matched real OHLCV; refusing snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False)
    manifest.write_text(json.dumps({
        "source": "Yahoo Finance historical raw OHLCV via yfinance",
        "url_pattern": "https://finance.yahoo.com/quote/{ticker}/history/",
        "source_price_file": str(source.relative_to(ROOT)).replace("\\", "/"),
        "cohort": args.cohort, "start": start, "end_exclusive": end,
        "ipo_count": int(joined.ipo_id.nunique()), "rows": len(joined),
        "ohlcv_row_coverage_pct": round(float(coverage * 100), 2),
        "close_conflict_over_2pct_rows": int((joined["close_conflict_pct"] > 2).sum()),
        "failed_tickers": failed,
        "note": "Fields are internet observations; missing cells are not imputed. Sessions after index 30 are outcomes only.",
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(joined)} rows for {joined.ipo_id.nunique()} IPOs; OHLCV coverage {coverage:.1%}")


if __name__ == "__main__":
    main()
