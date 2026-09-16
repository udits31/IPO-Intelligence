"""Fetch real mid-cap risk-appetite and India VIX histories for Module 2 research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/audit/module2_macro_2021_2026.csv"
START, END = "2021-01-01", "2026-10-17"
TICKERS = {"MID150BEES.NS": "mid150bees_close", "^INDIAVIX": "india_vix_close"}


def main() -> None:
    raw = yf.download(list(TICKERS), start=START, end=END, auto_adjust=False,
                      actions=False, progress=False, threads=False, timeout=20)
    if raw.empty:
        raise RuntimeError("No real market history returned")
    frame = pd.DataFrame({name: raw[("Close", ticker)]
                          for ticker, name in TICKERS.items()})
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame.reset_index(names="date")
    if frame[list(TICKERS.values())].notna().mean().min() < .9:
        raise RuntimeError("Macro history coverage is below 90%; refusing file")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    report = {"source": "Yahoo Finance historical raw Close via yfinance",
              "source_urls": [f"https://finance.yahoo.com/quote/{ticker}/history/" for ticker in TICKERS],
              "symbols": TICKERS, "start": START, "end_exclusive": END,
              "rows": len(frame),
              "coverage_pct": {col: round(float(frame[col].notna().mean() * 100), 2)
                               for col in TICKERS.values()},
              "csv_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
              "note": "Observed values only; missing dates and values are not filled"}
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
