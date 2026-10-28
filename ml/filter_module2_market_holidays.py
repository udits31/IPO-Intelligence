"""Remove documented non-trading carry-forward rows from the holdout snapshot.

The raw Yahoo download remains untouched. Unknown missing-index dates cause a
hard failure rather than being silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/audit/module2_holdout_prices_2025_2026.csv"
TARGET = ROOT / "data/audit/module2_holdout_prices_2025_2026_nse_calendar.csv"
REPORT = ROOT / "output/module2_market_calendar_correction.json"
VERIFIED_HOLIDAYS = {
    "2026-01-15": "https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf",
}


def filter_holidays(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = frame.nifty50_close.isna()
    dates = set(frame.loc[missing, "date"].astype(str))
    unknown = dates - VERIFIED_HOLIDAYS.keys()
    if unknown:
        raise ValueError(f"Missing Nifty history on unverified dates: {sorted(unknown)}")
    revised = frame.loc[~missing].copy()
    if revised.duplicated(["ipo_id", "date"]).any():
        raise ValueError("Duplicate stock/date rows")
    if not revised.ipo_id.unique().size == frame.ipo_id.unique().size:
        raise ValueError("Entire IPO history removed by calendar filter")
    report = {"source_rows": len(frame), "retained_rows": len(revised),
              "removed_rows": int(missing.sum()), "removed_by_date": frame.loc[missing, "date"].value_counts().to_dict(),
              "official_holiday_evidence": {date: VERIFIED_HOLIDAYS[date] for date in dates},
              "rule": "Remove stock rows without a Nifty close only when NSE officially confirms the market holiday. No price interpolation or replacement."}
    return revised, report


def main() -> None:
    original = pd.read_csv(SOURCE)
    revised, report = filter_holidays(original)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    revised.to_csv(TARGET, index=False)
    report["raw_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report["audited_sha256"] = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
