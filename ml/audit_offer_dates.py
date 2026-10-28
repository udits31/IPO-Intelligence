"""Audit offer-date provenance before any investor-time historical replay.

This is deliberately a read-only gate. A copied or missing date must be
replaced from a dated source, never inferred from the listing date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/canonical_listed_ipos.csv"


def audit_offer_dates(frame: pd.DataFrame) -> dict:
    required = {"id", "company_name", "issue_open_date", "issue_close_date", "listing_date"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = frame.copy()
    for column in ("issue_open_date", "issue_close_date", "listing_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    year = data["listing_date"].dt.year
    development = data[year.between(2021, 2024)].copy()
    later = data[year.between(2025, 2026)].copy()

    def inspect(cohort: pd.DataFrame) -> dict:
        opened = cohort["issue_open_date"]
        closed = cohort["issue_close_date"]
        listed = cohort["listing_date"]
        # An identical March 28–31 date pair appears across many unrelated
        # IPOs and years; flag it for source verification, not auto-correction.
        repeated_pair = opened.dt.strftime("%m-%d").eq("03-28") & closed.dt.strftime("%m-%d").eq("03-31")
        missing = opened.isna() | closed.isna()
        malformed = opened.gt(closed) | closed.ge(listed)
        long_gap = (listed - closed).dt.days.gt(30)
        suspect = repeated_pair | missing | malformed | long_gap
        examples = cohort.loc[suspect, [
            "id", "company_name", "issue_open_date", "issue_close_date", "listing_date"
        ]].head(8).copy()
        for column in ("issue_open_date", "issue_close_date", "listing_date"):
            examples[column] = examples[column].dt.strftime("%Y-%m-%d")
        examples = examples.astype(object).where(pd.notna(examples), None)
        return {
            "records": int(len(cohort)),
            "missing_offer_date": int(missing.sum()),
            "repeated_march_28_31_pair": int(repeated_pair.sum()),
            "chronology_invalid": int(malformed.sum()),
            "close_to_listing_over_30_days": int(long_gap.sum()),
            "records_requiring_source_verification": int(suspect.sum()),
            "records_without_these_flags": int((~suspect).sum()),
            "flagged_examples": examples.to_dict("records"),
        }

    return {
        "development_2021_2024": inspect(development),
        "later_2025_2026": inspect(later),
        "decision": "Do not use these offer dates to construct a pre-deadline backtest until source-verified replacements exist.",
        "scope_note": "The deployed subscription-only logistic model uses listing year and final bids, not offer dates; this audit does not revise its reported final-book score.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CANONICAL)
    parser.add_argument("--output", type=Path, help="Optional JSON audit path")
    args = parser.parse_args()
    report = audit_offer_dates(pd.read_csv(args.input))
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
