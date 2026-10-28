"""Compare retrospective day-wise source finals with the current training bids.

Differences are evidence for review, not permission to overwrite either feed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def audit(candidates: pd.DataFrame, stored: pd.DataFrame, offers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = candidates.merge(stored[["ipo_id", "observed_at", "qib_subscription",
                                      "hni_subscription", "retail_subscription"]],
                              on="ipo_id", suffixes=("_daywise", "_stored"), validate="one_to_one")
    joined = joined.merge(offers[["ipo_id", "issue_close_date", "listing_date"]], on="ipo_id",
                          suffixes=("", "_nse"), validate="one_to_one")
    observed = pd.to_datetime(joined.observed_at, errors="coerce", utc=True).dt.date
    close = pd.to_datetime(joined.issue_close_date_nse, errors="coerce").dt.date
    joined["stored_asof_matches_nse_close"] = observed.eq(close)
    for category in ("qib", "hni", "retail"):
        source = pd.to_numeric(joined[f"final_{category}_on_page"], errors="coerce")
        old = pd.to_numeric(joined[f"{category}_subscription_stored"], errors="coerce")
        joined[f"final_{category}_absolute_gap"] = (source - old).abs().round(3)
        joined[f"final_{category}_material_mismatch"] = (
            (source - old).abs().gt(0.25) &
            (source - old).abs().div(source.clip(lower=1.0)).gt(0.10)
        )
    joined["any_material_final_mismatch"] = joined[[
        f"final_{category}_material_mismatch" for category in ("qib", "hni", "retail")
    ]].any(axis=1)
    report = {
        "source_matched_records": int(len(joined)),
        "stored_timestamp_does_not_match_offer_close": int((~joined.stored_asof_matches_nse_close).sum()),
        "material_final_mismatch_any_category": int(joined.any_material_final_mismatch.sum()),
        "material_final_mismatch_by_category": {
            category: int(joined[f"final_{category}_material_mismatch"].sum())
            for category in ("qib", "hni", "retail")
        },
        "interpretation": "The later-published table is an independent source candidate, not automatically ground truth. Review large discrepancies against exchange/issuer basis records before retraining.",
    }
    return joined, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/daywise_subscription_candidates.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output/subscription_source_discrepancies.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output/subscription_source_audit.json")
    args = parser.parse_args()
    joined, report = audit(pd.read_csv(args.candidates),
                           pd.read_csv(ROOT / "data/subscription_observations.csv"),
                           pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
