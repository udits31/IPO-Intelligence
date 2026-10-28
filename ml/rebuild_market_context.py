"""Recompute IPO-close market context from verified NSE dates and real closes.

This produces a separate audited file; it never mutates the legacy table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def build_context(offers: pd.DataFrame, observations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    offers = offers.copy()
    offers["issue_close_date"] = pd.to_datetime(offers.issue_close_date, errors="coerce")
    offers["listing_date"] = pd.to_datetime(offers.listing_date, errors="coerce")
    observations = observations[observations.observation_type.eq("CLOSE")].copy()
    observations["observed_at"] = pd.to_datetime(observations.observed_at, errors="coerce")
    observations["value"] = pd.to_numeric(observations.value, errors="coerce")
    pivot = observations.pivot_table(index="observed_at", columns="instrument", values="value", aggfunc="first")
    pivot = pivot[["NIFTY50", "INDIAVIX"]].dropna().sort_index()
    if pivot.empty or pivot.index.has_duplicates:
        raise ValueError("Market close series is empty or has duplicate dates")
    rows = []
    skipped = []
    for offer in offers.itertuples():
        if not 2021 <= offer.listing_date.year <= 2024:
            continue
        if pd.isna(offer.issue_close_date) or offer.issue_close_date >= offer.listing_date:
            skipped.append({"ipo_id": int(offer.ipo_id), "reason": "invalid_offer_chronology"})
            continue
        prior = pivot.index.searchsorted(offer.issue_close_date, side="right") - 1
        if prior < 10 or (offer.issue_close_date - pivot.index[prior]).days > 4:
            skipped.append({"ipo_id": int(offer.ipo_id), "reason": "missing_nearby_ten_session_history"})
            continue
        current = pivot.iloc[prior]
        ten_before = float(pivot.iloc[prior - 10].NIFTY50)
        rows.append({
            "ipo_id": int(offer.ipo_id),
            "company_name": offer.company_name,
            "issue_close_date": offer.issue_close_date.strftime("%Y-%m-%d"),
            "cutoff_trading_date": pivot.index[prior].strftime("%Y-%m-%d"),
            "nifty_close": round(float(current.NIFTY50), 4),
            "nifty_10d_return": round((float(current.NIFTY50) / ten_before - 1) * 100, 4),
            "india_vix": round(float(current.INDIAVIX), 4),
            "date_source_url": offer.source_url,
            "date_source_response_sha256": offer.source_response_sha256,
            "market_source_url": "https://finance.yahoo.com/quote/%5ENSEI ; https://finance.yahoo.com/quote/%5EINDIAVIX",
            "stage": "offer_close_market_context_not_pre_deadline_intraday",
        })
    output = pd.DataFrame(rows)
    report = {
        "verified_development_offer_dates": int(offers.listing_date.dt.year.between(2021, 2024).sum()),
        "complete_market_context": int(len(output)),
        "skipped": skipped,
        "method": "Last Nifty/VIX close on or before verified NSE offer-close date; Nifty return against ten trading sessions earlier. No neutral fills.",
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offers", type=Path, default=ROOT / "data/nse_verified_offer_dates.csv")
    parser.add_argument("--series", type=Path, default=ROOT / "data/raw_daily_market_time_series.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/market_observations_nse_verified_2021_2024.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output/nse_market_context_rebuild.json")
    args = parser.parse_args()
    output, report = build_context(pd.read_csv(args.offers), pd.read_csv(args.series))
    if output.empty:
        raise ValueError("No verified market-context rows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
