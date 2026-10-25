"""Compare a day-before-close logistic model with a strictly dated context variant.

This is retrospective, source-limited research. It never updates the live bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.model_pipeline import summarize
from ml.research_daywise_model import prepare


ROOT = Path(__file__).resolve().parents[1]
BIDS = ["qib_subscription", "hni_subscription", "retail_subscription"]
CONTEXT = ["issue_price", "issue_size_cr", "nifty_10d_return", "india_vix"]


def attach_decision_context(frame: pd.DataFrame, offers: pd.DataFrame,
                            observations: pd.DataFrame) -> pd.DataFrame:
    """Attach only closes known at the end of the penultimate bidding session."""
    offers = offers[["ipo_id", "issue_open_date", "issue_close_date"]].copy()
    offers["issue_open_date"] = pd.to_datetime(offers.issue_open_date)
    offers["issue_close_date"] = pd.to_datetime(offers.issue_close_date)
    joined = frame.merge(offers, on="ipo_id", suffixes=("", "_verified"), validate="one_to_one")
    if not (pd.to_datetime(joined.issue_close_date) == joined.issue_close_date_verified).all():
        raise ValueError("Candidate and NSE-verified issue-close dates disagree")
    obs = observations[observations.observation_type.eq("CLOSE")].copy()
    obs["observed_at"] = pd.to_datetime(obs.observed_at)
    obs["value"] = pd.to_numeric(obs.value, errors="coerce")
    series = obs.pivot_table(index="observed_at", columns="instrument", values="value",
                             aggfunc="first")[["NIFTY50", "INDIAVIX"]].dropna().sort_index()
    if series.index.has_duplicates or series.empty:
        raise ValueError("Invalid market calendar")
    rows = []
    for offer in joined.itertuples():
        sessions = series.loc[offer.issue_open_date:offer.issue_close_date_verified]
        if len(sessions) < 2 or sessions.index[-1] != offer.issue_close_date_verified:
            continue
        decision_date = sessions.index[-2]
        market_pos = series.index.get_loc(decision_date)
        if market_pos < 10:
            continue
        nifty = float(series.iloc[market_pos].NIFTY50)
        prior = float(series.iloc[market_pos - 10].NIFTY50)
        rows.append({"ipo_id": int(offer.ipo_id),
                     "decision_market_date": decision_date.strftime("%Y-%m-%d"),
                     "nifty_10d_return": (nifty / prior - 1) * 100,
                     "india_vix": float(series.iloc[market_pos].INDIAVIX)})
    return joined.merge(pd.DataFrame(rows), on="ipo_id", validate="one_to_one")


def compare(frame: pd.DataFrame) -> dict:
    frame = frame.copy()
    for col in BIDS + ["issue_price", "issue_size_cr"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[frame[BIDS + CONTEXT].notna().all(axis=1) &
                  frame[BIDS].ge(0).all(axis=1) &
                  frame[["issue_price", "issue_size_cr", "india_vix"]].gt(0).all(axis=1)]
    for col in BIDS + ["issue_price", "issue_size_cr"]:
        frame["log_" + col] = np.log1p(frame[col])
    bid_cols = ["log_" + x for x in BIDS]
    all_cols = bid_cols + ["log_issue_price", "log_issue_size_cr", "nifty_10d_return", "india_vix"]
    folds = []
    for year in (2022, 2023, 2024):
        train, test = frame[frame.year.lt(year)], frame[frame.year.eq(year)]
        if min(len(train), len(test)) < 10 or train.target.nunique() < 2 or test.target.nunique() < 2:
            continue
        result = {"test_year": year, "train_rows": len(train), "test_rows": len(test)}
        for label, cols in (("bids_only", bid_cols), ("bids_plus_dated_context", all_cols)):
            scaler = StandardScaler().fit(train[cols])
            model = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
            model.fit(scaler.transform(train[cols]), train.target)
            result[label] = summarize(test.target, model.predict_proba(scaler.transform(test[cols]))[:, 1], 0.5)
        folds.append(result)
    return {"source_rows_after_complete_case_gate": len(frame), "folds": folds,
            "limitations": ["Retrospective day-wise bid tables are not certified point-in-time snapshots.",
                            "Nifty and VIX are previous-day closes, never final bidding-day closes.",
                            "Source coverage is non-random; this is a challenger research result, not live accuracy."]}


def main() -> None:
    canonical = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    daywise = pd.read_csv(ROOT / "data/daywise_subscription_research_cohort.csv")
    offers = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    market = pd.read_csv(ROOT / "data/raw_daily_market_time_series.csv")
    prepared, audit = prepare(daywise, canonical)
    prepared = prepared.merge(canonical[["id", "issue_size_cr"]], left_on="ipo_id", right_on="id", validate="one_to_one")
    prepared = attach_decision_context(prepared, offers, market)
    report = {"cohort_audit": audit, **compare(prepared)}
    path = ROOT / "output/daywise_logistic_context_research.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
