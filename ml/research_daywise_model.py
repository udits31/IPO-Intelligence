"""Exploratory penultimate-day Module 1 logistic replay on real day-wise bids.

Source pages were read retrospectively and may have been revised. This does
not establish an investor-time backtest or supersede the deployed classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from ml.model_pipeline import summarize
except ModuleNotFoundError:
    from model_pipeline import summarize


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("qib_subscription", "hni_subscription", "retail_subscription")
# NSE's issue archive includes these trust-unit offerings; they are outside
# the ordinary equity IPO population evaluated here.
NON_EQUITY_VEHICLE_IDS = {293, 307}  # PowerGrid InvIT, Brookfield India REIT


def prepare(candidates: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = candidates.merge(canonical[["id", "listing_date", "issue_price", "listing_open_price",
                                        "company_name"]], left_on="ipo_id", right_on="id",
                             suffixes=("", "_canonical"), validate="one_to_one")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["listing_date"] = pd.to_datetime(frame.listing_date, errors="coerce")
    frame["issue_close_date"] = pd.to_datetime(frame.issue_close_date, errors="coerce")
    frame["source_published_at"] = pd.to_datetime(frame.source_published_at, errors="coerce")
    frame["year"] = frame.listing_date.dt.year
    frame["target"] = (pd.to_numeric(frame.listing_open_price, errors="coerce")
                       > pd.to_numeric(frame.issue_price, errors="coerce")).astype(int)
    data_valid = frame[list(FEATURES)].notna().all(axis=1) & frame[list(FEATURES)].ge(0).all(axis=1)
    dated_page = frame.source_published_at.notna() & frame.source_published_at.le(frame.issue_close_date + pd.Timedelta(days=3))
    dated_page &= frame.source_published_at.lt(frame.listing_date)
    # Equitymaster has a date-stamped daily row but exposes no historical
    # page-publication timestamp. It is admissible only as retrospective
    # research, never as a certified investor-time observation.
    undated_page = (frame.get("source_tier", pd.Series("", index=frame.index)).eq("equitymaster_only")
                    & frame.source_url.str.startswith("https://www.equitymaster.com/"))
    source_valid = dated_page | undated_page
    included = data_valid & source_valid & frame.year.between(2021, 2024)
    included &= ~frame.ipo_id.isin(NON_EQUITY_VEHICLE_IDS)
    audit = {"source_rows": int(len(frame)), "source_with_valid_dates_and_features": int(included.sum()),
             "excluded_after_close_or_listing": int((~source_valid).sum()),
             "excluded_invalid_or_missing_features": int((~data_valid).sum()),
             "source_coverage_by_year": frame.groupby("year").size().to_dict()}
    return frame.loc[included].copy(), audit


def run_replay(frame: pd.DataFrame) -> dict:
    if len(frame) < 40:
        raise ValueError("Too few source-complete early-book observations")
    frame = frame.copy()
    for feature in FEATURES:
        frame["log_" + feature] = np.log1p(frame[feature])
    full_cols = ["log_" + feature for feature in FEATURES]
    qib_cols = ["log_qib_subscription"]
    folds = []
    for year in (2022, 2023, 2024):
        train = frame[frame.year.lt(year)]
        test = frame[frame.year.eq(year)]
        if len(train) < 10 or len(test) < 10 or train.target.nunique() < 2 or test.target.nunique() < 2:
            continue
        row = {"test_year": year, "train_rows": int(len(train)), "test_rows": int(len(test))}
        for name, columns in (("three_category_logistic", full_cols), ("qib_only_logistic", qib_cols)):
            scaler = StandardScaler().fit(train[columns])
            model = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
            model.fit(scaler.transform(train[columns]), train.target)
            score = model.predict_proba(scaler.transform(test[columns]))[:, 1]
            row[name] = summarize(test.target, score, 0.5)
        folds.append(row)
    return {
        "stage": "penultimate_bidding_day_end_values_from_retrospective_tables",
        "model_family": "StandardScaler + scikit-learn LogisticRegression(C=0.5)",
        "threshold": 0.5,
        "folds": folds,
        "limitations": [
            "Article pages may have been revised after the decision day; some have no publication timestamp. This is not prospective accuracy.",
            "Source coverage is non-random; score only applies to matched and validated pages.",
            "Final-day subscriptions, GMP, and future market prices are not features.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/daywise_subscription_candidates.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output/daywise_logistic_research.json")
    args = parser.parse_args()
    frame, audit = prepare(pd.read_csv(args.candidates), pd.read_csv(ROOT / "data/canonical_listed_ipos.csv"))
    report = {"cohort_audit": audit, **run_replay(frame)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
