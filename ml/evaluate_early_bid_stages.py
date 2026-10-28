"""Audit day-specific IPO probabilities without borrowing final-book observations.

Equitymaster pages are retrospective tables, not immutable investor-time ticks.
This is a research gate; passing it alone must not authorize production advice.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.fetch_equitymaster_daywise import fetch_page
from ml.fetch_daywise_subscriptions import HEADERS as IPOWATCH_HEADERS, POST_URL, table_days


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["qib", "hni", "retail"]
OFFER_TERMS = ["issue_size_cr", "issue_price"]


def collect() -> tuple[pd.DataFrame, dict]:
    matched = pd.read_csv(ROOT / "data/equitymaster_daywise_candidates.csv")
    canonical = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    canonical = canonical.set_index("id")
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for item in matched.itertuples():
            page_id = int(str(item.source_url).rstrip("/").split("/")[-1])
            futures[pool.submit(fetch_page, page_id)] = item
        for future in as_completed(futures):
            item = futures[future]
            page = future.result()
            if not page or page["source_url"] != item.source_url or page["issue_close_date"] != item.issue_close_date:
                failures.append(int(item.ipo_id))
                continue
            if int(item.ipo_id) not in canonical.index:
                failures.append(int(item.ipo_id))
                continue
            record = canonical.loc[int(item.ipo_id)]
            target = int(float(record.listing_open_price) > float(record.issue_price))
            year = pd.Timestamp(record.listing_date).year
            for day, bids in page["days"].items():
                if day >= max(page["days"]):
                    continue  # Never use final-day book for an early-bid audit.
                rows.append({"ipo_id": int(item.ipo_id), "year": year, "day": int(day),
                             "day_date": bids["date"], "source_url": page["source_url"],
                             "source_family": "equitymaster",
                             "source_response_sha256": page["response_sha256"],
                             "target": target,
                             "issue_size_cr": record.issue_size_cr, "issue_price": record.issue_price,
                             **{name: bids[name] for name in FEATURES}})
    ipowatch = pd.read_csv(ROOT / "data/daywise_subscription_candidates.csv")

    def fetch_ipowatch(item):
        try:
            response = requests.get(POST_URL.format(post_id=int(item.source_post_id)),
                                    headers=IPOWATCH_HEADERS, timeout=20)
            response.raise_for_status()
            page = response.json()
            if str(page.get("link", "")).rstrip("/") != str(item.source_url).rstrip("/"):
                return None
            return table_days(page["content"]["rendered"])
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return None

    ipowatch_failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_ipowatch, item): item for item in ipowatch.itertuples()}
        for future in as_completed(futures):
            item = futures[future]
            days = future.result()
            if not days or int(item.ipo_id) not in canonical.index:
                ipowatch_failures.append(int(item.ipo_id))
                continue
            record = canonical.loc[int(item.ipo_id)]
            target = int(float(record.listing_open_price) > float(record.issue_price))
            year = pd.Timestamp(record.listing_date).year
            for day, bids in days.items():
                if day >= max(days):
                    continue
                rows.append({"ipo_id": int(item.ipo_id), "year": year, "day": int(day),
                             "day_date": None, "source_url": item.source_url,
                             "source_family": "ipowatch", "source_response_sha256": None,
                             "target": target,
                             "issue_size_cr": record.issue_size_cr, "issue_price": record.issue_price,
                             **{name: bids.get(name) for name in FEATURES}})
    frame = pd.DataFrame(rows)
    accepted, conflicts = [], []
    for (ipo_id, day), group in frame.groupby(["ipo_id", "day"]):
        if len(group) > 1:
            first, second = group.iloc[0], group.iloc[1]
            disagreed = [name for name in FEATURES if pd.notna(first[name]) and pd.notna(second[name])
                          and abs(first[name] - second[name]) > max(0.25, 0.2 * max(first[name], second[name]))]
            if disagreed:
                conflicts.append({"ipo_id": int(ipo_id), "day": int(day), "categories": disagreed})
                continue
            accepted.append(group.loc[group.source_family.eq("ipowatch")].iloc[0])
        else:
            accepted.append(group.iloc[0])
    frame = pd.DataFrame(accepted)
    return frame, {"equitymaster_matched_pages": len(matched), "equitymaster_retrieval_failures": failures,
                   "ipowatch_matched_pages": len(ipowatch), "ipowatch_retrieval_failures": ipowatch_failures,
                   "same_day_source_conflicts": conflicts,
                   "rows_by_day": frame.groupby("day").size().to_dict() if not frame.empty else {}}


def evaluate(frame: pd.DataFrame) -> dict:
    folds = []
    for day in sorted(frame.day.unique()):
        stage = frame.loc[frame.day.eq(day)].copy()
        for model_name, columns in (("bids_only", FEATURES),
                                    ("bids_and_offer_terms", FEATURES + OFFER_TERMS)):
            for year in (2022, 2023, 2024):
                train = stage.loc[stage.year.lt(year)]
                test = stage.loc[stage.year.eq(year)]
                if len(train) < 25 or len(test) < 15 or train.target.nunique() < 2 or test.target.nunique() < 2:
                    continue
                # Missing source cells are missing data, never 0x bids. Fit all
                # preprocessing inside each earlier-year training fold.
                pipeline = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, random_state=42),
                )
                pipeline.fit(train[columns], train.target)
                probabilities = pipeline.predict_proba(test[columns])[:, 1]
                labels = (probabilities >= 0.5).astype(int)
                base = max(float(test.target.mean()), 1 - float(test.target.mean()))
                training_prior = float(train.target.mean())
                baseline_brier = float(brier_score_loss(test.target, np.full(len(test), training_prior)))
                folds.append({"model": model_name, "day": int(day), "test_year": year,
                              "train_rows": len(train), "test_rows": len(test),
                              "positive_rate": round(float(test.target.mean()), 4),
                              "accuracy": round(float((labels == test.target).mean()), 4),
                              "always_majority_accuracy": round(base, 4),
                              "balanced_accuracy": round(float(balanced_accuracy_score(test.target, labels)), 4),
                              "auc": round(float(roc_auc_score(test.target, probabilities)), 4),
                              "brier": round(float(brier_score_loss(test.target, probabilities)), 4),
                              "training_prior_brier": round(baseline_brier, 4),
                              "negative_recall": round(float(((labels == 0) & (test.target == 0)).sum() /
                                                             (test.target == 0).sum()), 4)})
    latest = [fold for fold in folds if fold["test_year"] == 2024]
    passes_research_gate = any(
        all(fold["accuracy"] > fold["always_majority_accuracy"]
            and fold["brier"] < fold["training_prior_brier"]
            and fold["negative_recall"] >= 0.3 for fold in latest if fold["model"] == model)
        for model in ("bids_only", "bids_and_offer_terms")
    )
    return {"method": "expanding-year replay, day-number matched, final-day excluded",
            "passes_research_gate": passes_research_gate,
            "folds": folds,
            "limitations": ["Retrospective articles may have been edited after the bidding day.",
                            "End-of-day rows do not validate intraday forecasts.",
                            "Source coverage is incomplete and non-random."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/early_bid_stage_gate.json")
    args = parser.parse_args()
    frame, audit = collect()
    report = {"collection": audit, **evaluate(frame)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
