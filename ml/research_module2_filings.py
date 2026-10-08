"""Chronological Module 2 logistic test with dated NSE filing features.

This is an exploratory comparison, not a deployed or prospective score.
All filing features use exchange dissemination times before the decision close.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.research_module2_ohlcv import BASE, IPO_CONTEXT, cases, model, score

FILINGS = ROOT / "data/audit/module2_dated_nse_filings.jsonl"
OUTPUT = ROOT / "output/module2_filings_research.json"
PRICE = BASE + IPO_CONTEXT
FILING_FEATURES = ["filings_log", "filings_last10_log", "financial_results_log",
                   "investor_meetings_log", "corporate_action_log",
                   "governance_log", "exchange_query_log", "risk_notice_log"]


def filing_features(row: dict) -> dict:
    items = row["filings"]
    cutoff = pd.Timestamp(row["cutoff_ist"])
    if any(pd.Timestamp(item["disseminated_at_ist"]) > cutoff for item in items):
        raise ValueError(f"Post-decision filing in IPO {row['ipo_id']}")
    last10 = cutoff - pd.Timedelta(days=10)
    subjects = [str(item.get("subject") or "").lower() for item in items]
    full_text = [f"{item.get('subject') or ''} {item.get('details') or ''}".lower()
                 for item in items]
    def count(words: tuple[str, ...], corpus: list[str] = subjects) -> int:
        return sum(any(word in text for word in words) for text in corpus)
    return {
        "ipo_id": int(row["ipo_id"]),
        "filings_log": float(np.log1p(len(items))),
        "filings_last10_log": float(np.log1p(sum(
            pd.Timestamp(item["disseminated_at_ist"]) >= last10 for item in items))),
        "financial_results_log": float(np.log1p(count(
            ("financial result", "investor presentation", "earnings")))),
        "investor_meetings_log": float(np.log1p(count(
            ("analysts/institutional", "investor meet", "conference call")))),
        "corporate_action_log": float(np.log1p(count(
            ("dividend", "bonus", "stock split", "merger", "allotment", "fund rais")))),
        "governance_log": float(np.log1p(count(
            ("resignation", "appointment", "auditor", "compliance officer")))),
        "exchange_query_log": float(np.log1p(count(
            ("spurt in volume", "price movement", "clarification", "rumor", "rumour")))),
        "risk_notice_log": float(np.log1p(count(
            ("default", "penalty", "litigation", "fraud", "insolvency", "winding up"),
            full_text))),
    }


def main() -> None:
    rows = [json.loads(line) for line in FILINGS.read_text(encoding="utf-8").splitlines()]
    verified = [row for row in rows if row["status"] == "verified_response"]
    features = pd.DataFrame(filing_features(row) for row in verified)
    prices, _ = cases()
    prices = prices.dropna(subset=PRICE)
    all_data = prices.merge(features, on="ipo_id", how="inner", validate="one_to_one")
    train = all_data[all_data.year.between(2021, 2023)]
    validation = all_data[all_data.year == 2024]
    dev = all_data[all_data.year.between(2021, 2024)]
    later = all_data[all_data.year.isin((2025, 2026))]
    if min(len(train), len(validation), len(later)) < 10:
        raise ValueError("Not enough verified filings across train, validation, and later years")
    candidates = []
    groups = {"price_ipo_context": PRICE,
              "price_ipo_context_filings": PRICE + FILING_FEATURES}
    for name, cols in groups.items():
        for c in (.01, .1, 1.0):
            fitted = model(c).fit(train[cols], train.up)
            candidates.append({"group": name, "C": c,
                               "validation": score(validation, fitted.predict_proba(validation[cols])[:, 1])})
    chosen = min(candidates, key=lambda x: x["validation"]["brier"])
    # Also report both best-in-group scores on the identical later-year cohort.
    comparisons = {}
    for name, cols in groups.items():
        group_choice = min((x for x in candidates if x["group"] == name),
                           key=lambda x: x["validation"]["brier"])
        fitted = model(group_choice["C"]).fit(dev[cols], dev.up)
        p = fitted.predict_proba(later[cols])[:, 1]
        comparisons[name] = {"C": group_choice["C"], "later": score(later, p),
                             "by_year": {str(year): score(later[later.year == year],
                                 p[later.year.to_numpy() == year]) for year in (2025, 2026)}}
    report = {
        "protocol": "Train 2021-23, select C and feature group on 2024 Brier, refit 2021-24, later-year exploratory check",
        "decision": "31st observed trading close; filings through 15:30 IST that day",
        "source": "NSE corporate announcements and Yahoo historical raw prices",
        "filing_jsonl_sha256": hashlib.sha256(FILINGS.read_bytes()).hexdigest(),
        "response_count": len(rows), "verified_response_count": len(verified),
        "filing_rows_before_cutoff": sum(r["eligible_rows"] or 0 for r in verified),
        "coverage": {"training_n": len(train), "validation_n": len(validation),
                     "development_n": len(dev), "later_n": len(later)},
        "candidates": candidates,
        "selected": {"group": chosen["group"], "C": chosen["C"]},
        "same_cohort_comparisons": comparisons,
        "limitations": [
            "NSE announcement response fetched retrospectively; timestamps are exchange dissemination times, but archive completeness is unverified",
            "Subject-count features are not true textual sentiment or filing content analysis",
            "Yahoo price history is secondary and unadjusted; exchange-price reconciliation remains pending",
            "2025-26 data were viewed in earlier model experiments, so this is exploratory, not blinded",
            "No deployment or paper accuracy claim follows from this test",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"coverage": report["coverage"], "selected": report["selected"],
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
