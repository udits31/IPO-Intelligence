"""Chronological logistic test of observed mid-cap and VIX conditions.

Uses only index/ETF closes available by each IPO's 31st trading-session close.
Research only; does not alter the deployed Module 2 rules.
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

MACRO = ROOT / "data/audit/module2_macro_2021_2026.csv"
OUT = ROOT / "output/module2_macro_research.json"
PRICE = BASE + IPO_CONTEXT
MACRO_COLS = ["midcap_5", "midcap_10", "midcap_30", "relative_midcap_10",
              "relative_midcap_30", "vix_level", "vix_change_10", "vix_change_30"]


def ret(series: pd.Series, index: int, lag: int) -> float:
    a, b = series.iloc[index], series.iloc[index-lag]
    return float((a / b - 1) * 100) if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def main() -> None:
    cases_frame, _ = cases()
    macro = pd.read_csv(MACRO, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    location = {day.strftime("%Y-%m-%d"): i for i, day in enumerate(macro.date)}
    mid = macro.mid150bees_close
    vix = macro.india_vix_close
    feature_rows = []
    for row in cases_frame.itertuples(index=False):
        i = location.get(row.decision_date)
        if i is None or i < 30:
            continue
        mid5, mid10, mid30 = (ret(mid, i, lag) for lag in (5, 10, 30))
        feature_rows.append({
            "ipo_id": int(row.ipo_id), "midcap_5": mid5, "midcap_10": mid10,
            "midcap_30": mid30,
            "relative_midcap_10": row.stock_10 - mid10,
            "relative_midcap_30": row.stock_30 - mid30,
            "vix_level": float(vix.iloc[i]) if pd.notna(vix.iloc[i]) else np.nan,
            "vix_change_10": ret(vix, i, 10),
            "vix_change_30": ret(vix, i, 30),
        })
    enriched = cases_frame.merge(pd.DataFrame(feature_rows), on="ipo_id", how="left",
                                 validate="one_to_one")
    missing = int(enriched[PRICE + MACRO_COLS].isna().any(axis=1).sum())
    enriched = enriched.dropna(subset=PRICE + MACRO_COLS)
    train = enriched[enriched.year.between(2021, 2023)]
    validation = enriched[enriched.year == 2024]
    dev = enriched[enriched.year.between(2021, 2024)]
    later = enriched[enriched.year.isin((2025, 2026))]
    if min(len(train), len(validation), len(later)) < 10:
        raise ValueError("Insufficient actual macro feature coverage")
    groups = {"price_ipo_context": PRICE,
              "price_ipo_context_macro": PRICE + MACRO_COLS}
    candidates = []
    for name, cols in groups.items():
        for c in (.01, .1, 1.0):
            fitted = model(c).fit(train[cols], train.up)
            candidates.append({"group": name, "C": c,
                               "validation": score(validation, fitted.predict_proba(validation[cols])[:, 1])})
    chosen = min(candidates, key=lambda x: x["validation"]["brier"])
    comparisons = {}
    for name, cols in groups.items():
        selected = min((x for x in candidates if x["group"] == name),
                       key=lambda x: x["validation"]["brier"])
        fitted = model(selected["C"]).fit(dev[cols], dev.up)
        p = fitted.predict_proba(later[cols])[:, 1]
        comparisons[name] = {"C": selected["C"], "later": score(later, p),
                             "by_year": {str(year): score(later[later.year == year],
                                 p[later.year.to_numpy() == year]) for year in (2025, 2026)}}
    report = {
        "protocol": "Train 2021-23, select C/feature group on 2024 Brier, refit 2021-24, test later years exploratorily",
        "decision": "31st observed IPO close, macro closes from that or earlier sessions",
        "source": "Yahoo Finance historical raw ETF and India VIX closes",
        "macro_csv_sha256": hashlib.sha256(MACRO.read_bytes()).hexdigest(),
        "missing_feature_cases_excluded": missing,
        "coverage": {"training_n": len(train), "validation_n": len(validation),
                     "development_n": len(dev), "later_n": len(later)},
        "candidates": candidates,
        "selected": {"group": chosen["group"], "C": chosen["C"]},
        "same_cohort_comparisons": comparisons,
        "limitations": ["Yahoo market data is retrospective and secondary; exchange cross-check pending",
                        "MID150BEES is an ETF proxy, not the official mid-cap index",
                        "No absent value is imputed; excluded cases reduce coverage",
                        "Later years have already been inspected in prior research and are not blind",
                        "Research only; no deployed policy change"],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"coverage": report["coverage"], "selected": report["selected"],
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
