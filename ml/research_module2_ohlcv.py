"""Research-only, chronological OHLCV challenger for Module 2.

The 31st observed close is the decision point. Only rows through that point
can become features; the close 30 sessions later is the target. This script
does not change the deployed policy or claim prospective accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data/audit/module2_development_ohlcv_2021_2024.csv"
HOLDOUT = ROOT / "data/audit/module2_holdout_ohlcv_2025_2026.csv"
IPO = ROOT / "data/canonical_listed_ipos.csv"
OUTPUT = ROOT / "output/module2_ohlcv_research.json"
CHECKPOINT = 30
HORIZON = 30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


BASE = ["stock_5", "stock_10", "stock_30", "sector_10", "sector_30",
        "nifty_10", "nifty_30", "relative_10", "relative_30", "drawdown_30",
        "volatility_20"]
FLOW = ["volume_ratio_5_20", "turnover_ratio_5_20", "close_location_5",
        "close_location_20", "intraday_range_5", "overnight_gap_5"]
IPO_CONTEXT = ["listing_open_premium_pct", "issue_size_log_cr", "qib_log",
               "retail_log", "hni_log"]


def pct(s: pd.Series, i: int, lag: int) -> float:
    a, b = s.iloc[i], s.iloc[i - lag]
    return float((a / b - 1) * 100) if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def mean_ratio(s: pd.Series, recent: slice, earlier: slice) -> float:
    a, b = s.iloc[recent].mean(), s.iloc[earlier].mean()
    return float(a / b) if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def cases() -> tuple[pd.DataFrame, dict]:
    ipos = pd.read_csv(IPO, parse_dates=["listing_date"]).set_index("id")
    subscriptions = pd.read_csv(ROOT / "data/subscription_observations.csv").set_index("ipo_id")
    rows = []
    rejection = {}
    for path in (DEV, HOLDOUT):
        prices = pd.read_csv(path, parse_dates=["date"])
        for ipo_id, g in prices.groupby("ipo_id"):
            ipo_id = int(ipo_id)
            if ipo_id not in ipos.index or ipo_id in (293, 307):
                continue
            g = g.sort_values("date").reset_index(drop=True)
            reason = None
            if len(g) <= CHECKPOINT + HORIZON:
                reason = "insufficient outcome history"
            elif (g.loc[0, "date"] - ipos.loc[ipo_id, "listing_date"]).days > 7:
                reason = "listing date mismatch"
            elif (g.loc[0, "date"].normalize() == ipos.loc[ipo_id, "listing_date"].normalize()
                  and pd.notna(ipos.loc[ipo_id, "listing_day_close_price"])
                  and abs(float(g.loc[0, "ohlcv_close"])
                          / float(ipos.loc[ipo_id, "listing_day_close_price"]) - 1) > .02):
                reason = "first-day close conflict"
            if reason:
                rejection[reason] = rejection.get(reason, 0) + 1
                continue
            i = CHECKPOINT
            close = g.ohlcv_close.astype(float)
            open_ = g.open.astype(float)
            high = g.high.astype(float)
            low = g.low.astype(float)
            volume = g.volume.astype(float)
            sector = g.sector_index_close.astype(float)
            nifty = g.nifty50_close.astype(float)
            stock_10, stock_30 = pct(close, i, 10), pct(close, i, 30)
            sector_10, sector_30 = pct(sector, i, 10), pct(sector, i, 30)
            band5 = (high.iloc[i-4:i+1] - low.iloc[i-4:i+1]).replace(0, np.nan)
            band20 = (high.iloc[i-19:i+1] - low.iloc[i-19:i+1]).replace(0, np.nan)
            prior = close.shift(1)
            gap = ((open_ / prior) - 1) * 100
            intraday_range = ((high - low) / close.replace(0, np.nan)) * 100
            future = float((close.iloc[i + HORIZON] / close.iloc[i] - 1) * 100)
            ipo = ipos.loc[ipo_id]
            subscription = subscriptions.loc[ipo_id] if ipo_id in subscriptions.index else None
            def log_subscription(field: str) -> float:
                if subscription is None:
                    return np.nan
                value = pd.to_numeric(subscription[field], errors="coerce")
                return float(np.log1p(value)) if pd.notna(value) and value >= 0 else np.nan
            rows.append({
                "ipo_id": ipo_id, "year": int(ipos.loc[ipo_id, "listing_date"].year),
                "decision_date": g.loc[i, "date"].date().isoformat(),
                "future_return_pct": future, "up": int(future > 0),
                "stock_5": pct(close, i, 5), "stock_10": stock_10,
                "stock_30": stock_30, "sector_10": sector_10,
                "sector_30": sector_30, "nifty_10": pct(nifty, i, 10),
                "nifty_30": pct(nifty, i, 30),
                "relative_10": stock_10 - sector_10,
                "relative_30": stock_30 - sector_30,
                "drawdown_30": float((close.iloc[i] / close.iloc[:i+1].max() - 1) * 100),
                "volatility_20": float(close.pct_change().iloc[i-19:i+1].std() * 100),
                "volume_ratio_5_20": mean_ratio(volume, slice(i-4, i+1), slice(i-24, i-4)),
                "turnover_ratio_5_20": mean_ratio(volume * close, slice(i-4, i+1), slice(i-24, i-4)),
                "close_location_5": float(((close.iloc[i-4:i+1] - low.iloc[i-4:i+1]) / band5).mean()),
                "close_location_20": float(((close.iloc[i-19:i+1] - low.iloc[i-19:i+1]) / band20).mean()),
                "intraday_range_5": float(intraday_range.iloc[i-4:i+1].mean()),
                "overnight_gap_5": float(gap.iloc[i-4:i+1].mean()),
                "listing_open_premium_pct": float((ipo.listing_open_price / ipo.issue_price - 1) * 100)
                if pd.notna(ipo.listing_open_price) and pd.notna(ipo.issue_price) and ipo.issue_price > 0 else np.nan,
                "issue_size_log_cr": float(np.log1p(ipo.issue_size_cr))
                if pd.notna(ipo.issue_size_cr) and ipo.issue_size_cr >= 0 else np.nan,
                "qib_log": log_subscription("qib_subscription"),
                "retail_log": log_subscription("retail_subscription"),
                "hni_log": log_subscription("hni_subscription"),
            })
    return pd.DataFrame(rows), rejection


def score(df: pd.DataFrame, probability: np.ndarray) -> dict:
    y = df.up.to_numpy()
    pred = probability >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "up_rate_pct": round(float(y.mean() * 100), 2),
        "accuracy_pct": round(float(accuracy_score(y, pred) * 100), 2),
        "balanced_accuracy_pct": round(float(balanced_accuracy_score(y, pred) * 100), 2),
        "brier": round(float(brier_score_loss(y, probability)), 4),
        "tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "always_sell_pct": round(float((1 - y.mean()) * 100), 2),
        "always_hold_pct": round(float(y.mean() * 100), 2),
    }


def model(c: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=c, random_state=42, max_iter=2000)),
    ])


def main() -> None:
    data, rejected = cases()
    # Strict complete-case research: no synthetic replacement for absent prices.
    missing_count = int(data[BASE + FLOW + IPO_CONTEXT].isna().any(axis=1).sum())
    data = data.dropna(subset=BASE + FLOW + IPO_CONTEXT).copy()
    train = data[data.year.between(2021, 2023)]
    validate = data[data.year == 2024]
    development = data[data.year.between(2021, 2024)]
    later = data[data.year.isin([2025, 2026])]
    candidates = []
    feature_groups = {
        "price": BASE,
        "price_volume_range": BASE + FLOW,
        "price_ipo_context": BASE + IPO_CONTEXT,
        "all_observed": BASE + FLOW + IPO_CONTEXT,
    }
    for feature_set, features in feature_groups.items():
        for c in (.01, .1, 1.0):
            fitted = model(c).fit(train[features], train.up)
            result = score(validate, fitted.predict_proba(validate[features])[:, 1])
            candidates.append({"feature_set": feature_set, "C": c, "validation": result})
    chosen = min(candidates, key=lambda x: x["validation"]["brier"])
    features = feature_groups[chosen["feature_set"]]
    fitted = model(chosen["C"]).fit(development[features], development.up)
    later_probability = fitted.predict_proba(later[features])[:, 1]
    report = {
        "protocol": "Train 2021-23, select feature group and regularization by 2024 Brier, refit 2021-24, test 2025-26 once",
        "checkpoint": "31st observed close; only prior 31 sessions used as features",
        "target": "stock close above checkpoint close 30 trading sessions later",
        "data_source": "Yahoo Finance raw historical OHLCV, retrospective download",
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            DEV, HOLDOUT, IPO, ROOT / "data/subscription_observations.csv")},
        "cohort_rejections": rejected,
        "missing_feature_cases_excluded": missing_count,
        "training_n": len(train), "validation_n": len(validate),
        "development_n": len(development), "later_n": len(later),
        "candidates": candidates, "selected": {"feature_set": chosen["feature_set"], "C": chosen["C"]},
        "later": score(later, later_probability),
        "by_year": {str(year): score(later[later.year == year], later_probability[later.year.to_numpy() == year])
                    for year in (2025, 2026)},
        "feature_missing_counts_development": development[BASE + FLOW + IPO_CONTEXT].isna().sum().to_dict(),
        "limitations": [
            "Retrospective secondary-market prices, not exchange-verified nor prospective decisions",
            "Any IPO with a missing candidate feature is excluded; no values are imputed",
            "No historically as-published news or filings",
            "Later-year cohort has been used in prior exploratory tests; it is no longer a fully blind holdout",
            "No deployed policy changes from this research script",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("training_n", "validation_n", "later_n", "selected", "later", "by_year")}, indent=2))


if __name__ == "__main__":
    main()
