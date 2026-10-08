"""Small, chronological Module 2 price-model comparison (research only).

Only pre-decision price history is used. The model is selected on 2024 and
refitted through 2024 before later-year evaluation. No production promotion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data" / "audit" / "module2_development_prices_2021_2024.csv"
LATER = ROOT / "data" / "audit" / "module2_holdout_prices_2025_2026.csv"
IPOS = ROOT / "data" / "canonical_listed_ipos.csv"
OUTPUT = ROOT / "output" / "module2_price_model_research.json"
CHECKPOINT, HORIZON = 30, 30
FEATURES = ["stock_5", "stock_10", "stock_30", "sector_5", "sector_10",
            "sector_30", "nifty_10", "nifty_30", "spread_10", "spread_30",
            "drawdown_30", "volatility_20"]


def ratio(series: pd.Series, i: int, lag: int) -> float:
    a, b = series.iloc[i], series.iloc[i - lag]
    return (a / b - 1) * 100 if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def cases(price_paths: tuple[Path, Path] | None = None) -> pd.DataFrame:
    ipos = pd.read_csv(IPOS, parse_dates=["listing_date"]).set_index("id")
    frames = [pd.read_csv(path, parse_dates=["date"])
              for path in (price_paths or (OLD, LATER))]
    rows = []
    for prices in frames:
        for ipo_id, group in prices.groupby("ipo_id"):
            ipo_id = int(ipo_id)
            if ipo_id not in ipos.index or ipo_id in (293, 307):
                continue
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) <= CHECKPOINT + HORIZON + 1 or (
                group.loc[0, "date"] - ipos.loc[ipo_id, "listing_date"]
            ).days > 7:
                continue
            if (group.loc[0, "date"].normalize() == ipos.loc[ipo_id, "listing_date"].normalize()
                and pd.notna(ipos.loc[ipo_id, "listing_day_close_price"])
                and abs(float(group.loc[0, "close_price_adjusted"])
                        / float(ipos.loc[ipo_id, "listing_day_close_price"]) - 1) > .02):
                continue
            stock = group["close_price_adjusted"].astype(float)
            sector = group["sector_index_close"].astype(float)
            nifty = group["nifty50_close"].astype(float)
            i = CHECKPOINT
            s10, s30 = ratio(stock, i, 10), ratio(stock, i, 30)
            sec10, sec30 = ratio(sector, i, 10), ratio(sector, i, 30)
            future = (stock.iloc[i + HORIZON] / stock.iloc[i] - 1) * 100
            rows.append({
                "ipo_id": ipo_id,
                "year": int(ipos.loc[ipo_id, "listing_date"].year),
                "decision_date": group.loc[i, "date"].date().isoformat(),
                "future_return_pct": future, "up": int(future > 0),
                "stock_5": ratio(stock, i, 5), "stock_10": s10, "stock_30": s30,
                "sector_5": ratio(sector, i, 5), "sector_10": sec10, "sector_30": sec30,
                "nifty_10": ratio(nifty, i, 10), "nifty_30": ratio(nifty, i, 30),
                "spread_10": s10 - sec10, "spread_30": s30 - sec30,
                "drawdown_30": (stock.iloc[i] / stock.iloc[:i + 1].max() - 1) * 100,
                "volatility_20": stock.pct_change().iloc[i - 19:i + 1].std() * 100,
            })
    return pd.DataFrame(rows)


def score(data: pd.DataFrame, probability: np.ndarray) -> dict:
    y = data.up.to_numpy()
    pred = probability >= .5
    return {"n": len(y), "up_rate_pct": round(y.mean() * 100, 2),
            "accuracy_pct": round(accuracy_score(y, pred) * 100, 2),
            "balanced_accuracy_pct": round(balanced_accuracy_score(y, pred) * 100, 2),
            "always_sell_pct": round((1 - y.mean()) * 100, 2),
            "always_hold_pct": round(y.mean() * 100, 2),
            "brier": round(brier_score_loss(y, probability), 4),
            "auc": round(roc_auc_score(y, probability), 4) if len(np.unique(y)) > 1 else None,
            "predicted_up_pct": round(pred.mean() * 100, 2)}


def model(c: float) -> Pipeline:
    return Pipeline([
        ("pre", ColumnTransformer([("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), FEATURES)])),
        ("logistic", LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])


def main() -> None:
    data = cases()
    training = data[data.year.between(2021, 2023)]
    validation = data[data.year == 2024]
    later = data[data.year.isin([2025, 2026])]
    candidates = []
    for c in (.01, .1, 1, 10):
        fitted = model(c).fit(training[FEATURES], training.up)
        probabilities = fitted.predict_proba(validation[FEATURES])[:, 1]
        candidates.append({"C": c, "validation": score(validation, probabilities)})
    selected = min(candidates, key=lambda x: x["validation"]["brier"])["C"]
    dev = data[data.year.between(2021, 2024)]
    fitted = model(selected).fit(dev[FEATURES], dev.up)
    later_p = fitted.predict_proba(later[FEATURES])[:, 1]
    results = {"protocol": "train 2021-23, select regularization by 2024 Brier, refit 2021-24, test 2025-26 once",
               "features": FEATURES, "training_n": len(training), "validation_n": len(validation),
               "candidates": candidates, "selected_C": selected,
               "later": score(later, later_p),
               "by_year": {str(y): score(later[later.year == y], later_p[later.year.to_numpy() == y])
                           for y in (2025, 2026)},
               "feature_medians_training": dev[FEATURES].median().to_dict(),
               "limitations": ["Both price cohorts use Yahoo unadjusted closes, retrieved after the fact; exchange verification remains pending",
                               "One price-only checkpoint per IPO; no as-published filings or news",
                               "Exploratory comparison after observing deployed-rule later-year aggregate; not blinded"]}
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ["training_n", "validation_n", "candidates", "selected_C", "later", "by_year"]}, indent=2))


if __name__ == "__main__":
    main()
