"""Replace secondary stock closes with exact NSE UDiFF closes for research.

Nifty and sector benchmark histories remain separately identified Yahoo
observations; stock closes and their provenance come from the NSE archive.
No forward fill or split normalization is performed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/audit/module2_holdout_prices_2025_2026_nse_audited.csv"
OFFICIAL = ROOT / "data/audit/module2_nse_full_closes_2025_2026.csv"
OUTPUT = ROOT / "data/audit/module2_holdout_prices_2025_2026_nse_full.csv"
REPORT = ROOT / "output/module2_nse_full_price_reconciliation.json"


def reconcile(base: pd.DataFrame, nse: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = base.copy()
    base["symbol"] = base.ticker.str.removesuffix(".NS")
    joined = base.merge(nse, on=["date", "symbol"], how="left", validate="one_to_one")
    joined["source_difference_pct"] = ((joined.close_price_adjusted / joined.nse_close - 1).abs() * 100)
    unmatched = joined.loc[joined.nse_close.isna(), ["ipo_id", "date", "symbol"]].to_dict("records")
    verified = joined[joined.nse_close.notna()].copy()
    verified["close_price_adjusted"] = verified.nse_close
    verified["price_basis"] = "Official NSE UDiFF CM unadjusted Close"
    if verified.duplicated(["ipo_id", "date"]).any() or verified.nifty50_close.isna().any():
        raise ValueError("Invalid official stock history")
    report = {"base_rows": len(base), "official_rows": len(verified),
              "unmatched_secondary_rows": unmatched,
              "stock_ipo_count": verified.ipo_id.nunique(),
              "stock_days_secondary_difference_gt_1pct": int(joined.source_difference_pct.gt(1).sum()),
              "stock_ipos_secondary_difference_gt_1pct": int(joined.loc[joined.source_difference_pct.gt(1), "ipo_id"].nunique()),
              "stock_ipos_with_large_adjustment": joined.loc[joined.source_difference_pct.gt(20), ["ipo_id", "symbol"]].drop_duplicates().to_dict("records"),
              "note": "Only stock prices are exchange-native. Sector and Nifty historical closes remain Yahoo-sourced; no stock-price interpolation."}
    cols = list(base.columns.drop("symbol"))
    return verified[cols].sort_values(["ipo_id", "date"]), report


def main() -> None:
    revised, report = reconcile(pd.read_csv(BASE), pd.read_csv(OFFICIAL))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    revised.to_csv(OUTPUT, index=False)
    report["base_sha256"] = hashlib.sha256(BASE.read_bytes()).hexdigest()
    report["official_source_sha256"] = hashlib.sha256(OFFICIAL.read_bytes()).hexdigest()
    report["reconciled_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
