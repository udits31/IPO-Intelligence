"""Collect official daily NSE closes for the Module 2 holdout cohort.

Uses the trading dates already present in the verified Nifty calendar. Raw
exchange archives are cached and hashed by audit_module2_nse_closes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ml.audit_module2_nse_closes import exchange_rows


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/audit/module2_holdout_prices_2025_2026_nse_audited.csv"
OUTPUT = ROOT / "data/audit/module2_nse_full_closes_2025_2026.csv"
REPORT = ROOT / "output/module2_nse_full_history_collection.json"


def requests_by_date(prices: pd.DataFrame, canonical: pd.DataFrame) -> dict[str, set[str]]:
    listing = canonical.set_index("id")[["listing_date", "nse_symbol"]]
    ids = sorted(prices.ipo_id.unique())
    ipo = listing.loc[ids].copy()
    ipo["listing_date"] = pd.to_datetime(ipo.listing_date)
    wanted = defaultdict(set)
    for date in sorted(prices.date.unique()):
        date_asof = pd.Timestamp(date)
        symbols = ipo.loc[ipo.listing_date.le(date_asof), "nse_symbol"]
        wanted[date.replace("-", "")].update(symbols.astype(str))
    return dict(wanted)


def main() -> None:
    prices = pd.read_csv(INPUT)
    canonical = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    wanted = requests_by_date(prices, canonical)
    rows, manifests = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(exchange_rows, date, symbols): date for date, symbols in wanted.items()}
        for index, future in enumerate(as_completed(futures), 1):
            found, manifest = future.result()
            rows.extend(found)
            manifests.append(manifest)
            if index % 40 == 0 or index == len(futures):
                print(f"Full NSE closes {index}/{len(futures)}", flush=True)
    if not rows:
        raise RuntimeError("No official daily stock closes available")
    output = pd.DataFrame(rows).sort_values(["date", "symbol"])
    if output.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate NSE symbol-date rows")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    report = {"source_page": "https://www.nseindia.com/all-reports",
              "dates_requested": len(wanted),
              "verified_archive_dates": sum(m["status"] == "verified_archive" for m in manifests),
              "official_stock_close_rows": len(output),
              "unique_symbols": output.symbol.nunique(),
              "failed_archives": [m for m in manifests if m["status"] != "verified_archive"],
              "limitations": "Official unadjusted closes, not intraday execution prices; corporate actions and non-trading stock days require separate reconciliation."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
