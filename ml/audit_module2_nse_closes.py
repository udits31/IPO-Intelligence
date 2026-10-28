"""Cross-check Module 2 checkpoint closes against official NSE UDiFF bhavcopies.

Read-only with respect to the deployed monitor and its Yahoo research inputs.
Raw exchange ZIPs are cached locally for reproducibility; only requested IPO
symbols/dates are exported to the audit ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/audit/nse_bhavcopy_udiff"
OUTPUT = ROOT / "output/module2_nse_close_audit.csv"
REPORT = ROOT / "output/module2_nse_close_audit.json"
LEDGER = ROOT / "output/module2_holdout_decision_ledger_repaired_benchmarks_v2.csv"
YAHOO = ROOT / "data/audit/module2_holdout_prices_2025_2026.csv"
URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/all-reports"}
CHECKPOINTS = ("listing_date", "decision_date", "execution_date", "outcome_date")


def exchange_rows(date: str, symbols: set[str]) -> tuple[list[dict], dict]:
    url = URL.format(date=date)
    path = CACHE / f"{date}.csv.zip"
    if path.exists():
        content = path.read_bytes()
    else:
        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(url, headers=HEADERS, timeout=25)
                response.raise_for_status()
                content = response.content
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    if len(archive.namelist()) != 1:
                        raise ValueError("Unexpected NSE archive layout")
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                break
            except (requests.RequestException, ValueError, zipfile.BadZipFile) as exc:
                last_error = str(exc)
                if attempt < 2:
                    time.sleep(attempt + 1)
        else:
            return [], {"date": date, "url": url, "status": "fetch_failed", "error": last_error}
    digest = hashlib.sha256(content).hexdigest()
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            table = pd.read_csv(archive.open(archive.namelist()[0]),
                                usecols=["TradDt", "TckrSymb", "SctySrs", "ClsPric"])
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        return [], {"date": date, "url": url, "status": "parse_failed", "error": str(exc),
                    "sha256": digest}
    if not table.TradDt.astype(str).eq(pd.Timestamp(date).strftime("%Y-%m-%d")).all():
        return [], {"date": date, "url": url, "status": "date_mismatch", "sha256": digest}
    table = table[table.TckrSymb.isin(symbols)].copy()
    rows = []
    for symbol, group in table.groupby("TckrSymb"):
        eq = group[group.SctySrs.eq("EQ")]
        selected = eq if len(eq) == 1 else group
        if len(selected) != 1:
            continue
        row = selected.iloc[0]
        rows.append({"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "symbol": symbol,
                     "series": row.SctySrs, "nse_close": float(row.ClsPric),
                     "source_url": url, "archive_sha256": digest})
    return rows, {"date": date, "url": url, "status": "verified_archive", "sha256": digest,
                  "requested_symbols": len(symbols), "matched_symbols": len(rows)}


def audit(ledger: pd.DataFrame, canonical: pd.DataFrame,
          yahoo: pd.DataFrame, exchange: pd.DataFrame) -> pd.DataFrame:
    symbols = canonical[["id", "nse_symbol"]].rename(columns={"id": "ipo_id", "nse_symbol": "symbol"})
    cases = ledger[["ipo_id", "company_name", *CHECKPOINTS]].merge(symbols, on="ipo_id", validate="many_to_one")
    long = cases.melt(id_vars=["ipo_id", "company_name", "symbol"], value_vars=CHECKPOINTS,
                      var_name="checkpoint", value_name="date")
    long["date"] = pd.to_datetime(long.date).dt.strftime("%Y-%m-%d")
    yahoo = yahoo[["ipo_id", "date", "close_price_adjusted"]].drop_duplicates(["ipo_id", "date"])
    merged = long.merge(yahoo, on=["ipo_id", "date"], how="left", validate="many_to_one")
    merged = merged.merge(exchange, on=["date", "symbol"], how="left", validate="many_to_one")
    merged["difference_pct"] = (merged.close_price_adjusted / merged.nse_close - 1).abs() * 100
    return merged.sort_values(["ipo_id", "checkpoint"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    ledger = pd.read_csv(args.ledger)
    canonical = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    yahoo = pd.read_csv(YAHOO)
    symbols = canonical.set_index("id").nse_symbol
    desired: dict[str, set[str]] = defaultdict(set)
    for row in ledger.itertuples():
        symbol = symbols.loc[row.ipo_id]
        for column in CHECKPOINTS:
            desired[pd.Timestamp(getattr(row, column)).strftime("%Y%m%d")].add(symbol)
    data, manifests = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(exchange_rows, date, symbols): date for date, symbols in desired.items()}
        for index, future in enumerate(as_completed(futures), 1):
            rows, manifest = future.result()
            data.extend(rows)
            manifests.append(manifest)
            if index % 25 == 0 or index == len(futures):
                print(f"NSE bhavcopies {index}/{len(futures)}", flush=True)
    if not data:
        raise RuntimeError("No NSE closes fetched; refusing empty audit")
    result = audit(ledger, canonical, yahoo, pd.DataFrame(data))
    output = OUTPUT.with_name(OUTPUT.stem + args.tag + OUTPUT.suffix)
    report_path = REPORT.with_name(REPORT.stem + args.tag + REPORT.suffix)
    output.write_text(result.to_csv(index=False), encoding="utf-8")
    paired = result.dropna(subset=["nse_close", "close_price_adjusted"])
    report = {"source": "Official NSE UDiFF CM bhavcopy final ZIP",
              "source_page": "https://www.nseindia.com/all-reports",
              "checkpoints_requested": len(result), "dates_requested": len(desired),
              "dates_fetched": sum(x["status"] == "verified_archive" for x in manifests),
              "nse_closes_matched": int(result.nse_close.notna().sum()),
              "yahoo_closes_present": int(result.close_price_adjusted.notna().sum()),
              "paired_closes": len(paired),
              "max_paired_discrepancy_pct": round(float(paired.difference_pct.max()), 4),
              "discrepancies_gt_1pct": int(result.difference_pct.gt(1).sum()),
              "discrepancies_gt_5pct": int(result.difference_pct.gt(5).sum()),
              "archive_manifest": sorted(manifests, key=lambda x: x["date"]),
              "limitations": ["Exchange close comparison does not establish intraday executable prices.",
                              "No corporate-action adjustment is applied; large discrepancies require case review."]}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "archive_manifest"}, indent=2))


if __name__ == "__main__":
    main()
