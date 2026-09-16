"""Fetch as-disseminated NSE filings for the Module 2 research cohort.

Research-only. The NSE endpoint is undocumented and may change or restrict
access. Every success/failure is recorded; missing responses are not empty
filing histories. No post-decision filing is admitted into model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, time as clock_time, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.research_module2_ohlcv import cases

URL = "https://www.nseindia.com/api/corporate-announcements"
OUT = ROOT / "data/audit/module2_dated_nse_filings.jsonl"
MANIFEST = ROOT / "data/audit/module2_dated_nse_filings_manifest.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IPO-Intelligence-Research/1.0)",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    "Accept": "application/json, text/plain, */*",
}


def dissemination_at(item: dict) -> datetime | None:
    raw = item.get("exchdisstime") or item.get("an_dt")
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%d-%b-%Y %H:%M:%S")
    except (TypeError, ValueError):
        return None


def fetch_one(session: requests.Session, ipo_id: int, symbol: str,
              listed: str, decision: str) -> dict:
    begin = pd.Timestamp(listed).strftime("%d-%m-%Y")
    end = pd.Timestamp(decision).strftime("%d-%m-%Y")
    cutoff = datetime.combine(pd.Timestamp(decision).date(), clock_time(15, 30))
    params = {"index": "equities", "symbol": symbol,
              "from_date": begin, "to_date": end}
    result = {"ipo_id": ipo_id, "symbol": symbol, "listing_date": listed,
              "decision_date": decision, "cutoff_ist": cutoff.isoformat(),
              "source_url": URL, "query": params, "retrieved_at_utc": None,
              "status": "unavailable", "http_status": None, "raw_sha256": None,
              "total_rows": None, "eligible_rows": None, "filings": [], "error": None}
    for attempt in range(3):
        try:
            response = session.get(URL, params=params, timeout=20)
            result["http_status"] = response.status_code
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            result["raw_sha256"] = hashlib.sha256(response.content).hexdigest()
            result["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("non-list announcement response")
            result["total_rows"] = len(data)
            seen = set()
            eligible = []
            for item in data:
                stamp = dissemination_at(item)
                if stamp is None or not pd.Timestamp(listed).to_pydatetime() <= stamp <= cutoff:
                    continue
                if str(item.get("symbol", symbol)).upper() != symbol.upper():
                    continue
                key = str(item.get("seq_id") or item.get("attchmntFile") or
                          (stamp.isoformat(), item.get("desc")))
                if key in seen:
                    continue
                seen.add(key)
                eligible.append({"disseminated_at_ist": stamp.isoformat(),
                                 "subject": item.get("desc"),
                                 "details": item.get("attchmntText"),
                                 "attachment_url": item.get("attchmntFile"),
                                 "exchange_sequence_id": item.get("seq_id")})
            result["eligible_rows"] = len(eligible)
            result["filings"] = sorted(eligible, key=lambda x: x["disseminated_at_ist"])
            result["status"] = "verified_response"
            result["error"] = None
            return result
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 means the full eligible cohort")
    args = parser.parse_args()
    cohort, _ = cases()
    cohort = cohort.dropna(subset=["sector_10", "sector_30", "relative_10", "relative_30"])
    ipos = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    cohort = cohort.merge(ipos[["id", "nse_symbol", "listing_date"]],
                          left_on="ipo_id", right_on="id", validate="one_to_one")
    cohort = cohort.sort_values(["year", "ipo_id"])
    if args.limit:
        cohort = cohort.head(args.limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prior[int(row["ipo_id"])] = row
    session = requests.Session()
    session.headers.update(HEADERS)
    results = []
    for i, row in enumerate(cohort.itertuples(index=False), start=1):
        ipo_id = int(row.ipo_id)
        if ipo_id in prior and prior[ipo_id]["status"] == "verified_response":
            result = prior[ipo_id]
        else:
            result = fetch_one(session, ipo_id, str(row.nse_symbol),
                               str(row.listing_date)[:10], str(row.decision_date))
            time.sleep(.5)
        results.append(result)
        if i % 20 == 0 or i == len(cohort):
            print(f"Filings {i}/{len(cohort)}: {sum(r['status'] == 'verified_response' for r in results)} successful", flush=True)
        # Checkpoint the exact cohort so an interrupted run can resume.
        if i % 20 == 0 or i == len(cohort):
            OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
                           encoding="utf-8")
    report = {"source": "NSE corporate announcements endpoint",
              "source_page": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
              "decision_cutoff": "15:30 IST on the 31st observed trading session",
              "requested": len(results),
              "verified_responses": sum(r["status"] == "verified_response" for r in results),
              "unavailable": sum(r["status"] != "verified_response" for r in results),
              "filings_before_cutoff": sum(r["eligible_rows"] or 0 for r in results),
              "notes": ["Response is retrieved after the fact; exchange dissemination timestamps are used to prevent future leakage",
                        "An empty successful response is distinct from a failed request",
                        "Undocumented endpoint; independent archival verification is pending"]}
    MANIFEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
