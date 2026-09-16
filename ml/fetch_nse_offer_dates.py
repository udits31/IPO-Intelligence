"""Recover source-dated IPO offer periods from NSE's public past-issues list.

The canonical training file is intentionally not rewritten here. Accepted
records form an overlay with a raw-response SHA and deterministic match gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PAGE_URL = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"
API_URL = "https://www.nseindia.com/api/public-past-issues"
DEFAULT_OUTPUT = ROOT / "data/nse_verified_offer_dates.csv"
DEFAULT_REPORT = ROOT / "output/nse_offer_date_match_report.json"
DEFAULT_RAW = ROOT / "data/nse_public_past_issues_raw.json"


def normalized_name(value: object) -> str:
    words = re.findall(r"[a-z0-9]+", str(value).lower())
    while words and words[-1] in {"ltd", "limited"}:
        words.pop()
    return " ".join(words)


def parse_nse_date(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, format="mixed", dayfirst=True, errors="coerce")


def fetch_records() -> tuple[list[dict], bytes]:
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": PAGE_URL,
    }
    session.get(PAGE_URL, headers=headers, timeout=25).raise_for_status()
    response = session.get(API_URL, headers=headers, timeout=30)
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list) or len(records) < 100:
        raise ValueError("NSE past-issues response is not a plausible historical list")
    return records, response.content


def match_records(canonical: pd.DataFrame, records: list[dict], raw_sha: str) -> tuple[pd.DataFrame, dict]:
    expected = {"id", "company_name", "nse_symbol", "listing_date", "issue_price"}
    if not expected.issubset(canonical.columns):
        raise ValueError(f"Canonical file lacks {sorted(expected - set(canonical.columns))}")
    nse = pd.DataFrame(records)
    for field in ("ipoStartDate", "ipoEndDate", "listingDate", "issuePrice", "symbol"):
        if field not in nse:
            raise ValueError(f"NSE data lacks {field}")
    nse["start"] = nse.ipoStartDate.map(parse_nse_date)
    nse["end"] = nse.ipoEndDate.map(parse_nse_date)
    nse["listed"] = nse.listingDate.map(parse_nse_date)
    nse["price"] = pd.to_numeric(nse.issuePrice.astype(str).str.strip(), errors="coerce")
    nse["name_key"] = nse.get("company", nse.get("companyName", pd.Series("", index=nse.index))).map(normalized_name)
    nse["symbol_key"] = nse.symbol.fillna("").astype(str).str.upper().str.strip()
    original = canonical.copy()
    original["listed"] = pd.to_datetime(original.listing_date, errors="coerce")
    original["price"] = pd.to_numeric(original.issue_price, errors="coerce")
    original["name_key"] = original.company_name.map(normalized_name)
    original["symbol_key"] = original.nse_symbol.fillna("").astype(str).str.upper().str.strip()
    original = original[original.listed.dt.year.between(2021, 2026)]
    accepted: list[dict] = []
    unresolved: list[dict] = []
    for row in original.itertuples():
        candidate = nse[nse.listed.eq(row.listed) & nse.price.sub(row.price).abs().le(0.01)]
        candidate = candidate[candidate.start.notna() & candidate.end.notna()
                              & candidate.start.le(candidate.end) & candidate.end.lt(candidate.listed)]
        identity = candidate.name_key.eq(row.name_key) | (
            candidate.symbol_key.eq(row.symbol_key) & candidate.symbol_key.ne("")
        )
        candidate = candidate[identity]
        if len(candidate) != 1:
            unresolved.append({"ipo_id": int(row.id), "company_name": row.company_name,
                               "candidate_count": int(len(candidate))})
            continue
        source = candidate.iloc[0]
        accepted.append({
            "ipo_id": int(row.id), "company_name": row.company_name,
            "issue_open_date": source.start.strftime("%Y-%m-%d"),
            "issue_close_date": source.end.strftime("%Y-%m-%d"),
            "listing_date": source.listed.strftime("%Y-%m-%d"),
            "issue_price": float(source.price),
            "nse_issue_symbol": str(source.symbol),
            "identity_match": "name_and_symbol" if source.name_key == row.name_key and source.symbol_key == row.symbol_key
                              else "name" if source.name_key == row.name_key else "symbol",
            "source_url": API_URL,
            "source_response_sha256": raw_sha,
        })
    output = pd.DataFrame(accepted)
    if not output.empty and output.ipo_id.duplicated().any():
        raise ValueError("Duplicate canonical IDs in accepted offer dates")
    report = {
        "source_url": API_URL,
        "source_response_sha256": raw_sha,
        "nse_response_records": int(len(nse)),
        "canonical_2021_2026_records": int(len(original)),
        "accepted": int(len(output)),
        "unresolved": int(len(unresolved)),
        "identity_match_counts": output.identity_match.value_counts().to_dict() if not output.empty else {},
        "unresolved_examples": unresolved[:30],
        "method": "Accept only one NSE record with identical listing date and issue price, valid offer chronology, and exact normalized company name or NSE symbol. No date inference or fuzzy match.",
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=ROOT / "data/canonical_listed_ipos.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    records, raw_bytes = fetch_records()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    matched, report = match_records(pd.read_csv(args.canonical), records, raw_sha)
    if matched.empty:
        raise ValueError("No NSE offer dates passed the identity gates")
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.raw.write_bytes(raw_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output, index=False)
    report["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
