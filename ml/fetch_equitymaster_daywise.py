"""Collect independent retrospective day-wise IPO tables for source comparison.

Only exact IPO identity and NSE-verified dates are joined. These pages do not
prove their historical publication time and are never treated as live ticks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from ml.fetch_nse_offer_dates import normalized_name
except ModuleNotFoundError:
    from fetch_nse_offer_dates import normalized_name


ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.equitymaster.com/research-it/ipo/current-ipo/{page_id}/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IPOResearchAudit/1.0)"}


def fetch_page(page_id: int) -> dict | None:
    try:
        response = requests.get(URL.format(page_id=page_id), headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    heading = soup.find("h1")
    if heading is None or "IPO" not in heading.get_text(" ", strip=True).upper():
        return None
    company_name = re.split(r"\s+IPO\s*:", heading.get_text(" ", strip=True), flags=re.I)[0]
    issue_close = listing = None
    table_rows: list[list[str]] | None = None
    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if "IPO Timetable" in text and "Bid/Offer Close Date" in text:
            pairs = [[cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                     for row in table.find_all("tr")]
            fields = {row[0]: row[1] for row in pairs if len(row) == 2}
            issue_close = pd.to_datetime(fields.get("Bid/Offer Close Date"), errors="coerce", dayfirst=True)
            listing = pd.to_datetime(fields.get("Listing Date"), errors="coerce", dayfirst=True)
        if "Date wise Subscription" in text and re.search(r"Day\s*2", text, re.I):
            table_rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                          for row in table.find_all("tr")]
    if issue_close is None or listing is None or pd.isna(issue_close) or pd.isna(listing) or not table_rows:
        return None
    header_index = next((i for i, row in enumerate(table_rows)
                         if len(row) >= 4 and row[0].lower() == "date"
                         and {"qib", "hni", "retail"}.issubset({x.lower() for x in row[1:]})), None)
    if header_index is None:
        return None
    header = [x.lower() for x in table_rows[header_index]]
    values = {}
    for row in table_rows[header_index + 1:]:
        match = re.match(r"Day\s*(\d+)\s*\(([^)]+)\)", row[0], re.I)
        if not match:
            continue
        day = int(match.group(1))
        date = pd.to_datetime(match.group(2), errors="coerce", dayfirst=True)
        if pd.isna(date):
            continue
        item = {"day": day, "date": date.strftime("%Y-%m-%d")}
        for category in ("qib", "hni", "retail"):
            raw = row[header.index(category)] if header.index(category) < len(row) else ""
            item[category] = float(raw.replace(",", "")) if re.fullmatch(r"\d+(?:\.\d+)?", raw) else None
        values[day] = item
    if len(values) < 2:
        return None
    return {"page_id": page_id, "company_name": company_name,
            "issue_close_date": issue_close.strftime("%Y-%m-%d"),
            "listing_date": listing.strftime("%Y-%m-%d"), "days": values,
            "source_url": URL.format(page_id=page_id),
            "response_sha256": hashlib.sha256(response.content).hexdigest()}


def match_pages(pages: list[dict], offers: pd.DataFrame, market_days: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    offers = offers.copy()
    offers["name_key"] = offers.company_name.map(normalized_name)
    offers["listing_date"] = pd.to_datetime(offers.listing_date)
    offers["issue_open_date"] = pd.to_datetime(offers.issue_open_date)
    offers["issue_close_date"] = pd.to_datetime(offers.issue_close_date)
    accepted, rejected = [], []
    for page in pages:
        close = pd.Timestamp(page["issue_close_date"])
        listed = pd.Timestamp(page["listing_date"])
        matches = offers[offers.name_key.eq(normalized_name(page["company_name"]))
                         & offers.listing_date.eq(listed) & offers.issue_close_date.eq(close)]
        if len(matches) != 1:
            rejected.append({"page_id": page["page_id"], "reason": "identity_or_nse_date_mismatch"})
            continue
        offer = matches.iloc[0]
        sessions = market_days[(market_days >= offer.issue_open_date) & (market_days <= close)]
        day_number = len(sessions) - 1
        if day_number < 1 or day_number not in page["days"] or len(sessions) not in page["days"]:
            rejected.append({"page_id": page["page_id"], "reason": "missing_penultimate_or_final_day"})
            continue
        prior = page["days"][day_number]
        final = page["days"][len(sessions)]
        if prior["date"] != sessions[-2].strftime("%Y-%m-%d") or final["date"] != sessions[-1].strftime("%Y-%m-%d"):
            rejected.append({"page_id": page["page_id"], "reason": "day_number_date_mismatch"})
            continue
        if any(prior[key] is None or final[key] is None for key in ("qib", "hni", "retail")):
            rejected.append({"page_id": page["page_id"], "reason": "missing_category"})
            continue
        accepted.append({
            "ipo_id": int(offer.ipo_id), "company_name": offer.company_name,
            "issue_close_date": close.strftime("%Y-%m-%d"), "listing_date": listed.strftime("%Y-%m-%d"),
            "penultimate_day_date": prior["date"],
            "qib_subscription": prior["qib"], "hni_subscription": prior["hni"],
            "retail_subscription": prior["retail"],
            "final_qib_on_page": final["qib"], "final_hni_on_page": final["hni"],
            "final_retail_on_page": final["retail"],
            "source_url": page["source_url"], "source_response_sha256": page["response_sha256"],
            "evidence_class": "retrospective_daywise_table_not_timestamped_day_snapshot",
        })
    output = pd.DataFrame(accepted)
    if not output.empty and output.ipo_id.duplicated().any():
        raise ValueError("Multiple Equitymaster pages matched the same IPO")
    return output, {"candidate_pages": len(pages), "accepted": len(accepted),
                    "rejection_counts": pd.Series([item["reason"] for item in rejected]).value_counts().to_dict(),
                    "rejected_examples": rejected[:25]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=320)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data/equitymaster_daywise_candidates.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output/equitymaster_daywise_collection.json")
    args = parser.parse_args()
    pages = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_page, page_id): page_id for page_id in range(args.start, args.end + 1)}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                pages.append(result)
            if index % 40 == 0:
                print(f"checked={index} parsed={len(pages)}", flush=True)
    offers = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    market = pd.read_csv(ROOT / "data/raw_daily_market_time_series.csv")
    market_days = pd.DatetimeIndex(pd.to_datetime(market.loc[market.instrument.eq("NIFTY50"), "observed_at"]))
    output, report = match_pages(pages, offers, market_days)
    report.update({"page_ids_checked": args.end - args.start + 1,
                   "limitations": "Retrospective cells may have been changed after each bidding day. Separate source agreement and dated publication evidence are needed before promotion."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejected_examples"}, indent=2))


if __name__ == "__main__":
    main()
