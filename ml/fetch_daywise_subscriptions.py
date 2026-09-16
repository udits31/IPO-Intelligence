"""Research-only collection of historical day-wise IPO subscription tables.

These tables are retrospective page content. They are not proof of an
intraday publication time and must not be promoted as prospective evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
SEARCH_URL = "https://ipowatch.in/wp-json/wp/v2/search"
POST_URL = "https://ipowatch.in/wp-json/wp/v2/posts/{post_id}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IPOResearchAudit/1.0)"}


def name_tokens(value: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return [token for token in tokens if token not in {"ltd", "limited", "private", "ipo"}]


def table_days(content: str) -> dict[int, dict[str, float]] | None:
    soup = BeautifulSoup(content, "html.parser")
    for table in soup.find_all("table"):
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                for row in table.find_all("tr")]
        rows = [row for row in rows if row]
        header_idx = next((i for i, row in enumerate(rows)
                           if len(row) >= 3 and "category" in row[0].lower()
                           and re.search(r"day\s*1", " ".join(row[1:]), re.I)), None)
        if header_idx is None:
            continue
        header = rows[header_idx]
        day_columns = [(index, int(match.group(1))) for index, cell in enumerate(header)
                       if (match := re.search(r"day\s*(\d+)", cell, re.I))]
        if len(day_columns) < 2 or len({day for _, day in day_columns}) != len(day_columns):
            continue
        values: dict[int, dict[str, float]] = {day: {} for _, day in day_columns}
        for row in rows[header_idx + 1:]:
            label = row[0].strip().lower()
            category = ("qib" if label.startswith("qib") else
                        "hni" if label.startswith(("nii", "hni")) else
                        "retail" if label.startswith(("rii", "retail")) else
                        "total" if label.startswith("total") else None)
            if category is None:
                continue
            for index, day in day_columns:
                if index >= len(row):
                    continue
                match = re.fullmatch(r"\s*([\d,]+(?:\.\d+)?)\s*(?:x|times)?\s*", row[index], re.I)
                if match:
                    from services.subscription_numbers import subscription_multiple
                    parsed=subscription_multiple(match.group(1))
                    if parsed is not None:values[day][category] = parsed
        if all({"qib", "hni", "retail"}.issubset(day) for day in values.values()):
            return values
    return None


def find_daywise(row: pd.Series, session: requests.Session, market_days: pd.DatetimeIndex) -> tuple[dict | None, dict]:
    name = str(row.company_name)
    issue_open = pd.Timestamp(row.issue_open_date)
    issue_close = pd.Timestamp(row.issue_close_date)
    listing = pd.Timestamp(row.listing_date)
    expected_days = int(((market_days >= issue_open) & (market_days <= issue_close)).sum())
    if expected_days < 2:
        return None, {"ipo_id": int(row.ipo_id), "reason": "fewer_than_two_bidding_sessions"}
    tokens = name_tokens(name)
    target = set(tokens)
    queries = [" ".join(tokens) + " IPO Subscription"]
    if len(tokens) > 1:
        queries.append(tokens[0] + " IPO Subscription")
    candidates = []
    try:
        for query in queries:
            response = session.get(SEARCH_URL, params={"search": query, "per_page": 10}, headers=HEADERS, timeout=25)
            response.raise_for_status()
            candidates.extend(response.json())
            if any("subscription" in html.unescape(item.get("title", "")).lower()
                   and tokens[0] in name_tokens(html.unescape(item.get("title", "")))
                   for item in candidates):
                break
    except (requests.RequestException, ValueError) as exc:
        return None, {"ipo_id": int(row.ipo_id), "reason": f"search_error:{type(exc).__name__}"}
    ranked = []
    for candidate in candidates:
        title = html.unescape(candidate.get("title", ""))
        if "<" in title:
            title = BeautifulSoup(title, "html.parser").get_text(" ", strip=True)
        if "subscription" not in title.lower() or "ipo" not in title.lower():
            continue
        prefix = title.lower().split("ipo", 1)[0]
        title_tokens = set(name_tokens(prefix))
        overlap = len(target & title_tokens) / max(len(target), 1)
        first_name_token = name_tokens(name)[0] if target else None
        short_exact = (len(title_tokens) == 1 and len(target) > 1
                       and len(first_name_token or "") >= 5 and first_name_token in title_tokens)
        if (overlap >= 0.65 or short_exact) and target and first_name_token in title_tokens:
            ranked.append((overlap, candidate))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if not ranked:
        return None, {"ipo_id": int(row.ipo_id), "reason": "no_matching_subscription_article"}
    for score, candidate in ranked[:3]:
        try:
            detail_response = session.get(POST_URL.format(post_id=candidate["id"]), headers=HEADERS, timeout=25)
            detail_response.raise_for_status()
            detail = detail_response.json()
        except (requests.RequestException, ValueError, KeyError):
            continue
        published = pd.to_datetime(detail.get("date", ""), errors="coerce")
        if pd.isna(published) or not issue_open - timedelta(days=21) <= published <= listing + timedelta(days=7):
            continue
        content = detail.get("content", {}).get("rendered", "")
        days = table_days(content)
        if not days or max(days) != expected_days or expected_days - 1 not in days:
            continue
        prior = days[expected_days - 1]
        final = days[expected_days]
        result = {
            "ipo_id": int(row.ipo_id), "company_name": name,
            "issue_open_date": issue_open.strftime("%Y-%m-%d"),
            "issue_close_date": issue_close.strftime("%Y-%m-%d"),
            "decision_stage": f"after_day_{expected_days - 1}_before_final_day",
            "qib_subscription": prior["qib"], "hni_subscription": prior["hni"],
            "retail_subscription": prior["retail"],
            "final_qib_on_page": final["qib"], "final_hni_on_page": final["hni"],
            "final_retail_on_page": final["retail"],
            "source_url": candidate["url"], "source_post_id": int(candidate["id"]),
            "source_published_at": detail.get("date"), "source_modified_at": detail.get("modified"),
            "source_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "source_identity_overlap": round(score, 3),
            "evidence_class": "retrospective_daywise_table_not_timestamped_day_snapshot",
        }
        return result, {"ipo_id": int(row.ipo_id), "reason": "accepted", "source_url": candidate["url"]}
    return None, {"ipo_id": int(row.ipo_id), "reason": "article_date_or_day_table_mismatch"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Initial IPOs to probe, 0 for all")
    parser.add_argument("--output", type=Path, default=ROOT / "data/daywise_subscription_candidates.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output/daywise_subscription_collection.json")
    args = parser.parse_args()
    offers = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    offers["listing_date"] = pd.to_datetime(offers.listing_date)
    offers = offers[offers.listing_date.dt.year.between(2021, 2024)].sort_values("listing_date")
    if args.limit:
        offers = offers.head(args.limit)
    market = pd.read_csv(ROOT / "data/raw_daily_market_time_series.csv")
    market_days = pd.DatetimeIndex(pd.to_datetime(market.loc[market.instrument.eq("NIFTY50"), "observed_at"]))
    session = requests.Session()
    accepted, audit = [], []
    for _, row in offers.iterrows():
        result, status = find_daywise(row, session, market_days)
        audit.append(status)
        if result:
            accepted.append(result)
        if len(audit) % 20 == 0:
            print(f"checked={len(audit)} accepted={len(accepted)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted).to_csv(args.output, index=False)
    report = {"checked": len(audit), "accepted": len(accepted),
              "reason_counts": pd.Series([item["reason"] for item in audit]).value_counts().to_dict(),
              "audit": audit,
              "limitations": "Retrospective daily cells can be revised later; this is not a certified point-in-time snapshot."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "audit"}, indent=2))


if __name__ == "__main__":
    main()
