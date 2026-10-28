"""Score frozen Module 2 decisions against later official NSE closes.

One first decision per IPO is scored so repeated daily observations do not
inflate the sample. Neither the ledger nor its source evidence is rewritten.
Include newly listed live-calendar IPOs without changing the paper's cohort.
"""

from __future__ import annotations

import json
import hashlib
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from ml.audit_module2_nse_closes import exchange_rows
from scrapers.live_ipo_scraper import load_scraped_cache
from services.corporate_action_service import official_actions, PriceBasisError


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "telemetry" / "module2_decisions.db"
OUTPUT = ROOT / "output" / "module2_prospective_accuracy.json"
LEDGER = ROOT / "output" / "module2_prospective_outcomes.csv"
HORIZON = 30


def comparable_outcome_closes(prices: pd.DataFrame, events: list[dict], entry_date: str) -> pd.Series:
    """Wealth per entry-date share from raw dated closes and explicit ratios.

    Outcomes can be reconciled after the fact; frozen decisions are unchanged.
    Cash dividends are excluded. Rights/mergers are not guessed from prices.
    """
    closes=prices.set_index('date')['nse_close'].astype(float).copy()
    if closes.index.duplicated().any() or closes.le(0).any() or closes.isna().any():
        raise PriceBasisError('Invalid or duplicate outcome closes')
    for event in events:
        if pd.Timestamp(event['ex_date'])<=pd.Timestamp(entry_date):
            continue  # Entry quote is already in its own dated share units.
        if event['kind']=='unsupported':
            raise PriceBasisError('Unresolved outcome capital event: '+str(event['subject']))
        if event['kind'] in ('bonus','split'):
            closes.loc[pd.to_datetime(closes.index)>=pd.Timestamp(event['ex_date'])]*=float(event['share_factor'])
    return closes


def _sell(verdict: str) -> bool:
    return "SELL" in verdict or "TAKE PROFIT" in verdict


def _metrics(frame: pd.DataFrame, column: str) -> dict:
    if frame.empty:
        return {"n": 0, "accuracy_pct": None}
    actual_up = frame["day30_return_pct"].gt(0)
    sell = frame[column].map(_sell)
    correct = sell.ne(actual_up)
    return {"n": len(frame),
            "accuracy_pct": round(float(correct.mean() * 100), 2),
            "sell_calls": int(sell.sum()),
            "hold_calls": int((~sell).sum())}


def _direction_metrics(frame: pd.DataFrame, return_column: str) -> dict:
    if frame.empty:
        return {"n": 0, "price_only_accuracy_pct": None,
                "news_adjusted_accuracy_pct": None,
                "always_sell_accuracy_pct": None, "always_hold_accuracy_pct": None}
    up = frame[return_column].gt(0)
    return {
        "n": len(frame),
        "price_only_accuracy_pct": round(float(frame.price_only_verdict.map(_sell).ne(up).mean() * 100), 2),
        "news_adjusted_accuracy_pct": round(float(frame.news_adjusted_verdict.map(_sell).ne(up).mean() * 100), 2),
        "always_sell_accuracy_pct": round(float((~up).mean() * 100), 2),
        "always_hold_accuracy_pct": round(float(up.mean() * 100), 2),
    }


def _identities() -> dict[int, tuple[str, str]]:
    canonical = pd.read_csv(ROOT / "data" / "canonical_listed_ipos.csv")
    identities = {int(row.id): (str(row.nse_symbol).upper(), str(row.company_name))
                  for row in canonical.itertuples() if pd.notna(row.nse_symbol)}
    for url, issue in load_scraped_cache().items():
        if issue.get("category") != "LISTED" or issue.get("source_observed") is not True:
            continue
        try:
            ipo_id = int(url.rstrip("/").split("/")[-1])
        except ValueError:
            continue
        symbol = str(issue.get("nse_symbol") or "").strip().upper()
        if symbol and ipo_id not in identities:
            identities[ipo_id] = (symbol, str(issue.get("company_name") or ""))
    return identities


def _exchange_history(decisions: pd.DataFrame, identities: dict[int, tuple[str, str]],
                      now: datetime) -> tuple[pd.DataFrame, list[dict]]:
    """Retrieve actual daily archives only when a decision could have matured."""
    eligible = decisions[pd.to_datetime(decisions.price_date).dt.date <=
                         now.date() - timedelta(days=35)]
    if eligible.empty:
        return pd.DataFrame(), []
    wanted = defaultdict(set)
    for stamp in pd.date_range(pd.to_datetime(eligible.price_date).min().date(), now.date(), freq="B"):
        wanted[stamp.strftime("%Y%m%d")].update(
            identities[int(ipo_id)][0] for ipo_id in eligible.ipo_id if int(ipo_id) in identities)
    rows, manifests = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(exchange_rows, stamp, symbols): stamp
                   for stamp, symbols in wanted.items()}
        for future in as_completed(futures):
            found, manifest = future.result()
            rows.extend(found)
            manifests.append(manifest)
    return pd.DataFrame(rows), sorted(manifests, key=lambda item: item["date"])


def main() -> None:
    if not DB.exists():
        raise SystemExit("No prospective Module 2 decision ledger exists yet")
    with sqlite3.connect(DB) as conn:
        decisions = pd.read_sql_query("""SELECT id, ipo_id, observed_at_utc, price_date,
            price_only_verdict, final_verdict, news_changed_decision, payload_sha256,
            payload_json FROM module2_decisions_v1 ORDER BY observed_at_utc, id""", conn)
    decisions = decisions.drop_duplicates("ipo_id", keep="first")
    now = datetime.now(timezone.utc)
    # Thirty NSE sessions cannot elapse in fewer than 35 calendar days.
    ready_to_check = decisions[
        pd.to_datetime(decisions["price_date"]).dt.date <= (now.date() - timedelta(days=35))
    ]
    identities = _identities()
    exchange, manifests = _exchange_history(decisions, identities, now)
    outcomes = []
    pending = len(decisions) - len(ready_to_check)
    excluded = []
    market_dates = sorted(item["date"] for item in manifests
                          if item["status"] == "verified_archive")
    transient_failures = {item["date"] for item in manifests
                          if item["status"] != "verified_archive"
                          and "404" not in str(item.get("error", ""))}
    for decision in ready_to_check.itertuples(index=False):
        if int(decision.ipo_id) not in identities:
            excluded.append({"ipo_id": int(decision.ipo_id), "reason": "verified NSE identity absent"})
            continue
        symbol, _ = identities[int(decision.ipo_id)]
        evidence = json.loads(decision.payload_json)
        if hashlib.sha256(decision.payload_json.encode('utf-8')).hexdigest()!=decision.payload_sha256:
            excluded.append({'ipo_id':int(decision.ipo_id),'reason':'frozen decision payload hash mismatch'})
            continue
        frozen_symbol = str(evidence.get("nse_symbol") or "").strip().upper()
        if frozen_symbol and frozen_symbol != symbol:
            excluded.append({"ipo_id": int(decision.ipo_id),
                             "reason": "frozen NSE symbol conflicts with current identity"})
            continue
        if exchange.empty:
            pending += 1
            continue
        sessions = [stamp for stamp in market_dates if stamp >= decision.price_date]
        if len(sessions) <= HORIZON + 1:
            pending += 1
            continue
        if any(decision.price_date <= stamp <= sessions[HORIZON + 1]
               for stamp in transient_failures):
            pending += 1
            continue
        prices = exchange.loc[exchange.symbol.eq(symbol)]
        if prices.date.duplicated().any():
            excluded.append({"ipo_id": int(decision.ipo_id), "reason": "duplicate exchange close"})
            continue
        prices = prices.set_index("date")
        if decision.price_date not in prices.index:
            excluded.append({"ipo_id": int(decision.ipo_id),
                             "reason": "decision-date exchange close absent"})
            continue
        if any(stamp not in prices.index for stamp in
               (sessions[1], sessions[HORIZON], sessions[HORIZON + 1])):
            excluded.append({"ipo_id": int(decision.ipo_id),
                             "reason": "exchange close absent at an outcome checkpoint"})
            continue
        frozen_close = float(evidence["decision_close"])
        decision_row = prices.loc[decision.price_date]
        day1_row = prices.loc[sessions[1]]
        outcome_row = prices.loc[sessions[HORIZON]]
        exit_row = prices.loc[sessions[HORIZON + 1]]
        if frozen_close <= 0 or abs(float(decision_row.nse_close) / frozen_close - 1) > .01:
            excluded.append({"ipo_id": int(decision.ipo_id),
                             "reason": "frozen price conflicts with NSE close"})
            continue
        try:
            actions,action_sources=official_actions(symbol,decision.price_date,sessions[HORIZON+1])
            comparable=comparable_outcome_closes(prices.reset_index(),actions,decision.price_date)
        except (PriceBasisError, OSError, ValueError, requests.RequestException) as exc:
            excluded.append({'ipo_id':int(decision.ipo_id),'reason':'outcome share basis unresolved: '+str(exc)})
            continue
        outcomes.append({
            "decision_id": int(decision.id), "ipo_id": int(decision.ipo_id),
            "nse_symbol": symbol, "observed_at_utc": decision.observed_at_utc,
            "price_date": decision.price_date, "decision_close": frozen_close,
            "decision_archive_sha256": decision_row.archive_sha256,
            "outcome_date": sessions[HORIZON],
            "outcome_archive_sha256": outcome_row.archive_sha256,
            "execution_date": sessions[1],
            "strategy_exit_date": sessions[HORIZON + 1],
            "day1_return_pct": round((float(comparable.loc[sessions[1]]) / frozen_close - 1) * 100, 4),
            "day30_return_pct": round((float(comparable.loc[sessions[HORIZON]]) / frozen_close - 1) * 100, 4),
            "executed_hold_return_pct": round((float(comparable.loc[sessions[HORIZON+1]]) /
                                                float(comparable.loc[sessions[1]]) - 1) * 100, 4),
            "share_events_json":json.dumps(actions,sort_keys=True),
            "share_source_windows_json":json.dumps(action_sources,sort_keys=True),
            "price_only_verdict": decision.price_only_verdict,
            "news_adjusted_verdict": decision.final_verdict,
            "news_changed_decision": bool(decision.news_changed_decision),
            "decision_payload_sha256": decision.payload_sha256,
        })
    frame = pd.DataFrame(outcomes)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(LEDGER, index=False)
    accuracy = {
        "protocol": "First frozen decision per IPO; 30 later trading sessions; no backfilled decisions",
        "outcome_source": "Official NSE UDiFF CM raw closes plus dated NSE split/bonus ratios; archive hashes saved. Cash dividends excluded; unresolved capital events excluded.",
        "retrieved_at_utc": now.isoformat(),
        "frozen_ipo_count": len(decisions), "matured_count": len(frame),
        "pending_count": pending, "excluded": excluded,
        "verified_archive_dates": sum(item["status"] == "verified_archive" for item in manifests),
        "archive_failures": [item for item in manifests if item["status"] != "verified_archive"],
        "price_only": _metrics(frame, "price_only_verdict") if not frame.empty else {"n": 0, "accuracy_pct": None},
        "news_adjusted": _metrics(frame, "news_adjusted_verdict") if not frame.empty else {"n": 0, "accuracy_pct": None},
        "first_day": _direction_metrics(frame, "day1_return_pct"),
        "day30": _direction_metrics(frame, "day30_return_pct"),
        "news_changed_decision_count": int(frame["news_changed_decision"].sum()) if not frame.empty else 0,
        "always_sell_accuracy_pct": round(float((frame["day30_return_pct"] <= 0).mean() * 100), 2) if not frame.empty else None,
        "always_hold_accuracy_pct": round(float((frame["day30_return_pct"] > 0).mean() * 100), 2) if not frame.empty else None,
        "warning": "No public accuracy claim should be made from an immature or small prospective sample.",
    }
    OUTPUT.write_text(json.dumps(accuracy, indent=2), encoding="utf-8")
    print(json.dumps(accuracy, indent=2))


if __name__ == "__main__":
    main()
