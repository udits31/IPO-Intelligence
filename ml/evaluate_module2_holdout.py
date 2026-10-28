"""One-checkpoint, later-year Module 2 price-only holdout evaluation.

The deployed rule is frozen. Each IPO contributes at most one decision, after
31 observed trading closes. Outcomes are read strictly after that decision.
"""

from __future__ import annotations

import hashlib
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.monitor_engine import HoldSellMonitor


IPOS = ROOT / "data" / "canonical_listed_ipos.csv"
COHORTS = {
    "holdout": (ROOT / "data" / "audit" / "module2_holdout_prices_2025_2026.csv",
                ROOT / "output" / "module2_holdout_accuracy_2025_2026.json",
                ROOT / "output" / "module2_holdout_decision_ledger.csv"),
    "development": (ROOT / "data" / "audit" / "module2_development_prices_2021_2024.csv",
                    ROOT / "output" / "module2_development_accuracy_2021_2024.json",
                    ROOT / "output" / "module2_development_decision_ledger.csv"),
}
CHECKPOINT = 30
HORIZON = 30
SELL_COST_PCT = 0.30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame: pd.DataFrame, outcome: str, rng: np.random.Generator) -> dict:
    actual_up = frame[outcome].to_numpy() > 0
    sell = frame["sell"].to_numpy(dtype=bool)
    trailing = frame["trailing_8_sell"].to_numpy(dtype=bool)
    n = len(frame)
    if not n:
        return {"n": 0}
    correct = np.where(sell, ~actual_up, actual_up)
    always_sell = ~actual_up
    always_hold = actual_up
    trailing_correct = np.where(trailing, ~actual_up, actual_up)
    tn, fp, fn, tp = confusion_matrix(actual_up, ~sell, labels=[False, True]).ravel()
    samples = rng.integers(0, n, size=(10000, n))
    gap_sell = (correct[samples].astype(int) - always_sell[samples].astype(int)).mean(axis=1) * 100
    gap_trailing = (correct[samples].astype(int) - trailing_correct[samples].astype(int)).mean(axis=1) * 100
    return {
        "n": n, "up_rate_pct": round(actual_up.mean() * 100, 2),
        "model_accuracy_pct": round(correct.mean() * 100, 2),
        "always_sell_accuracy_pct": round(always_sell.mean() * 100, 2),
        "always_hold_accuracy_pct": round(always_hold.mean() * 100, 2),
        "trailing_8_accuracy_pct": round(trailing_correct.mean() * 100, 2),
        "model_minus_always_sell_pp": round((correct.mean() - always_sell.mean()) * 100, 2),
        "model_minus_always_sell_95ci_pp": np.quantile(gap_sell, [.025, .975]).round(2).tolist(),
        "model_minus_trailing_8_pp": round((correct.mean() - trailing_correct.mean()) * 100, 2),
        "model_minus_trailing_8_95ci_pp": np.quantile(gap_trailing, [.025, .975]).round(2).tolist(),
        "tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "sell_calls": int(sell.sum()), "hold_calls": int((~sell).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, default="holdout")
    parser.add_argument("--tag", default="")
    parser.add_argument("--prices", type=Path, help="Audited alternate price snapshot; never overwrites the default")
    parser.add_argument("--listing-close-audit", type=Path,
                        help="Raw-bhavcopy-verified resolutions of canonical listing-close conflicts")
    args = parser.parse_args()
    price_path, base_output, base_ledger = COHORTS[args.cohort]
    if args.prices is not None:
        price_path = args.prices.resolve()
    output = base_output.with_name(base_output.stem + args.tag + base_output.suffix)
    ledger = base_ledger.with_name(base_ledger.stem + args.tag + base_ledger.suffix)
    prices = pd.read_csv(price_path, parse_dates=["date"])
    if prices["nifty50_close"].isna().any():
        raise ValueError("Price snapshot contains stock rows without a Nifty trading close; possible holiday carry-forward")
    ipos = pd.read_csv(IPOS, parse_dates=["listing_date"]).set_index("id")
    resolutions = {}
    if args.listing_close_audit:
        from ml.reconcile_module2_corporate_actions import load_listing_resolutions
        resolutions = load_listing_resolutions(args.listing_close_audit.resolve(),ipos)
    monitor = HoldSellMonitor(prices_path=str(price_path), ipos_path=str(IPOS))
    cases, excluded, resolved_conflicts = [], [], []
    for ipo_id, group in prices.groupby("ipo_id"):
        group = group.sort_values("date").reset_index(drop=True)
        ipo_id = int(ipo_id)
        reason = None
        if len(group) <= CHECKPOINT + HORIZON + 1:
            reason = "insufficient 30-session future"
        elif ipo_id not in ipos.index:
            reason = "IPO identity missing"
        elif (group.loc[0, "date"] - ipos.loc[ipo_id, "listing_date"]).days > 7:
            reason = "first close more than seven calendar days after listing"
        elif (group["close_price_adjusted"] <= 0).any():
            reason = "nonpositive close"
        elif (group.loc[0, "date"].normalize() == ipos.loc[ipo_id, "listing_date"].normalize()
              and pd.notna(ipos.loc[ipo_id, "listing_day_close_price"])
              and abs(float(group.loc[0, "close_price_adjusted"])
                      / float(ipos.loc[ipo_id, "listing_day_close_price"]) - 1) > .02):
            official = resolutions.get(ipo_id)
            if official is not None and np.isclose(float(group.loc[0,"close_price_adjusted"]),
                                                   official['nse_close'],atol=.0001):
                resolved_conflicts.append({'ipo_id':ipo_id,'canonical_close':float(ipos.loc[ipo_id,"listing_day_close_price"]),
                                           'verified_nse_close':official['nse_close']})
            else:
                reason = "first-day close conflicts with IPO listing record (possible split adjustment)"
        if reason:
            excluded.append({"ipo_id": ipo_id, "reason": reason})
            continue
        history = group.loc[:CHECKPOINT, "close_price_adjusted"].astype(float)
        issue_price = float(ipos.loc[ipo_id, "issue_price"])
        if not np.isfinite(issue_price) or issue_price <= 0:
            excluded.append({"ipo_id": ipo_id, "reason": "issue price unavailable"})
            continue
        decision_date = group.loc[CHECKPOINT, "date"].strftime("%Y-%m-%d")
        result = monitor.evaluate_holding(ipo_id, as_of_date=decision_date,
                                          disclosures=[], fetch_news=False)
        if "verdict" not in result:
            excluded.append({"ipo_id": ipo_id, "reason": str(result.get("error", "monitor failed"))})
            continue
        verdict = str(result["verdict"])
        sell = "SELL" in verdict or "TAKE PROFIT" in verdict
        decision_benchmark = result.get("attribution_30d") or {}
        benchmark_source = str(group.loc[CHECKPOINT].get("benchmark_source", "Unknown"))
        curr = float(group.loc[CHECKPOINT, "close_price_adjusted"])
        one = float(group.loc[CHECKPOINT + 1, "close_price_adjusted"])
        future = float(group.loc[CHECKPOINT + HORIZON, "close_price_adjusted"])
        entry = float(group.loc[CHECKPOINT + 1, "close_price_adjusted"])
        exit_price = float(group.loc[CHECKPOINT + HORIZON + 1, "close_price_adjusted"])
        future_window = group.loc[CHECKPOINT + 1:CHECKPOINT + HORIZON + 1, "close_price_adjusted"]
        peak = max(issue_price, float(history.max()))
        cases.append({
            "ipo_id": ipo_id, "company_name": ipos.loc[ipo_id, "company_name"],
            "listing_year": int(ipos.loc[ipo_id, "listing_date"].year),
            "listing_date": ipos.loc[ipo_id, "listing_date"].date().isoformat(),
            "decision_date": decision_date,
            "decision_close": curr, "verdict": verdict, "sell": sell,
            "confidence": result.get("confidence"),
            "risk_level": result.get("risk_level"),
            "attribution_category": decision_benchmark.get("category"),
            "stock_return_30d_pct": decision_benchmark.get("stock_return_pct"),
            "sector_return_30d_pct": decision_benchmark.get("sector_return_pct"),
            "nifty_return_30d_pct": decision_benchmark.get("nifty_return_pct"),
            "spread_vs_sector_30d_pct": decision_benchmark.get("spread_vs_sector_pct"),
            "cumulative_gain_pct": (result.get("price_waterfall") or {}).get("cumulative_gain_pct"),
            "drawdown_from_peak_pct": (result.get("price_waterfall") or {}).get("drawdown_from_peak_pct"),
            "benchmark_source": benchmark_source,
            "sector_comparison_available": decision_benchmark.get("sector_return_pct") is not None,
            "trailing_8_sell": curr <= peak * 0.92,
            "trailing_25_sell": curr <= peak * 0.75,
            "day1_return_pct": (one / curr - 1) * 100,
            "day30_return_pct": (future / curr - 1) * 100,
            "execution_date": group.loc[CHECKPOINT + 1, "date"].date().isoformat(),
            "outcome_date": group.loc[CHECKPOINT + HORIZON + 1, "date"].date().isoformat(),
            "executed_hold_return_pct": (exit_price / entry - 1) * 100,
            "future_min_return_from_entry_pct": (float(future_window.min()) / entry - 1) * 100,
            "future_max_drawdown_pct": float((future_window / future_window.cummax() - 1).min()*100),
            "future_single_day_drop_min_pct": float(group.loc[CHECKPOINT + 1:CHECKPOINT + HORIZON + 1,
                                                          "close_price_adjusted"].pct_change().min() * 100),
        })
    frame = pd.DataFrame(cases).sort_values(["listing_year", "ipo_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ledger, index=False)
    rng = np.random.default_rng(20261016)
    stock_price_bases = sorted(prices['price_basis'].dropna().astype(str).unique().tolist()) \
        if 'price_basis' in prices else ['unspecified']
    exchange_stock_prices = len(stock_price_bases) == 1 and stock_price_bases[0].startswith('Official NSE')
    actions_reconciled = exchange_stock_prices and 'dated NSE bonus/split reconciliation' in stock_price_bases[0]
    result = {
        "scope": f"{args.cohort} listed NSE-symbol IPOs with matured 31-session checkpoint and 30-session future",
        "rule": "core/monitor_engine.py at recorded SHA-256; SELL or TAKE PROFIT -> cash, else hold",
        "decision": "31st observed close (index 30); one decision per IPO",
        "execution": "next observed close (index 31)",
        "outcome": "30 sessions after decision for direction; 30 sessions after execution for strategy",
        "stock_price_basis": stock_price_bases,
        "limitations": ["NSE stock closes were retrieved retrospectively; original publication-time snapshots are not archived"
                        if exchange_stock_prices else
                        "Yahoo historical closes are secondary and retrieved after the fact; exchange-close audit is separate",
                        "No historically timestamped news or filings; price-only policy, not full live Module 2",
                        "Simple close-to-close execution and assumed 0.30% sell cost are illustrative",
                        ("Explicit NSE bonus/split events reconciled in listing-share units; cash dividends not included; source coverage is retrospective"
                         if actions_reconciled else "No corporate-action reconciliation; split-like days flagged in the ledger")],
        "input_sha256": {str(price_path.relative_to(ROOT)): sha256(price_path),
                         str(IPOS.relative_to(ROOT)): sha256(IPOS),
                         "core/monitor_engine.py": sha256(ROOT / "core" / "monitor_engine.py"),
                         "core/attribution_engine.py": sha256(ROOT / "core" / "attribution_engine.py")},
        "coverage": {"available_ipo_count": int(prices.ipo_id.nunique()),
                     "eligible_matured_count": len(frame), "excluded_count": len(excluded),
                     "excluded_by_reason": pd.Series([x["reason"] for x in excluded]).value_counts().to_dict(),
                     "sector_comparison_available_count": int(frame.sector_comparison_available.sum()),
                     "broad_market_proxy_count": int(frame.benchmark_source.str.contains("broad-market proxy").sum()),
                     "benchmark_source_counts": frame.benchmark_source.value_counts().to_dict(),
                     "split_like_future_count": int((frame.future_single_day_drop_min_pct < -40).sum())},
        "first_day": metrics(frame, "day1_return_pct", rng),
        "day30": metrics(frame, "day30_return_pct", rng),
        "by_year": {str(year): metrics(part, "day30_return_pct", rng)
                    for year, part in frame.groupby("listing_year")},
        "excluded": excluded,
        "resolved_listing_close_conflicts": resolved_conflicts,
    }
    if args.listing_close_audit:
        audit_path = args.listing_close_audit.resolve()
        result['input_sha256'][str(audit_path.relative_to(ROOT))] = sha256(audit_path)
    hold = frame["executed_hold_return_pct"].to_numpy()
    n = len(frame)
    for name, signal in [("model", frame["sell"]),
                         ("trailing_8", frame["trailing_8_sell"]),
                         ("trailing_25", frame["trailing_25_sell"])]:
        sold = signal.to_numpy(dtype=bool)
        net = np.where(sold, -SELL_COST_PCT, hold)
        result["day30"][f"{name}_strategy_net_mean_pct"] = round(float(net.mean()), 3)
        result["day30"][f"{name}_strategy_max_drawdown_mean_pct"] = round(
            float(np.where(sold, -SELL_COST_PCT, frame["future_max_drawdown_pct"]).mean()), 3)
    result["day30"]["always_hold_net_mean_pct"] = round(float(hold.mean()), 3)
    result["day30"]["always_sell_net_mean_pct"] = -SELL_COST_PCT
    diffs = np.where(frame["sell"].to_numpy(), -SELL_COST_PCT, hold) - hold
    draws = rng.integers(0, n, size=(10000, n))
    result["day30"]["model_minus_hold_net_95ci_pp"] = np.quantile(
        diffs[draws].mean(axis=1), [.025, .975]).round(3).tolist()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["coverage", "first_day", "day30", "by_year"]}, indent=2))


if __name__ == "__main__":
    main()
