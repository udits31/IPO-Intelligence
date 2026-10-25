"""Measure the effect of disputed final-bid rows without rewriting training data.

IPOWatch finals are retrospective source candidates, not certified exchange
records. The output is a sensitivity study, never a promoted model score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.model_pipeline import (EXCLUDED_NON_CORPORATE_IDS, FEATURE_COLS,
                               fit_model, prepare_dataset, score_model, summarize)


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("qib", "retail", "hni")


def source_agrees(left: pd.Series, right: pd.Series) -> bool:
    for category in CATEGORIES:
        a = float(left[f"final_{category}_on_page"])
        b = float(right[f"final_{category}_on_page"])
        if abs(a - b) > max(0.25, 0.10 * max(a, b)):
            return False
    return True


def build_overlay(ipowatch: pd.DataFrame, equitymaster: pd.DataFrame,
                  offers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    first = ipowatch.set_index("ipo_id", verify_integrity=True)
    second = equitymaster.set_index("ipo_id", verify_integrity=True)
    verified = offers.set_index("ipo_id", verify_integrity=True)
    rows, conflicts = [], []
    for ipo_id, candidate in first.iterrows():
        if ipo_id not in verified.index or int(ipo_id) in EXCLUDED_NON_CORPORATE_IDS:
            continue
        nse = verified.loc[ipo_id]
        if candidate.issue_close_date != nse.issue_close_date:
            continue
        publication = pd.to_datetime(candidate.source_published_at, errors="coerce")
        listing = pd.to_datetime(nse.listing_date)
        if pd.isna(publication) or publication >= listing:
            continue
        if ipo_id in second.index and not source_agrees(candidate, second.loc[ipo_id]):
            conflicts.append(int(ipo_id))
            continue
        values = {category: float(candidate[f"final_{category}_on_page"])
                  for category in CATEGORIES}
        if not all(np.isfinite(value) and value >= 0 for value in values.values()):
            continue
        rows.append({"ipo_id": int(ipo_id), "company_name": candidate.company_name,
                     "issue_close_date": candidate.issue_close_date,
                     **{f"{category}_subscription": value for category, value in values.items()},
                     "source_url": candidate.source_url,
                     "secondary_source_url": second.loc[ipo_id].source_url if ipo_id in second.index else None,
                     "source_class": "retrospective_final_table_not_exchange_adjudicated"})
    result = pd.DataFrame(rows)
    return result, {"overlay_candidates": len(result), "excluded_cross_source_final_conflicts": len(conflicts),
                    "conflict_ids": conflicts, "method": "IPOWatch final table; if Equitymaster also exists, require all three final categories within max(0.25x, 10%). No averaging."}


def overlay_frame(base: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    revised = base.copy()
    if candidates.empty:
        return revised
    source = candidates.set_index("ipo_id", verify_integrity=True)
    if not source.index.isin(revised.id).all():
        raise ValueError("Unknown IPO ID in final-bid overlay")
    for category in CATEGORIES:
        col = f"{category}_subscription"
        mask = revised.id.isin(source.index)
        revised.loc[mask, col] = revised.loc[mask, "id"].map(source[col]).to_numpy()
    for category, feature in (("qib", "log_qib"), ("retail", "log_retail"), ("hni", "log_hni")):
        revised[feature] = np.log1p(revised[f"{category}_subscription"].clip(lower=0.01))
    return revised


def replay(frame: pd.DataFrame) -> dict:
    development = frame[frame.year.between(2021, 2024) & ~frame.id.isin(EXCLUDED_NON_CORPORATE_IDS)]
    result = {}
    for year in (2022, 2023, 2024, 2025, 2026):
        train = development[development.year.le(min(year - 1, 2024))]
        test = frame[frame.year.eq(year)]
        scaler, model = fit_model(train)
        result[str(year)] = {"train_rows": len(train), **summarize(test.target_binary,
                               score_model(scaler, model, test), 0.5)}
    return result


def main() -> None:
    frame = prepare_dataset()
    first = pd.read_csv(ROOT / "data/daywise_subscription_candidates.csv")
    second = pd.read_csv(ROOT / "data/equitymaster_daywise_candidates.csv")
    offers = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    candidates, audit = build_overlay(first, second, offers)
    revised = overlay_frame(frame, candidates)
    changed = (~np.isclose(frame[FEATURE_COLS], revised[FEATURE_COLS])).any(axis=1)
    report = {"overlay_audit": audit,
              "changed_development_rows": int(changed[frame.year.between(2021, 2024).to_numpy()].sum()),
              "stored_data_replay": replay(frame), "candidate_source_overlay_replay": replay(revised),
              "warning": "Retrospective source-sensitivity only. Do not call this verified accuracy or promote the overlay without primary-source adjudication."}
    output = ROOT / "output/final_book_source_sensitivity.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
