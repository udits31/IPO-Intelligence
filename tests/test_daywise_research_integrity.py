"""Guardrails for retrospective Module 1 research; never a deployed-model test."""

from pathlib import Path

import pandas as pd

from ml.combine_daywise_sources import combine
from ml.research_daywise_model import prepare
from ml.research_early_book_context import attach_decision_context
from ml.research_final_book_source_sensitivity import build_overlay, overlay_frame
from ml.model_pipeline import prepare_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_combined_sources_never_average_or_accept_material_conflict():
    first = pd.read_csv(ROOT / "data/daywise_subscription_candidates.csv")
    second = pd.read_csv(ROOT / "data/equitymaster_daywise_candidates.csv")
    cohort, report = combine(first, second)
    assert report["excluded_material_conflicts"] > 0
    assert not cohort.ipo_id.duplicated().any()
    assert cohort.source_tier.isin(
        ["two_source_agreement", "ipowatch_only", "equitymaster_only"]
    ).all()
    first_by_id = first.set_index("ipo_id")
    second_by_id = second.set_index("ipo_id")
    for row in cohort.itertuples():
        source = first_by_id.loc[row.ipo_id] if row.ipo_id in first_by_id.index else second_by_id.loc[row.ipo_id]
        for category in ("qib", "hni", "retail"):
            assert getattr(row, f"{category}_subscription") == source[f"{category}_subscription"]


def test_replay_uses_only_preclose_context_and_verified_offer_dates():
    canonical = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    candidates = pd.read_csv(ROOT / "data/daywise_subscription_research_cohort.csv")
    offers = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    market = pd.read_csv(ROOT / "data/raw_daily_market_time_series.csv")
    frame, _ = prepare(candidates, canonical)
    dated = attach_decision_context(frame, offers, market)
    assert len(dated) > 100
    assert (pd.to_datetime(dated.decision_market_date) < pd.to_datetime(dated.issue_close_date)).all()
    assert (pd.to_datetime(dated.issue_close_date) == pd.to_datetime(dated.issue_close_date_verified)).all()


def test_final_book_sensitivity_leaves_authoritative_training_frame_unchanged():
    first = pd.read_csv(ROOT / "data/daywise_subscription_candidates.csv")
    second = pd.read_csv(ROOT / "data/equitymaster_daywise_candidates.csv")
    offers = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    overlay, audit = build_overlay(first, second, offers)
    assert audit["excluded_cross_source_final_conflicts"] > 0
    assert not overlay.ipo_id.duplicated().any()
    assert set(overlay.ipo_id).isdisjoint(audit["conflict_ids"])
    baseline = prepare_dataset()
    before = baseline[["id", "qib_subscription", "retail_subscription", "hni_subscription"]].copy(deep=True)
    challenger = overlay_frame(baseline, overlay)
    pd.testing.assert_frame_equal(before, baseline[before.columns])
    assert baseline["id"].equals(challenger["id"])
    assert (baseline["qib_subscription"] != challenger["qib_subscription"]).any()


def test_adjudication_ledger_tracks_real_disagreements_without_overwriting_data():
    ledger = pd.read_csv(ROOT / "output/final_subscription_adjudication.csv")
    stored = pd.read_csv(ROOT / "data/subscription_observations.csv")
    dates = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    joined = ledger.merge(stored, on="ipo_id", suffixes=("_audit", "_stored"), validate="one_to_one")
    joined = joined.merge(dates[["ipo_id", "issue_close_date"]], on="ipo_id", validate="one_to_one")
    assert not joined.ipo_id.duplicated().any()
    assert (joined.nse_issue_close_date == joined.issue_close_date).all()
    assert (joined.stored_observed_at == joined.observed_at).all()
    for category in ("qib", "hni", "retail"):
        assert (joined[f"stored_{category}"] == joined[f"{category}_subscription"]).all()
    assert joined[joined.ipo_id.eq(303)].evidence_grade.item() == "issuer_and_exchange_filing"
