from pathlib import Path

import pandas as pd

from core.similarity_engine import SimilarityEngine
from ml.rebuild_market_context import build_context


ROOT = Path(__file__).resolve().parents[1]


def test_rebuilt_context_uses_actual_zomato_close_and_no_future_data() -> None:
    offers = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    observations = pd.read_csv(ROOT / "data/raw_daily_market_time_series.csv")
    rebuilt, report = build_context(offers, observations)
    zomato = rebuilt[rebuilt.company_name.eq("Zomato Ltd.")].iloc[0]
    assert zomato.issue_close_date == "2021-07-16"
    assert zomato.cutoff_trading_date == "2021-07-16"
    assert abs(zomato.india_vix - 11.71) < 0.01
    assert (pd.to_datetime(rebuilt.cutoff_trading_date) <= pd.to_datetime(rebuilt.issue_close_date)).all()
    assert report["complete_market_context"] == len(rebuilt) == 241


def test_analogs_are_source_complete_and_development_only() -> None:
    engine = SimilarityEngine()
    assert len(engine.df_universe) >= 200
    assert pd.to_datetime(engine.df_universe.listing_date).dt.year.max() == 2024
    assert engine.df_universe[engine.feature_names].notna().all().all()
    assert abs(engine.df_universe.loc[engine.df_universe.company_name.eq("Zomato Ltd."), "india_vix"].iloc[0] - 11.71) < 0.01
    result = engine.evaluate_ipo(qib_sub=20.0, retail_sub=5.0, gmp_pct=None,
                                 vix=14.0, nifty_10d=1.0)
    assert result["total_analogs_count"] == 15
