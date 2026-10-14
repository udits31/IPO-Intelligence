from core.monitor_engine import HoldSellMonitor
import numpy as np


def test_missing_disclosures_do_not_imply_anchor_investors_held():
    monitor = HoldSellMonitor()
    result = monitor.evaluate_holding(364, as_of_date="2024-03-15",
                                      disclosures=[], fetch_news=False)
    assert "error" not in result
    assert "ACCUMULATE / STRONG BUY" != result["verdict"]
    assert "Anchor Filings Unverified" == result["thesis_breakdown"][2]["verdict_badge"]
    text = " ".join(result["key_reasons"] + [result["thesis_breakdown"][2]["explanation"]])
    assert "unverified" in text.lower()
    assert "zero institutional" not in text.lower()


def test_observed_anchor_sale_still_blocks_accumulation():
    monitor = HoldSellMonitor()
    result = monitor.evaluate_holding(
        364, as_of_date="2024-03-15", fetch_news=False,
        disclosures=[{"discloser_type": "anchor", "transaction_type": "SELL",
                      "pct_change": -3.2}],
    )
    assert result["verdict"] == "SELL / REDUCE"
    assert result["thesis_breakdown"][2]["verdict_badge"] == "Anchor Sale Detected"


def test_missing_sector_history_is_not_presented_as_zero_or_relative_strength():
    monitor = HoldSellMonitor()
    prices = monitor.attribution_engine.df_prices
    prices.loc[prices["ipo_id"] == 364, "sector_index_close"] = np.nan
    result = monitor.evaluate_holding(364, as_of_date="2024-03-15",
                                      disclosures=[], fetch_news=False)
    assert "error" not in result
    attribution = result["attribution_30d"]
    assert attribution["sector_return_pct"] is None
    assert attribution["spread_vs_sector_pct"] is None
    assert attribution["category"] == "INSUFFICIENT_BENCHMARK"
    assert result["thesis_breakdown"][1]["verdict_badge"] == "Sector Comparison Unavailable"


def test_historical_monitor_does_not_pull_present_day_news(monkeypatch):
    monitor = HoldSellMonitor()
    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("Present-day RSS was queried for a historical decision")
    monkeypatch.setattr(monitor.news_service, "get_company_news", unexpected_fetch)
    result = monitor.evaluate_holding(364, as_of_date="2024-03-15",
                                      disclosures=[], fetch_news=True)
    assert "error" not in result
    assert result["news_sentiment"]["status"] == "historical_news_unavailable"
    assert result["news_sentiment"]["score"] is None
