"""Regression checks for the real NSE holiday and checkpoint-close audit."""

from pathlib import Path

import pandas as pd

from ml.filter_module2_market_holidays import filter_holidays


ROOT = Path(__file__).resolve().parents[1]


def test_market_holiday_carry_forward_is_removed_without_price_imputation():
    raw = pd.read_csv(ROOT / "data/audit/module2_holdout_prices_2025_2026.csv")
    revised, report = filter_holidays(raw)
    assert report["removed_by_date"] == {"2026-01-15": 80}
    assert report["removed_rows"] == 80
    assert revised.nifty50_close.notna().all()
    assert not revised.date.eq("2026-01-15").any()
    unchanged = raw[~raw.date.eq("2026-01-15")].reset_index(drop=True)
    pd.testing.assert_frame_equal(revised.reset_index(drop=True), unchanged)


def test_nse_bhavcopy_audit_covers_checkpoint_closes():
    audit = pd.read_csv(ROOT / "output/module2_nse_close_audit.csv")
    assert len(audit) == 512
    assert audit.nse_close.notna().sum() == 511
    assert audit.loc[audit.nse_close.isna(), "date"].tolist() == ["2026-01-15"]
    paired = audit.dropna(subset=["nse_close", "close_price_adjusted"])
    assert len(paired) == 507
    assert paired.difference_pct.max() < 1.0
