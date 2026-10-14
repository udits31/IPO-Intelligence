"""Guard the exchange-identity gate for newly listed Module 2 issues."""

import hashlib
from datetime import date, datetime, timezone

import pytest

from scrapers.live_ipo_scraper import load_scraped_cache
from services.module2_recent_exchange import (
    ARCHIVES, _archive_rows, _identity_matches, recent_exchange_monitors,
)
from services.module2_decision_ledger import record_module2_decision, get_module2_event_history


def test_exchange_name_gate_rejects_unrelated_symbol_collisions():
    assert _identity_matches("SRIT India", "SRIT INDIA LIMITED")
    assert _identity_matches("Nityas Gems & Jewellery", "NITYAS GEMS N JEWELLERY L")
    assert not _identity_matches("SRIT India", "SRI TECH INDUSTRIES LTD")


def test_recent_ipo_uses_real_nse_archive_when_available():
    archive = ARCHIVES / "20261016.csv.zip"
    if not archive.exists():
        pytest.skip("Official 16 October NSE archive is not present in this checkout")
    if not any(item.get("company_name") == "SRIT India" for item in load_scraped_cache().values()):
        pytest.skip("The dated SRIT live-calendar observation is not present")
    rows = _archive_rows(archive, {"SRIT"})
    assert len(rows) == 1
    assert rows.iloc[0].ISIN == "INE307K01028"
    assert float(rows.iloc[0].OpnPric) > 0
    assert rows.iloc[0].archive_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    monitors, coverage = recent_exchange_monitors(as_of=date(2026, 10, 20))
    assert coverage["exchange_verified"] >= 1
    monitor = monitors[2858]
    result = monitor.evaluate_holding(2858, as_of_date="2026-10-16", fetch_news=False)
    assert result["price_source"] == "NSE UDiFF CM"
    assert result["price_basis"] == "Unadjusted daily close"
    assert result["price_archive_sha256"] == rows.iloc[0].archive_sha256


def test_real_exchange_decision_is_frozen_once_and_event_is_deduplicated(tmp_path, monkeypatch):
    # The real archive simulates a 20 October observation. Advancing the host
    # date must not silently turn that fixture into a stale-price test.
    import core.attribution_engine as attribution_module
    observed = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    class ObservationClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed.astimezone(tz) if tz else observed.replace(tzinfo=None)
    monkeypatch.setattr(attribution_module, 'datetime', ObservationClock)
    archive = ARCHIVES / "20261016.csv.zip"
    if not archive.exists():
        pytest.skip("Official NSE archive is not present in this checkout")
    monitors, _ = recent_exchange_monitors(as_of=date(2026, 10, 20))
    if 2858 not in monitors:
        pytest.skip("SRIT live calendar identity is not available")
    result = monitors[2858].evaluate_holding(2858, fetch_news=False)
    db = tmp_path / "decisions.db"
    assert record_module2_decision(result, db_path=db, observed_at=observed)
    assert not record_module2_decision(result, db_path=db, observed_at=observed)
    events = get_module2_event_history(2858, db_path=db)["events"]
    assert len(events) == 1
    assert events[0]["decision"]["nse_symbol"] == "SRIT"
    assert events[0]["decision"]["price_archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
