import pandas as pd
from pathlib import Path

from ml.audit_offer_dates import audit_offer_dates


def test_flags_repeated_placeholder_and_missing_dates_without_rewriting() -> None:
    frame = pd.DataFrame([
        {"id": 1, "company_name": "A", "issue_open_date": "2021-03-28",
         "issue_close_date": "2021-03-31", "listing_date": "2021-12-01"},
        {"id": 2, "company_name": "B", "issue_open_date": "2025-01-01",
         "issue_close_date": "2025-01-03", "listing_date": "2025-01-10"},
        {"id": 3, "company_name": "C", "issue_open_date": None,
         "issue_close_date": None, "listing_date": "2025-02-10"},
    ])
    report = audit_offer_dates(frame)
    assert report["development_2021_2024"]["repeated_march_28_31_pair"] == 1
    assert report["development_2021_2024"]["records_requiring_source_verification"] == 1
    assert report["later_2025_2026"]["missing_offer_date"] == 1
    assert report["later_2025_2026"]["records_without_these_flags"] == 1
    assert pd.isna(frame.loc[2, "issue_open_date"])


def test_flags_impossible_chronology() -> None:
    frame = pd.DataFrame([{"id": 1, "company_name": "A", "issue_open_date": "2024-04-05",
                           "issue_close_date": "2024-04-06", "listing_date": "2024-04-06"}])
    report = audit_offer_dates(frame)["development_2021_2024"]
    assert report["chronology_invalid"] == 1
    assert report["records_requiring_source_verification"] == 1


def test_sourced_seed_rows_match_canonical_identities() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = pd.read_csv(root / "data/canonical_listed_ipos.csv").set_index("id")
    seed = pd.read_csv(root / "ml/verified_offer_dates_seed.csv")
    assert seed.ipo_id.is_unique
    for row in seed.itertuples():
        assert canonical.loc[row.ipo_id, "company_name"] == row.company_name
        assert row.source_url.startswith("https://")
        assert pd.Timestamp(row.issue_open_date) <= pd.Timestamp(row.issue_close_date)
        assert pd.Timestamp(row.issue_close_date) < pd.Timestamp(canonical.loc[row.ipo_id, "listing_date"])
