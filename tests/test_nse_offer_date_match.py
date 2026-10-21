import pandas as pd
import hashlib
import json
from pathlib import Path

from ml.fetch_nse_offer_dates import match_records


ROOT = Path(__file__).resolve().parents[1]


def test_requires_date_price_and_identity_and_rejects_ambiguous() -> None:
    canonical = pd.DataFrame([
        {"id": 1, "company_name": "Alpha Limited", "nse_symbol": "ALPHA",
         "listing_date": "2024-06-28", "issue_price": 100},
        {"id": 2, "company_name": "Renamed Limited", "nse_symbol": "NEWNAME",
         "listing_date": "2024-07-01", "issue_price": 70},
        {"id": 3, "company_name": "Wrong Price Limited", "nse_symbol": "WRONG",
         "listing_date": "2024-07-02", "issue_price": 75},
    ])
    records = [
        {"company": "Alpha Limited", "symbol": "ALPHA", "ipoStartDate": "21-JUN-2024",
         "ipoEndDate": "25-JUN-2024", "listingDate": "28-JUN-2024", "issuePrice": "100"},
        {"company": "Renamed Limited", "symbol": "OLDNAME", "ipoStartDate": "24-JUN-2024",
         "ipoEndDate": "26-JUN-2024", "listingDate": "01-JUL-2024", "issuePrice": "70"},
        {"company": "Wrong Price Limited", "symbol": "WRONG", "ipoStartDate": "25-JUN-2024",
         "ipoEndDate": "27-JUN-2024", "listingDate": "02-JUL-2024", "issuePrice": "76"},
    ]
    matched, report = match_records(canonical, records, "sha")
    assert matched.ipo_id.tolist() == [1, 2]
    assert matched.identity_match.tolist() == ["name_and_symbol", "name"]
    assert report["unresolved"] == 1
    ambiguous, report = match_records(canonical.iloc[:1], [records[0], records[0]], "sha")
    assert ambiguous.empty
    assert report["unresolved"] == 1


def test_rejects_post_listing_or_reversed_offer_period() -> None:
    canonical = pd.DataFrame([{"id": 1, "company_name": "Alpha Ltd.", "nse_symbol": "ALPHA",
                               "listing_date": "2024-06-28", "issue_price": 100}])
    record = {"company": "Alpha Limited", "symbol": "ALPHA", "ipoStartDate": "28-JUN-2024",
              "ipoEndDate": "29-JUN-2024", "listingDate": "28-JUN-2024", "issuePrice": "100"}
    matched, report = match_records(canonical, [record], "sha")
    assert matched.empty
    assert report["unresolved"] == 1


def test_archived_nse_response_matches_verified_overlay() -> None:
    raw = (ROOT / "data/nse_public_past_issues_raw.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    overlay = pd.read_csv(ROOT / "data/nse_verified_offer_dates.csv")
    canonical = pd.read_csv(ROOT / "data/canonical_listed_ipos.csv")
    rebuilt, report = match_records(canonical, json.loads(raw), digest)
    assert report["accepted"] == len(overlay) == 422
    assert set(rebuilt.ipo_id) == set(overlay.ipo_id)
    assert set(overlay.source_response_sha256) == {digest}
