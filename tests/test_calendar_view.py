from datetime import date
from services.calendar_view import reconcile_calendar


def test_listed_record_replaces_stale_closed_duplicate():
    url = 'https://www.chittorgarh.com/ipo/example/123/'
    active, listed = reconcile_calendar([{'company_name':'Example','url':url,'category':'CLOSED'}],
        [{'company_name':'Example Ltd','url':url.rstrip('/'),'category':'LISTED','listing_date':'2026-10-08'}],
        date(2026,10,21))
    assert not active
    assert len(listed) == 1


def test_future_or_invalid_listing_does_not_promote_an_issue():
    source = [{'company_name':'Example','category':'CLOSED'}]
    for value in ['2026-10-23','Listed','']:
        active, listed = reconcile_calendar(source,[{'company_name':'Example','listing_date':value}],date(2026,10,21))
        assert active == source and not listed
