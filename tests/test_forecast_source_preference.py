from services.snapshot_service import preferred_forecast
from scrapers.live_ipo_scraper import parse_subscription_published_at


def row(source, recorded, observed, publisher=None):
    return {'recorded_at_ist':f'2026-10-21T{recorded}:00+05:30',
            'source_asof_ist':f'2026-10-21T{observed}:00+05:30',
            'source_family':source,'inputs':{'source_published_at_ist':publisher}}


def test_old_aggregator_refetch_does_not_replace_fresh_publisher_dated_book():
    nse = row('nse_exchange','15:05','15:00')
    secondary = row('chittorgarh_investorgain','15:06','15:06')
    assert preferred_forecast([secondary,nse]) is nse


def test_newer_verified_publisher_time_can_win_regardless_of_source():
    nse = row('nse_exchange','15:05','15:00')
    secondary = row('chittorgarh_investorgain','15:06','15:04','2026-10-21T15:04:00+05:30')
    assert preferred_forecast([secondary,nse]) is secondary


def test_old_exchange_book_does_not_permanently_block_fallback():
    nse = row('nse_exchange','14:00','13:59')
    secondary = row('chittorgarh_investorgain','15:06','15:06')
    assert preferred_forecast([secondary,nse]) is secondary


def test_subscription_clock_not_gmp_or_fetch_clock():
    html = '<p>GMP updated Oct 21, 2026 15:05</p><p>The Jio Platforms IPO is subscribed 0.85x as of Oct 21, 2026 10:07.</p>'
    assert parse_subscription_published_at(html) == '2026-10-21T10:07:00+05:30'
    assert parse_subscription_published_at('<p>GMP as of Oct 21, 2026 15:05</p>') is None


def test_exchange_capture_time_is_not_assumed_to_be_a_publisher_clock_when_flagged():
    dated = row('nse_exchange', '15:05', '15:00')
    undated = row('nse_exchange', '15:06', '15:06')
    undated['inputs']['publisher_time_unverified'] = True
    assert preferred_forecast([undated, dated]) is dated
