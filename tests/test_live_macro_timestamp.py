"""NSE market observations must retain the exchange's as-of time."""

import unittest

from scrapers.live_ipo_scraper import _parse_nse_market_timestamp


class LiveMacroTimestampTest(unittest.TestCase):
    def test_exchange_timestamp_is_ist_market_close(self):
        # Timestamp observed in the NSE allIndices response on 18 Oct 2026.
        stamp = _parse_nse_market_timestamp('16-Oct-2026 15:30')
        self.assertEqual(stamp.isoformat(), '2026-10-16T15:30:00+05:30')

    def test_missing_exchange_timestamp_is_not_replaced_with_fetch_time(self):
        with self.assertRaises(ValueError):
            _parse_nse_market_timestamp(None)


if __name__ == '__main__':
    unittest.main()
