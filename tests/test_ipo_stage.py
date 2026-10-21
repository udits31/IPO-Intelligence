import unittest
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from scrapers.live_ipo_scraper import classify_ipo_stage


class IpoStageTest(unittest.TestCase):
    def test_daily_session_end_does_not_close_offer_before_final_date(self):
        issue = {"date_range": "15 - 19 Oct", "listing_date": "2026-10-23"}
        with patch("scrapers.live_ipo_scraper.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 16, 19, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
            self.assertEqual(classify_ipo_stage(issue)[0], "OPEN")

    def test_final_day_after_cutoff_closes_offer(self):
        issue = {"date_range": "15 - 19 Oct", "listing_date": "2026-10-23"}
        with patch("scrapers.live_ipo_scraper.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 19, 19, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
            self.assertEqual(classify_ipo_stage(issue)[0], "CLOSED")


if __name__ == "__main__":
    unittest.main()
