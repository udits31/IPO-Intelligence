"""Market-window subscription capture must preserve failures and point-in-time evidence."""

import gzip
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from scrapers import live_ipo_scraper as live
from services import snapshot_service as snapshots
from services import nse_subscription_service as nse_service

IST = ZoneInfo('Asia/Kolkata')
IPO_URL = 'https://www.chittorgarh.com/ipo/example-ipo/123/'


class SubscriptionCaptureTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.directory.name) / 'snapshots.db')
        snapshots.init_db(self.db_path)
        self.cache = {IPO_URL: {
            'company_name': 'Example', 'category': 'OPEN', 'source_observed': True,
            'date_range': '13 - 15 Oct', 'qib_subscription': 7.0,
            'subscription_observed': True,  # old cache must never become new evidence
        }}
        self.clock = datetime(2026, 10, 14, 14, 30, tzinfo=IST)

    def tearDown(self):
        self.directory.cleanup()

    def test_window_and_aligned_final_hour(self):
        self.assertEqual(snapshots.subscription_slot(self.clock), '2026-10-14_1430')
        self.assertEqual(snapshots.subscription_slot(self.clock.replace(minute=34)), '2026-10-14_1430')
        self.assertEqual(snapshots.subscription_slot(self.clock.replace(hour=16, minute=58)), '2026-10-14_1655')
        self.assertIsNone(snapshots.subscription_slot(self.clock.replace(hour=17, minute=0)))
        self.assertIsNone(snapshots.subscription_slot(self.clock.replace(hour=17, minute=1)))
        self.assertIsNone(snapshots.subscription_slot(self.clock.replace(hour=17, minute=15)))
        self.assertIsNone(snapshots.subscription_slot(self.clock.replace(hour=19, minute=15)))
        self.assertIsNone(snapshots.subscription_slot(self.clock.replace(hour=19, minute=16)))
        self.assertIsNone(snapshots.subscription_slot(self.clock.replace(hour=9)))

    def test_failed_poll_is_recorded_without_copying_stale_qib(self):
        with patch.object(live, 'fetch_subscription_evidence', return_value={
            'status': 'no_subscription_table', 'http_status': 200,
            'raw_html': '<table>allocation only</table>', 'raw_sha256': 'abc',
        }):
            self.assertEqual(snapshots.capture_active_subscriptions(
                now_ist=self.clock, cache=self.cache, db_path=self.db_path, capture_exchange=False), 1)
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute('SELECT status, qib_x, raw_html_gzip FROM subscription_capture_v2').fetchone()
        self.assertEqual(row[0], 'no_subscription_table')
        self.assertIsNone(row[1])
        self.assertEqual(gzip.decompress(row[2]).decode(), '<table>allocation only</table>')
        self.assertEqual(self.cache[IPO_URL]['qib_subscription'], 7.0)
        decision = snapshots.get_automatic_decision_rows(company='Example', db_path=self.db_path)['decisions'][0]
        self.assertIsNone(decision['inputs']['qib_subscription'])
        self.assertIsNone(decision['positive_listing_probability'])
        self.assertEqual(decision['decision_kind'], 'provisional_scorecard')
        self.assertIsNone(decision['model_version'])
        self.assertEqual(decision['decision_engine_version'], 'prebid_rules_2026_10_v1')

    def test_completed_slot_is_not_refetched_after_restart(self):
        evidence = {'status': 'no_subscription_table', 'http_status': 200,
                    'raw_html': '<p>Not published</p>', 'raw_sha256': 'abc'}
        with patch.object(live, 'fetch_subscription_evidence', return_value=evidence) as fetch:
            first = snapshots.capture_active_subscriptions(now_ist=self.clock,
                cache=self.cache, db_path=self.db_path, capture_exchange=False)
            second = snapshots.capture_active_subscriptions(now_ist=self.clock,
                cache=self.cache, db_path=self.db_path, capture_exchange=False)
        self.assertEqual((first, second), (1, 0))
        self.assertEqual(fetch.call_count, 1)
        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM subscription_capture_v2').fetchone()[0], 1)

    def test_recovered_external_readings_do_not_become_forecasts(self):
        tonbo = snapshots.get_automatic_decision_rows(
            company='Tonbo Imaging India', db_path=self.db_path)
        self.assertEqual(tonbo['count'], 0)
        self.assertEqual(len(tonbo['recovered_observations']), 2)
        self.assertTrue(all(not row['eligible_for_prospective_accuracy']
                            for row in tonbo['recovered_observations']))
        self.assertEqual(snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)['recovered_observations'], [])
        self.assertEqual(snapshots.get_intraday_forecast_scorecard(
            db_path=self.db_path, cache={})['summary']['ipos_with_forecast'], 0)

    def test_closed_ipo_replay_is_locked_before_listing_but_not_counted_as_live_forecast(self):
        closed = {IPO_URL: {**self.cache[IPO_URL], 'category': 'CLOSED',
                            'date_range': '13 - 15 Oct', 'listing_date': '2026-10-21'}}
        capture_time = self.clock.replace(hour=16, minute=45)
        evidence = {'status': 'verified', 'http_status': 200,
                    'fetched_at_ist': capture_time.isoformat(),
                    'subscription_url': live._subscription_url(IPO_URL),
                    'raw_html': '<table>observed</table>', 'raw_sha256': 'replay-evidence',
                    'values': {'qib_subscription': 4.2, 'hni_subscription': 1.3,
                               'retail_subscription': 0.8}}
        with patch.object(live, 'fetch_subscription_evidence', return_value=evidence):
            snapshots.capture_active_subscriptions(
                now_ist=capture_time, cache=self.cache, db_path=self.db_path,
                capture_exchange=False)
        replay_clock = datetime(2026, 10, 20, 1, 30, tzinfo=IST)
        self.assertEqual(snapshots.record_prelisting_replays(
            cache=closed, now_ist=replay_clock, db_path=self.db_path), 1)
        self.assertEqual(snapshots.record_prelisting_replays(
            cache=closed, now_ist=replay_clock, db_path=self.db_path), 0)
        history = snapshots.get_automatic_decision_rows(company='Example', db_path=self.db_path)
        replay = history['prelisting_replay']
        self.assertEqual(replay['evidence_status'], 'app_capture_verified')
        self.assertEqual(replay['qib_x'], 4.2)
        self.assertEqual(replay['recorded_at_ist'], replay_clock.isoformat(timespec='seconds'))
        self.assertFalse(any(row['decision_kind'] == 'live_bid_forecast' for row in history['decisions']))
        self.assertEqual(snapshots.get_intraday_forecast_scorecard(
            db_path=self.db_path, cache=closed)['summary']['ipos_with_forecast'], 0)
        replay_scorecard = snapshots.get_prelisting_replay_scorecard(
            db_path=self.db_path, cache=closed)
        self.assertEqual(replay_scorecard['summary']['replays_locked'], 1)
        self.assertEqual(replay_scorecard['summary']['pending_open'], 1)
        self.assertIsNone(replay_scorecard['summary']['direction_accuracy_app_captured_pct'])

    def test_only_pre_deadline_nse_capture_becomes_prelisting_outcome_forecast(self):
        closed = {IPO_URL: {**self.cache[IPO_URL], 'category': 'CLOSED',
                            'date_range': '13 - 15 Oct', 'listing_date': '2026-10-21'}}
        capture_time = datetime(2026, 10, 15, 16, 55, tzinfo=IST)
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute("""
                INSERT INTO subscription_capture_v2 (
                    ipo_id, company_name, ipo_url, attempted_at_ist,
                    fetched_at_ist, slot_key, closing_date, snapshot_stage,
                    status, qib_x, nii_x, retail_x, raw_sha256,
                    raw_html_gzip, source_family, capture_phase
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ('chitto_123', 'Example', IPO_URL, capture_time.isoformat(),
                  capture_time.isoformat(), '2026-10-15_1655', '2026-10-15',
                  'FINAL_CLOSE', 'verified', 4.2, 1.3, 0.8,
                  hashlib.sha256(b'<table>verified NSE</table>').hexdigest(),
                  gzip.compress(b'<table>verified NSE</table>'),
                  'nse_exchange', 'pre_deadline'))
            capture_id = cursor.lastrowid
            conn.execute("""INSERT INTO subscription_capture_v2 (
                ipo_id, company_name, ipo_url, attempted_at_ist, fetched_at_ist,
                slot_key, closing_date, snapshot_stage, status, qib_x, nii_x,
                retail_x, raw_sha256, raw_html_gzip, source_family, capture_phase)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                'chitto_123', 'Example', IPO_URL,
                capture_time.replace(hour=18).isoformat(),
                capture_time.replace(hour=18).isoformat(), '2026-10-15_1855',
                '2026-10-15', 'POST_CLOSE_RECONCILIATION', 'verified', 20.0,
                30.0, 40.0, hashlib.sha256(b'<table>after close</table>').hexdigest(),
                gzip.compress(b'<table>after close</table>'), 'nse_exchange',
                'post_close_reconciliation'))
            conn.commit()
        lock_clock = datetime(2026, 10, 20, 12, 0, tzinfo=IST)
        self.assertEqual(snapshots.record_prelisting_outcome_forecasts(
            cache=closed, now_ist=lock_clock, db_path=self.db_path), 1)
        self.assertEqual(snapshots.record_prelisting_outcome_forecasts(
            cache=closed, now_ist=lock_clock, db_path=self.db_path), 0)
        forecast = snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)['prelisting_outcome_forecast']
        self.assertEqual(forecast['capture_id'], capture_id)
        self.assertEqual(forecast['recorded_at_ist'], lock_clock.isoformat(timespec='seconds'))
        self.assertEqual(forecast['evidence_sha256'],
                         hashlib.sha256(b'<table>verified NSE</table>').hexdigest())
        self.assertEqual(forecast['capture_phase'], 'pre_deadline')
        self.assertGreater(forecast['positive_listing_probability'], 0)
        self.assertEqual(snapshots.get_intraday_forecast_scorecard(
            db_path=self.db_path, cache=closed)['summary']['ipos_with_forecast'], 0)
        outcome = snapshots.get_prelisting_outcome_scorecard(
            db_path=self.db_path, cache=closed)
        self.assertEqual(outcome['summary']['forecasts_locked'], 1)
        self.assertEqual(outcome['summary']['openings_verified'], 0)
        self.assertIsNone(outcome['summary']['direction_accuracy_pct'])
        listed = {IPO_URL: {**closed[IPO_URL], 'category': 'LISTED',
                            'issue_price': 100, 'listing_open_price': 110}}
        resolved = snapshots.get_prelisting_outcome_scorecard(
            db_path=self.db_path, cache=listed)
        self.assertEqual(resolved['summary']['openings_verified'], 0)
        self.assertIsNone(resolved['summary']['direction_accuracy_pct'])

    def test_no_prelisting_outcome_forecast_after_listing_or_without_verified_capture(self):
        closed = {IPO_URL: {**self.cache[IPO_URL], 'category': 'CLOSED',
                            'listing_date': '2026-10-21'}}
        self.assertEqual(snapshots.record_prelisting_outcome_forecasts(
            cache=closed, now_ist=datetime(2026, 10, 20, tzinfo=IST),
            db_path=self.db_path), 0)
        self.assertEqual(snapshots.record_prelisting_outcome_forecasts(
            cache=closed, now_ist=datetime(2026, 10, 21, tzinfo=IST),
            db_path=self.db_path), 0)

    def test_two_polls_same_hour_are_append_only(self):
        evidence = {'status': 'verified', 'http_status': 200,
                    'fetched_at_ist': self.clock.isoformat(),
                    'subscription_url': live._subscription_url(IPO_URL),
                    'raw_html': '<table>observed</table>', 'raw_sha256': 'def',
                    'values': {'qib_subscription': 1.2, 'hni_subscription': 2.3,
                               'retail_subscription': 3.4}}
        with patch.object(live, 'fetch_subscription_evidence', return_value=evidence):
            snapshots.capture_active_subscriptions(now_ist=self.clock, cache=self.cache, db_path=self.db_path, capture_exchange=False)
            snapshots.capture_active_subscriptions(now_ist=self.clock.replace(minute=35),
                                                   cache=self.cache, db_path=self.db_path, capture_exchange=False)
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute('SELECT slot_key, qib_x FROM subscription_capture_v2 ORDER BY id').fetchall()
        self.assertEqual(rows, [('2026-10-14_1430', 1.2), ('2026-10-14_1435', 1.2)])
        history = snapshots.get_automatic_decision_rows(company='Example', db_path=self.db_path)
        decisions = history['decisions']
        self.assertEqual({row['slot_key'] for row in decisions},
                         {'2026-10-14_1430', '2026-10-14_1435'})
        self.assertTrue(all(row['positive_listing_probability'] is None for row in decisions))
        self.assertEqual(len(history['pre_deadline_captures']), 2)
        self.assertTrue(all(row['raw_sha256'] == 'def' for row in history['pre_deadline_captures']))

    def test_1645_live_book_forecast_is_saved_with_source_and_inputs(self):
        clock = self.clock.replace(hour=16, minute=45)
        values = {'qib_subscription': 1.2, 'hni_subscription': 2.3,
                  'retail_subscription': 3.4}
        snapshots.record_automatic_decision(
            item=self.cache[IPO_URL], ipo_url=IPO_URL, slot='2026-10-14_1645',
            stage='HOURLY', now_ist=clock, values=values,
            source_family='nse_exchange', source_status='verified',
            source_asof=clock.replace(minute=40).isoformat(),
            post_close=False, decision_at_ist=clock, db_path=self.db_path,
        )
        decisions = snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)['decisions']
        forecast = next(row for row in decisions if row['decision_kind'] == 'live_bid_forecast')
        self.assertGreater(forecast['positive_listing_probability'], 0)
        self.assertLess(forecast['positive_listing_probability'], 1)
        self.assertEqual(forecast['conviction'], 'UNVALIDATED')
        self.assertEqual(forecast['inputs']['qib_subscription'], 1.2)
        self.assertEqual(forecast['inputs']['source_asof_ist'], clock.replace(minute=40).isoformat())
        self.assertEqual(len(decisions), 2)
        pending = snapshots.get_intraday_forecast_scorecard(
            db_path=self.db_path, cache=self.cache)
        self.assertEqual(pending['summary']['pending_open'], 1)
        listed = {IPO_URL: {**self.cache[IPO_URL], 'category': 'LISTED',
                            'issue_price': 100, 'listing_open_price': 110}}
        resolved = snapshots.get_intraday_forecast_scorecard(
            db_path=self.db_path, cache=listed)
        self.assertEqual(resolved['summary']['ipos_with_verified_open'], 0)
        self.assertIsNone(resolved['daily_records'][0]['actual_open_return_pct'])
        self.assertIsNone(resolved['summary']['direction_accuracy_pct'])

    def test_midday_complete_book_is_saved_as_experimental_forecast(self):
        clock = self.clock.replace(hour=11, minute=30)
        snapshots.record_automatic_decision(
            item=self.cache[IPO_URL], ipo_url=IPO_URL, slot='2026-10-14_1130',
            stage='HOURLY', now_ist=clock,
            values={'qib_subscription': 0.8, 'hni_subscription': 1.5,
                    'retail_subscription': 1.1},
            source_family='nse_exchange', source_status='verified',
            source_asof=clock.replace(minute=25).isoformat(), post_close=False,
            decision_at_ist=clock, db_path=self.db_path,
        )
        history = snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)
        forecast = next(row for row in history['decisions']
                        if row['decision_kind'] == 'live_bid_forecast')
        self.assertEqual(forecast['inputs']['decision_stage'], 'before_final_application_cutoff')
        self.assertEqual(forecast['recorded_at_ist'], clock.isoformat(timespec='seconds'))
        self.assertEqual(forecast['conviction'], 'UNVALIDATED')

    def test_no_forecast_after_offering_closing_date_even_if_cache_says_open(self):
        clock = self.clock.replace(day=16, hour=11, minute=30)
        snapshots.record_automatic_decision(
            item=self.cache[IPO_URL], ipo_url=IPO_URL, slot='2026-10-16_1130',
            stage='HOURLY', now_ist=clock,
            values={'qib_subscription': 0.8, 'hni_subscription': 1.5,
                    'retail_subscription': 1.1},
            source_family='nse_exchange', source_status='verified',
            source_asof=clock.replace(minute=25).isoformat(), post_close=False,
            decision_at_ist=clock, db_path=self.db_path,
        )
        history = snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)
        self.assertFalse(any(row['decision_kind'] == 'live_bid_forecast'
                             for row in history['decisions']))

    def test_last_published_book_is_usable_before_close_but_reconciliation_is_excluded(self):
        clock = self.clock.replace(hour=16, minute=45)
        values = {'qib_subscription': 1.2, 'hni_subscription': 2.3,
                  'retail_subscription': 3.4}
        snapshots.record_automatic_decision(
            item=self.cache[IPO_URL], ipo_url=IPO_URL, slot='2026-10-14_1645',
            stage='HOURLY', now_ist=clock, values=values,
            source_family='nse_exchange', source_status='verified',
            source_asof=clock.replace(hour=15).isoformat(),
            post_close=False, decision_at_ist=clock, db_path=self.db_path,
        )
        snapshots.record_automatic_decision(
            item=self.cache[IPO_URL], ipo_url=IPO_URL, slot='2026-10-14_1900',
            stage='POST_CLOSE_RECONCILIATION', now_ist=clock.replace(hour=19, minute=0),
            values=values, source_family='nse_exchange', source_status='verified',
            source_asof=clock.replace(hour=19, minute=0).isoformat(),
            post_close=True, db_path=self.db_path,
        )
        decisions = snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)['decisions']
        forecasts = [row for row in decisions if row['decision_kind'] == 'live_bid_forecast']
        self.assertEqual(len(forecasts), 1)
        self.assertFalse(forecasts[0]['inputs']['snapshot_is_current'])
        self.assertEqual(forecasts[0]['inputs']['snapshot_age_seconds'], 3600)

    def test_scrape_finishing_after_cutoff_cannot_backdate_forecast(self):
        clock = self.clock.replace(day=15, hour=16, minute=59)
        values = {'qib_subscription': 1.2, 'hni_subscription': 2.3,
                  'retail_subscription': 3.4}
        snapshots.record_automatic_decision(
            item=self.cache[IPO_URL], ipo_url=IPO_URL, slot='2026-10-14_1655',
            stage='HOURLY', now_ist=clock, values=values,
            source_family='nse_exchange', source_status='verified',
            source_asof=clock.replace(hour=17, minute=0).isoformat(),
            post_close=False, decision_at_ist=clock.replace(hour=17, minute=1),
            db_path=self.db_path,
        )
        decisions = snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)['decisions']
        self.assertFalse(any(row['decision_kind'] == 'live_bid_forecast' for row in decisions))

    def test_allocation_is_not_subscription(self):
        html = '<table><tr><th>Investor Category</th><th>Shares</th></tr><tr><td>QIB</td><td>50%</td></tr></table>'
        self.assertIsNone(live.parse_subscription_table(html))
        response = type('Response', (), {'text': html, 'status_code': 200,
            'raise_for_status': lambda self: None})()
        with patch.object(live.httpx, 'get', return_value=response):
            evidence = live.fetch_subscription_evidence(IPO_URL)
        self.assertEqual(evidence['status'], 'no_subscription_table')
        self.assertIsNone(evidence['values'])
        self.assertEqual(evidence['raw_sha256'], hashlib.sha256(html.encode()).hexdigest())

    def test_partial_live_table_keeps_observed_categories_without_inventing_qib(self):
        html = '''<table><tr><th>Investor Category</th><th>Subscription (times)</th></tr>
            <tr><td>Non Institutional</td><td>1.74x</td></tr>
            <tr><td>Retail Individual</td><td>0.86x</td></tr></table>'''
        response = type('Response', (), {'text': html, 'status_code': 200,
            'raise_for_status': lambda self: None})()
        with patch.object(live.httpx, 'get', return_value=response):
            evidence = live.fetch_subscription_evidence(IPO_URL)
        self.assertEqual(evidence['status'], 'partial_subscription_table')
        self.assertEqual(evidence['values']['hni_subscription'], 1.74)
        self.assertEqual(evidence['values']['retail_subscription'], 0.86)
        self.assertNotIn('qib_subscription', evidence['values'])

    def test_partial_capture_replaces_old_book_but_does_not_verify_model_inputs(self):
        evidence = {'status': 'partial_subscription_table', 'http_status': 200,
                    'fetched_at_ist': self.clock.isoformat(),
                    'subscription_url': live._subscription_url(IPO_URL),
                    'raw_html': '<table>partial</table>', 'raw_sha256': 'partial',
                    'values': {'hni_subscription': 1.74, 'retail_subscription': 0.86}}
        with patch.object(live, 'fetch_subscription_evidence', return_value=evidence):
            snapshots.capture_active_subscriptions(now_ist=self.clock, cache=self.cache, db_path=self.db_path, capture_exchange=False)
        item = self.cache[IPO_URL]
        self.assertIsNone(item['qib_subscription'])
        self.assertEqual(item['hni_subscription'], 1.74)
        self.assertEqual(item['retail_subscription'], 0.86)
        self.assertFalse(item['subscription_observed'])
        self.assertEqual(item['subscription_observation_status'], 'partial_subscription_table')

    def test_scheduled_subscription_fetch_stops_after_session(self):
        evidence = {'status': 'verified', 'http_status': 200,
                    'fetched_at_ist': self.clock.replace(day=15, hour=19, minute=0).isoformat(),
                    'subscription_url': live._subscription_url(IPO_URL),
                    'raw_html': '<table>after close</table>', 'raw_sha256': 'afterclose',
                    'values': {'qib_subscription': 8.0, 'hni_subscription': 4.0,
                               'retail_subscription': 3.0}}
        with patch.object(live, 'fetch_subscription_evidence', return_value=evidence) as fetch:
            self.assertEqual(snapshots.capture_active_subscriptions(
                now_ist=self.clock.replace(day=15, hour=19, minute=0),
                cache=self.cache, db_path=self.db_path, capture_exchange=False), 0)
        fetch.assert_not_called()
        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM subscription_capture_v2').fetchone()[0], 0)

    def test_poll_finishing_after_closing_cutoff_is_reconciliation(self):
        started = self.clock.replace(day=15, hour=16, minute=55)
        fetched = started.replace(hour=17, minute=0)
        evidence = {'status': 'verified', 'http_status': 200,
                    'fetched_at_ist': fetched.isoformat(),
                    'raw_html': '<table>after cutoff</table>',
                    'values': {'qib_subscription': 8.0, 'hni_subscription': 4.0,
                               'retail_subscription': 3.0}}
        with patch.object(live, 'fetch_subscription_evidence', return_value=evidence):
            snapshots.capture_active_subscriptions(
                now_ist=started, cache=self.cache, db_path=self.db_path,
                capture_exchange=False)
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute('SELECT capture_phase, snapshot_stage '
                               'FROM subscription_capture_v2').fetchone()
        self.assertEqual(row, ('post_close_reconciliation', 'POST_CLOSE_RECONCILIATION'))
        self.assertEqual(self.cache[IPO_URL]['subscription_capture_phase'],
                         'post_close_reconciliation')
        decision = snapshots.get_automatic_decision_rows(
            company='Example', db_path=self.db_path)['decisions'][0]
        self.assertEqual(decision['decision_kind'], 'reconciliation_only')
        self.assertIsNone(decision['positive_listing_probability'])

    def test_independent_nse_capture_replaces_display_book_not_history(self):
        chitto = {'status': 'partial_subscription_table', 'http_status': 200,
                  'fetched_at_ist': self.clock.isoformat(), 'raw_html': '{}',
                  'values': {'hni_subscription': 1.74, 'retail_subscription': 0.86}}
        nse = {'status': 'verified', 'http_status': 200,
               'fetched_at_ist': self.clock.isoformat(),
               'source_displayed_at_ist': self.clock.isoformat(),
               'raw_html': '{"symbol":"EXAMPLE"}', 'raw_sha256': 'nsehash',
               'nse_symbol': 'EXAMPLE', 'values': {
                   'qib_subscription': 0.0042, 'hni_subscription': 5.06,
                   'retail_subscription': 2.20}}
        with patch.object(live, 'fetch_subscription_evidence', return_value=chitto), \
             patch.object(nse_service, 'fetch_nse_subscription_evidence', return_value=nse):
            self.assertEqual(snapshots.capture_active_subscriptions(
                now_ist=self.clock, cache=self.cache, db_path=self.db_path), 2)
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute('SELECT source_family, qib_x, nii_x FROM subscription_capture_v2 ORDER BY id').fetchall()
        self.assertEqual(rows, [('chittorgarh_investorgain', None, 1.74),
                                ('nse_exchange', 0.0042, 5.06)])
        self.assertEqual(self.cache[IPO_URL]['hni_subscription'], 5.06)
        self.assertEqual(self.cache[IPO_URL]['subscription_source'], nse_service.SOURCE_NAME)
        self.assertFalse(self.cache[IPO_URL]['subscription_is_final'])
        decision = snapshots.get_automatic_decision_rows(company='Example', db_path=self.db_path)['decisions'][0]
        self.assertEqual(decision['source_family'], 'nse_exchange')
        self.assertEqual(decision['inputs']['qib_subscription'], 0.0042)
        self.assertIsNone(decision['positive_listing_probability'])
        self.assertEqual(decision['decision_kind'], 'provisional_scorecard')

    def test_closed_offer_does_not_pull_more_subscription_data(self):
        item = {**self.cache[IPO_URL], 'category': 'CLOSED',
                'date_range': '13 - 15 Oct', 'nse_symbol': 'EXAMPLE',
                'nse_symbol_source': 'nse_current_issue',
                'subscription_source': nse_service.SOURCE_NAME,
                'subscription_is_final': False}
        evidence = {'status': 'verified', 'historical_close': True,
                    'values': {'qib_subscription': 4.0, 'hni_subscription': 2.0,
                               'retail_subscription': 3.0},
                    'source_displayed_at_ist': '2026-10-15T19:00:00+05:30',
                    'fetched_at_ist': '2026-10-19T19:00:00+05:30',
                    'subscription_url': 'https://www.nseindia.com/market-data/issue-information?symbol=EXAMPLE'}
        with patch.object(nse_service, 'fetch_nse_subscription_evidence', return_value=evidence) as fetch:
            refreshed = live.refresh_cached_subscriptions(IPO_URL, item)
        fetch.assert_not_called()
        self.assertEqual(refreshed, item)


if __name__ == '__main__':
    unittest.main()
