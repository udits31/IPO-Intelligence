"""Unverified cache records and missing final bids must not become predictions."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import api.main as api
from api.main import ScrapeAndEvaluateRequest
from scrapers import live_ipo_scraper as live


class LiveProvenanceGateTest(unittest.TestCase):
    def test_stale_published_book_remains_visible_but_is_not_scored(self):
        observed_at = (datetime.now(ZoneInfo('Asia/Kolkata')) - timedelta(hours=2)).isoformat()
        record = {
            'company_name': 'Last Published Ltd', 'category': 'OPEN',
            'subscription_source': live.NSE_SUBSCRIPTION_SOURCE,
            'subscription_source_family': 'nse_exchange',
            'subscription_observed': True, 'subscription_observed_at': observed_at,
            'subscription_source_displayed_at': observed_at,
            'qib_subscription': 1.04, 'hni_subscription': 11.38,
            'retail_subscription': 3.78, 'issue_price': 100,
            'issue_size_cr': 100,
        }
        with patch.object(api.resolver, 'resolve', return_value={
            'success': True, 'source': 'Cached exchange observation', 'data': record,
        }), patch.object(api, 'fetch_live_macro_regime', return_value={}):
            result = api.scrape_and_evaluate_ipo(ScrapeAndEvaluateRequest(
                url_or_name='Last Published Ltd', fetch_news=False,
            ))
        displayed = result['scraped_data']
        self.assertEqual((displayed['qib_subscription'], displayed['hni_subscription'],
                          displayed['retail_subscription']), (1.04, 11.38, 3.78))
        self.assertEqual(displayed['subscription_display_status'], 'last_published')
        self.assertFalse(displayed['subscription_observed'])
        self.assertIsNone(result['evaluation']['hybrid_ml_scores']['hybrid_positive_listing_prob'])

    def test_open_partial_book_is_displayed_but_not_scored(self):
        record = {
            'company_name': 'Partial Book Ltd', 'category': 'OPEN',
            'subscription_source': live.SUBSCRIPTION_SOURCE,
            'subscription_observed': False,
            'subscription_observed_at': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
            'qib_subscription': None, 'hni_subscription': 1.74,
            'retail_subscription': 0.86, 'issue_price': 271,
            'issue_size_cr': 712,
        }
        with patch.object(api.resolver, 'resolve', return_value={
            'success': True, 'source': 'Test fetched page', 'data': record,
        }), patch.object(api, 'fetch_live_macro_regime', return_value={}):
            result = api.scrape_and_evaluate_ipo(ScrapeAndEvaluateRequest(
                url_or_name='Partial Book Ltd', fetch_news=False,
            ))
        self.assertEqual(result['scraped_data']['hni_subscription'], 1.74)
        self.assertEqual(result['scraped_data']['retail_subscription'], 0.86)
        self.assertIsNone(result['scraped_data']['qib_subscription'])
        self.assertFalse(result['scraped_data']['subscription_observed'])
        self.assertIsNone(result['evaluation']['hybrid_ml_scores']['hybrid_positive_listing_prob'])

    def test_allocation_table_is_not_subscription_demand(self):
        html = '''<h1>Example IPO Details</h1><table>
            <tr><th>Investor Category</th><th>Shares Offered</th></tr>
            <tr><td>QIB Shares Offered</td><td>Not more than 50% of the Net Issue</td></tr>
            <tr><td>Retail Shares Offered</td><td>Not less than 35% of the Net Issue</td></tr>
            <tr><td>NII Shares Offered</td><td>Not less than 15% of the Net Issue</td></tr>
        </table>'''
        response = type('Response', (), {'text': html, 'raise_for_status': lambda self: None})()
        with patch.object(live.httpx, 'get', return_value=response):
            parsed = live.parse_chittorgarh_html('https://www.chittorgarh.com/ipo/example-ipo/123/')
            observed = live.fetch_chittorgarh_subscriptions('https://www.chittorgarh.com/ipo/example-ipo/123/')
        self.assertIsNone(parsed['qib_subscription'])
        self.assertFalse(parsed['subscription_observed'])
        self.assertIsNone(observed)

    def test_real_subscription_table_is_verified(self):
        html = '''<table><tr><th>Investor Category</th><th>Subscription (times)</th></tr>
            <tr><td>Qualified Institutional</td><td>1.25x</td></tr>
            <tr><td>Non Institutional</td><td>3.40x</td></tr>
            <tr><td>Retail Individual</td><td>2.10x</td></tr></table>'''
        response = type('Response', (), {'text': html, 'raise_for_status': lambda self: None})()
        with patch.object(live.httpx, 'get', return_value=response):
            observed = live.fetch_chittorgarh_subscriptions('https://www.chittorgarh.com/ipo/example-ipo/123/')
        self.assertEqual(observed['qib_subscription'], 1.25)
        self.assertEqual(observed['hni_subscription'], 3.4)
        self.assertTrue(observed['subscription_observed'])
        self.assertEqual(__import__('datetime').datetime.fromisoformat(observed['subscription_observed_at']).utcoffset().total_seconds(), 19800)

    def test_old_misparsed_cache_subscription_is_discarded(self):
        rows = live.reconcile_cache_stages({'url': {'company_name': 'Example', 'category': 'OPEN',
            'qib_subscription': 0.0, 'retail_subscription': 5.0, 'hni_subscription': 5.0,
            'subscription_observed': True}})
        self.assertIsNone(rows['url']['qib_subscription'])
        self.assertFalse(rows['url']['subscription_observed'])

    def test_closed_ipo_does_not_promote_interim_bids_to_final_model(self):
        record = {
            'company_name': 'Closing Book Ltd', 'category': 'CLOSED', 'bidding_ended': True,
            'subscription_source': live.SUBSCRIPTION_SOURCE, 'subscription_is_final': False,
            'subscription_observed': True, 'qib_subscription': 2.0,
            'retail_subscription': 3.0, 'hni_subscription': 4.0,
            'issue_price': 100, 'issue_size_cr': 100,
        }
        with patch.object(api.resolver, 'resolve', return_value={
            'success': True, 'source': 'Test fetched page', 'data': record,
        }), patch.object(api, 'fetch_live_macro_regime', return_value={}):
            result = api.scrape_and_evaluate_ipo(ScrapeAndEvaluateRequest(
                url_or_name='Closing Book Ltd', fetch_news=False,
            ))
        self.assertEqual(result['evaluation']['verdict'], 'WAIT FOR DATA')
        self.assertIsNone(result['scraped_data']['qib_subscription'])

    def test_background_refresh_updates_open_ipo_only_in_source_window(self):
        rows = {'https://www.chittorgarh.com/ipo/example-ipo/123/': {
            'company_name': 'Example', 'category': 'OPEN', 'source_observed': True,
            'subscription_observed': False,
        }}
        observed = {'qib_subscription': 1.2, 'retail_subscription': 0.8,
                    'hni_subscription': 0.6, 'subscription_observed': True,
                    'subscription_source': live.SUBSCRIPTION_SOURCE}
        with patch.object(live, '_subscription_window_open', return_value=False), \
             patch.object(live, 'load_scraped_cache', return_value=rows), \
             patch.object(live, 'fetch_chittorgarh_subscriptions') as fetch:
            self.assertEqual(live.refresh_open_ipo_subscriptions(), 0)
            fetch.assert_not_called()
        with patch.object(live, '_subscription_window_open', return_value=True), \
             patch.object(live, 'load_scraped_cache', return_value=rows), \
             patch.object(live, 'fetch_chittorgarh_subscriptions', return_value=observed), \
             patch.object(live, 'save_scraped_cache') as save:
            self.assertEqual(live.refresh_open_ipo_subscriptions(), 1)
            self.assertEqual(save.call_args.args[0][next(iter(rows))]['qib_subscription'], 1.2)

    def test_legacy_registry_cache_is_not_served_as_observed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / 'cache.json'
            cache_path.write_text(json.dumps({
                'legacy': {'company_name': 'Invented Ltd', 'issue_price': 100},
                'observed': {'company_name': 'Fetched Ltd', 'issue_price': 200,
                             'source_observed': True, 'category': 'OPEN'},
            }), encoding='utf-8')
            with patch.object(live, 'CACHE_FILE_PATH', str(cache_path)):
                result = live.load_scraped_cache()
                self.assertNotIn('legacy', result)
                self.assertIn('observed', result)
                live.save_scraped_cache(result)
                persisted = json.loads(cache_path.read_text(encoding='utf-8'))
                self.assertIn('legacy', persisted)  # retained for audit

    def test_scraped_missing_final_book_never_gets_probability(self):
        record = {
            'company_name': 'Pending Book Ltd', 'category': 'CLOSED',
            'bidding_ended': True, 'subscription_observed': False,
            'issue_price': None, 'issue_size_cr': None,
            'qib_subscription': 9.0, 'retail_subscription': 2.0,
            'hni_subscription': None, 'promoter_ofs_ratio': None,
            'risk_factor_count': None, 'objects_capex_ratio': None,
        }
        with patch.object(api.resolver, 'resolve', return_value={
            'success': True, 'source': 'Test fetched page', 'data': record,
        }), patch.object(api, 'fetch_live_macro_regime', return_value={}):
            result = api.scrape_and_evaluate_ipo(ScrapeAndEvaluateRequest(
                url_or_name='Pending Book Ltd', fetch_news=False,
            ))
        self.assertIsNone(result['evaluation']['hybrid_ml_scores']['hybrid_positive_listing_prob'])
        self.assertEqual(result['evaluation']['verdict'], 'WAIT FOR DATA')
        self.assertIsNone(result['scraped_data']['issue_price'])
        self.assertIsNone(result['scraped_data']['hni_subscription'])

    def test_listed_quote_fetched_after_open_is_not_a_prelisting_signal(self):
        record = {
            'company_name': 'SRIT India', 'category': 'LISTED', 'is_listed': True,
            'listing_date': '2026-10-06', 'listing_open_price': 148.0,
            'issue_price': 130.0, 'issue_size_cr': 218.0,
            'qib_subscription': 4.0, 'retail_subscription': .7058,
            'hni_subscription': .9925, 'subscription_observed': True,
            'promoter_ofs_ratio': 0.0, 'gmp_pct': 45.38,
            'gmp_observed_at': '2026-10-07 19:17:41',
        }
        with patch.object(api.resolver, 'resolve', return_value={
            'success': True, 'source': 'Verified fetched page', 'data': record,
        }), patch.object(api, 'fetch_live_macro_regime', return_value={'india_vix': 13.93}):
            result = api.scrape_and_evaluate_ipo(ScrapeAndEvaluateRequest(
                url_or_name='SRIT India', fetch_news=False,
            ))
        self.assertIsNone(result['scraped_data']['gmp_pct'])
        self.assertEqual(result['scraped_data']['gmp_source'], 'post_listing_quote_excluded')
        self.assertEqual(result['scraped_data']['listing_open_price'], 148.0)
        diagnostics = result['evaluation']['conviction_diagnostics']
        self.assertEqual([row['status'] for row in diagnostics],
                         ['WARN', 'FAIL', 'FAIL', 'CONTEXT', 'CONTEXT', 'CONTEXT'])
        score = result['evaluation']['hybrid_ml_scores']
        self.assertEqual(result['evaluation']['verdict'], 'ELEVATED RISK')
        self.assertLess(score['sampling_sensitivity_low_pct'], 50)
        self.assertGreater(score['sampling_sensitivity_high_pct'], 50)


if __name__ == '__main__':
    unittest.main()
