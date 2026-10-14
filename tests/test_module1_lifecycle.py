"""
Test Suite: Module 1 Lifecycle State Machine & Empirical Quantile Invariants
===========================================================================
Verifies that:
1. Active / Day-1 IPOs with provisional subscriptions NEVER receive premature ML SKIP/APPLY.
2. QIB >= 2.0x during active bidding does NOT bypass the lifecycle gate (strict lifecycle rule).
3. Upcoming IPOs prior to book close emit BIDDING IN PROGRESS.
4. Finalized deals below 0.50 show elevated downside risk without a return forecast.
5. Borderline deals (0.50 <= P < 0.82) receive a caution.
6. APPLY requires the development-selected 0.82 threshold.
7. Historical return statistics are generated from out-of-year predictions.
8. Exchange-listed deals evaluate under finalized regime.
9. Empirical quantile JSON is monotonic and covers [0.0, 1.01) continuously.
"""

import os
import json
import unittest
from core.predictor import HybridIPOPredictor

class TestModule1Lifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = HybridIPOPredictor()

    def test_invariant_1_open_day1_runwal_weak_sentiment(self):
        """Runwal (+1.6% GMP, 0% OFS) must evaluate to MIXED / WEAK SENTIMENT SETUP, never HIGH-RISK."""
        res = self.predictor.predict(
            company_name='Runwal Enterprises',
            qib_sub=0.7,
            retail_sub=2.5,
            hni_sub=1.0,
            gmp_pct=1.64,
            promoter_ofs_ratio=0.0,
            category='OPEN',
            is_listed=False,
            bidding_ended=False
        )
        self.assertEqual(res['verdict'], 'MIXED / WEAK SENTIMENT SETUP')
        self.assertTrue(res['is_preliminary'])
        self.assertEqual(res['conviction'], 'LOW')
        self.assertIsNotNone(res['pre_bidding_scorecard'])
        self.assertEqual(res['pre_bidding_scorecard']['setup_type'], 'MIXED / WEAK SENTIMENT SETUP')
        
        scores = res['hybrid_ml_scores']
        self.assertIsNone(scores['hybrid_positive_listing_prob'])
        self.assertIsNone(scores['expected_return_median_pct'])
        self.assertEqual(scores['display_probability'], 'Scorecard Based (No Fake Prob)')
        self.assertIn('INTERIM', res['conviction_diagnostics'][0]['explanation'])

    def test_invariant_1b_open_missing_gmp_sentiment_unverified(self):
        """When GMP is missing/unobserved, must output SENTIMENT UNVERIFIED SETUP, explicitly distinguishing from observed weakness."""
        res = self.predictor.predict(
            company_name='Early Stage Offering Ltd',
            qib_sub=0.5,
            retail_sub=1.2,
            hni_sub=0.4,
            gmp_pct=None,
            promoter_ofs_ratio=0.15,
            category='OPEN',
            is_listed=False,
            bidding_ended=False
        )
        self.assertEqual(res['verdict'], 'SENTIMENT UNVERIFIED SETUP')
        self.assertTrue(res['is_preliminary'])
        self.assertEqual(res['conviction'], 'LOW')
        self.assertIsNotNone(res['pre_bidding_scorecard'])
        self.assertEqual(res['pre_bidding_scorecard']['setup_type'], 'SENTIMENT UNVERIFIED SETUP')
        self.assertIn('Secondary grey market observations are pending', res['reasoned_explanation'])

    def test_invariant_2_open_orient_cables_favorable_setup(self):
        """Orient Cables (+36.8% GMP, 42% OFS) evaluates to FAVORABLE SETUP with ML probabilities gated."""
        res = self.predictor.predict(
            company_name='Orient Cables',
            qib_sub=0.7,
            retail_sub=2.5,
            hni_sub=1.5,
            gmp_pct=36.76,
            promoter_ofs_ratio=0.42,
            category='OPEN',
            is_listed=False,
            bidding_ended=False
        )
        self.assertEqual(res['verdict'], 'FAVORABLE SETUP')
        self.assertTrue(res['is_preliminary'])
        self.assertEqual(res['conviction'], 'HIGH')
        self.assertIsNotNone(res['pre_bidding_scorecard'])
        self.assertEqual(res['pre_bidding_scorecard']['setup_type'], 'FAVORABLE SETUP')
        self.assertIsNone(res['hybrid_ml_scores']['hybrid_positive_listing_prob'])
        self.assertIn('INTERIM', res['conviction_diagnostics'][0]['explanation'])

    def test_invariant_3a_cash_out_dumping_is_high_risk(self):
        """Heavy promoter cash-out (>60% OFS with <5% GMP) maps strictly to HIGH-RISK SETUP."""
        res = self.predictor.predict(
            company_name='Cash Out Promoters Ltd',
            qib_sub=0.5,
            retail_sub=1.0,
            hni_sub=0.5,
            gmp_pct=2.0,
            promoter_ofs_ratio=0.85,
            category='OPEN',
            is_listed=False,
            bidding_ended=False
        )
        self.assertEqual(res['verdict'], 'HIGH-RISK SETUP')
        self.assertTrue(res['is_preliminary'])
        self.assertEqual(res['bHNI_sizing_tier'], 'Zero Allocation (Avoid)')

    def test_invariant_3b_negative_gmp_is_high_risk(self):
        """Negative GMP (<0%) maps strictly to HIGH-RISK SETUP."""
        res = self.predictor.predict(
            company_name='Discount Risk Ltd',
            qib_sub=0.5,
            retail_sub=1.0,
            hni_sub=0.5,
            gmp_pct=-4.0,
            promoter_ofs_ratio=0.10,
            category='OPEN',
            is_listed=False,
            bidding_ended=False
        )
        self.assertEqual(res['verdict'], 'HIGH-RISK SETUP')
        self.assertTrue(res['is_preliminary'])

    def test_strong_unofficial_gmp_with_high_ofs_names_the_actual_risk(self):
        res = self.predictor.predict(
            company_name='HD Fire Protect', qib_sub=0.0044, retail_sub=2.31,
            hni_sub=5.36, gmp_pct=31.4, promoter_ofs_ratio=1.0,
            category='OPEN', is_listed=False, bidding_ended=False,
        )
        self.assertEqual(res['verdict'], 'MIXED / CAPITAL STRUCTURE RISK')
        self.assertIn('100.0% of the offer', res['reasoned_explanation'])
        self.assertNotIn('muted', res['reasoned_explanation'].lower())
        self.assertIsNone(res['hybrid_ml_scores']['hybrid_positive_listing_prob'])

    def test_invariant_3c_interim_qib_factual_neutrality(self):
        """Low interim QIB on Day 1 is factual context and does NOT penalize or downgrade clean offerings."""
        res = self.predictor.predict(
            company_name='Clean Setup Day 1',
            qib_sub=0.1,
            retail_sub=1.2,
            hni_sub=0.4,
            gmp_pct=6.0,
            promoter_ofs_ratio=0.20,
            category='OPEN',
            is_listed=False,
            bidding_ended=False
        )
        # With 6% GMP, low OFS, and 0.1x interim QIB, must NOT be HIGH-RISK; maps honestly to MIXED
        self.assertEqual(res['verdict'], 'MIXED / WEAK SENTIMENT SETUP')
        self.assertNotEqual(res['verdict'], 'HIGH-RISK SETUP')
        self.assertIn('INTERIM', res['conviction_diagnostics'][0]['explanation'])

    def test_invariant_4_finalized_deal_with_low_probability_outputs_skip_with_weak_edge(self):
        """A low score flags downside while retaining the measured uncertainty."""
        res = self.predictor.predict(
            company_name='Weak Demand Ltd',
            qib_sub=0.3,
            retail_sub=4.5,
            hni_sub=0.4,
            gmp_pct=-3.0,
            category='CLOSED',
            is_listed=False,
            bidding_ended=True
        )
        self.assertEqual(res['verdict'], 'ELEVATED RISK')
        self.assertFalse(res['is_preliminary'])
        self.assertEqual(res['bHNI_sizing_tier'], 'Do not auto-allocate; review weak final demand')
        
        scores = res['hybrid_ml_scores']
        self.assertLess(scores['hybrid_positive_listing_prob'], 50.0)
        self.assertEqual(scores['conviction_label'], 'Weak bid signal; outcome uncertain')
        self.assertEqual(scores['tone'], 'negative')
        self.assertEqual(scores['expected_return_median_pct'], 0.1)
        self.assertEqual(scores['expected_range_50_low'], -1.7)
        self.assertEqual(scores['expected_range_50_high'], 4.9)
        self.assertEqual(scores['downside_p10'], -6.9)
        self.assertIsNone(scores['predicted_listing_return_pct'])

    def test_invariant_5_production_threshold_is_selected_from_earlier_years(self):
        """Final-book APPLY uses the saved 0.82 operating threshold."""
        res = self.predictor.predict(
            company_name='Solid Demand Co',
            qib_sub=45.0,
            retail_sub=12.0,
            hni_sub=35.0,
            gmp_pct=18.0,
            category='CLOSED',
            bidding_ended=True
        )
        self.assertEqual(res['verdict'], 'APPLY')
        self.assertFalse(res['is_preliminary'])
        scores = res['hybrid_ml_scores']
        self.assertGreaterEqual(scores['hybrid_positive_listing_prob'], 82.0)
        self.assertEqual(scores['tone'], 'positive')
        self.assertEqual(scores['model_version'], 'module1-corporate-subscription-lr-2021-2024-v2')

    def test_invariant_6_strong_score_does_not_claim_aggressive_allocation(self):
        """A strong model score retains moderate conviction and group statistics."""
        res = self.predictor.predict(
            company_name='Blockbuster IPO Ltd',
            qib_sub=65.0,
            retail_sub=15.0,
            hni_sub=40.0,
            gmp_pct=35.0,
            category='CLOSED',
            bidding_ended=True
        )
        self.assertEqual(res['verdict'], 'APPLY')
        self.assertEqual(res['conviction'], 'MODERATE')
        self.assertEqual(res['bHNI_sizing_tier'], 'Review allocation against personal risk limits')
        
        scores = res['hybrid_ml_scores']
        self.assertGreaterEqual(scores['hybrid_positive_listing_prob'], 82.0)
        self.assertEqual(scores['conviction_label'], 'Stronger historical listing signal')
        self.assertEqual(scores['historical_sample_size'], 102)
        self.assertEqual(scores['expected_return_median_pct'], 34.9)
        self.assertEqual(scores['expected_range_50_low'], 13.4)
        self.assertEqual(scores['expected_range_50_high'], 55.4)
        self.assertEqual(scores['downside_p10'], 0.8)

    def test_invariant_7_listed_deals_evaluated_under_finalized_regime(self):
        """A listed deal with is_listed=True or category='LISTED' bypasses provisional gating."""
        res = self.predictor.predict(
            company_name='Tata Technologies',
            qib_sub=45.0,
            retail_sub=12.0,
            hni_sub=35.0,
            gmp_pct=22.0,
            category='LISTED',
            is_listed=True,
            bidding_ended=True
        )
        self.assertEqual(res['verdict'], 'APPLY')
        self.assertFalse(res['is_preliminary'])
        self.assertIsNotNone(res['hybrid_ml_scores']['hybrid_positive_listing_prob'])

    def test_invariant_8_empirical_tiers_json_integrity(self):
        """The active bundle binds training years, threshold and group outcomes."""
        with open('ml/model_bundle.json', 'r', encoding='utf-8') as source:
            data = json.load(source)

        self.assertEqual(data['training_years'], [2021, 2022, 2023, 2024])
        self.assertEqual(data['training_samples'], 252)
        self.assertEqual(data['decision_threshold'], 0.82)
        tiers = data['empirical_tiers']
        self.assertIn('skip', tiers)
        self.assertIn('caution', tiers)
        self.assertIn('apply', tiers)
        self.assertEqual(sum(tier['sample_size'] for tier in tiers.values()), 188)
        self.assertEqual(tiers['skip']['median_return_pct'], 0.1)
        self.assertEqual(tiers['apply']['median_return_pct'], 34.9)

    def test_missing_final_subscription_fields_never_create_a_model_score(self):
        res = self.predictor.predict(
            company_name='Unverified final book', category='CLOSED', bidding_ended=True,
            qib_sub=1.0, retail_sub=1.0, hni_sub=1.0, subscription_verified=False,
        )
        self.assertEqual(res['verdict'], 'WAIT FOR DATA')
        self.assertIsNone(res['hybrid_ml_scores']['hybrid_positive_listing_prob'])

    def test_invariant_9_snapshot_service_deduplication(self):
        """Verify SQLite snapshot logger enforces ipo_id uniqueness, WAL mode, closing checkpoints, and active-only filtering."""
        import tempfile
        import sqlite3
        from datetime import datetime, date
        from zoneinfo import ZoneInfo
        from services.snapshot_service import SnapshotService, determine_snapshot_stage, extract_ipo_id
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_db = tmp.name
        try:
            svc = SnapshotService(db_path=tmp_db)
            
            # 1. Verify WAL mode and busy timeout
            with sqlite3.connect(tmp_db) as conn:
                journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
                self.assertEqual(journal.lower(), 'wal')
                busy = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
                self.assertEqual(busy, 5000)

            # 2. Test ipo_id uniqueness across name formatting differences
            clean_id = extract_ipo_id({'url': 'https://www.chittorgarh.com/ipo/sample-ipo/9988/'})
            self.assertEqual(clean_id, 'chitto_9988')
            
            first = svc.record_snapshot(company_name='Sample IPO Ltd.', ipo_id=clean_id, hour_bucket='2026-09-25_11', gmp_pct=15.0)
            self.assertTrue(first)
            
            # Second insert with slightly different name string but identical (ipo_id, hour_bucket) -> deduplicated
            second = svc.record_snapshot(company_name='Sample IPO Limited', ipo_id=clean_id, hour_bucket='2026-09-25_11', gmp_pct=16.0)
            self.assertFalse(second)
            self.assertEqual(svc.get_snapshot_count(), 1)

            # 3. Test closing day checkpoints (T-120, T-60, T-30, FINAL_CLOSE)
            ist = ZoneInfo('Asia/Kolkata')
            close_day = date(2026, 9, 25)
            
            # T-120 checkpoint at 15:00 IST
            st120, b120 = determine_snapshot_stage(close_day, datetime(2026, 9, 25, 15, 0, tzinfo=ist))
            self.assertEqual(st120, 'T_MINUS_120')
            self.assertEqual(b120, '2026-09-25_T120')
            
            # T-60 checkpoint at 16:00 IST
            st60, b60 = determine_snapshot_stage(close_day, datetime(2026, 9, 25, 16, 0, tzinfo=ist))
            self.assertEqual(st60, 'T_MINUS_60')
            self.assertEqual(b60, '2026-09-25_T60')

            # T-30 checkpoint at 16:30 IST
            st30, b30 = determine_snapshot_stage(close_day, datetime(2026, 9, 25, 16, 30, tzinfo=ist))
            self.assertEqual(st30, 'T_MINUS_30')
            self.assertEqual(b30, '2026-09-25_T30')

            # Final close checkpoint at 17:00 IST
            stc, bc = determine_snapshot_stage(close_day, datetime(2026, 9, 25, 17, 0, tzinfo=ist))
            self.assertEqual(stc, 'FINAL_CLOSE')
            self.assertEqual(bc, '2026-09-25_CLOSE')

            # Standard non-closing day or outside window -> HOURLY
            sth, bh = determine_snapshot_stage(date(2026, 9, 29), datetime(2026, 9, 25, 11, 20, tzinfo=ist))
            self.assertEqual(sth, 'HOURLY')
            self.assertEqual(bh, '2026-09-25_11')

            # 4. Test active-only filtering in scheduled harvester
            test_items = [
                {'company_name': 'Active Co 1', 'category': 'OPEN', 'bidding_ended': False, 'url': 'https://chittorgarh.com/ipo/a1/1001/'},
                {'company_name': 'Closed Co 2', 'category': 'CLOSED', 'bidding_ended': True, 'url': 'https://chittorgarh.com/ipo/a2/1002/'},
                {'company_name': 'Listed Co 3', 'category': 'LISTED', 'is_listed': True, 'url': 'https://chittorgarh.com/ipo/a3/1003/'},
            ]
            inserted_count = svc.record_scheduled_snapshots(test_items, macro={'india_vix': 14.5})
            # Only the 1 active offering is logged
            self.assertEqual(inserted_count, 1)

        finally:
            try:
                if os.path.exists(tmp_db):
                    os.remove(tmp_db)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
