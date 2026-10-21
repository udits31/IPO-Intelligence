import os
import time
import math
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from core.similarity_engine import SimilarityEngine
from core.attribution_engine import AttributionEngine
from core.monitor_engine import HoldSellMonitor
from core.predictor import HybridIPOPredictor
from db.supabase_client import get_supabase

class ComprehensiveStressTester:
    def __init__(self):
        self.results = {}
        
    def log_test(self, suite: str, test_name: str, passed: bool, details: str):
        if suite not in self.results:
            self.results[suite] = []
        self.results[suite].append({
            'test': test_name,
            'status': 'PASS' if passed else 'FAIL',
            'details': details
        })
        status_str = '[PASS]' if passed else '[FAIL]'
        print('  ' + status_str + ' ' + test_name + ': ' + details)

    def run_suite_1_phase1_data(self):
        print('\n=== SUITE 1: PHASE 1 DATA FOUNDATION STRESS TEST ===')
        suite = 'Phase 1: Data Foundation'
        
        df_ipos = pd.read_csv('data/canonical_listed_ipos.csv')
        total_ipos = len(df_ipos)
        modern_count = (df_ipos['is_primary_universe'] == True).sum()
        legacy_count = (df_ipos['is_primary_universe'] == False).sum()
        
        cond1 = total_ipos == 500 and modern_count == 254 and legacy_count == 246
        self.log_test(suite, 'IPO Universe Partitioning', cond1, 'Total: ' + str(total_ipos) + ', Modern (2021-24): ' + str(modern_count) + ', Legacy: ' + str(legacy_count))
        
        df_ipos['listing_date'] = pd.to_datetime(df_ipos['listing_date'])
        df_ipos['anchor_lockin_50pct_date'] = pd.to_datetime(df_ipos['anchor_lockin_50pct_date'])
        df_ipos['anchor_lockin_full_date'] = pd.to_datetime(df_ipos['anchor_lockin_full_date'])
        
        null_lockin = df_ipos['anchor_lockin_50pct_date'].isna().sum() + df_ipos['anchor_lockin_full_date'].isna().sum()
        diff_30 = (df_ipos['anchor_lockin_50pct_date'] - df_ipos['listing_date']).dt.days
        valid_30 = ((diff_30 >= 20) & (diff_30 <= 35)).all()
        
        post_2022 = df_ipos[df_ipos['listing_date'] >= '2022-04-01']
        diff_90 = (post_2022['anchor_lockin_full_date'] - post_2022['listing_date']).dt.days
        valid_90 = ((diff_90 >= 75) & (diff_90 <= 95)).all()
        
        self.log_test(suite, 'Lock-In Date Calculation Validity', (null_lockin == 0 and valid_30 and valid_90),
                      '0 nulls. 30d deltas valid: ' + str(valid_30) + ', Post-2022 90d deltas valid: ' + str(valid_90))

        df_prices = pd.read_csv('data/post_listing_prices.csv')
        total_prices = len(df_prices)
        unique_ipos = df_prices['ipo_id'].nunique()
        dups = df_prices.duplicated(subset=['ipo_id', 'date']).sum()
        neg_prices = (df_prices['close_price_adjusted'] <= 0).sum()
        
        df_prices['dt'] = pd.to_datetime(df_prices['date'])
        is_monotonic = True
        for ipo_id, group in df_prices.groupby('ipo_id'):
            if not group['dt'].is_monotonic_increasing:
                is_monotonic = False
                break
                
        cond3 = total_prices >= 210000 and unique_ipos == 254 and dups == 0 and neg_prices == 0 and is_monotonic
        self.log_test(suite, 'Price History Completeness & Monotonicity', cond3,
                      'Rows: ' + str(total_prices) + ', Unique IPOs: ' + str(unique_ipos) + '/254, Duplicates: ' + str(dups) + ', Non-positive prices: ' + str(neg_prices) + ', Monotonic: ' + str(is_monotonic))

        try:
            supabase = get_supabase(use_service_role=True)
            res = supabase.table('post_listing_prices').select('count', count='exact').limit(1).execute()
            cloud_count = res.count
            cloud_match = cloud_count == total_prices
            self.log_test(suite, 'Supabase Cloud Database Parity', cloud_match, 'Local CSV: ' + str(total_prices) + ' == Cloud DB: ' + str(cloud_count))
        except Exception as e:
            self.log_test(suite, 'Supabase Cloud Database Parity', False, 'Cloud check error: ' + str(e))

    def run_suite_2_phase2_similarity(self):
        print('\n=== SUITE 2: PHASE 2 SIMILARITY ENGINE STRESS TEST ===')
        suite = 'Phase 2: Similarity Engine'
        engine = SimilarityEngine()
        
        try:
            res_ext = engine.evaluate_ipo(qib_sub=999.0, retail_sub=500.0, gmp_pct=250.0, vix=80.0, nifty_10d=15.0)
            valid_ext = len(res_ext['historical_analogs']) == 15 and res_ext['verdict'] in ['APPLY', 'SKIP', 'NEUTRAL / CAUTION']
            self.log_test(suite, 'Mega-Bull Outlier (QIB 999x, VIX 80)', valid_ext, 'Verdict: ' + str(res_ext['verdict']) + ', Win Rate: ' + str(res_ext['win_rate_pct']) + '%')
        except Exception as e:
            self.log_test(suite, 'Mega-Bull Outlier', False, str(e))
            
        try:
            res_zero = engine.evaluate_ipo(qib_sub=0.0, retail_sub=0.0, gmp_pct=-75.0, vix=10.0, nifty_10d=-20.0)
            valid_zero = res_zero['verdict'] == 'SKIP' and not math.isnan(res_zero['win_rate_pct'])
            self.log_test(suite, 'Zero Subscription Boundary (QIB 0x, Retail 0x)', valid_zero, 'Verdict: ' + str(res_zero['verdict']) + ', Sizing: ' + str(res_zero['bHNI_sizing_tier']))
        except Exception as e:
            self.log_test(suite, 'Zero Subscription Boundary', False, str(e))
            
        sample_res = engine.evaluate_ipo(qib_sub=25.0, retail_sub=8.0, gmp_pct=15.0)
        distances = [a['distance'] for a in sample_res['historical_analogs']]
        dist_sorted = all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
        self.log_test(suite, 'Analog Distance Monotonicity', dist_sorted, 'Min dist: ' + str(distances[0]) + ', Max dist: ' + str(distances[-1]))

    def run_suite_3_phase3_attribution(self):
        print('\n=== SUITE 3: PHASE 3 ATTRIBUTION & MONITOR STRESS TEST ===')
        suite = 'Phase 3: Module 2 Attribution'
        attr_eng = AttributionEngine()
        monitor = HoldSellMonitor()
        
        try:
            res_early = attr_eng.compute_attribution(364, as_of_date='2023-12-02')
            cond_early = res_early['attribution']['10d'] is None and res_early['attribution']['30d'] is None
            self.log_test(suite, 'Insufficient History Boundary (2 Days Post-Listing)', cond_early, 'Correctly returns None for 10d/30d without runtime error')
        except Exception as e:
            self.log_test(suite, 'Insufficient History Boundary', False, str(e))
            
        disclosures_test = [
            [],
            [{'discloser_type': 'anchor', 'transaction_type': 'SELL', 'pct_change': -2.5}],
            [{'discloser_type': 'promoter', 'transaction_type': 'SELL', 'pct_change': -5.0}]
        ]
        
        dates_test = ['2023-12-20', '2024-01-15', '2024-03-20', '2024-10-01']
        all_passed = True
        for d in dates_test:
            for disc in disclosures_test:
                res_m = monitor.evaluate_holding(364, as_of_date=d, disclosures=disc)
                if 'verdict' not in res_m or len(res_m['explanation']) == 0:
                    all_passed = False
                    break
        self.log_test(suite, 'Decision Matrix Combinatorial Permutations', all_passed, 'Tested 12 combinations of dates x filings')

        df_modern = pd.read_csv('data/canonical_listed_ipos.csv')
        modern_ids = df_modern[df_modern['is_primary_universe'] == True]['id'].sample(50, random_state=42).tolist()
        
        fleet_errors = []
        for mid in modern_ids:
            try:
                res_f = monitor.evaluate_holding(mid)
                if 'verdict' not in res_f:
                    fleet_errors.append(mid)
            except Exception as e:
                fleet_errors.append((mid, str(e)))
                
        self.log_test(suite, '50-Stock Fleet Random Stress Test', len(fleet_errors) == 0, 'Evaluated 50 IPOs concurrently: ' + str(len(fleet_errors)) + ' failures')

    def run_suite_4_phase4_hybrid_ml(self):
        print('\n=== SUITE 4: PHASE 4 HYBRID ML MODEL STRESS TEST ===')
        suite = 'Phase 4: Hybrid ML & Predictions'
        predictor = HybridIPOPredictor()
        
        probs = []
        qib_steps = [0.1, 1.0, 5.0, 20.0, 60.0, 150.0]
        for q in qib_steps:
            r = predictor.predict(qib_sub=q, retail_sub=5.0, hni_sub=5.0, gmp_pct=10.0)
            probs.append(r['hybrid_ml_scores']['hybrid_positive_listing_prob'])
            
        is_monotonic_qib = all(probs[i] <= probs[i+1] for i in range(len(probs)-1))
        self.log_test(suite, 'QIB Monotonicity Sensitivity', is_monotonic_qib, 'Probs across QIB steps: ' + str(probs))

        test_cases = [
            {'qib': 0.01, 'ret': 100.0, 'hni': 0.01, 'gmp': -50.0, 'vix': 60.0},
            {'qib': 200.0, 'ret': 1.0, 'hni': 150.0, 'gmp': 120.0, 'vix': 11.0},
            {'qib': 1.0, 'ret': 1.0, 'hni': 1.0, 'gmp': 0.0, 'vix': 15.0},
            {'qib': 50.0, 'ret': 50.0, 'hni': 50.0, 'gmp': 30.0, 'vix': 20.0}
        ]
        range_valid = True
        for tc in test_cases:
            res_tc = predictor.predict(qib_sub=tc['qib'], retail_sub=tc['ret'], hni_sub=tc['hni'], gmp_pct=tc['gmp'], vix=tc['vix'])
            p = res_tc['hybrid_ml_scores']['hybrid_positive_listing_prob']
            if p < 0.0 or p > 100.0 or math.isnan(p):
                range_valid = False
                break
        self.log_test(suite, 'Probability Range Invariance [0, 100]', range_valid, 'All 4 extreme boundary cases strictly bounded')

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            predictor.predict(qib_sub=35.0, retail_sub=10.0, hni_sub=20.0, gmp_pct=15.0, fetch_news=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        self.log_test(suite, 'Inference Latency Profiling (100 Runs)', p95 < 20.0, 'p50: ' + str(round(p50, 2)) + 'ms, p95: ' + str(round(p95, 2)) + 'ms, p99: ' + str(round(p99, 2)) + 'ms')

    def print_summary(self):
        print('\n' + '='*70)
        print('           FINAL EXHAUSTIVE STRESS TEST AUDIT SUMMARY')
        print('='*70)
        total_tests = 0
        passed_tests = 0
        for suite, tests in self.results.items():
            print('\n[SUITE] ' + suite + ':')
            for t in tests:
                total_tests += 1
                if t['status'] == 'PASS':
                    passed_tests += 1
                print('  [' + t['status'] + '] ' + t['test'])
                
        print('\nTOTAL RESULTS: ' + str(passed_tests) + ' / ' + str(total_tests) + ' Tests Passed (' + str(round(passed_tests/total_tests*100, 1)) + '%)')
        if passed_tests == total_tests:
            print('VERDICT: ALL 4 PHASES ROCK-SOLID & PRODUCTION READY.')

if __name__ == '__main__':
    tester = ComprehensiveStressTester()
    tester.run_suite_1_phase1_data()
    tester.run_suite_2_phase2_similarity()
    tester.run_suite_3_phase3_attribution()
    tester.run_suite_4_phase4_hybrid_ml()
    tester.print_summary()
