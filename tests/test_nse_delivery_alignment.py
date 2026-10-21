"""Regression checks against the real NSE pilot, with no fabricated market rows."""

import unittest

import pandas as pd

from ml.fetch_module2_nse_delivery import ROOT, audit_alignment, cohort_requests
from ml.research_module2_nse_delivery import assemble_cases


class NSEDeliveryAlignmentTest(unittest.TestCase):
    def test_real_nse_rows_reject_symbol_mismatch(self):
        prices = pd.read_csv(ROOT / 'data/audit/module2_development_prices_2021_2024.csv',
                             parse_dates=['date'])
        canonical = pd.read_csv(ROOT / 'data/canonical_listed_ipos.csv')
        observed = pd.read_csv(ROOT / 'data/audit/module2_nse_delivery_development.csv')
        planned = cohort_requests(prices, canonical)[:8]
        audit = audit_alignment(observed, prices, planned)
        self.assertIn(247, audit['eligible_ipo_ids'])  # CMSINFO agrees with NSE close
        self.assertNotIn(249, audit['eligible_ipo_ids'])  # HPAL history is incomplete/mismatched
        self.assertEqual(len(audit['eligible_ipo_ids']) + len(audit['rejected']), 8)

    def test_delivery_percentage_matches_exchange_quantities(self):
        observed = pd.read_csv(ROOT / 'data/audit/module2_nse_delivery_development.csv')
        self.assertGreater(len(observed), 200)
        gap = (observed.deliverable_qty / observed.traded_qty * 100 - observed.deliverable_pct).abs()
        self.assertLess(gap.max(), 0.02)

    def test_research_features_stop_before_decision_session(self):
        cases, _coverage = assemble_cases()
        self.assertGreater(len(cases), 200)
        self.assertTrue((pd.to_datetime(cases.latest_delivery_session) <
                         pd.to_datetime(cases.decision_date)).all())
        delivery = cases[['delivery_5_mean_pct', 'delivery_20_mean_pct']]
        self.assertTrue((delivery.ge(0) & delivery.le(100)).all().all())


if __name__ == '__main__':
    unittest.main()
