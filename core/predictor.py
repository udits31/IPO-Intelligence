import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from core.similarity_engine import SimilarityEngine
from services.news_service import fetch_company_news
from ml.model_pipeline import BUNDLE_PATH, FEATURE_COLS, MODEL_VERSION

class HybridIPOPredictor:
    def __init__(self, bundle_path=BUNDLE_PATH):
        self.similarity_engine = SimilarityEngine()
        with open(bundle_path, 'r', encoding='utf-8') as source:
            bundle = json.load(source)
        if (bundle.get('model_version') != MODEL_VERSION or bundle.get('feature_cols') != FEATURE_COLS
                or bundle.get('training_years') != [2021, 2022, 2023, 2024]
                or bundle.get('training_samples') != 252):
            raise ValueError('Module 1 model bundle does not match the approved training cohort.')
        self.model_bundle = bundle
        self.model_metadata = {
            'model_name': bundle['model_name'], 'model_version': bundle['model_version'],
            'training_years': bundle['training_years'], 'training_samples': bundle['training_samples'],
            'decision_threshold': bundle['decision_threshold'],
        }
        self.empirical_tiers = {'tiers': bundle['empirical_tiers']}
        self.training_input_reference = bundle['training_input_reference']
        self.operating_threshold = float(bundle['decision_threshold'])
        self.avoid_threshold = float(bundle['avoid_threshold'])
        self._mean = np.asarray(bundle['scaler_mean'], dtype=float)
        self._scale = np.asarray(bundle['scaler_scale'], dtype=float)
        self._coef = np.asarray(bundle['coefficients'], dtype=float)
        self._intercept = float(bundle['intercept'])
        refits = bundle['bootstrap_refits']
        self._bootstrap_mean = np.asarray([row['mean'] for row in refits], dtype=float)
        self._bootstrap_scale = np.asarray([row['scale'] for row in refits], dtype=float)
        self._bootstrap_coef = np.asarray([row['coef'] for row in refits], dtype=float)
        self._bootstrap_intercept = np.asarray([row['intercept'] for row in refits], dtype=float)
        if (len(self._mean) != len(FEATURE_COLS) or len(self._scale) != len(FEATURE_COLS)
                or len(self._coef) != len(FEATURE_COLS) or not np.isfinite(self._coef).all()
                or not np.isfinite(self._scale).all() or (self._scale <= 0).any()):
            raise ValueError('Module 1 model coefficients are invalid.')
        
        # Canonical dataset used for historical sector track records
        canonical_path = 'data/canonical_listed_ipos.csv'
        sub_path = 'data/subscription_observations.csv'
        if os.path.exists(canonical_path):
            df_can = pd.read_csv(canonical_path)
            if os.path.exists(sub_path):
                df_sub = pd.read_csv(sub_path)[['ipo_id', 'qib_subscription']]
                df_can = df_can.merge(df_sub, left_on='id', right_on='ipo_id', how='left')
            self.df_canonical = df_can
        else:
            self.df_canonical = pd.DataFrame()

    def score_interim_bid_snapshot(self, *, qib_sub: float, retail_sub: float,
                                   hni_sub: float) -> Dict[str, Any]:
        """Experimental point-in-time score; never return an allocation verdict.

        The deployed coefficients were fitted on closing books, so their
        historical accuracy does not apply to a live bid table.
        We save this score prospectively to measure that gap rather than
        silently promoting it as a validated model.
        """
        bids = (qib_sub, retail_sub, hni_sub)
        if any(value is None or not np.isfinite(value) or value < 0 for value in bids):
            raise ValueError('A complete non-negative QIB, retail and HNI snapshot is required')
        features = np.log1p(np.maximum(np.asarray(bids, dtype=float), 0.01))
        linear = float(np.dot((features - self._mean) / self._scale, self._coef) + self._intercept)
        probability = float(1.0 / (1.0 + np.exp(-np.clip(linear, -50, 50))))
        return {
            'positive_listing_probability': probability,
            'model_version': self.model_metadata['model_version'],
            'evaluation_mode': 'experimental_intraday_snapshot',
            'validated_for_intraday': False,
        }
        
    def predict(self, company_name: str = 'Target IPO', sector: str = 'Technology',
                qib_sub: float = 1.0, retail_sub: float = 1.0, hni_sub: float = 1.0,
                issue_size_cr: float = 500.0, gmp_pct: Optional[float] = None,
                vix: Optional[float] = None, nifty_10d: Optional[float] = None, sector_10d: float = 0.0,
                promoter_ofs_ratio: Optional[float] = None, risk_factor_count: float = 50.0,
                objects_capex_ratio: float = 0.3, price_band_spread_pct: float = 5.0,
                category: str = 'CLOSED', is_listed: bool = False, bidding_ended: bool = False,
                subscription_verified: Optional[bool] = None,
                fetch_news: bool = False) -> Dict[str, Any]:
        
        cat_upper = (category or '').strip().upper()
        is_deal_listed = is_listed or cat_upper == 'LISTED'
        is_deal_closed = bidding_ended or cat_upper == 'CLOSED' or is_deal_listed
        subscription_data_available = (
            bool(subscription_verified) if subscription_verified is not None
            else all(np.isfinite(v) and v >= 0 for v in (qib_sub, retail_sub, hni_sub))
        )

        # Lifecycle Horizon:
        # Horizon 1 (Active Bidding): Rule-based Pre-Bidding Scorecard on verified facts.
        # Horizon 2 (Closed Book): audited subscription model and historical cohorts.
        bidding_in_progress = (not is_deal_closed) and (cat_upper in ['UPCOMING', 'OPEN'])

        gmp_available = gmp_pct is not None and np.isfinite(float(gmp_pct))
        gmp_value = float(gmp_pct) if gmp_available else 0.0
        vix_observed = vix is not None and np.isfinite(float(vix))
        vix_context = float(vix) if vix_observed else 15.0
        nifty_observed = nifty_10d is not None and np.isfinite(float(nifty_10d))
        ofs_observed = (promoter_ofs_ratio is not None and
                        np.isfinite(float(promoter_ofs_ratio)) and 0 <= float(promoter_ofs_ratio) <= 1)
        ofs_context = float(promoter_ofs_ratio) if ofs_observed else 0.25
        sim_res = (
            self.similarity_engine.evaluate_ipo(
                qib_sub=qib_sub, retail_sub=retail_sub, gmp_pct=gmp_value,
                vix=float(vix), nifty_10d=float(nifty_10d), sector_10d=sector_10d,
                promoter_ofs=ofs_context
            ) if is_deal_closed and subscription_data_available and vix_observed and nifty_observed else {
                'win_rate_pct': None, 'median_return_pct': None,
                'worst_drawdown_pct': None, 'total_analogs_count': 0,
                'historical_analogs': [],
            }
        )
        
        qib_val = max(qib_sub, 0.01)
        retail_val = max(retail_sub, 0.01)
        hni_val = max(hni_sub, 0.01)
        
        log_qib = np.log1p(qib_val)
        log_retail = np.log1p(retail_val)
        log_hni = np.log1p(hni_val)
        qib_ret_ratio = (qib_val + 0.1) / (retail_val + 0.1)
        log_issue_size = np.log1p(max(issue_size_cr, 10.0))
        
        feature_dict = {
            'log_qib': log_qib,
            'log_retail': log_retail,
            'log_hni': log_hni,
            'qib_retail_ratio': qib_ret_ratio,
            'log_issue_size': log_issue_size,
            'india_vix': vix_context,
            'nifty_10d_return': nifty_10d,
            'sector_10d_return': sector_10d,
            'promoter_ofs_ratio': ofs_context,
            'risk_factor_count': min(max(risk_factor_count, 10.0), 150.0),
            'objects_capex_ratio': min(max(objects_capex_ratio, 0.0), 1.0),
            'price_band_spread_pct': min(max(price_band_spread_pct, 0.0), 20.0)
        }
        
        news_items = fetch_company_news(company_name, limit=3) if fetch_news else []
        
        # 1. Diagnostic Pillars
        ofs_pct = round(ofs_context * 100, 1) if ofs_observed else None
        ofs_status = ('PASS' if ofs_context <= 0.35 else ('WARN' if ofs_context <= 0.60 else 'FAIL')) if ofs_observed else 'CONTEXT'
        ofs_reason = 'Offer-for-sale share is unverified; capital-allocation context is unavailable.' if not ofs_observed else (
            f"{100-ofs_pct:.1f}% fresh issue component (OFS {ofs_pct}%). Refer to the offer's stated use of proceeds." if ofs_status == 'PASS' else (
                f"Balanced split: {ofs_pct}% OFS exit vs {100-ofs_pct:.1f}% fresh issue" if ofs_status == 'WARN' else
                f"Heavy promoter cash-out ({ofs_pct}% OFS); majority proceeds exit company"
            )
        )

        if not gmp_available:
            gmp_status = 'WARN'
            gmp_reason = 'No point-in-time verified GMP observation is available; secondary sentiment excluded.'
            gmp_metric = 'Unavailable'
        else:
            gmp_status = 'PASS' if gmp_value >= 15.0 else ('WARN' if gmp_value >= 3.0 else 'FAIL')
            gmp_reason = (
                f"Strong secondary demand with +{gmp_value:.1f}% grey market premium" if gmp_status == 'PASS' else (
                    f"Modest listing buffer of +{gmp_value:.1f}%" if gmp_status == 'WARN' else
                    f"Muted or negative grey market premium ({gmp_value:.1f}%) signals high listing risk"
                )
            )
            gmp_metric = f"{gmp_value:+.1f}% GMP"

        vix_status = ('PASS' if vix_context < 16.0 else ('WARN' if vix_context < 22.0 else 'FAIL')) if vix_observed else 'CONTEXT'
        vix_reason = ('India VIX is unavailable; no volatility signal is claimed.' if not vix_observed else (
            f"Benign volatility (India VIX {vix_context:.2f}) supports primary liquidity" if vix_status == 'PASS' else (
                f"Elevated volatility (India VIX {vix_context:.2f}) increases pricing sensitivity" if vix_status == 'WARN' else
                f"Extreme volatility (India VIX {vix_context:.2f}) triggers systematic risk-off flows"
            )
        ))

        if bidding_in_progress:
            # INTERIM QIB: Pure factual reporting. NEVER used as a negative signal during active bidding.
            if subscription_data_available:
                qib_status = 'PASS' if qib_sub >= 2.0 else 'WARN'
                qib_metric = f"QIB: {qib_sub:.1f}x · INTERIM"
                qib_reason = f"Observed INTERIM QIB demand is {qib_sub:.1f}x; the final institutional book is pending."
            else:
                qib_status = 'CONTEXT'
                qib_metric = 'Awaiting update'
                qib_reason = 'No verified interim subscription reading is available. Missing bids are not zero bids.'
        elif not subscription_data_available:
            qib_status = 'WARN'
            qib_metric = 'Pending'
            qib_reason = 'Subscription data has not been published yet; institutional demand cannot be confirmed.'
        else:
            qib_ref = self.training_input_reference['qib_subscription']
            qib_status = 'PASS' if qib_sub >= qib_ref['median'] else ('WARN' if qib_sub >= qib_ref['p25'] else 'FAIL')
            qib_metric = f"QIB {qib_sub:.1f}x · {qib_ret_ratio:.2f}x vs retail"
            qib_reason = (
                f"Absolute QIB demand is {qib_sub:.1f}x versus the 2021–24 training median of {qib_ref['median']:.1f}x. "
                f"The {qib_ret_ratio:.2f}x QIB/retail ratio alone is not a strong-listing signal; it can rise when retail demand is weak."
            )

        diagnostics = [
            {'pillar': '1. Final QIB demand' if is_deal_closed else '1. Interim QIB demand',
             'status': qib_status, 'role': 'MODEL INPUT' if is_deal_closed else 'INTERIM CONTEXT',
             'metric': qib_metric, 'explanation': qib_reason},
        ]
        for index, (label, value, source) in enumerate((
            ('Retail demand', retail_sub, 'retail_subscription'),
            ('HNI demand', hni_sub, 'hni_subscription'),
        ), start=2):
            ref = self.training_input_reference[source]
            status = ('PASS' if value >= ref['median'] else
                      'WARN' if value >= ref['p25'] else 'FAIL') if is_deal_closed and subscription_data_available else 'CONTEXT'
            diagnostics.append({
                'pillar': f'{index}. {label}', 'status': status,
                'role': 'MODEL INPUT' if is_deal_closed else 'INTERIM CONTEXT',
                'metric': f'{value:.1f}x' if subscription_data_available else 'Pending',
                'explanation': (
                    f'Final {label.lower()} is {value:.1f}x versus the 2021–24 training median of {ref["median"]:.1f}x.'
                    if is_deal_closed and subscription_data_available else
                    f'Final {label.lower()} is not verified yet; it cannot be used for a model decision.'
                ),
            })
        diagnostics.extend([
            {'pillar': '4. Capital allocation & OFS', 'status': 'CONTEXT' if is_deal_closed else ofs_status,
             'role': 'CONTEXT · NOT MODEL INPUT', 'metric': f"{ofs_pct}% OFS" if ofs_observed else 'Unavailable', 'explanation': ofs_reason},
            {'pillar': '5. Grey market premium', 'status': 'CONTEXT' if is_deal_closed else gmp_status,
             'role': 'CONTEXT · NOT MODEL INPUT', 'metric': gmp_metric, 'explanation': gmp_reason},
            {'pillar': '6. Macro volatility', 'status': 'CONTEXT' if is_deal_closed else vix_status,
             'role': 'CONTEXT · NOT MODEL INPUT', 'metric': f"VIX {vix_context:.2f}" if vix_observed else 'Unavailable', 'explanation': vix_reason},
        ])

        # 2. Decision Logic and Quantile Mapping
        tiers = self.empirical_tiers.get('tiers', {})
        scorecard = None

        if bidding_in_progress:
            is_preliminary = True

            # --- HORIZON 1: PRE-BIDDING SCORECARD HEURISTICS ---
            # Rule 1: High-Risk Setup strictly reserved for hard adverse evidence
            if (gmp_available and gmp_value < 0.0) or (ofs_observed and ofs_context > 0.60 and (not gmp_available or gmp_value < 5.0)):
                final_verdict = 'HIGH-RISK SETUP'
                conviction = 'HIGH'
                conviction_label = 'High Governance / Downside Risk'
                sizing = 'Zero Allocation (Avoid)'
                tone = 'negative'
                status_explanation = (
                    f"HIGH-RISK SETUP - Adverse risk factors detected. "
                    f"{'Negative grey market sentiment signals high discount risk.' if gmp_available and gmp_value < 0 else f'Heavy promoter cash-out ({ofs_pct}% OFS) with weak secondary market buffer.'} "
                    f"Capital preservation recommended."
                )
            # Rule 2: Favorable Setup requires solid secondary buffer and clean capital structure (OFS <= 50%)
            elif gmp_available and gmp_value >= 15.0 and ofs_observed and ofs_context <= 0.50:
                final_verdict = 'FAVORABLE SETUP'
                conviction = 'HIGH' if gmp_value >= 25.0 else 'MODERATE'
                conviction_label = 'Strong Pre-Bidding Setup'
                sizing = 'Standard Retail / bHNI Allocation'
                tone = 'positive'
                status_explanation = (
                    f"FAVORABLE SETUP - Supported by robust grey market demand (+{gmp_value:.1f}% GMP) "
                    f"and clean capital allocation ({100-ofs_pct:.1f}% fresh issue proceeds). "
                    f"Favorable risk-reward for retail bidding prior to broker mandate cut-offs."
                )
            # Rule 3: Missing / Unobserved GMP -> explicitly distinguish lack of observation from observed weakness
            elif not gmp_available:
                final_verdict = 'SENTIMENT UNVERIFIED SETUP'
                conviction = 'LOW'
                conviction_label = 'Secondary Sentiment Unobserved'
                sizing = 'Minimum Application Only / Selective'
                tone = 'warning'
                status_explanation = (
                    "SENTIMENT UNVERIFIED SETUP - Secondary grey market observations are pending or unrecorded. "
                    "Absence of secondary quote is not proof of weakness, but provides no margin of safety buffer. "
                    "Capital allocation requires disciplined sizing."
                )
            elif not ofs_observed:
                final_verdict = 'CAPITAL STRUCTURE UNVERIFIED'
                conviction = 'LOW'
                conviction_label = 'Offer Terms Unverified'
                sizing = 'Minimum Application Only / Selective'
                tone = 'warning'
                status_explanation = (
                    'CAPITAL STRUCTURE UNVERIFIED - An unofficial GMP quote exists, but the offer-for-sale '
                    'share is not verified. This is not sufficient for a favorable setup assessment.'
                )
            elif gmp_value >= 15.0 and ofs_observed and ofs_context > 0.50:
                final_verdict = 'MIXED / CAPITAL STRUCTURE RISK'
                conviction = 'LOW'
                conviction_label = 'Strong Unofficial Sentiment; High OFS'
                sizing = 'Minimum Application Only / Selective'
                tone = 'warning'
                status_explanation = (
                    f'MIXED / CAPITAL STRUCTURE RISK - Unofficial GMP is +{gmp_value:.1f}%, '
                    f'but {ofs_pct}% of the offer is a sale by existing holders. '
                    'The quote is not a verified listing forecast and the final demand book is pending.'
                )
            # Rule 4: Observed modest/low positive GMP maps to Mixed / Weak Sentiment
            else:
                final_verdict = 'MIXED / WEAK SENTIMENT SETUP'
                conviction = 'LOW'
                conviction_label = 'Muted Listing Buffer'
                sizing = 'Minimum Application Only / Selective'
                tone = 'warning'
                status_explanation = (
                    f"MIXED / WEAK SENTIMENT SETUP - Secondary grey market premium is muted (+{gmp_value:.1f}% GMP), "
                    f"providing a narrow margin of safety on listing. "
                    f"Capital allocation requires disciplined sizing."
                )

            scorecard = {
                'setup_type': final_verdict,
                'tone': tone,
                'gmp_pct': gmp_value if gmp_available else None,
                'promoter_ofs_ratio': promoter_ofs_ratio if ofs_observed else None,
                'fresh_capital_pct': round((1.0 - ofs_context) * 100, 1) if ofs_observed else None,
                'interim_qib_subscription': qib_sub if subscription_data_available else None,
                'interim_retail_subscription': retail_sub if subscription_data_available else None,
                'retail_sizing_recommendation': sizing,
                'broker_upi_cutoff_note': "Check the exchange issue schedule and your broker's application and UPI-mandate cut-offs. Your broker may close applications before the exchange's final bid-position update.",
                'sebi_anchor_disclosure': "Under SEBI ICDR guidelines, up to 60% of the QIB portion is allocated to anchor investors 24 hours prior to public open.",
                'listing_distribution_status': "No validated investor-time return distribution is available for this bidding stage."
            }

            hybrid_scores = {
                'hybrid_positive_listing_prob': None,
                'xgboost_prob': None,
                'hist_gradient_boost_prob': None,
                'logistic_baseline_prob': None,
                'predicted_listing_return_pct': None,
                'conviction_tier': conviction,
                'conviction_label': conviction_label,
                'tone': tone,
                'expected_return_median_pct': None,
                'expected_range_50_low': None,
                'expected_range_50_high': None,
                'expected_range_80_low': None,
                'expected_range_80_high': None,
                'downside_p10': None,
                'decision_threshold': self.operating_threshold,
                'is_preliminary': True,
                'display_probability': 'Scorecard Based (No Fake Prob)',
                'empirical_range_text': 'Early-bidding return model has not passed validation.'
            }
            explanation = status_explanation

        elif not subscription_data_available:
            is_preliminary = True
            final_verdict = 'WAIT FOR DATA'
            conviction = 'LOW'
            conviction_label = 'Data Pending'
            sizing = 'No Allocation Until Subscription Data Is Available'
            tone = 'warning'

            hybrid_scores = {
                'hybrid_positive_listing_prob': None,
                'xgboost_prob': None,
                'hist_gradient_boost_prob': None,
                'logistic_baseline_prob': None,
                'predicted_listing_return_pct': None,
                'conviction_tier': 'PENDING',
                'conviction_label': conviction_label,
                'tone': tone,
                'expected_return_median_pct': None,
                'expected_range_50_low': None,
                'expected_range_50_high': None,
                'expected_range_80_low': None,
                'expected_range_80_high': None,
                'downside_p10': None,
                'decision_threshold': self.operating_threshold,
                'is_preliminary': True,
                'display_probability': 'Awaiting Subscription Data',
                'empirical_range_text': 'Awaiting subscription figures'
            }
            explanation = (
                'WAIT FOR DATA (Confidence: Low) - Subscription figures are not available yet. '
                'GMP and macro conditions alone are not sufficient for a capital-allocation decision.'
            )

        else:
            # Final-book scores use only QIB, retail and HNI. The live JSON
            # bundle and the displayed benchmark share this model version.
            is_preliminary = False
            values = np.asarray([feature_dict[name] for name in FEATURE_COLS], dtype=float)
            linear_score = float(np.dot((values - self._mean) / self._scale, self._coef) + self._intercept)
            prob_positive = float(1.0 / (1.0 + np.exp(-np.clip(linear_score, -50, 50))))
            refit_scores = np.sum(
                ((values - self._bootstrap_mean) / self._bootstrap_scale) * self._bootstrap_coef,
                axis=1,
            ) + self._bootstrap_intercept
            refit_probabilities = 1.0 / (1.0 + np.exp(-np.clip(refit_scores, -50, 50)))
            sensitivity_low, sensitivity_high = [round(float(value) * 100, 1)
                                                  for value in np.percentile(refit_probabilities, [10, 90])]
            if prob_positive < self.avoid_threshold:
                final_verdict, tier_key = 'ELEVATED RISK', 'skip'
                sizing = 'Do not auto-allocate; review weak final demand'
                conviction, tone = 'LOW', 'negative'
                conviction_label = 'Weak bid signal; outcome uncertain'
            elif prob_positive < self.operating_threshold:
                final_verdict, tier_key = 'NEUTRAL / CAUTION', 'caution'
                sizing = 'Wait or limit exposure'
                conviction, tone = 'LOW', 'neutral'
                conviction_label = 'Uncertain listing edge'
            else:
                final_verdict, tier_key = 'APPLY', 'apply'
                sizing = 'Review allocation against personal risk limits'
                conviction, tone = ('MODERATE' if sensitivity_low >= self.operating_threshold * 100 else 'LOW'), 'positive'
                conviction_label = ('Stronger historical listing signal' if conviction == 'MODERATE'
                                    else 'Positive score; threshold-sensitive across refits')

            tier_data = tiers.get(tier_key, {})
            emp_median = tier_data.get('median_return_pct')
            emp_range_50 = tier_data.get('range_50_pct')
            emp_range_80 = tier_data.get('range_80_pct')
            emp_downside_p10 = tier_data.get('downside_p10_pct')
            sample_size = int(tier_data.get('sample_size', 0))
            win_rate = tier_data.get('win_rate_pct')
            historical_text = (
                f"{sample_size} earlier-year IPOs in this score band had a {win_rate:.1f}% positive opening rate "
                f"and a {emp_median:+.1f}% median opening return."
                if sample_size >= 20 and win_rate is not None and emp_median is not None
                else 'This score band has too few historical examples for a return estimate.'
            )
            hybrid_scores = {
                'model_name': self.model_metadata['model_name'],
                'model_version': self.model_metadata['model_version'],
                'model_inputs': ['Final QIB demand', 'Final retail demand', 'Final HNI demand'],
                'sampling_sensitivity_low_pct': sensitivity_low,
                'sampling_sensitivity_high_pct': sensitivity_high,
                'sampling_sensitivity_note': self.model_bundle['bootstrap_note'],
                'decision_rationale': (
                    f'Final QIB {qib_sub:.1f}x, retail {retail_sub:.1f}x and HNI {hni_sub:.1f}x are compared with '
                    f'2021–24 medians of {self.training_input_reference["qib_subscription"]["median"]:.1f}x, '
                    f'{self.training_input_reference["retail_subscription"]["median"]:.1f}x and '
                    f'{self.training_input_reference["hni_subscription"]["median"]:.1f}x. '
                    'A high QIB/retail ratio can result from weak retail demand. GMP, OFS and VIX are context only.'
                ),
                'hybrid_positive_listing_prob': round(prob_positive * 100, 1),
                'xgboost_prob': None,
                'hist_gradient_boost_prob': None,
                'logistic_baseline_prob': round(prob_positive * 100, 1),
                'predicted_listing_return_pct': None,
                'historical_median_listing_return_pct': emp_median if sample_size >= 20 else None,
                'historical_sample_size': sample_size,
                'historical_positive_rate_pct': win_rate if sample_size >= 20 else None,
                'conviction_tier': conviction,
                'conviction_label': conviction_label,
                'tone': tone,
                'expected_return_median_pct': emp_median if sample_size >= 20 else None,
                'expected_range_50_low': emp_range_50[0] if sample_size >= 20 and emp_range_50 else None,
                'expected_range_50_high': emp_range_50[1] if sample_size >= 20 and emp_range_50 else None,
                'expected_range_80_low': emp_range_80[0] if sample_size >= 20 and emp_range_80 else None,
                'expected_range_80_high': emp_range_80[1] if sample_size >= 20 and emp_range_80 else None,
                'downside_p10': emp_downside_p10 if sample_size >= 20 else None,
                'decision_threshold': self.operating_threshold,
                'avoid_threshold': self.avoid_threshold,
                'is_preliminary': False,
                'display_probability': f'{prob_positive * 100:.1f}%',
                'empirical_range_text': historical_text,
            }
            explanation = (
                f'{final_verdict} (Confidence: {conviction.capitalize()}) - '
                f'The final-subscription model estimates a {prob_positive * 100:.1f}% chance of opening above issue price. '
                f'{historical_text} This is a risk signal, not a guaranteed listing outcome.'
            )

        sector_record = self.compute_sector_track_record(sector)

        return {
            'verdict': final_verdict,
            'conviction': conviction,
            'is_preliminary': is_preliminary,
            'bHNI_sizing_tier': sizing,
            'pre_bidding_scorecard': scorecard,
            'conviction_diagnostics': diagnostics,
            'sector_track_record': sector_record,
            'hybrid_ml_scores': hybrid_scores,
            'similarity_baseline': {
                'analog_win_rate_pct': sim_res['win_rate_pct'],
                'analog_median_return_pct': sim_res['median_return_pct'],
                'analog_worst_drawdown_pct': sim_res['worst_drawdown_pct'],
                'total_analogs': sim_res['total_analogs_count'],
                'top_analogs': sim_res['historical_analogs'][:5]
            },
            'reasoned_explanation': explanation,
            'qualitative_news_context': news_items,
            'raw_features': feature_dict if is_deal_closed and subscription_data_available else None
        }

    def compute_sector_track_record(self, sector_name: str) -> Dict[str, Any]:
        df = self.df_canonical.copy()
        clean_sec = (sector_name or 'Technology').lower().strip()
        
        matched = df[df['sector'].str.lower().str.contains(clean_sec, na=False)]
        if len(matched) < 3:
            keywords = ['tech', 'industr', 'auto', 'health', 'consum', 'financ', 'materi', 'energ', 'infra']
            for kw in keywords:
                if kw in clean_sec:
                    matched = df[df['sector'].str.lower().str.contains(kw, na=False)]
                    break
        if len(matched) == 0:
            matched = df.head(15)
            
        avg_ret = round(float(matched['listing_return_pct'].mean()), 2)
        win_rate = round(float((matched['listing_return_pct'] > 0).mean() * 100), 1)
        cols_to_keep = [c for c in ['company_name', 'listing_date', 'issue_price', 'qib_subscription', 'listing_return_pct'] if c in matched.columns]
        top_peers = matched.sort_values('listing_return_pct', ascending=False).head(6)[cols_to_keep].to_dict(orient='records')
        
        return {
            'sector_name': sector_name,
            'total_ipos_in_sector': len(matched),
            'avg_listing_return_pct': avg_ret,
            'win_rate_pct': win_rate,
            'top_sector_peers': top_peers
        }
