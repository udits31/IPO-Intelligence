import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from core.attribution_engine import AttributionEngine
from services.news_service import NewsService

class HoldSellMonitor:
    def __init__(self, prices_path='data/post_listing_prices.csv',
                 ipos_path='data/canonical_listed_ipos.csv', *,
                 prices_frame: Optional[pd.DataFrame] = None,
                 ipos_frame: Optional[pd.DataFrame] = None,
                 prefer_supplied_prices: bool = False):
        self.attribution_engine = AttributionEngine(
            prices_path=prices_path, ipos_path=ipos_path,
            prices_frame=prices_frame, ipos_frame=ipos_frame,
            prefer_supplied_prices=prefer_supplied_prices)
        self.df_ipos = ipos_frame.copy() if ipos_frame is not None else pd.read_csv(ipos_path)
        self.news_service = NewsService()
        
    def evaluate_holding(self, ipo_id: int, as_of_date: Optional[str] = None,
                          disclosures: Optional[List[Dict[str, Any]]] = None,
                          fetch_news: bool = True,
                          disclosure_coverage_verified: bool = False,
                          historical_news: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if historical_news is not None and as_of_date is None:
            raise ValueError('Archived news requires an explicit historical decision date')
        attr_res = self.attribution_engine.compute_attribution(ipo_id, as_of_date=as_of_date)
        if attr_res.get('status')=='PRICE_BASIS_UNVERIFIED':
            return attr_res
        if 'error' in attr_res:
            return attr_res
        if as_of_date is None and attr_res.get('price_is_stale'):
            return {'status': 'DATA_STALE', 'company_name': attr_res['company_name'],
                    'evaluation_date': attr_res['evaluation_date'],
                    'current_price_adjusted': attr_res['current_price_adjusted'],
                    'price_source': attr_res.get('price_source'),
                    'message': ('The price refresh did not supply a recent dated close. '
                                f"Last observation: {attr_res['evaluation_date']}. "
                                'The old price is retained for reference; a current hold/sell call is paused.')}
            
        company_name = attr_res['company_name']
        sector = attr_res['sector']
        eval_date = attr_res['evaluation_date']
        curr_price = attr_res['current_price_adjusted']
        lockin_info = attr_res['lockin_status']
        lockin_dates_verified = lockin_info.get('dates_verified') is True
        attr_30d = attr_res['attribution'].get('30d')
        attr_10d = attr_res['attribution'].get('10d')
        attr_since_listing = attr_res['attribution'].get('since_listing')
        decision_attr = attr_30d or attr_since_listing
        comparison_label = '30 trading days' if attr_30d else 'since listing'
        
        anchor_sell_detected = False
        promoter_sell_detected = False
        anchor_exit_pct = 0.0
        
        if disclosures:
            for d in disclosures:
                dtype = str(d.get('discloser_type', '')).lower()
                ttype = str(d.get('transaction_type', '')).upper()
                p_change = float(d.get('pct_change', 0.0))
                if 'anchor' in dtype and ('SELL' in ttype or p_change < 0):
                    anchor_sell_detected = True
                    anchor_exit_pct += abs(p_change)
                if 'promoter' in dtype and ('SELL' in ttype or p_change < 0):
                    promoter_sell_detected = True
                    
        reasons = []
        stock_ret_30d = float(decision_attr['stock_return_pct']) if decision_attr else 0.0
        sec_ret_30d = float(decision_attr['sector_return_pct']) if decision_attr and decision_attr.get('sector_return_pct') is not None else None
        spread_30d = float(decision_attr['spread_vs_sector_pct']) if decision_attr and decision_attr.get('spread_vs_sector_pct') is not None else None
        category_30d = decision_attr.get('category', 'NEUTRAL_MOVE') if decision_attr else 'NEUTRAL_MOVE'
        
        if not lockin_dates_verified:
            reasons.append('Anchor unlock dates are unverified. The stored calendar alone cannot establish that anchor shares remain locked or are eligible for sale.')
        elif lockin_info['is_30d_unlocked']:
            if anchor_sell_detected:
                reasons.append('Institutional exit detected: Anchor investors trimmed ~' + str(round(anchor_exit_pct, 1)) + '% stake following lock-in expiry (' + str(lockin_info['lockin_30d_date']) + ').')
            elif disclosure_coverage_verified:
                reasons.append('No anchor sale appears in the verified disclosure coverage supplied for this evaluation.')
            else:
                reasons.append('Anchor lock-in has expired; anchor sale status is unverified because complete dated disclosure coverage was not supplied.')
        else:
            days_left = lockin_info['days_to_30d_lockin']
            if days_left <= 7:
                reasons.append('Upcoming 30-day anchor lock-in expiry in ' + str(days_left) + ' days (' + str(lockin_info['lockin_30d_date']) + '); anticipate potential short-term volume volatility.')
            else:
                reasons.append('Anchor shares remain locked under 30-day SEBI restriction until ' + str(lockin_info['lockin_30d_date']) + ' (' + str(days_left) + ' days remaining).')
                
        if category_30d == 'SECTOR_PULLBACK':
            reasons.append(f"Stock is down {stock_ret_30d:.1f}% {comparison_label}, but the {sector} sector index is down {sec_ret_30d:.1f}% over the same period — this is a sector-wide macro move, not specific to {company_name}.")
        elif category_30d == 'STOCK_SPECIFIC_LAG':
            reasons.append(f"Stock is lagging its industry benchmark by -{abs(spread_30d):.1f}% {comparison_label} ({company_name} {stock_ret_30d:+.1f}% vs {sector} {sec_ret_30d:+.1f}%) — indicates company-specific selling pressure.")
        elif category_30d == 'OUTPERFORMING_SECTOR':
            reasons.append(f"Stock is outperforming the {sector} sector index by +{spread_30d:.1f}% {comparison_label} ({company_name} {stock_ret_30d:+.1f}% vs {sector} {sec_ret_30d:+.1f}%), showing robust relative strength.")
        elif category_30d == 'BROAD_MARKET_CORRECTION':
            reasons.append(f"Stock pullback ({stock_ret_30d:.1f}%) is driven by broad Nifty 50 index correction rather than idiosyncratic company weakness.")
        elif sec_ret_30d is None:
            reasons.append(f"{company_name} moved {stock_ret_30d:+.1f}% {comparison_label}; a reliable sector comparison is unavailable.")
        else:
            reasons.append(f"Stock return ({stock_ret_30d:+.1f}%) is tracking within normal historical spread boundaries vs the {sector} benchmark ({sec_ret_30d:+.1f}%).")
            
        waterfall = attr_res.get('price_waterfall', {})
        issue_p = float(waterfall.get('issue_price', curr_price))
        peak_p = float(waterfall.get('peak_high_price', curr_price))
        cum_ret = float(waterfall.get('cumulative_gain_pct', ((curr_price - issue_p) / issue_p) * 100.0))
        dd_from_peak = float(waterfall.get('drawdown_from_peak_pct', ((curr_price - peak_p) / peak_p) * 100.0 if peak_p > 0 else 0.0))
        decision_code = None
        decision_summary = None
        
        # Historical price-only accuracy remains under evaluation; these rules
        # are risk heuristics, not a validated forecast of future returns.
        # 1. Structural Breakdown / Severely Underwater / Anchor Exit
        if anchor_sell_detected or (cum_ret <= -12.0) or (cum_ret <= -5.0 and category_30d == 'STOCK_SPECIFIC_LAG' and spread_30d <= -6.0) or (curr_price < issue_p * 0.88):
            decision_code = 'reported_anchor_sale' if anchor_sell_detected else 'loss_or_relative_lag_rule'
            verdict = 'SELL / REDUCE'
            confidence = 'High' if (anchor_sell_detected or cum_ret <= -20.0) else 'Medium'
            risk_level = 'High'
            action = 'EXIT / CUT LOSS'
            entry_zone = 'N/A (Preserve Capital)'
            stop_loss = f'Immediate exit at current price (₹{curr_price:.2f})'
            decision_summary = (f'Supplied disclosures report anchor selling of approximately {anchor_exit_pct:.1f}%.'
                if anchor_sell_detected else f'The close is {cum_ret:+.1f}% versus the issue price. It meets the fixed loss/relative-lag risk rule; this is not a forecast of the next return.')
            reasons.append(decision_summary)

        elif lockin_dates_verified and lockin_info['is_30d_unlocked'] and disclosure_coverage_verified and not anchor_sell_detected and spread_30d is not None and spread_30d >= 5.0 and curr_price >= issue_p:
            # Track 1: ACCUMULATE (Compounding Runners)
            # A trailing level is a risk rule, not a guaranteed profit floor.
            trail_stop_level = peak_p * 0.75
            has_ratchet = peak_p >= issue_p * 1.25
            if has_ratchet:
                # Capital Ratchet Guarantee: Lock in break-even capital preservation (+5% over issue price) once gain exceeded +25%
                trail_stop_level = max(trail_stop_level, issue_p * 1.05)

            if curr_price <= trail_stop_level:
                decision_code = 'accumulation_trailing_rule'
                verdict = 'WATCH / TAKE PROFIT'
                confidence = 'High'
                risk_level = 'Moderate-High'
                action = 'TAKE 50-100% PROFIT'
                entry_zone = 'Wait for post-pullback stabilization'
                stop_loss = f'₹{trail_stop_level:.2f} (Breached)'
                reasons.append(f'Trailing profit protection: Price retraced {abs(dd_from_peak):.1f}% from peak (₹{peak_p:.2f}), breaching the 25% trailing stop zone. Exit position to protect accumulated gains.')
            else:
                decision_code = 'verified_relative_strength_setup'
                verdict = 'ACCUMULATE / STRONG BUY'
                confidence = 'High'
                risk_level = 'Low-Moderate'
                action = 'ACCUMULATE / BUY ON DIPS'
                entry_zone = f'₹{curr_price * 0.96:.2f} - ₹{curr_price * 1.01:.2f}'
                stop_loss = f'₹{trail_stop_level:.2f} (25% trail from peak ₹{peak_p:.2f}{" with capital ratchet" if has_ratchet else ""})'
                reasons.append(f'Positive sector-relative price performance ({spread_30d:+.1f}%). No anchor sale appears in the verified disclosure coverage supplied; that does not prove future selling is absent.')

        elif curr_price >= issue_p:
            # Track 2: HOLD / CONSOLIDATION (Post-Listing Drift Risk)
            # This trailing-stop level is a heuristic; future execution may gap.
            trail_stop_level = peak_p * 0.92
            if curr_price <= trail_stop_level or cum_ret < -12.0:
                decision_code = 'unvalidated_trailing_alert'
                verdict = 'WATCH / PRICE PULLBACK'
                confidence = 'Unvalidated'
                risk_level = 'Review required'
                action = 'REVIEW POSITION — NO AUTOMATIC EXIT'
                entry_zone = 'No new entry recommendation'
                stop_loss = f'Legacy 8% reference: ₹{trail_stop_level:.2f}; not a validated exit level'
                decision_summary = (f'The latest close of ₹{curr_price:.2f} is {abs(dd_from_peak):.1f}% below '
                    f'the highest observed closing price of ₹{peak_p:.2f}, but remains {cum_ret:.1f}% above '
                    f'the ₹{issue_p:.2f} issue price. The old 8% cutoff was manually chosen and has not '
                    'been validated for this stock. The pullback is flagged for review, not an automatic sell.')
                reasons.append(decision_summary)
            else:
                decision_code = 'no_price_exit_rule_triggered'
                verdict = 'HOLD / CONSOLIDATION'
                confidence = 'Medium'
                risk_level = 'Moderate'
                action = 'HOLD EXISTING POSITION'
                entry_zone = f'₹{curr_price * 0.94:.2f} - ₹{curr_price:.2f}'
                stop_loss = f'₹{trail_stop_level:.2f} (8% trailing stop from peak ₹{peak_p:.2f})'
                spread_note = f'Recent spread vs {sector} ({spread_30d:+.1f}%). ' if spread_30d is not None else 'Sector-relative spread unavailable. '
                reasons.append(f'Profitable consolidation: Stock trades at ₹{curr_price:.2f} (+{cum_ret:.1f}% above ₹{issue_p:.2f} issue price). {spread_note}The 8% trailing level is a risk rule, not a return forecast.')
        else:
            decision_code = 'below_issue_watch'
            verdict = 'HOLD (WATCH)'
            confidence = 'Medium'
            risk_level = 'Moderate'
            action = 'HOLD / MONITOR'
            entry_zone = 'Hold existing allocation'
            stop_loss = f'Issue Price (₹{issue_p:.2f})'
            reasons.append(f'The close is {cum_ret:+.1f}% versus the issue price. No configured exit rule fired; watch does not mean a recovery is predicted.')
        if decision_summary is None:
            decision_summary = reasons[-1]
        price_only_verdict = verdict
            
        # Ingest corporate & event news for the holding
        news_articles = []
        news_sentiment = {'score': None, 'label': 'UNAVAILABLE', 'article_count': 0,
                          'status': 'historical_news_unavailable' if as_of_date else 'not_fetched'}
        if historical_news is not None or (fetch_news and company_name and as_of_date is None):
            try:
                news_res = (self.news_service.summarize_historical_snapshot(company_name,as_of_date,historical_news)
                            if historical_news is not None else
                            self.news_service.get_company_news(company_name, max_results=6))
                news_articles = news_res.get('articles', [])
                raw_score = news_res.get('sentiment_score')
                score = float(raw_score) if raw_score is not None else None
                if score is None:
                    label = 'UNAVAILABLE'
                elif score >= 0.25:
                    label = 'BULLISH CATALYST'
                elif score <= -0.25:
                    label = 'BEARISH EVENT RISK'
                else:
                    label = 'NEUTRAL / BALANCED'
                news_sentiment = {
                    'score': score,
                    'label': label,
                    'article_count': len(news_articles),
                    'status': news_res.get('status', 'unknown'),
                    'freshness': news_res.get('freshness'),
                    'checked_at': news_res.get('checked_at'),
                    'last_success_at': news_res.get('last_success_at'),
                    'error': news_res.get('error'),
                    'source_url':news_res.get('source_url'),
                    'raw_sha256':news_res.get('raw_sha256'),
                    'rejected_by_asof_guard':news_res.get('rejected_by_asof_guard'),
                    **{key:news_res.get(key) for key in ('scorer_version','scorer_sha256','identity_policy_version',
                                                       'identity_code_sha256','issuer_master_sha256')},
                }
            except ValueError:
                if historical_news is not None:
                    raise  # Invalid historical provenance must not become an apparently valid no-news replay.
                news_sentiment = {'score': None,'label':'UNAVAILABLE','article_count':0,'status':'fetch_failed'}
            except Exception as e:
                if historical_news is not None:
                    raise
                news_sentiment = {'score': None, 'label': 'UNAVAILABLE', 'article_count': 0,
                                  'status': 'fetch_failed'}

        # Event-driven News Sentiment Governance
        if news_sentiment['score'] is not None and news_sentiment['score'] <= -0.30:
            if 'BUY' in verdict or 'ACCUMULATE' in verdict:
                verdict = 'WATCH / CAUTION (NEWS OVERHANG)'
                confidence = 'Medium'
                risk_level = 'Elevated'
                action = 'HOLD / AWAIT EVENT CLARITY'
                entry_zone = 'Wait for news event overhang to settle'
                decision_code = 'news_blocks_accumulation'
                decision_summary = ('The price rules indicated an accumulation setup, but negative dated news '
                    'blocked adding to the position. This is an event-risk caution, not a measured probability of a fall.')
            reasons.append(f"Negative corporate/event news sentiment ({news_sentiment['score']:+.2f}) detected: Active headline overhang creates short-term downside volatility.")
        elif news_sentiment['score'] is not None and news_sentiment['score'] >= +0.30:
            reasons.append(f"Positive corporate news catalysts ({news_sentiment['score']:+.2f}) reinforce underlying business expansion.")

        explanation_text = verdict + ' (Confidence: ' + confidence + ') - ' + ' '.join(['(' + str(i+1) + ') ' + r for i, r in enumerate(reasons)])
        
        playbook = {
            'action': action,
            'entry_zone': entry_zone,
            'stop_loss_basis': stop_loss
        }
        
        # 4-Pillar Human-Readable Reasoning Thesis
        is_pos_alpha = spread_30d is not None and spread_30d >= 0
        alpha_text = (f"+{spread_30d:.1f}% higher" if is_pos_alpha else f"{abs(spread_30d):.1f}% lower") if spread_30d is not None else None
        
        p1_badge_color = 'EMERALD' if ('BUY' in verdict or 'ACCUMULATE' in verdict or 'HOLD' in verdict) else ('ROSE' if 'SELL' in verdict else 'AMBER')
        
        p2_badge = ('Outperforming Sector' if is_pos_alpha else 'Lagging Sector Peers') if alpha_text else 'Sector Comparison Unavailable'
        p2_badge_color = ('EMERALD' if is_pos_alpha else 'ROSE') if alpha_text else 'AMBER'
        
        if anchor_sell_detected:
            p3_badge = 'Anchor Sale Detected'
            p3_badge_color = 'ROSE'
            p3_exp = f'Supplied disclosures report anchor selling of approximately {anchor_exit_pct:.1f}%. The date calendar is a separate evidence question.'
        elif not lockin_dates_verified:
            p3_badge = 'Anchor Filings Unverified' if lockin_info['is_30d_unlocked'] else 'Anchor Dates Unverified'
            p3_badge_color = 'AMBER'
            p3_exp = 'The stored unlock dates have not been matched to an official allotment or unlock record. They do not prove that anchor shares remain locked, or that an anchor investor has sold.'
        elif lockin_info['is_30d_unlocked']:
            p3_badge = 'Anchor Sale Detected' if anchor_sell_detected else ('No Sale in Checked Filings' if disclosure_coverage_verified else 'Anchor Filings Unverified')
            p3_badge_color = 'ROSE' if anchor_sell_detected else ('EMERALD' if disclosure_coverage_verified else 'AMBER')
            p3_exp = (f"Anchor selling appears in supplied disclosures: ~{anchor_exit_pct:.1f}% stake reduction."
                      if anchor_sell_detected else
                      ('No anchor sale appears in the verified dated disclosure coverage supplied.'
                       if disclosure_coverage_verified else
                       'The lock-in has expired, but complete dated anchor-sale disclosure coverage was not supplied.'))
        else:
            d_left = lockin_info['days_to_30d_lockin']
            p3_badge = f'{d_left} Days to Unlock'
            p3_badge_color = 'EMERALD' if d_left > 7 else 'AMBER'
            p3_exp = f"Anchor investor shares remain legally locked until {lockin_info['lockin_30d_date']} ({d_left} days remaining). Zero institutional selling allowed before this date."

        thesis_breakdown = [
            {
                'title': '1. Plain-English Recommendation',
                'subtitle': 'What to do right now',
                'verdict_badge': verdict,
                'badge_color': p1_badge_color,
                'explanation': decision_summary
            },
            {
                'title': '2. Price Movement & Attribution',
                'subtitle': 'Why is the price moving?',
                'verdict_badge': p2_badge,
                'badge_color': p2_badge_color,
                'explanation': (f"{comparison_label.capitalize()}, {company_name} moved {stock_ret_30d:+.1f}% while the {sector} industry index moved {sec_ret_30d:+.1f}%. This indicates {'strong company-specific resilience and buyer demand.' if is_pos_alpha else 'stock-specific selling pressure relative to its industry peers.'}" if sec_ret_30d is not None else f"{company_name} moved {stock_ret_30d:+.1f}% {comparison_label}. No verified sector return is available for the same dates.")
            },
            {
                'title': '3. Smart Money & Lock-in Safety',
                'subtitle': 'Institutional safety check',
                'verdict_badge': p3_badge,
                'badge_color': p3_badge_color,
                'explanation': p3_exp
            },
            {
                'title': '4. Execution Playbook & Safety Net',
                'subtitle': 'Exact price zones to trade',
                'verdict_badge': action,
                'badge_color': 'BLUE',
                'explanation': (f'{action}. {stop_loss}.' if decision_code == 'unvalidated_trailing_alert' else
                                f'Target entry zone: {entry_zone}. Risk reference: {stop_loss}. These fixed rule levels are not validated return forecasts.')
            }
        ]

        return {
            'ipo_id': ipo_id,
            'company_name': company_name,
            'nse_symbol': attr_res.get('nse_symbol'),
            'sector': sector,
            'listing_date': attr_res.get('listing_date'),
            'evaluation_date': eval_date,
            'price_source': attr_res.get('price_source', 'Local market history'),
            'price_basis': attr_res.get('price_basis'),
            'price_basis_adjustment':attr_res.get('price_basis_adjustment'),
            'history_price_basis':attr_res.get('history_price_basis'),
            'price_archive_sha256': attr_res.get('price_archive_sha256'),
            'price_is_live': bool(attr_res.get('price_is_live', False)),
            'price_age_calendar_days': attr_res.get('price_age_calendar_days'),
            'benchmark_source': attr_res.get('benchmark_source', f'{sector} sector history'),
            'current_price_adjusted': curr_price,
            'price_waterfall': attr_res.get('price_waterfall'),
            'recent_trajectory': attr_res.get('recent_trajectory'),
            'actionable_playbook': playbook,
            'news_sentiment': news_sentiment,
            'recent_news': news_articles,
            'verdict': verdict,
            'decision_policy_version': 'monitor-risk-alert-governance-20261023-v1',
            'decision_reason': {
                'code': decision_code,
                'summary': decision_summary,
                'method': 'Fixed risk rules and supplied evidence; not an ML-learned exit threshold',
                'automatic_exit_supported': False if decision_code == 'unvalidated_trailing_alert' else None,
                'reference_price_kind': 'highest_observed_daily_close',
                'trailing_reference_price': round(peak_p * .92, 2) if decision_code == 'unvalidated_trailing_alert' else None,
                'history_note': 'Less than 30 trading sessions; assessment uses available since-listing history.' if attr_30d is None else None,
            },
            'price_only_verdict': price_only_verdict,
            'news_changed_decision': verdict != price_only_verdict,
            'confidence': confidence,
            'risk_level': risk_level,
            'explanation': explanation_text,
            'key_reasons': reasons,
            'thesis_breakdown': thesis_breakdown,
            'lockin_status': lockin_info,
            'attribution_30d': attr_30d,
            'attribution_10d': attr_10d,
            'attribution_90d': attr_res['attribution'].get('90d'),
            'attribution_since_listing': attr_since_listing,
            'trendlyne_data': attr_res.get('trendlyne_data'),
        }

if __name__ == '__main__':
    monitor = HoldSellMonitor()
    print('=== TEST 1: Tata Technologies (Normal Post-Lockin Hold) ===')
    res1 = monitor.evaluate_holding(364, as_of_date='2024-03-15')
    print('Verdict: ' + str(res1['verdict']) + ' (' + str(res1['confidence']) + ')')
    print('Explanation: ' + str(res1['explanation']))
    
    print('')
    print('=== TEST 2: Simulated Anchor Exit on Post-Lockin Stock ===')
    mock_disclosure = [{'discloser_type': 'non_promoter_anchor', 'transaction_type': 'SELL', 'pct_change': -3.2}]
    res2 = monitor.evaluate_holding(364, as_of_date='2024-03-15', disclosures=mock_disclosure)
    print('Verdict: ' + str(res2['verdict']) + ' (' + str(res2['confidence']) + ')')
    print('Explanation: ' + str(res2['explanation']))
