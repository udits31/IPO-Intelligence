import os
from typing import Dict, Any, List, Optional


def _bid(value: float) -> str:
    return f'{value:.4f}x' if 0 < value < 0.01 else f'{value:.2f}x'

def generate_ai_investment_memo(company_name: str, sector: str, qib_sub: Optional[float], retail_sub: Optional[float],
                                hni_sub: Optional[float], issue_size_cr: float, issue_price: Optional[float], gmp_pct: Optional[float],
                                win_prob: Optional[float], predicted_return_pct: Optional[float], sizing_tier: str,
                                verdict: str, conviction: str, news_items: List[Dict[str, str]],
                                analogs: List[Dict[str, Any]]) -> Dict[str, Any]:
    subscription_available = all(v is not None for v in (qib_sub, retail_sub, hni_sub))
    qib_ret_ratio = round((qib_sub + 0.1) / (retail_sub + 0.1), 2) if subscription_available else None
    historical_median = predicted_return_pct

    if 'FAVORABLE' in verdict:
        gmp_context = (f' Unofficial GMP context: {gmp_pct:+.1f}%; quote time unverified.') if gmp_pct is not None else ''
        thesis_headline = f"Favorable pre-bidding setup supported by secondary demand and disciplined capital structure.{gmp_context}"
        recommendation_action = f"Review {sizing_tier} against your risk limits and verify the broker's current cut-off before applying."
    elif 'HIGH-RISK' in verdict:
        thesis_headline = "High risk profile characterized by adverse secondary sentiment or promoter capital extraction."
        recommendation_action = 'Avoid allocating until the risk factors and live source data have been checked.'
    elif 'SENTIMENT UNVERIFIED' in verdict:
        thesis_headline = "Secondary market sentiment is unverified; no grey market buffer has been observed."
        recommendation_action = "Disciplined sizing recommended (1 lot or observe). Re-evaluate if secondary quotes appear prior to broker cut-off."
    elif 'CAPITAL STRUCTURE RISK' in verdict:
        thesis_headline = (
            f'Unofficial GMP is +{gmp_pct:.1f}%, but the offer-for-sale share is high; '
            'final bids and the offer document must be reviewed.'
            if gmp_pct is not None else
            'A high offer-for-sale share warrants caution while final bids are pending.'
        )
        recommendation_action = 'Keep sizing conservative; an unofficial GMP quote is not a listing-price forecast.'
    elif 'MIXED' in verdict or 'WEAK' in verdict:
        thesis_headline = f"Muted pre-bidding market sentiment ({gmp_pct:+.1f}% GMP) provides narrow margin of safety." if gmp_pct is not None else "Observed secondary market sentiment is weak."
        recommendation_action = "Selective participation only. Apply minimum 1 lot or observe Day 2 subscription pace."
    elif verdict == 'WAIT FOR DATA' or not subscription_available:
        thesis_headline = 'Decision deferred because live subscription figures are not yet available.'
        recommendation_action = 'Do not allocate capital yet. Re-run analysis once bidding opens.'
    elif verdict == 'APPLY':
        gmp_context = (' and unofficial GMP context of ' + f'{gmp_pct:+.1f}%') if gmp_pct is not None else ''
        thesis_headline = 'Primary market opportunity supported by institutional sponsorship (QIB ' + str(qib_sub) + 'x)' + gmp_context + '.'
        recommendation_action = 'Review the offer document and your risk limits before applying. A model score is not an allotment or return guarantee.'
    elif verdict == 'ELEVATED RISK':
        thesis_headline = 'Final subscription demand is weak relative to earlier-year IPOs; a positive opening is still possible.'
        recommendation_action = 'Do not treat this warning as a certain discount. Review the offer document and your loss limit before deciding.'
    else:
        thesis_headline = 'Mixed risk-reward profile with moderate interest across tranches. Requires selective application sizing.'
        recommendation_action = 'Apply with minimum application limits only. Monitor institutional closing trends before full allocation.'

    if not subscription_available:
        partial_readings = [
            f'{label} {_bid(value)}' for label, value in
            (('QIB', qib_sub), ('Retail', retail_sub), ('HNI', hni_sub)) if value is not None
        ]
        institutional_context = (
            'Interim subscription table: ' + ', '.join(partial_readings) +
            '. Missing categories have not been published; they are not zero bids.'
            if partial_readings else
            'Institutional demand: No subscription reading is available yet.'
        )
        listing_catalyst = 'The final QIB, retail and HNI book is still required for a model decision.'
    elif 'FAVORABLE' in verdict:
        institutional_context = f"Interim Institutional Demand: QIB demand stands at {_bid(qib_sub)}. The institutional book is not final."
        listing_catalyst = f"Unofficial GMP context: {gmp_pct:+.1f}%; quote time unverified." if gmp_pct is not None else "Secondary Market Discovery: GMP observations pending."
    elif 'HIGH-RISK' in verdict or 'MIXED' in verdict:
        institutional_context = f"Interim Demand: Bidding in progress ({_bid(qib_sub)} QIB vs {_bid(retail_sub)} Retail). Final tallies are pending."
        listing_catalyst = f"Unofficial GMP context: {gmp_pct:+.1f}%; quote time unverified." if gmp_pct is not None else "GMP unobserved."
    elif not subscription_available:
        institutional_context = 'Institutional Demand: Subscription figures are pending, so QIB conviction cannot yet be assessed.'
        listing_catalyst = 'Listing expectation unlocks when subscription data arrives.'
    elif verdict == 'ELEVATED RISK':
        institutional_context = f"Institutional Conviction: QIB subscription stands at {_bid(qib_sub)} with an informed-money ratio of {qib_ret_ratio:.2f}x relative to retail demand."
        listing_catalyst = "Risk Profile: The model places this IPO in an elevated downside-risk band; the warning is uncertain and can include positive listings."
    else:
        institutional_context = f"Institutional Conviction: QIB subscription stands at {_bid(qib_sub)} with an informed-money ratio of {qib_ret_ratio:.2f}x relative to retail demand."
        listing_catalyst = (
            f"Historical Context: Earlier-year IPOs in this model score band had a {historical_median:+.1f}% median opening return. "
            "This is a group statistic, not a price target."
            if historical_median is not None else
            "Historical Context: Too few comparable out-of-time examples for a return estimate."
        )

    bull_catalysts = [
        institutional_context,
        listing_catalyst,
        'Market conditions are separate context; the listing-risk model uses only final QIB, retail, and HNI demand.'
    ]

    bear_risks = [
        'SEBI Anchor Lock-in Expiry: Under SEBI ICDR regulations, 50% of the anchor allocation unlocks 30 days post-listing and the remainder at 90 days, creating potential short-term supply pressure.',
        'Market Broadening Risk: Any sharp macro drawdown or sudden spike in India VIX before listing day could compress grey market premiums.',
        f"Valuation & Execution Scrutiny: Review the verified issue price and offer-document assumptions for the {sector} sector."
    ]

    pending_note = 'Wait for verified subscription figures before placing an application.'
    is_active = any(k in verdict for k in ['FAVORABLE', 'MIXED', 'HIGH-RISK'])
    bidding_strategy = {
        'retail_investor': {
            'budget': 'One lot only if it fits your risk budget',
            'strategy': ('Check the verified price band, lot size, and your broker cut-off before bidding.' if is_active or subscription_available else pending_note)
        },
        'small_hni': {
            'budget': 'Not determined by this model',
            'strategy': 'Check current category rules and your own allocation limits.' if (is_active or subscription_available) else pending_note
        },
        'big_hni': {
            'budget': 'Not determined by this model',
            'strategy': f"{sizing_tier}. Check current category rules and funding costs independently." if (is_active or subscription_available) else pending_note
        }
    }

    return {
        'company_name': company_name,
        'sector': sector,
        'executive_summary': {
            'verdict': verdict,
            'conviction': conviction,
            'win_probability_pct': win_prob,
            'issue_price': issue_price,
            'target_listing_price': None,
            'expected_gain_pct': None,
            'historical_median_open_return_pct': historical_median,
            'thesis_headline': thesis_headline,
            'recommended_action': recommendation_action
        },
        'bull_catalysts': bull_catalysts,
        'bear_risks': bear_risks,
        'bidding_strategy': bidding_strategy,
        'live_news_summary': [n.get('title', '') for n in news_items[:3]]
    }

def ask_ai_copilot(company_name: str, question: str, context_data: Dict[str, Any]) -> str:
    q = question.lower().strip()
    scores = context_data.get('hybrid_ml_scores') or {}
    inputs = context_data.get('input_multiples') or {}
    verdict = context_data.get('verdict') or 'WAIT FOR DATA'
    probability = scores.get('hybrid_positive_listing_prob')
    median = scores.get('historical_median_listing_return_pct')
    gmp = inputs.get('gmp_pct')
    demand = ', '.join(
        f'{label} {float(inputs[key]):.1f}x' for key, label in
        [('qib', 'QIB'), ('retail', 'retail'), ('hni', 'HNI')]
        if inputs.get(key) is not None
    ) or 'subscription figures unavailable'
    basis = context_data.get('forecast_basis')
    signal = (
        f'The final-book model estimates {probability:.1f}% probability of opening above issue price.'
        if probability is not None else
        'There is no final-book model probability yet; the book or verified inputs are pending.'
    )
    if basis in ('saved_preclose', 'historical_reconstruction'):
        label = 'reconstructed historical' if basis == 'historical_reconstruction' else 'saved pre-close'
        signal = (f'The {label} estimate is {probability:.1f}% probability of opening above issue price, '
                  f"using bids as of {context_data.get('forecast_asof_ist') or 'an unknown source time'}. "
                  'This is provisional, not validated interim accuracy.') if probability is not None else 'No saved probability is available.'
        if basis == 'historical_reconstruction':
            signal += ' It was calculated later, not issued live at that historical time.'
        if context_data.get('forecast_evidence_warnings'):
            signal += ' Its source evidence needs verification.'
    elif basis == 'no_preclose_forecast':
        signal = 'No saved pre-close forecast is available. Later closing-book research must not be described as an earlier prediction.'
    return_forecast = context_data.get('return_forecast') or {}
    historical = (
        f'Comparable earlier-year IPOs in this score band had a {median:+.1f}% median opening return. '
        'That is a group statistic, not a company price target.'
        if median is not None else 'No reliable company-level opening-return estimate is provided.'
    )
    if return_forecast.get('central_return_pct') is not None:
        historical = (f"The experimental return model estimates {return_forecast['central_return_pct']:+.1f}% centrally, "
                      f"with an empirical residual band {return_forecast['lower_return_pct']:+.1f}% to "
                      f"{return_forecast['upper_return_pct']:+.1f}%. This is not a guaranteed range or proven allocation edge. "
                      'It was trained on earlier daily bid books; live intraday validation is pending.')
    gmp_text = f'{float(gmp):+.1f}%' if gmp is not None else 'unavailable'
    if any(word in q for word in ('risk', 'danger', 'downside', 'drawdown')):
        return (
            f'{company_name}: {verdict}. {signal} {historical} '
            f'GMP is {gmp_text}. ' + ('A dated quote is an input to this return model. ' if return_forecast.get('track') == 'bids_plus_gmp' else 'GMP is not an input to this forecast. ') +
            'A risk warning can be wrong; check the offer document, liquidity, and your loss limit.'
        )
    if any(word in q for word in ('sizing', 'how much', 'budget', 'capital', 'apply')):
        return (
            f'{company_name}: {verdict}. {signal} The model does not determine lot count or capital allocation. '
            'Use the verified price band and lot size, your own risk budget, and your broker’s current deadlines.'
        )
    return f'{company_name}: {verdict}. {signal} Observed demand: {demand}. GMP: {gmp_text}. {historical}'
