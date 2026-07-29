"""Allotment/capital scenarios. Share multiples alone are not exact lottery odds.

No broker order is placed. Financing is charged on blocked/borrowed capital,
including unsuccessful allotment outcomes.
"""
import math

BOUNCER_VERSION='institutional-overlay-20261026-v1'


def institutional_bouncer(*,p_win,gmp_pct,qib,nii,retail,ofs_ratio,retail_quota_pct,
                          total_mult,original_lot_size,offer_cap_price,bnii_mult=None,
                          funding_apr_pct=9.5,funding_days=5,borrowed_amount=None):
    """Frozen Layer 1 plus user-specified selection policy; utility is a scenario.

    retail_quota_pct uses percentage points (35 means 35%). No change to ML calls.
    Category thresholds are product policy, not SEBI rules or learned parameters.
    """
    def finite(v):
        try:return float(v) if math.isfinite(float(v)) else None
        except (ValueError,TypeError):return None
    values={k:finite(v) for k,v in dict(p_win=p_win,gmp_pct=gmp_pct,qib=qib,nii=nii,retail=retail,
        ofs_ratio=ofs_ratio,retail_quota_pct=retail_quota_pct,total_mult=total_mult,
        original_lot_size=original_lot_size,offer_cap_price=offer_cap_price).items()}
    missing=[k for k,v in values.items() if v is None]
    for key in ('qib','nii','retail','total_mult'):
        if values[key] is not None and values[key]<0:raise ValueError('Negative subscription multiple')
    if values['p_win'] is not None and not 0<=p_win<=1:raise ValueError('Score outside [0,1]')
    if values['ofs_ratio'] is not None and not 0<=ofs_ratio<=1:raise ValueError('OFS must be a fraction')
    if values['retail_quota_pct'] is not None and not 0<=retail_quota_pct<=100:raise ValueError('Quota uses percentage points')
    flags={
        'undersubscribed_category':None if any(values[k] is None for k in ('qib','nii','retail')) else min(qib,nii,retail)<1.,
        'high_ofs_weak_gmp':None if any(values[k] is None for k in ('ofs_ratio','gmp_pct','qib')) else ofs_ratio>=.75 and gmp_pct<8 and qib<5,
        'retail_heavy_quota_weak_demand':None if any(values[k] is None for k in ('retail_quota_pct','retail','gmp_pct')) else retail_quota_pct>=50 and retail<2 and gmp_pct<10,
        'low_conviction_book':None if any(values[k] is None for k in ('total_mult','gmp_pct')) else total_mult<2 and gmp_pct<6,
    }
    result=dict(policy=BOUNCER_VERSION,model_score=values['p_win'],standard_binary_call=None if values['p_win'] is None else 'POSITIVE' if p_win>=.5 else 'FLAT_OR_NEGATIVE',
        flags=flags,missing_inputs=missing,probability_method='RECIPROCAL_APPROXIMATION',
        utility_basis='Uncalibrated score treated as probability for scenario only; 6.5% loss, 0.35% friction, Rs9 fixed cost are assumptions.')
    if missing:return dict(**result,tier=None,verdict='DATA_PENDING',reasons=['Missing inputs: '+', '.join(missing)])
    if original_lot_size<=0 or int(original_lot_size)!=original_lot_size:raise ValueError('Positive integer original lot size required')
    sizes=category_lots(offer_cap_price,original_lot_size)
    lot=sizes['lot_cost'];sparcel=sizes['snii_min_lots']*lot;bapp=sizes['bnii_min_lots']*lot;ratio=bapp/sparcel
    net=p_win*(max(2.,gmp_pct)-.35)-(1-p_win)*6.5
    probability=min(1.,1/max(1.,retail));eu=probability*lot*net/100-9.
    reasons=[k for k,v in flags.items() if v is True]
    if p_win<.5:reasons.append('score_below_0.50')
    if eu<=0:reasons.append('nonpositive_retail_utility_scenario')
    if reasons:tier=3
    elif p_win>=.68 and min(qib,nii,retail)>=1.5 and qib>=3 and gmp_pct>=8:tier=1
    else:tier=2
    borrowed=bapp if borrowed_amount is None else float(borrowed_amount)
    if not 0<=borrowed<=bapp or min(funding_apr_pct,funding_days)<0:raise ValueError('Invalid financing scenario')
    interest=borrowed*(funding_apr_pct/100)*funding_days/365
    bmult=finite(bnii_mult)
    if bmult is not None and bmult<0:raise ValueError('Negative bNII subscription')
    bp=min(1.,ratio/max(ratio,bmult)) if bmult is not None else None
    breakeven=100*interest/(bp*sparcel)+.35 if bp else None
    return dict(**result,tier=tier,verdict={1:'APPLY_TIER_1',2:'APPLY_TIER_2',3:'HARD_SKIP'}[tier],reasons=reasons,
        rii_lot_rupees=lot,snii_min_lots=sizes['snii_min_lots'],snii_parcel_rupees=sparcel,
        bnii_min_lots=sizes['bnii_min_lots'],bnii_app_rupees=bapp,parcel_ratio=ratio,
        approx_rii_allot_prob=probability,approx_bnii_allot_prob=bp,net_expected_return_pct=net,eu_rupees_rii=eu,
        nbfc_interest_rupees=interest,nbfc_breakeven_open_pct=breakeven,
        bnii_nbfc_leverage_viable=bool(tier==1 and breakeven is not None and net>=breakeven+4),
        financing=dict(apr_pct=funding_apr_pct,days=funding_days,borrowed_rupees=borrowed),
        limitations=['Aggregate subscribed shares do not identify valid application counts or bid-size distribution.',
            'An undersubscribed category does not guarantee full allotment or a negative listing.',
            'Minimum-parcel EU does not model residual/proportionate allocation or winner-dependent allotment.',
            'Tier 1 refers to eligible individual applicants; no duplicate PAN or pooled-identity application plan.'])


def capital_plan(*,score,regime_adjusted_gmp_pct,qib,hni,retail,cap_price,lot_size,
                 issue_size_cr=None,snii=None,bnii=None,financing_apr_pct=0.,champion_qualified=False):
    """Requested tier policy, with explicit scenario assumptions and stress loss.

    No probability calibration or exact lottery claim is implied by a score.
    The 6.5% miss loss, 0.35% friction, Rs9 and 7 days are user proposals.
    """
    values=(score,regime_adjusted_gmp_pct,qib,hni,retail)
    if any(v is None or not math.isfinite(float(v)) for v in values):return dict(tier=3,label='PRESERVE CAPITAL',deployable=False,reasons=['Current dated score, GMP and category readings are required.'])
    if not 0<=score<=1 or min(qib,hni,retail)<0:raise ValueError('Invalid score or subscription multiple')
    win_gain=max(2.,float(regime_adjusted_gmp_pct));net=score*(win_gain-.35)-(1-score)*6.5
    sizes=category_lots(cap_price,lot_size);lot=sizes['lot_cost'];p=min(1.,1/max(1.,retail));eu=p*lot*net/100-9.
    demand=min(qib,hni,retail);tier=3;reasons=[]
    if score<.55 or eu<=0 or (demand<1.25 and issue_size_cr is not None and issue_size_cr>200):reasons.append('The proposed score, net utility or thin-book exclusion failed.')
    elif score>=.70 and demand>=2 and regime_adjusted_gmp_pct>=8 and eu>150:tier=1
    elif score>=.55 and demand>=1.25 and eu>0:tier=2
    else:reasons.append('The proposed minimum demand/utility hurdles were not met.')
    categories={}
    for category,mult in [('retail',retail),('snii',snii),('bnii',bnii)]:
        if mult is None:categories[category]=dict(status='split_subscription_unavailable');continue
        scenario=assess(category=category,cap_price=cap_price,lot_size=lot_size,expected_return_pct=net,
            subscription_mult=mult,financing_apr_pct=financing_apr_pct if category=='bnii' else 0,
            blocked_days=7,funded_fraction=1 if category=='bnii' and financing_apr_pct else 0,
            fixed_cost_rupees=9,allow_multiple_approximation=True)
        # Do not assume that winning and losing IPOs give identical allotments.
        stress=score*scenario['allotment_probability']*scenario['conditional_minimum_allotment_rupees']*(win_gain-.35)/100-(1-score)*scenario['blocked_capital_rupees']*.065-scenario['financing_cost_rupees']-9
        scenario['full_allotment_loss_stress_EU_rupees']=stress
        scenario['funded_viable_under_scenario']=bool(category=='bnii' and stress>0 and scenario['bnii_funded_breakeven_return_pct'] is not None and net>=scenario['bnii_funded_breakeven_return_pct']+4)
        categories[category]=scenario
    return dict(tier=tier,label={1:'HIGH-CONVICTION SCENARIO',2:'ONE RETAIL LOT SCENARIO',3:'PRESERVE CAPITAL'}[tier],
        deployable=False,champion_qualified=champion_qualified,model_score=score,net_expected_return_scenario_pct=net,retail_EU_scenario_rupees=eu,
        category_scenarios=categories,reasons=reasons,
        assumptions=dict(miss_loss_pct=6.5,friction_pct=.35,fixed_cost_rupees=9,financing_days=7,win_gain_floor_pct=2.,score_treated_as_probability_for_scenario_only=True),
        limitations=['Allotment reciprocals are proxies, not exact SEBI application-level odds.','Loss size and fees here are proposed assumptions, not freshly measured model evidence.','Score/return calibration and allotment-outcome dependence remain unverified.','Only separate eligible investors may make their own PAN-linked applications; no duplicate or identity-pooled application plan is produced.'])


def category_lots(price,lot_size):
    cost=float(price)*int(lot_size)
    if cost<=0:raise ValueError('Positive price and lot size are required')
    return dict(retail_lots=1,snii_min_lots=math.floor(200000/cost)+1,
                bnii_min_lots=math.floor(1000000/cost)+1,lot_cost=cost)


def assess(*,category,cap_price,lot_size,expected_return_pct,subscription_mult,
           application_lots=None,valid_applications=None,category_shares=None,
           minimum_allotment_shares=None,allotment_basis_verified=False,
           financing_apr_pct=0.,blocked_days=0,funded_fraction=0.,
           sale_cost_pct=0.,fixed_cost_rupees=0.,probability_positive=None,
           precision_hurdle=None,demand=None,allow_multiple_approximation=False):
    if category not in ('retail','snii','bnii'):raise ValueError('Unsupported investor category')
    sizes=category_lots(cap_price,lot_size)
    lots=int(application_lots or (1 if category=='retail' else sizes[category+'_min_lots']))
    applied=lots*sizes['lot_cost']
    if category=='retail' and applied>200000:raise ValueError('Retail application exceeds category amount')
    if category=='snii' and not 200000<applied<=1000000:raise ValueError('sNII application must exceed 2 lakh and not exceed 10 lakh')
    if category=='bnii' and applied<=1000000:raise ValueError('bNII application must exceed 10 lakh')
    if not 0<=funded_fraction<=1 or min(financing_apr_pct,blocked_days,sale_cost_pct,fixed_cost_rupees)<0:raise ValueError('Invalid cost scenario')
    mult=float(subscription_mult) if subscription_mult is not None else None
    if mult is not None and mult<0:raise ValueError('Negative subscription multiple')
    p=None;method='unavailable';warnings=[]
    minshares=int(minimum_allotment_shares or (int(lot_size) if category=='retail' else sizes['snii_min_lots']*int(lot_size)))
    if allotment_basis_verified and valid_applications is not None and category_shares is not None:
        if valid_applications<=0 or category_shares<0:raise ValueError('Invalid allotment basis counts')
        slots=math.floor(category_shares/minshares)
        p=min(1.,slots/valid_applications);method='verified_minimum_allotment_lottery_basis'
        warnings.append('EU uses the minimum-allotment component; residual/proportionate allocation needs the complete issue basis.')
    elif allow_multiple_approximation and mult is not None:
        p=min(1.,(5. if category=='bnii' else 1.)/max(1.,mult));method='share_multiple_scenario_not_exact_sebi_probability'
        warnings.append('Share-subscription ratio does not identify valid application counts or their sizes. The bNII factor 5 is a scenario assumption, not a universal cap.')
    financing=applied*funded_fraction*(financing_apr_pct/100)*(blocked_days/365)
    conditional_allotment=min(applied,minshares*float(cap_price))
    if mult is not None and mult<=1 and method=='verified_minimum_allotment_lottery_basis':
        warnings.append('Undersubscription does not by itself prove this application receives its full bid; allocation spillovers/validity require the final basis.')
    expected_capital=p*conditional_allotment if p is not None else None
    eu=(expected_capital*(float(expected_return_pct)-sale_cost_pct)/100-financing-fixed_cost_rupees) if expected_capital is not None and expected_return_pct is not None else None
    breakeven=100*(financing+fixed_cost_rupees)/expected_capital+sale_cost_pct if expected_capital and expected_capital>0 else None
    reasons=[];policy='WATCH'
    if demand and all(demand.get(k) is not None for k in ('qib','hni','retail')) and min(demand[k] for k in ('qib','hni','retail'))<1:
        policy='SKIP';reasons.append('Conservative thin-book gate: one demand category is below 1x; this does not imply a certain loss.')
    elif eu is not None and eu<=0:policy='SKIP';reasons.append('Expected net rupee scenario is nonpositive after financing and costs.')
    elif eu is None:reasons.append('Valid allotment odds or a mean-return estimate is unavailable.')
    elif probability_positive is None or precision_hurdle is None:reasons.append('Walk-forward selection hurdle is unavailable.')
    elif probability_positive<precision_hurdle:policy='SKIP';reasons.append('Probability is below the previously frozen hurdle.')
    elif method!='verified_minimum_allotment_lottery_basis':reasons.append('Approximate allotment scenario cannot establish an exact APPLY recommendation.')
    else:policy='APPLY';reasons.append('Positive modeled minimum-allotment utility and the frozen probability hurdle passed.')
    warnings.append('Mean return conditional on allotment may differ from unconditional model return; adverse-selection dependence is not identified by this scenario.')
    return dict(category=category,application_lots=lots,blocked_capital_rupees=applied,
        allotment_probability=p,allotment_method=method,conditional_minimum_allotment_rupees=conditional_allotment,
        expected_minimum_allotment_rupees=expected_capital,financing_cost_rupees=financing,
        EU_rupees=eu,bnii_funded_breakeven_return_pct=breakeven if category=='bnii' else None,
        policy=policy,reasons=reasons,warnings=warnings,costs=dict(apr_pct=financing_apr_pct,days=blocked_days,funded_fraction=funded_fraction,sale_cost_pct=sale_cost_pct,fixed_cost_rupees=fixed_cost_rupees))
