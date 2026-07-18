// One forecast contract for research, comparison and the research assistant.
// Later closing-book research is deliberately not a fallback prediction.
export function forecastView(result, history = null) {
  const frozen = result?.frozen_forecast;
  if (frozen) {
    const replay = !frozen.saved_call_found && frozen.historical_backtest;
    const call = replay || frozen;
    const probability = call.probability;
    // An explicit empty saved-call record must not fall back to a legacy or
    // later closing-book score from a different model/evidence stage.
    if (probability == null || !Number.isFinite(Number(probability)) || probability < 0 || probability > 1) return null;
    const features = call.inputs?.features || call.features || {};
    const inputs = {...(call.inputs || {})};
    for (const [field, key] of [['qib_subscription','log_qib'],['hni_subscription','log_hni'],['retail_subscription','log_retail']]) {
      if (inputs[field] == null && features[key] != null && Number.isFinite(Number(features[key]))) inputs[field] = Math.expm1(Number(features[key]));
    }
    if (features.gmp_pct != null) inputs.gmp_pct = features.gmp_pct;
    const evidence = call.inputs?.evidence || {};
    const legacy = !!call.legacy_model || call.mode === 'legacy_saved_pre_close';
    return {
      forecast: call, probability: Number(probability),
      band: probability >= .6 ? 'HIGH' : probability <= .4 ? 'LOW' : 'MIXED',
      basis: replay ? 'historical_backtest' : ['saved_pre_close','legacy_saved_pre_close'].includes(call.mode) ? 'saved_preclose' : call.mode || 'current_estimate',
      asOf: evidence.sources?.bids?.published_at_ist || evidence.sources?.bids?.source_asof_ist || evidence.sources?.bids?.source_reported_at_ist || evidence.asof_ist || call.bid?.observed_at_ist || call.source_asof_ist || call.recorded_at_ist,
      recordedAt: call.recorded_at_ist, inputs, warnings: evidence.missing_features || call.missing_features || [], returnForecast: null,
      modelLabel: legacy ? 'Earlier subscription model · saved call' : 'Frozen five-expert model',
      modelFamily: legacy ? 'subscription_logistic' : 'five_expert_stack',
      modelSha: call.model_sha || null,
      applicationPlan: call.inputs?.capital_plan || call.capital_plan || null,
    };
  }
  const issued = history?.application_forecast || result?.application_forecast;
  const recovered = history?.historical_bid_forecast || result?.historical_bid_forecast;
  const forecast = issued || recovered;
  if (!forecast || forecast.positive_listing_probability == null) return null;
  const returnForecast = issued?.inputs?.return_forecast;
  const returnProbability = returnForecast?.outside_training_features?.length ? null : returnForecast?.positive_listing_probability;
  const probability = Number(returnProbability ?? forecast.positive_listing_probability);
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) return null;
  return {
    forecast, probability,
    band: probability >= .6 ? 'HIGH' : probability <= .4 ? 'LOW' : 'MIXED',
    basis: issued ? 'saved_preclose' : 'historical_reconstruction',
    asOf: forecast.source_reported_at_ist || forecast.source_asof_ist || forecast.observed_on,
    inputs: forecast.inputs || {},
    warnings: forecast.inputs?.evidence_warnings || [],
    returnForecast: issued ? forecast.inputs?.return_forecast || null : null,
    modelLabel: 'Earlier subscription model', modelFamily: 'subscription_logistic',
    applicationPlan: null,
  };
}

const screenReasons = {
  undersubscribed_category: 'At least one investor category is below 1× demand.',
  high_ofs_weak_gmp: 'A large offer-for-sale share is paired with weak GMP and QIB demand.',
  retail_heavy_quota_weak_demand: 'The retail allocation is large relative to observed demand.',
  low_conviction_book: 'Total demand and GMP are below the application policy hurdles.',
  'score_below_0.50': 'The model score is below the frozen 50% direction cutoff.',
  nonpositive_retail_utility_scenario: 'The retail payoff scenario is not positive after assumed costs.',
};
const inputNames={gmp_pct:'a dated GMP quote',p_win:'a model score',qib:'QIB demand',nii:'HNI demand',retail:'retail demand',
  ofs_ratio:'the offer-for-sale share',retail_quota_pct:'the retail reservation',total_mult:'total subscription',
  original_lot_size:'the original lot size',offer_cap_price:'the offer price'};

export function applicationScreenView(result) {
  const view = forecastView(result);
  const plan = view?.applicationPlan;
  const tier = plan?.tier;
  const pending = ![1,2,3].includes(tier);
  const historical = ['CLOSED','LISTED'].includes(result?.scraped_data?.category) || ['historical_backtest','retrospective_estimate'].includes(view?.basis);
  return {
    plan, tier, pending, historical,
    label: pending ? 'Awaiting screen inputs' : tier === 3 ? 'Skip' : tier === 1 ? 'Apply · Tier 1' : 'Apply · Tier 2',
    tone: pending ? 'warning' : tier === 3 ? 'negative' : 'positive',
    reasons: pending ? [plan?.missing_inputs?.length ? `Needs ${plan.missing_inputs.map(k=>inputNames[k]||k.replaceAll('_',' ')).join(', ')}.` : 'No application screen was recorded with this forecast.']
      : plan.reasons?.length ? plan.reasons.map(k=>screenReasons[k] || k.replaceAll('_',' '))
      : [tier === 1 ? 'Passed the higher-conviction demand, GMP and retail-payoff checks.' : 'Passed the application checks; limited to a one-lot retail scenario.'],
    lotCost: plan?.rii_lot_rupees ?? null,
    utility: plan?.eu_rupees_rii ?? null,
    probability: plan?.approx_rii_allot_prob ?? null,
    policy: plan?.policy || null,
  };
}

export function assistantContext(result) {
  const view = forecastView(result);
  const evaluation = result?.evaluation || {};
  return {
    ...evaluation,
    verdict: applicationScreenView(result).plan ? applicationScreenView(result).label : view?.band || 'NO PRE-CLOSE FORECAST',
    model_label: view?.modelLabel,
    application_screen: applicationScreenView(result),
    forecast_basis: view?.basis || 'no_preclose_forecast',
    forecast_asof_ist: view?.asOf,
    forecast_evidence_warnings: view?.warnings || [],
    return_forecast: view?.returnForecast || null,
    input_multiples: {qib: view?.inputs.qib_subscription, hni: view?.inputs.hni_subscription,
      retail: view?.inputs.retail_subscription,
      gmp_pct: view?.returnForecast?.input_features?.gmp_pct ?? result?.scraped_data?.gmp_pct},
    hybrid_ml_scores: {
      hybrid_positive_listing_prob: view ? view.probability * 100 : null,
      historical_median_listing_return_pct: null,
    },
  };
}
