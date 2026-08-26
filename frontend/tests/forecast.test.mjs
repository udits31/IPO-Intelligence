import test from 'node:test';
import assert from 'node:assert/strict';
import {forecastView, assistantContext, applicationScreenView} from '../src/lib/forecast.mjs';

test('comparison never silently substitutes a later closing-book probability', () => {
  const result = {evaluation: {verdict:'APPLY',hybrid_ml_scores:{hybrid_positive_listing_prob:99}}};
  assert.equal(forecastView(result),null);
  assert.equal(assistantContext(result).hybrid_ml_scores.hybrid_positive_listing_prob,null);
});
test('research, comparison and assistant share the issued pre-close estimate', () => {
  const result = {application_forecast:{positive_listing_probability:.49,source_asof_ist:'2026-10-21T15:00:00+05:30',
    inputs:{qib_subscription:7.09}},evaluation:{verdict:'FAVORABLE SETUP'}};
  assert.equal(forecastView(result).band,'MIXED');
  assert.equal(assistantContext(result).hybrid_ml_scores.hybrid_positive_listing_prob,49);
  assert.equal(assistantContext(result).input_multiples.qib,7.09);
});
test('reconstructions are explicitly labelled and retain evidence warnings', () => {
  const result = {historical_bid_forecast:{positive_listing_probability:.717,inputs:{evidence_warnings:['clock conflict']}}};
  assert.equal(assistantContext(result).forecast_basis,'historical_reconstruction');
  assert.deepEqual(forecastView(result).warnings,['clock conflict']);
});

test('new early-book probability and return model share their own saved evidence', () => {
  const result = {application_forecast:{positive_listing_probability:.99, inputs:{
    return_forecast:{positive_listing_probability:.55, central_return_pct:12, track:'bids_plus_gmp'}}}};
  assert.equal(forecastView(result).probability,.55);
  assert.equal(assistantContext(result).return_forecast.central_return_pct,12);
  assert.ok(Math.abs(assistantContext(result).hybrid_ml_scores.hybrid_positive_listing_prob - 55) < 1e-10);
});

test('current frozen forecast takes precedence over legacy fields and a return companion',()=>{
 const r={application_forecast:{positive_listing_probability:.99,inputs:{return_forecast:{positive_listing_probability:.95}}},
  frozen_forecast:{mode:'saved_pre_close',probability:.42,model_sha:'frozen',recorded_at_ist:'2026-10-30T14:00:00+05:30',inputs:{features:{log_qib:Math.log1p(3),gmp_pct:4},evidence:{sources:{bids:{published_at_ist:'2026-10-30T13:57:00+05:30'}}}}}};
 const view=forecastView(r);
 assert.equal(view.probability,.42);assert.equal(view.modelFamily,'five_expert_stack');
 assert.equal(view.inputs.qib_subscription,3);assert.equal(view.inputs.gmp_pct,4);
 assert.equal(view.asOf,'2026-10-30T13:57:00+05:30');assert.equal(view.returnForecast,null);
 assert.equal(assistantContext(r).hybrid_ml_scores.hybrid_positive_listing_prob,42);
});

test('positive model does not override a skipped application or turn unknown into apply',()=>{
 const r={frozen_forecast:{mode:'saved_pre_close',probability:.91,inputs:{capital_plan:{tier:3,verdict:'HARD_SKIP',eu_rupees_rii:-10,reasons:['undersubscribed_category'],policy:'locked'}}}};
 const screen=applicationScreenView(r);
 assert.equal(forecastView(r).band,'HIGH');assert.equal(screen.label,'Skip');assert.equal(screen.utility,-10);
 assert.match(screen.reasons[0],/below 1/);assert.equal(assistantContext(r).verdict,'Skip');
 assert.equal(applicationScreenView({frozen_forecast:{mode:'saved_pre_close',probability:.91}}).pending,true);
});

test('explicit no-call record cannot silently use an old forecast',()=>{
 const r={frozen_forecast:{mode:'no_saved_pre_close',probability:null},application_forecast:{positive_listing_probability:.99}};
 assert.equal(forecastView(r),null);assert.equal(applicationScreenView(r).pending,true);
});

test('saved legacy model identity and closed application window stay visible',()=>{
 const r={scraped_data:{category:'LISTED'},frozen_forecast:{mode:'legacy_saved_pre_close',probability:.69,inputs:{qib_subscription:15.95}}};
 assert.equal(forecastView(r).modelFamily,'subscription_logistic');assert.equal(forecastView(r).probability,.69);
 assert.equal(applicationScreenView(r).historical,true);assert.equal(applicationScreenView(r).pending,true);
});

test('historical backtest carries its own input book and policy',()=>{
 const r={frozen_forecast:{mode:'no_saved_pre_close',probability:null,historical_backtest:{probability:.7,features:{gmp_pct:5,log_qib:Math.log1p(2)},capital_plan:{tier:2,reasons:[],eu_rupees_rii:25}}}};
 assert.equal(forecastView(r).basis,'historical_backtest');assert.equal(forecastView(r).inputs.gmp_pct,5);
 assert.equal(applicationScreenView(r).tier,2);assert.equal(applicationScreenView(r).historical,true);
});

test('an earlier IPO uses its performance benchmark snapshot rather than the final scraped book',()=>{
 const r={scraped_data:{category:'LISTED',qib_subscription:110.38,gmp_pct:null},
   frozen_forecast:{mode:'no_saved_pre_close',saved_call_found:false,probability:null,historical_backtest:{
    mode:'historical_benchmark',probability:.9887216394926118,bid:{observed_at_ist:'2026-08-31T17:00:00+05:30'},
    features:{log_qib:Math.log1p(.42),log_hni:Math.log1p(72.06),log_retail:Math.log1p(19.38),gmp_pct:83.2168},
    missing_features:['some_other_field'],capital_plan:{tier:3,reasons:['undersubscribed_category']}}}};
 const view=forecastView(r);
 assert.equal(view.probability,.9887216394926118);
 assert.ok(Math.abs(view.inputs.qib_subscription-.42)<1e-9);
 assert.equal(view.inputs.gmp_pct,83.2168);assert.equal(view.asOf,'2026-08-31T17:00:00+05:30');
 assert.deepEqual(view.warnings,['some_other_field']);assert.equal(applicationScreenView(r).label,'Skip');
 assert.equal(view.basis,'historical_backtest');
});
