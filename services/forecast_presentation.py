"""Describe saved forecasts without relabelling old models or changing scores."""
import math


def forecast_identity(forecast):
    legacy=bool(forecast.get('legacy_model') or forecast.get('mode')=='legacy_saved_pre_close')
    return dict(model_configuration=forecast.get('model_version') if legacy else 'M1_V1_FROZEN_WITH_RECOVERED_INPUTS',
        model_family='subscription_logistic' if legacy else 'five_expert_stack',
        model_label='Earlier subscription model' if legacy else 'Frozen five-expert model',
        artifact_sha256=forecast.get('model_sha'),
        target='listing_open_above_issue_price',calibration='uncalibrated',
        score_is_expected_return=False,score_is_asset_quality=False)


def saved_demand(forecast):
    inputs=forecast.get('inputs') or {};features=inputs.get('features') or {}
    if not forecast.get('saved_call_found') and forecast.get('historical_backtest'):
        features=forecast['historical_backtest'].get('features') or features
    result={}
    for name,log_name in [('qib_subscription','log_qib'),('hni_subscription','log_hni'),('retail_subscription','log_retail')]:
        value=inputs.get(name)
        if value is None and features.get(log_name) is not None:
            try:value=math.expm1(float(features[log_name]))
            except (TypeError,ValueError,OverflowError):value=None
        try:value=float(value)
        except (TypeError,ValueError):value=None
        result[name]=value if value is not None and math.isfinite(value) and value>=0 else None
    return result
