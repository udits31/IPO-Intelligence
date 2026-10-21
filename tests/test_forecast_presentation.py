import math
from services.forecast_presentation import forecast_identity,saved_demand


def test_legacy_call_keeps_its_actual_model_identity():
    legacy={'mode':'legacy_saved_pre_close','model_version':'module1-corporate-subscription-lr-2021-2024-v2'}
    identity=forecast_identity(legacy)
    assert identity['model_family']=='subscription_logistic'
    assert identity['model_configuration']==legacy['model_version']
    assert identity['score_is_expected_return'] is False
    assert identity['score_is_asset_quality'] is False
    assert identity['model_configuration']!=forecast_identity({'mode':'saved_pre_close','model_sha':'frozen'})['model_configuration']


def test_memo_uses_saved_demand_instead_of_claiming_closed_offer_has_no_bids():
    frozen={'inputs':{'features':{'log_qib':math.log1p(13),'log_hni':math.log1p(4.5),'log_retail':math.log1p(.97)}}}
    values=saved_demand(frozen)
    assert abs(values['qib_subscription']-13)<1e-10
    assert abs(values['retail_subscription']-.97)<1e-10
    assert saved_demand({'inputs':{'qib_subscription':0,'retail_subscription':None}})['qib_subscription']==0
    assert saved_demand({'inputs':{'features':{'log_qib':float('nan')}}})['qib_subscription'] is None
