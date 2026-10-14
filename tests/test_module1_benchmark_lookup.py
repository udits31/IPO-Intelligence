import json
import math
import pytest
from services import module1_benchmark_lookup as benchmark
from services.forecast_presentation import saved_demand


@pytest.mark.skipif(not benchmark.LOCK.exists(),reason='Locked benchmark assets unavailable')
def test_esds_research_uses_the_actual_benchmark_prediction_inputs_and_open():
    r=benchmark.lookup(dict(company_name='ESDS Software Solution',issue_price=429,listing_date='2026-09-04'))
    assert r['mode']=='historical_benchmark'
    assert math.isclose(r['probability'],.9887216394926118)
    assert r['outcome']['actual_open']==757 and r['outcome']['issue_price']==429
    assert r['correct'] is True and r['input_stage']=='Day 2 close'
    assert math.isclose(r['bid']['qib'],.42) and math.isclose(r['bid']['retail'],19.38)
    assert math.isclose(r['features']['gmp_pct'],83.21678321678321)
    assert r['missing_features']==[]
    assert r['summary']['correct']==56 and r['summary']['evaluated']==69
    assert not r.get('recorded_before_cutoff')


@pytest.mark.skipif(not benchmark.LOCK.exists(),reason='Locked benchmark assets unavailable')
def test_entire_performance_cohort_has_consistent_bound_predictions_and_outcomes():
    locked=json.loads(benchmark.LOCK.read_text(encoding='utf8'))
    originals={r['issuer_key']:r for r in locked['module1']['records']}
    rows=benchmark.catalog()
    assert len(rows)==len(originals)==93
    for r in rows:
        assert r['probability']==originals[r['issuer_key']]['probability']
        assert r['outcome'] is not None
        assert math.isclose(r['outcome']['return_pct'],originals[r['issuer_key']]['actual_return_pct'],abs_tol=.001)
        assert r['capital_plan']==originals[r['issuer_key']]['plan']
    earlier=[r for r in rows if r['cohort']=='test_69']
    recent=[r for r in rows if r['cohort']!='test_69']
    assert len(earlier)==69 and sum(r['correct'] for r in earlier)==56
    assert len(recent)==24 and sum(r['correct'] for r in recent)==17


@pytest.mark.skipif(not benchmark.LOCK.exists(),reason='Locked benchmark assets unavailable')
def test_wrong_offer_identity_cannot_borrow_a_benchmark_score():
    assert benchmark.lookup(dict(company_name='ESDS Software Solution',issue_price=430)) is None
    assert benchmark.lookup(dict(company_name='ESDS Software Solution',listing_date='2025-09-04')) is None
    assert benchmark.lookup(dict(company_name='ESDS Different Company')) is None


def test_saved_demand_prefers_bound_historical_snapshot_but_preserves_real_calls():
    r=dict(inputs=dict(features={'log_qib':math.log1p(110)}),
           historical_backtest=dict(features={'log_qib':math.log1p(.42)}))
    assert math.isclose(saved_demand(r)['qib_subscription'],.42)
    r['saved_call_found']=True
    assert math.isclose(saved_demand(r)['qib_subscription'],110)


def test_listed_analysis_attaches_benchmark_without_fabricating_a_live_call(monkeypatch):
    from services import prospective_module1 as prospective,saved_preclose_service,preclose_context
    monkeypatch.setattr(prospective,'resolve_calendar',lambda item,url:item)
    monkeypatch.setattr(preclose_context,'restore_preclose_gmp',lambda item,db:item)
    monkeypatch.setattr(benchmark,'lookup',lambda item:{'mode':'historical_benchmark','probability':.9})
    original=dict(mode='no_saved_pre_close',saved_call_found=False,probability=None)
    monkeypatch.setattr(saved_preclose_service,'comparison',lambda *args:dict(original))
    result=prospective.analysis_card(dict(company_name='Example',category='LISTED'),'source')
    assert result['probability'] is None and result['saved_call_found'] is False
    assert result['historical_backtest']['probability']==.9
    original.update(mode='saved_pre_close',saved_call_found=True,probability=.6)
    result=prospective.analysis_card(dict(company_name='Example',category='LISTED'),'source')
    assert result['probability']==.6 and result['saved_call_found'] is True
    assert 'historical_backtest' not in result and result['benchmark_comparison']['probability']==.9
