"""Presentation metrics must agree with actual ledgers and the active bundle."""
import json
import numpy as np
import pandas as pd
import services.performance_service as service
from ml.model_pipeline import summarize


def test_displayed_operating_and_direction_scores_are_not_interchanged():
    evidence=service.performance_evidence()
    assert evidence['status']=='verified_current_snapshot'
    ledger=pd.read_csv(service.ROOT/'output/performance_model_comparison_ledger.csv')
    bundle=json.loads((service.ROOT/'ml/model_bundle.json').read_text(encoding='utf-8'))
    current=evidence['module1']
    assert current['later_n']==len(ledger)==179
    for key,threshold in [('direction',bundle['benchmark']['direction_threshold']),('application_policy',bundle['decision_threshold'])]:
        actual=summarize(ledger.actual_positive,ledger.retained_probability,threshold)
        assert current[key]==actual
    assert current['application_policy']['apply_precision_pct']==90.
    assert current['application_policy']['classifier_accuracy_pct']==73.2
    assert current['direction']['classifier_accuracy_pct']==76.
    assert len(evidence['complexity']['rows'])==20


def test_module2_headline_uses_same_replayed_cases_and_horizon():
    evidence=service.performance_evidence()
    ledger=pd.read_csv(service.ROOT/'output/module2_holdout_decision_ledger_governance_20261023.csv')
    truth=ledger.day30_return_pct.gt(0)
    current=evidence['module2']['day30']
    assert current['n']==len(ledger)==132
    assert np.isclose(current['model_accuracy_pct'],ledger.sell.ne(truth).mean()*100,atol=.005)
    assert np.isclose(current['always_sell_accuracy_pct'],(~truth).mean()*100,atol=.005)
    change=evidence['tested_accuracy_changes'][0]
    legacy=json.loads((service.ROOT/'output/module2_holdout_accuracy_2025_2026presentation_20261022.json').read_text(encoding='utf-8'))
    assert change['n']==current['n'] and change['before_pct']==legacy['day30']['model_accuracy_pct']
    assert 'earlier' in change['change'].lower()
    assert evidence['module2']['review_calls']==ledger.verdict.str.contains('WATCH',case=False).sum()
    assert change['paired_change']['bootstrap_95_pp'][0]<0


def test_changed_inputs_hide_scores_instead_of_using_slide_constants(monkeypatch,tmp_path):
    report=json.loads(service.REPORT.read_text(encoding='utf-8'))
    report['input_sha256']['ml/model_bundle.json']='changed'
    path=tmp_path/'evidence.json'
    path.write_text(json.dumps(report),encoding='utf-8')
    monkeypatch.setattr(service,'REPORT',path)
    result=service.performance_evidence()
    assert result['status']=='outdated'
    assert 'module1' not in result
    monkeypatch.setattr(service,'REPORT',tmp_path/'absent.json')
    assert service.performance_evidence()['status']=='unavailable'
