import json
import hashlib
from pathlib import Path
import pytest
from services.performance_dashboard import build_dashboard,ROOT


def test_dashboard_keeps_raw_accuracy_selection_and_constant_baselines_separate():
    d=build_dashboard()
    expected={'test_69':(56,69,18,18,28),'recent_23':(16,23,12,14,5),'recent_24_verified':(17,24,12,14,6)}
    for name,(right,n,selected_winners,selected,skipped_winners) in expected.items():
        c=d['module1'][name];m=c['model_direction'];s=c['application_screen']
        assert m['accuracy_pct']==pytest.approx(100*right/n)
        assert m['model_direction_missed']==n-right
        assert s['selected_apply_winners']==selected_winners and s['selected_apply_total']==selected
        assert s['skipped_winners_opportunity_cost']==skipped_winners
        assert c['always_positive_accuracy_pct']==pytest.approx(100*sum(r['actual_positive'] for r in c['records'])/n)
    assert d['module1']['recent_24_verified']['accuracy_minus_baseline_pp']<0
    assert d['module2']['active']['correct']==63 and d['module2']['baseline']['correct']==77


def test_actual_open_display_is_bound_to_the_locked_return_and_all_rows_covered():
    d=build_dashboard()
    for c in d['module1'].values():
        for r in c['records']:
            assert r['issue_price']>0 and r['actual_open']>0
            assert 100*(r['actual_open']/r['issue_price']-1)==pytest.approx(r['actual_return_pct'],abs=.001)
            assert r['model_direction_correct']==(r['model_predicted_positive']==r['actual_positive'])


def test_exploratory_followthrough_never_replaces_the_frozen_monitor_accuracy():
    d=build_dashboard()
    assert d['intraday']['evaluated']==52 and d['intraday']['matched']==23
    assert d['module2']['active']['accuracy_pct']==pytest.approx(100*63/132)
    assert 'direction_accuracy_pct' in d['prospective']['summary']
    json.dumps(d,allow_nan=False)
    assert d['locked_sha256']==hashlib.sha256((ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json').read_bytes()).hexdigest()
