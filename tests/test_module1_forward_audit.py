"""Evidence integrity for fixed-cohort forward updates and standard classifier."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def test_updated_test_never_contains_training_or_meta_calibration_issuers():
    report=json.loads((ROOT/'output/module1_standard_classifier_audit_20261025.json').read_text())
    stack=json.loads((ROOT/'output/module1_stacking_20261025.json').read_text())
    selection=stack['selections']['2025']
    assert set(selection['meta_training_ids'])<=set(report['training_ids'])
    assert not set(report['training_ids'])&set(report['test_ids'])
    frame=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv').set_index('ipo_id')
    train=frame.loc[report['training_ids']];test=frame.loc[report['test_ids']]
    assert train.year.between(2021,2025).all()
    assert test.year.eq(2026).all()
    assert pd.to_datetime(train.listing_date).max()<pd.to_datetime(test.decision_market_date).min()
    for fold in selection['trace']:
        assert not set(fold['fit_ids'])&set(fold['held_ids'])


def test_all_original_155_preserved_and_2025_updated_scores_are_absent():
    original=pd.read_csv(ROOT/'output/module1_blended_fix_20261024_predictions.csv').set_index('ipo_id')
    audit=pd.read_csv(ROOT/'output/module1_standard_classifier_audit_20261025_all_155_status.csv').set_index('ipo_id')
    assert audit.index.is_unique and set(audit.index)==set(original.index)
    assert audit[audit.year.eq(2025)].standard_call.isna().all()
    assert audit[audit.year.eq(2026)].standard_call.notna().all()
    assert np.allclose(audit.loc[original.index].exchange_open,original.exchange_open)
    assert np.allclose(audit.loc[original.index].strict_2021_2024_probability,original.p_blended_fix)


def test_accuracy_counts_and_complete_20_error_review():
    report=json.loads((ROOT/'output/module1_standard_classifier_audit_20261025.json').read_text())
    predictions=pd.read_csv(ROOT/'output/module1_standard_classifier_audit_20261025_2026_predictions.csv')
    assert predictions.ipo_id.is_unique and len(predictions)==69
    assert predictions.standard_call.eq(predictions.standard_probability.ge(.5).astype(int)).all()
    correct=predictions.standard_call.eq(predictions.target_binary)
    assert int(correct.sum())==report['correct']==report['metrics']['tn']+report['metrics']['tp']
    assert round(100*correct.mean(),1)==report['metrics']['classifier_accuracy_pct']
    assert predictions.target_binary.eq(predictions.exchange_open.gt(predictions.issue_price).astype(int)).all()
    review=pd.read_csv(ROOT/'output/module1_standard_classifier_audit_20261025_20_missed_positive_review.csv')
    assert len(review)==20
    assert review.year.eq(2025).sum()==12
    assert review.year.eq(2026).sum()==8
    assert report['limitations'][0].startswith('The standard 50% classification comparison was added after')


def test_source_fingerprints_and_past_only_observations():
    report=json.loads((ROOT/'output/module1_standard_classifier_audit_20261025.json').read_text())
    for relative,expected in report['input_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    frame=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv')
    decision=pd.to_datetime(frame.decision_market_date)
    assert pd.to_datetime(frame.macro_source_date).lt(decision).all()
    observed=frame.gmp_date.notna()
    assert pd.to_datetime(frame.loc[observed,'gmp_date']).lt(decision[observed]).all()
    observed=frame.pub_annual_period.notna()
    assert pd.to_datetime(frame.loc[observed,'pub_annual_period']).lt(decision[observed]).all()


def test_forward_selectors_reject_future_labels_in_training_runtime():
    code='''import pandas as pd
from ml.research_module1_forward_update import select
from ml.research_module1_recency import weights
from ml.research_module1_refit_calibration import choose_cutoff
late=pd.DataFrame([dict(year=2026,target_binary=1)])
for call in (lambda:select([],late,2025),lambda:weights(late,2025,1),lambda:choose_cutoff(late,2025,'p')):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError('Future label accepted')
print('future-label guards passed')
'''
    result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-c',code],cwd=ROOT,check=True,capture_output=True,text=True)
    assert 'future-label guards passed' in result.stdout
