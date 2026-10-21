"""Paired comparisons must not hide sample changes or future information."""
import hashlib
import json

import numpy as np
import pandas as pd

from ml.research_module1_gmp_fundamentals import OUT,VARIANTS,ROOT


def load():
    return (json.loads(OUT.read_text()),
        pd.read_csv(OUT.with_name(OUT.stem+'_predictions.csv')))


def test_all_variants_use_identical_test_cases_and_correct_counts():
    report,ledger=load()
    test=ledger[ledger.year.eq(2024)]
    assert len(test)==test.ipo_id.nunique()==report['confirmation_n']==39
    assert set(report['results'])==set(VARIANTS)|{'gmp_sign','always_positive'}
    y=test.target_binary.astype(bool)
    base=test.gmp_sign.ge(.5).eq(y)
    for name,item in report['results'].items():
        correct=test[name].ge(.5).eq(y)
        assert item['metrics']['sample_size']==len(test)
        assert item['metrics']['classifier_accuracy_pct']==round(100*correct.mean(),1)
        assert np.isclose(item['vs_gmp_sign']['change_pp'],100*(correct.mean()-base.mean()))
        assert item['vs_gmp_sign']['corrected_errors']==int((correct&~base).sum())
        assert item['vs_gmp_sign']['introduced_errors']==int((~correct&base).sum())


def test_selection_uses_development_only_and_no_future_quote_or_outcome():
    report,ledger=load()
    selected=min(VARIANTS,key=lambda name:(report['development'][name]['brier_score'],name))
    assert selected==report['selected_variant']
    for trace in report['trace']:
        assert trace['latest_training_outcome']<trace['first_test_decision']
        assert not set(trace['training_ids'])&set(trace['test_ids'])
    assert pd.to_datetime(ledger.gmp_date).lt(pd.to_datetime(ledger.decision_market_date)).all()
    assert all(item['metrics']['threshold']==.5 for item in report['results'].values())
    assert report['live_promoted'] is False


def test_complete_error_ledger_and_probability_explanations():
    report,ledger=load()
    errors=pd.read_csv(OUT.with_name(OUT.stem+'_errors.csv'))
    test=ledger[ledger.year.eq(2024)].set_index('ipo_id')
    assert set(errors.ipo_id)==set(test.index)
    assert errors.selected_vs_gmp.value_counts().to_dict()==report['selected_error_counts']
    contributions=pd.read_csv(OUT.with_name(OUT.stem+'_contributions.csv'))
    totals=contributions.groupby('ipo_id').log_odds_contribution.sum()
    intercept=contributions.groupby('ipo_id').intercept.first()
    reconstructed=1/(1+np.exp(-(totals+intercept)))
    assert np.allclose(reconstructed,test.loc[reconstructed.index,report['selected_variant']])


def test_sources_and_research_artifact_match_saved_evidence():
    report,_=load()
    for relative,expected in report['input_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    assert hashlib.sha256(OUT.with_suffix('.joblib').read_bytes()).hexdigest()==report['artifact_sha256']
    assert set(report['selected_training']['features'])==set(VARIANTS[report['selected_variant']])
