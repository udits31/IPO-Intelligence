"""Leakage and source-stage contracts for the later-year comparison."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.collect_module1_later_holdout import bid_history, ROOT
from ml.evaluate_module1_later_holdout import validate, current_probability


def test_bid_parser_does_not_treat_blanks_as_zero_or_parentheses_as_negative():
    rows=[dict(key='Day 1 - 01-01-25 05:00 PM',value=dict(QIB='(0.00)',NII='(1.20)',RII='(3.40)')),
          dict(key='Day 2 - 02-01-25 05:00 PM',value=dict(QIB='--',NII='(2.0)',RII='(4.0)'))]
    result=bid_history(rows,'2025-01-01','2025-01-03')
    assert len(result)==1
    assert result[0]['qib_subscription']==0
    assert result[0]['hni_subscription']==1.2


def test_bid_parser_rejects_rows_outside_verified_offer_dates():
    row=dict(key='Day 1 - 01-01-25 05:00 PM',value=dict(QIB='(1)',NII='(2)',RII='(3)'))
    assert bid_history([row],'2025-02-01','2025-02-03')==[]


def cohorts():
    common=dict(gmp_date=pd.Timestamp('2024-01-01'),decision_market_date=pd.Timestamp('2024-01-02'),
                issue_close_date=pd.Timestamp('2024-01-03'),log_qib=1.,log_hni=1.,log_retail=1.,gmp_pct=2.,log_offer_cap_price=4.)
    train=pd.DataFrame([dict(common,ipo_id=1,year=2024,listing_date=pd.Timestamp('2024-01-05'))])
    test=pd.DataFrame([dict(common,ipo_id=2,year=2025,listing_date=pd.Timestamp('2025-01-05'),
                           gmp_date=pd.Timestamp('2025-01-01'),decision_market_date=pd.Timestamp('2025-01-02'),
                           issue_close_date=pd.Timestamp('2025-01-03'))])
    canonical=pd.DataFrame([dict(id=1,company_name='Alpha Limited',nse_symbol='ALPHA'),
                            dict(id=2,company_name='Beta Ltd',nse_symbol='BETA')]).set_index('id')
    return train,test,canonical


def test_guard_rejects_identity_overlap_even_with_different_ids():
    train,test,canonical=cohorts()
    canonical.loc[2,'company_name']='Alpha Ltd.'
    with pytest.raises(ValueError,match='identity leakage'):
        validate(train,test,canonical)


def test_guard_rejects_final_book_and_same_day_gmp():
    train,test,canonical=cohorts()
    test['gmp_date']=test.decision_market_date
    with pytest.raises(ValueError,match='GMP'):
        validate(train,test,canonical)
    test['gmp_date']=pd.Timestamp('2025-01-01')
    test['decision_market_date']=test.issue_close_date
    with pytest.raises(ValueError,match='Final book'):
        validate(train,test,canonical)


def test_saved_comparison_counts_and_current_bundle_replay():
    path=ROOT/'output/module1_2025_2026_paired_holdout_20261024.json'
    report=json.loads(path.read_text())
    ledger=pd.read_csv(path.with_name(path.stem+'_predictions.csv'))
    assert not set(report['train_ids']) & set(report['test_ids'])
    assert ledger.ipo_id.is_unique
    assert set(ledger.year)<=set([2025,2026])
    assert (ledger.target_binary==(ledger.exchange_open>ledger.issue_price).astype(int)).all()
    for name,metrics in report['results'].items():
        correct=ledger[name].ge(.5).eq(ledger.target_binary)
        assert metrics['correct']==int(correct.sum())
        assert metrics['n']==len(ledger)==report['paired_test_n']
    bundle=json.loads((ROOT/'ml/model_bundle.json').read_text())
    assert np.allclose(current_probability(bundle,ledger),ledger.p_current_interim,atol=1e-12)
