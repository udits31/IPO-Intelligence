"""The new challenger cannot borrow future prices or change the test target."""
import json
import numpy as np
import pandas as pd
import pytest
from ml import research_monitor_daily_learning as research
from ml import research_module1_recency as recency


def test_feature_extraction_cannot_see_prices_after_decision():
    prices=pd.read_csv(research.DEV,parse_dates=['date'])
    group=next(group.sort_values('date').reset_index(drop=True) for _,group in prices.groupby('ipo_id') if len(group)>=62)
    before=research.features_at(group,30)
    modified=group.copy()
    # Test perturbation only. Never enters a training file or model fit.
    modified.loc[31:,['close_price_adjusted','sector_index_close','nifty50_close']]=np.nan
    after=research.features_at(modified,30)
    assert before.keys()==after.keys()
    np.testing.assert_allclose(list(before.values()),list(after.values()),equal_nan=True)


def test_real_policy_drop_is_exactly_four_correct_calls():
    old=pd.read_csv(research.OLD).set_index('ipo_id')
    current=pd.read_csv(research.CURRENT).set_index('ipo_id').loc[old.index]
    up=current.day30_return_pct.gt(0)
    changed=old.sell.ne(current.sell)
    assert len(current)==132
    assert changed.sum()==32
    assert old.sell[changed].ne(up[changed]).sum()==18
    assert current.sell[changed].ne(up[changed]).sum()==14
    assert current.sell.ne(up).sum()==62


def test_extended_candidate_retains_every_later_ipo_and_same_target():
    path=research.ROOT/'output/module2_extended_history_learning_20261023_predictions.csv'
    candidate=pd.read_csv(path).set_index('ipo_id')
    original=pd.read_csv(research.CURRENT).set_index('ipo_id')
    assert set(candidate.index)==set(original.index)
    assert not candidate.index.duplicated().any()
    assert np.allclose(candidate.future_return_pct,original.loc[candidate.index].day30_return_pct)
    assert candidate.correct.sum()==79
    report=json.loads(path.with_name('module2_extended_history_learning_20261023.json').read_text())
    assert report['final_examples']==2643
    assert report['final_issuers']==214
    assert report['live_promoted'] is False
    assert report['paired_comparisons']['vs_always_sell']['bootstrap_95_pp'][0]<0


def test_module1_fit_rejects_training_after_its_reference_year():
    frame=pd.read_csv(recency.ROOT/'output/module1_forward_coverage_20261024_features.csv')
    sample=frame[frame.year.eq(2024)].iloc[:2]
    with pytest.raises(ValueError,match='Future training row'):
        recency.fit(sample,recency.configs()[0],2023)
