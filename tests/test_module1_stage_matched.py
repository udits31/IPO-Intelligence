from datetime import date
import pandas as pd
import pytest
from ml.build_stage_matched_cohorts_20261028 import stage_for
from ml.test_stage_matched_models_20261028 import select_quote,fit_stage,evaluate,quote_event_date


def test_final_day_17_is_never_preclose_and_days_are_distinct():
    opened=date(2025,12,22);closed=date(2025,12,24)
    assert stage_for(1,'2025-12-22T17:00:00+05:30',opened,closed)=='DAY1_EOD'
    assert stage_for(2,'2025-12-23T17:00:00+05:30',opened,closed)=='DAY2_EOD'
    assert stage_for(3,'2025-12-24T16:59:59+05:30',opened,closed)=='CLOSING_PRE17'
    assert stage_for(3,'2025-12-24T17:00:00+05:30',opened,closed) is None
    assert stage_for(3,'2025-12-24T19:00:00+05:30',opened,closed) is None
    assert stage_for(1,'2025-12-23T17:00:00+05:30',opened,closed) is None
    assert stage_for(2,'2025-12-23T17:00:00',opened,closed) is None


def test_quote_selection_rejects_future_and_conflicting_values():
    quotes=pd.DataFrame({'timestamp':pd.to_datetime(['2025-12-22T15:00:00+05:30','2025-12-23T10:00:00+05:30']), 'gmp_rs':[5.,25.]})
    asof=pd.Timestamp('2025-12-22T17:00:00+05:30')
    assert select_quote(quotes,asof).gmp_rs==5
    assert select_quote(quotes,pd.Timestamp('2025-12-21T17:00:00+05:30')) is None
    conflict=pd.concat([quotes,quotes.iloc[[0]].assign(gmp_rs=8.)],ignore_index=True)
    assert select_quote(conflict,asof) is None


def test_stage_training_rejects_wrong_stage_or_test_issuer_before_fitting():
    train=pd.DataFrame({'ipo_id':[1],'year':[2025],'stage':['DAY1_EOD'],'listing_date':pd.to_datetime(['2025-12-20'])})
    held=pd.DataFrame({'ipo_id':[2],'year':[2026],'stage':['DAY2_EOD'],'decision_market_date':pd.to_datetime(['2026-01-01'])})
    with pytest.raises(ValueError,match='stage mismatch'):fit_stage(train,held,{})
    with pytest.raises(ValueError,match='leakage'):fit_stage(train,held.assign(ipo_id=1),{})


def test_missing_gmp_is_excluded_from_raw_sign_not_converted_to_zero():
    rows=pd.DataFrame({'target_binary':[1,0,1], 'gmp_pct':[10.,-1.,None]})
    result=evaluate(rows,[.8,.2,.8],[.7,.2,.7])
    assert result['model']['sample_size']==3
    assert result['raw_gmp']['sample_size']==2
    assert result['raw_gmp']['correct']==2


def test_quote_age_uses_event_date_not_conservative_availability_date():
    quote=pd.Series(dict(timestamp=pd.Timestamp('2025-12-23T00:00:00+05:30'),event_date=date(2025,12,22)))
    assert (date(2025,12,23)-quote_event_date(quote)).days==1
