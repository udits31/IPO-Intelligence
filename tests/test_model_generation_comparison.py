import numpy as np
import pandas as pd
import pytest
from ml.compare_model_generations_20261028 import guard,return_metrics,paired


def frame(year,ids,names):
    return pd.DataFrame({'year':[year]*len(ids),'ipo_id':ids,'company_name':names,
      'listing_date':pd.to_datetime([f'{year}-07-01']*len(ids)),
      'decision_market_date':pd.to_datetime([f'{year}-06-25']*len(ids))})


def test_split_rejects_later_training_and_issuer_overlap():
    train=frame(2024,[1],['Example Ltd']);held=frame(2025,[2],['Other Ltd'])
    guard(train,held)
    with pytest.raises(ValueError,match='year'):guard(frame(2025,[1],['Example']),held)
    with pytest.raises(ValueError,match='ID leakage'):guard(train,frame(2025,[1],['Other']))
    with pytest.raises(ValueError,match='name leakage'):guard(train,frame(2025,[2],['Example Limited']))


def test_weekly_selection_uses_predictions_not_the_known_winner():
    rows=frame(2026,[1,2],['Winner','Loser']);rows['target_return']=[50.,-10.]
    result=return_metrics(rows,np.array([1.,5.]))
    selection=result['weekly_ranking']['groups'][0]
    assert selection['selected_id']=='2' and selection['selected_return']==-10.
    assert selection['best_return']==50.


def test_paired_comparison_retains_both_directions_of_disagreement():
    rows=frame(2026,[1,2,3,4],['A','B','C','D']);rows['target_binary']=[1,0,1,0]
    r=paired(rows,np.array([.8,.2,.8,.8]),np.array([.8,.8,.2,.2]))
    assert r['candidate_only_correct']==2 and r['baseline_only_correct']==1
    assert r['accuracy_difference_pp']==25.
