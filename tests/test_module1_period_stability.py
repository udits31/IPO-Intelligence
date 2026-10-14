import numpy as np
import pandas as pd
from ml.evaluate_module1_period_stability_20261028 import compare,scope_metrics


def test_single_class_small_quarter_has_no_invented_auc():
    rows=pd.DataFrame({'target_binary':[1,1]})
    result=scope_metrics(rows,np.array([.8,.4]))
    assert result['correct']==1 and result['roc_auc'] is None and result['balanced_accuracy_pct'] is None


def test_baselines_and_model_are_scored_on_identical_issuers():
    rows=pd.DataFrame({'year':[2025]*4,'target_binary':[0,0,1,1],
        'model_score':[.2,.8,.7,.8],'gmp_score':[.7,.8,.7,.8],
        'always_positive':[1.]*4,'gmp_pct':[-1,2,3,4]})
    result=compare(rows)
    assert all(result[k]['sample_size']==4 for k in ['model','gmp_only','always_positive','raw_gmp_positive_rule'])
    assert result['model']['correct']==3
    assert result['gmp_only']['correct']==2
    assert result['raw_gmp_positive_rule']['correct']==3
    assert result['paired_model_vs_gmp']['candidate_only_correct']==1
