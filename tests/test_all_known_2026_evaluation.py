from ml.evaluate_all_known_2026 import metrics,noncorporate


def test_trust_scope_does_not_exclude_normal_corporate_names():
    assert noncorporate('Cube Highways Trust')
    assert noncorporate('Bagmane Prime Office REIT')
    assert not noncorporate('SBI Funds Management Ltd.')
    assert not noncorporate('Asset Reconstruction Co.(India) Ltd.')


def test_pending_outcomes_not_counted_as_wrong_and_gmp_uses_same_pairs():
    rows=[dict(correct=True,actual_positive=True,gmp_correct=False),
          dict(correct=False,actual_positive=False,gmp_correct=True),
          dict(correct=True,actual_positive=True,gmp_correct=None),
          dict(correct=None,actual_positive=None,gmp_correct=None)]
    result=metrics(rows)
    assert result['records']==4 and result['evaluated']==3
    assert result['correct']==2 and result['always_positive_correct']==2
    assert result['gmp_paired']==2 and result['gmp_correct']==1
    assert result['model_correct_on_gmp_pairs']==1
