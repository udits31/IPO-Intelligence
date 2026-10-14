import pandas as pd
import pytest
from ml.finalize_module1_stability_20261028 import select_and_confirm, POLICIES
from services.prospective_module1 import validate_direction_policy


def test_release_rejects_a_silent_cutoff_change():
    validate_direction_policy({'model':{'cutoff':.5}}, {'threshold':.5})
    for config,model in [(.625,.5),(.5,.625),(None,.5),(.5,None)]:
        with pytest.raises(ValueError,match='versioned release'):
            validate_direction_policy({'model':{'cutoff':config}}, {'threshold':model})


def test_selection_rejects_test_year_labels():
    with pytest.raises(ValueError,match='2024 selection'):
        select_and_confirm(pd.DataFrame({'year':[2024,2026]}))


def test_failed_confirmation_reverts_to_incumbent_without_test_results():
    rows=pd.DataFrame({'year':[2024]*4+[2025]*4,'target_binary':[0,0,1,1]*2,
        POLICIES[0]:[.4,.6,.7,.8,.4,.6,.7,.8],
        POLICIES[1]:[.1,.2,.8,.9,.8,.9,.2,.1],
        POLICIES[2]:[.4,.6,.7,.8,.4,.6,.7,.8]})
    result=select_and_confirm(rows)
    assert result['selection']==POLICIES[1]
    assert not result['confirmation_passed']
    assert result['locked_policy']==POLICIES[0]
