"""Real quarterly result source checks and unchanged-cohort rule tests."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.collect_module2_financial_evidence import extract_current_quarter
from ml.reparse_module2_financials import verify_detail
from ml.replay_module2_financial_rules import (EVIDENCE,archive_bytes,load_evidence,assemble_financial,
    financial_calls,select,GRID)
from ml.calibrate_module2_rules import DEFAULT,decisions


def elin():
    rows=load_evidence()
    return rows[313]


def test_duplicate_quarter_ytd_dates_require_exact_source_confirmation():
    row=elin();item=row['selected_filing']
    raw=archive_bytes(row['document_source'])
    with pytest.raises(ValueError,match='conflicting quarterly revenue'):
        extract_current_quarter(raw,item,'ELIN')
    quarter=extract_current_quarter(raw,item,'ELIN',confirmed_current_context='OneD')
    assert quarter['revenue_inr']==2022480000.
    assert quarter['profit_inr']==45850000.
    assert quarter['revenue_inr']!=8067050000.  # Actual YTD value in the same source is not substituted.
    detail=json.loads(archive_bytes(row['quarter_crosscheck']['source']))
    proof=verify_detail(detail,item,quarter)
    assert proof['detail_revenue']==20224.8
    assert proof['detail_profit']==458.5
    assert 'period' in proof['detail_metadata_from_selected_result']
    assert np.isclose(quarter['profit_margin_pct'],458.5/20224.8*100)


def test_matching_sequence_and_quarter_are_not_optional():
    row=elin();detail=json.loads(archive_bytes(row['quarter_crosscheck']['source']))
    with pytest.raises(ValueError,match='sequence'):
        verify_detail({**detail,'seqnum':'wrong'},row['selected_filing'],row['financials'])
    with pytest.raises(ValueError,match='quarterly'):
        verify_detail({**detail,'period':'Annual'},row['selected_filing'],row['financials'])


def test_financial_missingness_does_not_remove_ipos_or_create_zero_profits():
    data=assemble_financial('holdout',load_evidence())
    assert len(data)==132 and data.quarter_profit_margin_pct.notna().sum()==110
    missing=data.quarter_profit_margin_pct.isna().to_numpy()
    assert missing.sum()==22
    baseline=decisions(data,DEFAULT)
    for parameters in GRID:
        calls=financial_calls(data,parameters)
        assert len(calls)==132
        np.testing.assert_array_equal(calls[missing],baseline[missing])


def test_later_and_unmatured_labels_cannot_select_financial_settings():
    evidence=load_evidence()
    with pytest.raises(ValueError,match='matured 2021'):
        select(assemble_financial('holdout',evidence),'2024-12-31')
    development=assemble_financial('development',evidence)
    with pytest.raises(ValueError,match='matured 2021'):
        select(development,'2024-12-31')
    known=development[development.outcome_date.le(pd.Timestamp('2024-12-31'))]
    assert len(known)==214
    chosen,candidates=select(known,'2024-12-31')
    assert chosen in candidates and len(candidates)==11
