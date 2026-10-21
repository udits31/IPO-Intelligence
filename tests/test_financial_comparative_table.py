"""Use the complete, visually inspected original ELIN result table."""
import copy
import json

import numpy as np
import pytest

from ml.financial_comparative_table import OCR,FINANCIALS,extract


def observed():
    if not OCR.exists():
        pytest.skip('Actual original PDF page OCR is not installed')
    ocr=json.loads(OCR.read_text(encoding='utf-8'))
    row=next(json.loads(line) for line in FINANCIALS.read_text(encoding='utf-8').splitlines() if json.loads(line)['ipo_id']==313)
    return ocr,row['financials']


def test_actual_year_earlier_quarter_not_ytd_or_previous_quarter():
    ocr,financial=observed()
    result=extract(ocr,financial)
    assert result['current_quarter_end']=='2022-12-31'
    assert result['prior_year_quarter_end']=='2021-12-31'
    assert result['revenue_prior_year_quarter_inr']==2585230000.
    assert result['profit_prior_year_quarter_inr']==69320000.
    assert result['revenue_prior_year_quarter_inr']!=7767690000.  # Actual YTD comparison is not a quarter.
    assert np.isclose(result['revenue_yoy_change_pct'],-21.767889123985096)
    assert np.isclose(result['profit_change_scaled_to_abs_prior_pct'],-33.857472590882864)


def test_low_confidence_or_mismatched_current_value_does_not_become_a_feature():
    ocr,financial=observed()
    corrupted=copy.deepcopy(ocr)
    match=next(token for token in corrupted['tokens'] if token['text']=='2,585.23')
    match['confidence']=.5
    with pytest.raises(ValueError,match='Prior-year PDF revenue'):
        extract(corrupted,financial)
    with pytest.raises(ValueError,match='independent XBRL'):
        extract(ocr,{**financial,'revenue_inr':financial['revenue_inr']+1000000})


def test_missing_quarter_header_or_wrong_basis_is_rejected():
    ocr,financial=observed()
    corrupted=copy.deepcopy(ocr)
    corrupted['tokens']=[token for token in corrupted['tokens'] if token['text']!='Quarter Ended']
    with pytest.raises(ValueError,match='Quarter headings'):
        extract(corrupted,financial)
    with pytest.raises(ValueError,match='consolidation basis'):
        extract(ocr,{**financial,'basis':'Standalone'})
