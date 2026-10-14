"""Actual embedded-PDF text and integrity checks; no invented confidence."""
import copy
import json

import pytest

from ml.financial_comparative_table import extract
from ml.financial_pdf_native import verify_native_tokens
from ml.extract_module2_native_financials import OUTPUT, ROOT, FINANCIAL


def observed():
    if not OUTPUT.exists():
        pytest.skip('Actual original native-PDF extraction is not installed')
    record = next(json.loads(s) for s in OUTPUT.read_text(encoding='utf-8').splitlines()
                  if json.loads(s)['ipo_id'] == 286)
    payload = json.loads((ROOT / record['raw_extraction_path']).read_text(encoding='utf-8'))
    financial = next(json.loads(s) for s in FINANCIAL.read_text(encoding='utf-8').splitlines()
                     if json.loads(s)['ipo_id'] == 286)['financials']
    return payload, financial


def test_original_text_is_replayed_and_does_not_fabricate_an_ocr_score():
    payload, financial = observed()
    verify_native_tokens(payload, ROOT / payload['source_pdf_path'])
    assert all(token['confidence'] is None for token in payload['tokens'])
    result = extract(payload, financial)
    assert result['extraction_method'] == 'embedded_pdf_text'
    assert result['current_quarter_header']['label_confidence'] is None
    assert result['revenue_current_inr'] == pytest.approx(financial['revenue_inr'])


def test_altered_embedded_text_or_fabricated_certainty_is_rejected():
    payload, _ = observed()
    changed = copy.deepcopy(payload)
    changed['tokens'][0]['text'] += ' altered'
    with pytest.raises(ValueError, match='differ from the original PDF'):
        verify_native_tokens(changed, ROOT / payload['source_pdf_path'])
    changed = copy.deepcopy(payload)
    changed['tokens'][0]['confidence'] = 1.0
    with pytest.raises(ValueError, match='differ from the original PDF'):
        verify_native_tokens(changed, ROOT / payload['source_pdf_path'])


def test_native_tokens_cannot_bypass_ocr_gates_by_hiding_their_origin():
    payload, financial = observed()
    changed = copy.deepcopy(payload)
    changed['extraction_method'] = 'rapidocr'
    with pytest.raises(ValueError, match='explicit native-source provenance'):
        extract(changed, financial)
