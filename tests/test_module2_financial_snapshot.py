"""Pin an actual accepted native/PDF snapshot while alternative OCR arrives."""
import copy
import json

import pytest

from ml.recheck_module2_financial_comparatives import ROOT, MANIFEST, REVIEWS, recheck
from ml.replay_module2_financial_rules import EVIDENCE as QUARTERS
from ml.replay_module2_earnings_rules import verify_comparative_snapshot


def native_snapshot():
    native=ROOT/'data/audit/financial_pdf_ocr/92b546ec97e50cc47c521a36447e901c942886fbcde310c78afc43dfd1eec2ad_page8_native.json'
    if not native.exists():
        pytest.skip('Actual native Indiqube source extraction is not installed')
    record=next(json.loads(s) for s in MANIFEST.read_text(encoding='utf-8').splitlines() if json.loads(s)['ipo_id']==577)
    quarter=next(json.loads(s) for s in QUARTERS.read_text(encoding='utf-8').splitlines() if json.loads(s)['ipo_id']==577)
    reviews=json.loads(REVIEWS.read_text(encoding='utf-8'))['reviews']
    snapshot=recheck(record,quarter,reviews,only_extraction_path=native)
    return snapshot,quarter,record,reviews


def test_actual_native_snapshot_stays_valid_when_equivalent_ocr_cache_exists():
    snapshot,quarter,record,reviews=native_snapshot()
    assert snapshot['feature_accepted'] and snapshot['extraction_method']=='embedded_pdf_text'
    verified=verify_comparative_snapshot(snapshot,quarter,record,reviews)
    assert verified['raw_ocr_path']==snapshot['raw_ocr_path']
    assert verified['comparative_financials']['profit_current_inr']==-367550000.


def test_audited_snapshot_cannot_change_its_values_or_source_hash():
    snapshot,quarter,record,reviews=native_snapshot()
    changed=copy.deepcopy(snapshot)
    changed['comparative_financials']['profit_current_inr']*=-1
    with pytest.raises(ValueError,match='frozen raw source'):
        verify_comparative_snapshot(changed,quarter,record,reviews)
    changed=copy.deepcopy(snapshot)
    changed['raw_ocr_sha256']='0'*64
    with pytest.raises(ValueError,match='frozen raw source'):
        verify_comparative_snapshot(changed,quarter,record,reviews)
