"""Source-based gates using actual archived filings, not generated IPOs."""
import copy
import json

import pytest

from ml.recheck_module2_financial_comparatives import MANIFEST, REVIEWS, ROOT, recheck


def observed(ipo_id=248):
    if not MANIFEST.exists():
        pytest.skip('Real source-document archive is not installed')
    record = next(json.loads(s) for s in MANIFEST.read_text(encoding='utf-8').splitlines()
                  if json.loads(s)['ipo_id'] == ipo_id)
    financial = next(json.loads(s) for s in (ROOT / 'data/audit/module2_dated_financial_evidence_quarter_verified_20261022.jsonl').read_text(encoding='utf-8').splitlines()
                     if json.loads(s)['ipo_id'] == ipo_id)
    reviews = json.loads(REVIEWS.read_text(encoding='utf-8'))['reviews']
    return record, financial, reviews


@pytest.mark.parametrize('ipo_id', [248, 252, 256, 276, 282, 284, 286, 313, 352, 380, 381, 394, 401, 402, 441, 442, 450, 504, 511, 514, 515, 518, 523, 530, 535, 549, 557, 558, 577])
def test_visually_reviewed_real_tables_match_independent_quarter_source(ipo_id):
    record, financial, reviews = observed(ipo_id)
    result = recheck(record, financial, reviews)
    assert result['status'] == 'source_and_visual_checked'
    assert result['feature_accepted'] is True
    assert result['comparative_financials']['current_quarter_end'] == financial['financials']['period_end']


def test_merged_quarter_caption_does_not_turn_last_quarter_into_annual():
    record, financial, reviews = observed(284)
    values = recheck(record, financial, reviews)['comparative_financials']
    assert values['prior_year_quarter_end'] == '2020-06-30'
    assert values['prior_year_header']['kind'] == 'quarter'
    assert values['profit_prior_year_quarter_inr'] == pytest.approx(67100000.)


def test_real_prior_loss_keeps_its_sign_and_is_not_called_conventional_growth():
    record, financial, reviews = observed(276)
    values = recheck(record, financial, reviews)['comparative_financials']
    assert values['profit_prior_year_quarter_inr'] == -913800000.
    assert values['profit_change_scaled_to_abs_prior_pct'] > 0
    assert 'not a conventional growth rate' in values['profit_change_note']


def test_unlabelled_standalone_table_reports_metadata_authority_and_rejects_wrong_basis():
    from ml.financial_comparative_table import extract
    record, financial, reviews = observed(514)
    result = recheck(record, financial, reviews)
    assert 'PDF table title is unlabelled' in result['comparative_financials']['consolidation_basis_authority']
    ocr = json.loads((ROOT / result['raw_ocr_path']).read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='consolidation basis'):
        extract(ocr, {**financial['financials'], 'basis': 'Consolidated'})


def test_missing_visual_review_is_not_an_accepted_financial_feature():
    record, financial, _ = observed()
    result = recheck(record, financial, [])
    assert result['status'] == 'source_crosschecked_pending_visual_review'
    assert result['feature_accepted'] is False


def test_review_disagreement_or_missing_ocr_is_not_imputed(tmp_path):
    record, financial, reviews = observed()
    damaged = copy.deepcopy(reviews)
    next(r for r in damaged if r['ipo_id'] == record['ipo_id'])['observed_values']['profit_prior_year_quarter_inr'] *= -1
    result = recheck(record, financial, damaged)
    assert result['status'] == 'visual_review_values_conflict'
    assert result['feature_accepted'] is False
    missing = recheck(record, financial, reviews, cache=tmp_path)
    assert missing['status'] == 'raw_ocr_not_yet_available'
    assert missing['ipo_id'] == record['ipo_id']
    assert missing['feature_accepted'] is False


def test_real_bluestone_prior_revenue_ocr_error_is_not_accepted():
    record, financial, reviews = observed(564)
    # Exact archived extraction remains pinned even as the collector adds pages.
    path = ROOT / 'data/audit/financial_pdf_ocr' / (record['source_pdf_sha256'] + '_page5_batch.json')
    result = recheck(record, financial, reviews, only_extraction_path=path)
    assert result['status'] == 'visual_review_values_conflict'
    assert result['feature_accepted'] is False
    visual = next(r for r in reviews if r['ipo_id'] == 564)
    assert visual['observed_values']['revenue_prior_year_quarter_inr'] == 3482440000


def test_later_publication_and_wrong_checkpoint_cannot_enter_features():
    record, financial, reviews = observed()
    later = copy.deepcopy(record)
    later['selected_notice']['disseminated_at_ist'] = financial['decision_date'] + 'T16:00:00'
    with pytest.raises(ValueError, match='published after'):
        recheck(later, financial, reviews)
    with pytest.raises(ValueError, match='checkpoint differs'):
        recheck({**record, 'symbol': 'METROBRAND'}, financial, reviews)
