"""Reparse archived OCR offline and gate financial features on source review.

The OCR collector may keep running: this process reads complete cached JSON
pages and writes a different, all-case audit. No missing IPO is removed, no
current-quarter dates are inferred, and unchecked OCR is not training data.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from ml.financial_comparative_table import extract
from ml.replay_module2_financial_rules import load_evidence

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/audit/module2_original_financial_pdfs_20261022.jsonl'
REVIEWS = ROOT / 'data/audit/module2_financial_table_visual_reviews_20261022.json'
CACHE = ROOT / 'data/audit/financial_pdf_ocr'
OUTPUT = ROOT / 'data/audit/module2_comparative_financials_reviewed_20261022.jsonl'
REPORT = ROOT / 'output/module2_financial_comparative_recheck_20261022.json'
VALUE_FIELDS = ('current_quarter_end', 'prior_year_quarter_end', 'basis',
                'unit_multiplier_inr', 'revenue_current_inr',
                'revenue_prior_year_quarter_inr', 'profit_current_inr',
                'profit_prior_year_quarter_inr')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(relative, parents):
    path = (ROOT / relative).resolve()
    if not any(path.is_relative_to(parent.resolve()) for parent in parents):
        raise ValueError('Source path is outside its expected archive')
    return path


def equal_values(actual, expected):
    if any(key not in expected for key in VALUE_FIELDS):
        return False
    def comparable(key, value):
        if key == 'basis' and str(value).casefold() in ('standalone', 'non-consolidated'):
            return 'standalone'
        return value
    return all(math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=.01)
               if isinstance(actual[key], (int, float)) else
               comparable(key, actual[key]) == comparable(key, expected[key])
               for key in VALUE_FIELDS)


def profit_comparison_qualification(values,review):
    before_tax=review.get('reviewed_profit_before_tax_inr')
    if before_tax is None:
        return {'eligible_for_directional_event':True,'status':'reported_PAT_only_no_explicit_before_tax_direction_check'}
    current,prior=before_tax['current'],before_tax['prior_year_quarter']
    if not all(isinstance(value,(int,float)) and math.isfinite(value) for value in (current,prior)):
        raise ValueError('Reviewed before-tax facts are not finite source amounts')
    pat_change=values['profit_current_inr']-values['profit_prior_year_quarter_inr']
    opposite=(current-prior)*pat_change<0
    return {'eligible_for_directional_event':not opposite,
            'status':'before_and_after_tax_changes_conflict' if opposite else 'reviewed_before_and_after_tax_directions_agree',
            'reviewed_before_tax_facts_inr':before_tax,
            'note':'Reported PAT is unchanged. Conflicting directions qualify the comparative event, not the IPO, current quarter loss or price history. No adjusted profit is inferred.'}


def recheck(record, financial, reviews, cache=CACHE, *, only_extraction_path=None):
    result = {key: record[key] for key in ('ipo_id', 'symbol', 'decision_date')}
    result.update(status=record['status'], feature_accepted=False)
    if record['status'] != 'verified_original_pdf':
        return result
    if financial['status'] != 'verified_quarterly_metrics':
        raise ValueError('Original PDF has no verified current-quarter source')
    if any(record[key] != financial[key] for key in ('ipo_id', 'symbol', 'decision_date')):
        raise ValueError('PDF and XBRL security or decision checkpoint differs')
    cutoff = datetime.fromisoformat(financial['cutoff_ist'])
    if any(datetime.fromisoformat(value) > cutoff for value in (
        financial['financials']['published_at_ist'], record['selected_notice']['disseminated_at_ist'])):
        raise ValueError('Financial evidence was published after the decision')
    sha = record['source_pdf_sha256']
    if not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('Invalid source PDF archive identity')
    pdf = source_path(record['source_pdf_path'], [ROOT / 'data/archive/module2_financials/documents'])
    if digest(pdf) != sha:
        raise ValueError('Original PDF hash changed')
    result.update(source_pdf_path=record['source_pdf_path'], source_pdf_sha256=sha,
                  source_url=record['source_url'],
                  financial_published_at_ist=financial['financials']['published_at_ist'],
                  status='raw_ocr_not_yet_available', page_attempts=[])
    matches = []
    if only_extraction_path is None:
        paths=sorted(cache.glob(sha + '_page*.json'))
    else:
        pinned=source_path(only_extraction_path,[cache])
        if not pinned.name.startswith(sha+'_page') or pinned.suffix!='.json':
            raise ValueError('Pinned extraction is not a cache for the selected PDF')
        paths=[pinned]
    for path in paths:
        try:
            ocr = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            # The live worker may still be writing this page; do not restart
            # it or label its temporarily incomplete file a source failure.
            result['page_attempts'].append({'raw_ocr_path': path.relative_to(ROOT).as_posix(),
                                            'reason': 'page_cache_write_in_progress'})
            continue
        if ocr['source_pdf_sha256'] != sha or source_path(ocr['source_pdf_path'], [pdf.parent]) != pdf:
            raise ValueError('OCR does not belong to the selected original PDF')
        render = source_path(ocr['page_render_path'], [ROOT / 'data/archive/module2_financials/ocr_renders', ROOT / 'tmp/pdfs'])
        if digest(render) != ocr['render_sha256']:
            raise ValueError('Source page render changed since OCR')
        if ocr.get('extraction_method')=='embedded_pdf_text':
            from ml.financial_pdf_native import verify_native_tokens
            verify_native_tokens(ocr,pdf)
        try:
            values = extract(ocr, financial['financials'])
        except (ValueError, ZeroDivisionError) as exc:
            result['page_attempts'].append({'page_number': ocr['page_number'], 'reason': str(exc)})
            continue
        matches.append((path, ocr, values))
    if not matches:
        if result['page_attempts']:
            result['status'] = 'comparative_table_not_validated'
        return result
    if any(not equal_values(values, matches[0][2]) for _, _, values in matches):
        result['status'] = 'conflicting_comparative_tables_require_review'
        return result
    path, ocr, values = matches[0]
    result.update(status='source_crosschecked_pending_visual_review',
                  page_number=ocr['page_number'], comparative_financials=values,
                  raw_ocr_path=path.relative_to(ROOT).as_posix(), raw_ocr_sha256=digest(path),
                  extraction_method=ocr.get('extraction_method','rapidocr'),
                  render_path=ocr['page_render_path'], render_sha256=ocr['render_sha256'])
    reviews_for_page = [r for r in reviews if r['ipo_id'] == record['ipo_id'] and
                        r['source_pdf_sha256'] == sha and r['page_number'] == ocr['page_number']]
    if len(reviews_for_page) > 1:
        raise ValueError('Multiple visual reviews for one source table')
    if reviews_for_page:
        review = reviews_for_page[0]
        if not equal_values(values, review['observed_values']):
            result['status'] = 'visual_review_values_conflict'
            return result
        result.update(status='source_and_visual_checked', feature_accepted=True,
                      visual_review=review,
                      profit_comparison_qualification=profit_comparison_qualification(values,review))
    return result


def run():
    financial = load_evidence()  # Verify every current-quarter raw source too.
    manifest = [json.loads(s) for s in MANIFEST.read_text(encoding='utf-8').splitlines()]
    if len({r['ipo_id'] for r in manifest}) != len(manifest):
        raise ValueError('Duplicate PDF-source checkpoint')
    reviews = json.loads(REVIEWS.read_text(encoding='utf-8'))['reviews']
    by_id = {r['ipo_id']: r for r in manifest}
    rows = []
    for ipo_id, row in sorted(financial.items()):
        if ipo_id in by_id:
            rows.append(recheck(by_id[ipo_id], row, reviews))
        else:
            rows.append({key: row[key] for key in ('ipo_id', 'symbol', 'decision_date')} |
                        {'status': row['status'], 'feature_accepted': False})
    OUTPUT.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    report = {'all_unchanged_cases': len(rows), 'statuses': dict(Counter(r['status'] for r in rows)),
              'accepted_source_tables': sum(r['feature_accepted'] for r in rows),
              'source_collection_still_running': 'This is an offline cache snapshot, not a collector completion claim.',
              'training_or_live_policy_changed': False, 'synthetic_training_rows': 0,
              'input_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in
                               (MANIFEST, REVIEWS, Path(__file__), ROOT / 'ml/financial_comparative_table.py',
                                ROOT / 'data/audit/module2_dated_financial_evidence_quarter_verified_20261022.jsonl',
                                ROOT / 'ml/collect_module2_financial_evidence.py',
                                ROOT / 'ml/reparse_module2_financials.py',
                                ROOT / 'ml/replay_module2_financial_rules.py',
                                ROOT / 'ml/financial_pdf_native.py',
                                ROOT / 'ml/extract_module2_native_financials.py')},
              'output_sha256': digest(OUTPUT),
              'limitations': ['Visual review is assistant inspection, not an independent human audit.',
                              'Prior-year comparatives are as presented in the dated current filing and may be restated.',
                              'Missing features retain original rules in any later experiment; IPOs are not removed.',
                              'Coverage alone does not establish predictive improvement.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
