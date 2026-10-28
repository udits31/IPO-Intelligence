"""Recover comparative financial tables from original PDF text, not OCR guesses.

Run with the local Python containing PyMuPDF. Only source-matching candidates
are rendered/cached, and every candidate still requires full-page visual review
before use. Image-only or garbled PDF text remains for the existing OCR jobs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import fitz

from ml.financial_comparative_table import extract
from ml.financial_pdf_native import METHOD, native_tokens
from ml.financial_pdf_pages import candidate_pages

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/audit/module2_original_financial_pdfs_20261022.jsonl'
FINANCIAL = ROOT / 'data/audit/module2_dated_financial_evidence_quarter_verified_20261022.jsonl'
OUTPUT = ROOT / 'data/audit/module2_native_comparative_candidates_20261022.jsonl'
REPORT = ROOT / 'output/module2_native_financial_extraction_20261022.json'


def run():
    files = [json.loads(s) for s in MANIFEST.read_text(encoding='utf-8').splitlines()]
    financial = {r['ipo_id']: r for r in (json.loads(s) for s in FINANCIAL.read_text(encoding='utf-8').splitlines())}
    helper = ROOT / 'ml/financial_pdf_native.py'
    rows = []
    for index, record in enumerate(files, 1):
        result = {'ipo_id': record['ipo_id'], 'status': record['status'], 'feature_accepted': False}
        if record['status'] != 'verified_original_pdf':
            rows.append(result)
            continue
        path = (ROOT / record['source_pdf_path']).resolve()
        path.relative_to((ROOT / 'data/archive/module2_financials/documents').resolve())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != record['source_pdf_sha256']:
            raise ValueError('Original financial PDF hash changed')
        source = financial[record['ipo_id']]['financials']
        result.update(symbol=record['symbol'], status='native_text_not_validated', page_attempts=[])
        with fitz.open(path) as document:
            for page_number in candidate_pages(document, source['basis']):
                page = document[page_number - 1]
                payload = {'extraction_method': METHOD, 'source_pdf_path': record['source_pdf_path'],
                           'source_pdf_sha256': sha, 'page_number': page_number,
                           'text_extractor_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
                           'pymupdf_version': fitz.VersionBind, 'tokens': native_tokens(page),
                           'confidence_note': 'Exact embedded PDF text; no OCR score exists or is fabricated. Font maps may be wrong, so source parity and visual review remain mandatory.'}
                try:
                    values = extract(payload, source)
                except (ValueError, ZeroDivisionError) as exc:
                    result['page_attempts'].append({'page_number': page_number, 'reason': str(exc)})
                    continue
                render = ROOT / f'data/archive/module2_financials/ocr_renders/{sha}_page{page_number}_native.png'
                render.parent.mkdir(parents=True, exist_ok=True)
                image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image.save(render)
                payload.update(page_render_path=render.relative_to(ROOT).as_posix(),
                               render_sha256=hashlib.sha256(render.read_bytes()).hexdigest())
                cache = ROOT / f'data/audit/financial_pdf_ocr/{sha}_page{page_number}_native.json'
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
                result.update(status='native_source_crosschecked_pending_visual_review',
                              page_number=page_number, comparative_financials=values,
                              raw_extraction_path=cache.relative_to(ROOT).as_posix(),
                              raw_extraction_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                              render_path=payload['page_render_path'])
                break
        rows.append(result)
        if index % 20 == 0 or index == len(files):
            OUTPUT.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
            print(json.dumps({'processed': index, 'total': len(files),
                              'statuses': dict(Counter(row['status'] for row in rows))}), flush=True)
    OUTPUT.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    report = {'cases': len(rows), 'statuses': dict(Counter(row['status'] for row in rows)),
              'features_accepted_for_training': 0, 'synthetic_training_rows': 0, 'live_model_changed': False,
              'input_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (MANIFEST, FINANCIAL, helper, Path(__file__),
                                            ROOT / 'ml/financial_comparative_table.py', ROOT / 'ml/financial_pdf_pages.py')},
              'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              'limitations': ['Native text is not an OCR score and may contain faulty font maps.',
                              'Every candidate must be replayed from the original PDF and visually reviewed before becoming an input.',
                              'Unsupported layouts remain missing and do not remove IPOs from performance tests.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run()
