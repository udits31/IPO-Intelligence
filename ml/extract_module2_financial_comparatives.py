"""Source-check comparative earnings tables; never invent missing periods.

Raw OCR is generated in a separate dependency environment, leaving the live
predictor runtime unchanged. Successful tables still await visual review;
they are not silently used as accepted training data or accuracy results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pymupdf
from rapidocr import RapidOCR

from ml.financial_comparative_table import extract
from ml.financial_pdf_pages import candidate_pages
from ml.ocr_financial_pdf import run as source_ocr

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/audit/module2_original_financial_pdfs_20261022.jsonl'
FINANCIAL=ROOT/'data/audit/module2_dated_financial_evidence_quarter_verified_20261022.jsonl'
OUTPUT=ROOT/'data/audit/module2_comparative_financial_tables_20261022.jsonl'
REPORT=ROOT/'output/module2_comparative_financial_extraction_20261022.json'


def run(cohort='all'):
    # Capture the parser actually loaded at startup. A long OCR run must not
    # claim it used a newer on-disk parser edited while that run was alive.
    source_paths=(MANIFEST,FINANCIAL,Path(__file__),ROOT/'ml/financial_comparative_table.py',ROOT/'ml/ocr_financial_pdf.py',ROOT/'ml/financial_pdf_pages.py')
    output,report_path=OUTPUT,REPORT
    if cohort!='all':
        if cohort not in ('development','later'):
            raise ValueError('Unknown collection cohort')
        # This only partitions source collection. No later outcome value is
        # used, no table is accepted without the same review gate, and the
        # combined audit retains every original evaluation case.
        ledger=ROOT/('output/module2_development_decision_ledger_goal_corporate_actions_20261021.csv'
                     if cohort=='development' else
                     'output/module2_holdout_decision_ledger_goal_corporate_actions_20261021.csv')
        with ledger.open(encoding='utf-8',newline='') as stream:
            selected_ids={int(row['ipo_id']) for row in csv.DictReader(stream)}
        source_paths=(*source_paths,ledger)
        output=OUTPUT.with_name(OUTPUT.stem+'_'+cohort+OUTPUT.suffix)
        report_path=REPORT.with_name(REPORT.stem+'_'+cohort+REPORT.suffix)
    startup_hashes={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    parser_sha=startup_hashes['ml/financial_comparative_table.py']
    files=[json.loads(line) for line in MANIFEST.read_text(encoding='utf-8').splitlines()]
    if cohort!='all':
        files=[row for row in files if row['ipo_id'] in selected_ids]
    financial={r['ipo_id']:r for r in (json.loads(line) for line in FINANCIAL.read_text(encoding='utf-8').splitlines())}
    previous={}
    prior_path=output if output.exists() else OUTPUT
    if prior_path.exists():
        previous={r['ipo_id']:r for r in (json.loads(line) for line in prior_path.read_text(encoding='utf-8').splitlines())}
    engine=RapidOCR(params={'EngineConfig.onnxruntime.intra_op_num_threads':2,
        'EngineConfig.onnxruntime.inter_op_num_threads':2,'Global.log_level':'warning'})
    rows=[]
    for index,record in enumerate(files,1):
        if record['status']!='verified_original_pdf':
            rows.append({'ipo_id':record['ipo_id'],'status':record['status'],'feature_accepted':False})
            continue
        row=financial[record['ipo_id']]
        path=(ROOT/record['source_pdf_path']).resolve()
        path.relative_to((ROOT/'data/archive/module2_financials/documents').resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['source_pdf_sha256']:
            raise ValueError('Original PDF changed since its source download')
        prior=previous.get(record['ipo_id'])
        if (prior and prior.get('source_pdf_sha256')==record['source_pdf_sha256'] and
            prior.get('parser_sha256')==parser_sha and
            prior.get('financial_input_sha256')==startup_hashes[FINANCIAL.relative_to(ROOT).as_posix()] and
            prior.get('status')=='source_crosschecked_pending_visual_review'):
            rows.append(prior);continue
        result={'ipo_id':record['ipo_id'],'symbol':record['symbol'],'decision_date':record['decision_date'],
            'source_pdf_path':record['source_pdf_path'],'source_pdf_sha256':record['source_pdf_sha256'],
            'source_url':record['source_url'],'financial_published_at_ist':row['financials']['published_at_ist'],
            'parser_sha256':parser_sha,'financial_input_sha256':startup_hashes[FINANCIAL.relative_to(ROOT).as_posix()],
            'status':'comparative_table_not_validated','feature_accepted':False,'page_attempts':[]}
        with pymupdf.open(path) as document:
            pages=candidate_pages(document,row['financials']['basis'])
        for page_number in pages:
            ocr_path=ROOT/'data/audit/financial_pdf_ocr'/f"{record['source_pdf_sha256']}_page{page_number}_batch.json"
            if ocr_path.exists():
                ocr=json.loads(ocr_path.read_text(encoding='utf-8'))
            else:
                ocr=source_ocr(path,page_number,engine=engine,quiet=True,artifact_tag='batch')
            try:
                values=extract(ocr,row['financials'])
            except (ValueError,ZeroDivisionError) as exc:
                result['page_attempts'].append({'page_number':page_number,'reason':str(exc)})
                continue
            result.update(status='source_crosschecked_pending_visual_review',page_number=page_number,
                comparative_financials=values,raw_ocr_path=ocr_path.relative_to(ROOT).as_posix(),
                raw_ocr_sha256=hashlib.sha256(ocr_path.read_bytes()).hexdigest(),render_path=ocr['page_render_path'])
            break
        rows.append(result)
        output.write_text('\n'.join(json.dumps(item) for item in rows)+'\n',encoding='utf-8')
        print(json.dumps({'case':index,'total':len(files),'ipo_id':record['ipo_id'],'status':result['status'],
            'pending_visual_review':sum(r['status']=='source_crosschecked_pending_visual_review' for r in rows)}),flush=True)
    output.write_text('\n'.join(json.dumps(item) for item in rows)+'\n',encoding='utf-8')
    from collections import Counter
    report={'collection_partition':cohort,'source_cases':len(files),'statuses':dict(Counter(r['status'] for r in rows)),
        'features_accepted_for_training':0,'prediction_model_changed':False,
        'input_sha256':startup_hashes,
        'source_code_changed_during_run':[path.relative_to(ROOT).as_posix() for path in source_paths
            if hashlib.sha256(path.read_bytes()).hexdigest()!=startup_hashes[path.relative_to(ROOT).as_posix()]],
        'output_path':output.relative_to(ROOT).as_posix(),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'next_gate':'Visually review every accepted source table and verify same-quarter columns before any prediction experiment.',
        'limitations':['Conservative parser may reject valid but unsupported layouts; absent features do not remove IPOs from future evaluations.',
           'OCR confidence is not independent assurance of numeric correctness.',
           'Earnings changes are not consensus surprises and do not establish predictive value by themselves.']}
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=('all','development','later'),default='all')
    run(parser.parse_args().cohort)
