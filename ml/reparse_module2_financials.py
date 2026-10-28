"""Resolve defective NSE WEB-XBRL quarter/YTD contexts using dated details.

Some real WEB exports give OneD and FourD identical dates despite different
quarter and year-to-date values. We do not choose the better-looking number.
The selected filing's exact sequence detail must confirm the current-quarter
profit/revenue ratio. Original responses and the old extraction are preserved.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from ml.collect_module2_financial_evidence import (ROOT,OUTPUT as ORIGINAL,PAGE,HEADERS,
    source_get,AccessRestricted,extract_current_quarter)

DETAIL='https://www.nseindia.com/api/corporates-financial-results-data'
OUTPUT=ROOT/'data/audit/module2_dated_financial_evidence_quarter_verified_20261022.jsonl'
REPORT=ROOT/'output/module2_financial_quarter_recheck_20261022.json'


def verify_detail(detail,item,financials):
    if str(detail.get('seqnum'))!=str(item['seqNumber']):
        raise ValueError('Financial detail sequence differs from selected filing')
    if (detail.get('period') or item.get('period'))!='Quarterly' or str(
            detail.get('cumlativeNonCumlative') or item.get('cumulative')).casefold()!='non-cumulative':
        raise ValueError('Detail does not identify a non-cumulative quarterly result')
    if pd.to_datetime(detail.get('periodEndDT'),format='%d-%b-%Y')!=pd.Timestamp(item['period_end']):
        raise ValueError('Detail period differs from selected filing')
    if detail.get('conNonCon') is not None and str(detail['conNonCon']).casefold()!=str(item.get('consolidated')).casefold():
        raise ValueError('Consolidation basis differs between financial documents')
    data=detail.get('resultsData2') or {}
    revenue=float(data['re_net_sale'])
    # This explicit field is profit including discontinued operations, not
    # EPS, comprehensive income, or a selected segment's profit.
    profit=float(data['dis_opr_aftr_tax_plus_ord_act'])
    if not np.isfinite([revenue,profit]).all() or revenue<=0:
        raise ValueError('Current-quarter detail values unavailable')
    if not np.isclose(profit/revenue*100,financials['profit_margin_pct'],atol=.001):
        raise ValueError('Current-quarter JSON and XML profit/revenue ratio conflict')
    return {'detail_revenue':revenue,'detail_profit':profit,
            'detail_revenue_key':'re_net_sale','detail_profit_key':'dis_opr_aftr_tax_plus_ord_act',
            'detail_metadata_from_selected_result':[key for key in ('period','cumlativeNonCumlative','conNonCon') if detail.get(key) is None],
            'check':'Same filing sequence, quarter, consolidation basis and profit/revenue ratio; XML retains its explicit INR values. Missing detail labels use the exact sequence metadata, never invented labels.'}


def run():
    old=[json.loads(line) for line in ORIGINAL.read_text(encoding='utf-8').splitlines()]
    previous={}
    if OUTPUT.exists():
        previous={row['ipo_id']:row for row in (json.loads(line) for line in OUTPUT.read_text(encoding='utf-8').splitlines())}
    session=requests.Session();session.headers.update(HEADERS)
    session.get(PAGE,timeout=20).raise_for_status()
    cached={}
    for path in (ROOT/'data/archive/module2_financials').glob('*.gz'):
        try:
            with gzip.open(path,'rb') as stream:
                content=stream.read()
            if content.startswith(b'{'):
                detail=json.loads(content)
                if detail.get('seqnum') and detail.get('resultsData2'):
                    cached[str(detail['seqnum'])]=(detail,path,hashlib.sha256(content).hexdigest())
        except (OSError,ValueError):
            continue
    rows=[];stop=None
    for row in old:
        row=dict(row)
        prior=previous.get(row['ipo_id'])
        if prior and prior.get('quarter_crosscheck'):
            rows.append(prior);continue
        if row['status']=='document_unusable' and row.get('error')=='Missing or conflicting quarterly revenue':
            item=row['selected_filing']
            if item.get('seqNumber'):
                source=row['document_source']
                with gzip.open(ROOT/source['raw_archive_path'],'rb') as stream:
                    raw=stream.read()
                if hashlib.sha256(raw).hexdigest()!=source['raw_sha256']:
                    raise ValueError('Original financial XML archive changed')
                try:
                    metrics=extract_current_quarter(raw,item,row['symbol'],confirmed_current_context='OneD')
                    params={
                        'index':'equities','params':item['params'],'seq_id':item['seqNumber'],
                        'industry':item['industry'],'frOldNewFlag':item['oldNewFlag'],
                        'ind':item['reInd'],'format':item['format']}
                    if str(item['seqNumber']) in cached:
                        detail,path,digest=cached[str(item['seqNumber'])]
                        detail_source={'raw_archive_path':path.relative_to(ROOT).as_posix(),'raw_sha256':digest,
                            'source_url':DETAIL+'?'+urlencode(params),'source_url_is_reconstructed_query':True,
                            'note':'Original saved response reused offline by exact filing sequence; original receipt clock was not preserved for this earlier rejected extraction.'}
                    else:
                        response,detail_source=source_get(session,DETAIL,params)
                        detail=response.json()
                    row['quarter_detail_attempt_source']=detail_source
                    proof=verify_detail(detail,item,metrics)
                    row.update(status='verified_quarterly_metrics',financials=metrics,error=None,
                               quarter_crosscheck={**proof,'source':detail_source},
                               extraction_note='WEB-XBRL duplicate quarter/YTD dates resolved only after exact current-quarter detail confirmation.')
                except AccessRestricted as exc:
                    stop=str(exc)
                    row['quarter_recheck_error']=stop
                except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
                    row['quarter_recheck_error']=str(exc)
                time.sleep(.25)
        rows.append(row)
        if len(rows)%25==0:
            print(f"{len(rows)}/{len(old)} inspected; {sum(bool(r.get('quarter_crosscheck')) for r in rows)} exact-quarter recoveries",flush=True)
            OUTPUT.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
        if stop:
            rows.extend(old[len(rows):]);break
    OUTPUT.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    report={'original_snapshot_sha256':hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        'reprocessed_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'cases':len(rows),
        'original_verified':sum(r['status']=='verified_quarterly_metrics' for r in old),
        'verified_after_recheck':sum(r['status']=='verified_quarterly_metrics' for r in rows),
        'recovered_from_exact_quarter_detail':sum(bool(r.get('quarter_crosscheck')) for r in rows),
        'statuses':pd.Series([r['status'] for r in rows]).value_counts().to_dict(),
        'stopped_on_access_restriction':stop,
        'rule':'No YTD/annual period used as a quarter; exact NSE detail and raw XML must agree. Missing values remain missing.',
        'limitations':['Retrospective source retrieval; latest served filing attachments may not be immutable historical snapshots.',
                       'No earnings surprise, forward guidance or year-over-year growth is inferred.']}
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    run()
