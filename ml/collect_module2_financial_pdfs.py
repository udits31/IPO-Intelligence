"""Archive original quarterly-result PDFs for comparative earnings extraction.

Select source documents by issuer, explicit reporting period and publication
cutoff, not by their later stock return. Original board-meeting outcomes often
contain the results even when the announcement subject is not 'Financial'.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import re
import time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from ml.collect_module2_financial_evidence import ROOT,HEADERS,AccessRestricted
from ml.reparse_module2_financials import OUTPUT as FINANCIAL

ANNOUNCEMENTS=ROOT/'data/audit/module2_dated_nse_filings_exchange_decisions.jsonl'
OUTPUT=ROOT/'data/audit/module2_original_financial_pdfs_20261022.jsonl'
REPORT=ROOT/'output/module2_financial_pdf_collection_20261022.json'
DIRECTORY=ROOT/'data/archive/module2_financials/documents'


def mentions_period(text,stamp):
    stamp=pd.Timestamp(stamp)
    month=stamp.strftime('%B');short=stamp.strftime('%b')
    day=str(stamp.day);year=str(stamp.year)
    return bool(re.search(rf'\b(?:{day}(?:st|nd|rd|th)?[\s.,/-]*(?:{month}|{short})[\s.,/-]*{year}|'
                          rf'(?:{month}|{short})\s*{day}(?:st|nd|rd|th)?[\s.,/-]*{year}|'
                          rf'{day}[./-]0?{stamp.month}[./-]{year})\b',text,re.I))


def candidates(row,announcements):
    cutoff=pd.Timestamp(row['cutoff_ist'])
    result=[]
    for notice in announcements.get(row['ipo_id'],{}).get('filings',[]):
        text=str(notice.get('details') or '')
        subject=str(notice.get('subject') or '')
        if (pd.Timestamp(notice['disseminated_at_ist'])>cutoff or
                not re.search(r'financial (?:results|statements)',text,re.I) or
                not re.search(r'submitted|approved',text,re.I) or
                not mentions_period(text,row['financials']['period_end'])):
            continue
        url=str(notice.get('attachment_url') or '')
        if urlparse(url).hostname not in ('nsearchives.nseindia.com','archives.nseindia.com') or not urlparse(url).path.lower().endswith('.pdf'):
            continue
        result.append({**notice,'source_announcement_subject':subject})
    # Prefer explicit financial-result submission over merely a board outcome.
    # Never select by a future price or by a desired growth sign.
    return sorted(result,key=lambda notice:(bool(re.search(r'has submitted.*financial results',str(notice.get('details')),re.I)),
                                             notice['disseminated_at_ist']),reverse=True)


def run():
    financial=[json.loads(line) for line in FINANCIAL.read_text(encoding='utf-8').splitlines()]
    eligible=[row for row in financial if row['status']=='verified_quarterly_metrics']
    announcements={row['ipo_id']:row for row in (json.loads(line) for line in ANNOUNCEMENTS.read_text(encoding='utf-8').splitlines())}
    previous={}
    if OUTPUT.exists():
        previous={row['ipo_id']:row for row in (json.loads(line) for line in OUTPUT.read_text(encoding='utf-8').splitlines())}
    session=requests.Session();session.headers.update(HEADERS)
    rows=[];stop=None
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(eligible,1):
        prior=previous.get(row['ipo_id'])
        if prior and prior.get('status')=='verified_original_pdf' and prior.get('decision_date')==row['decision_date']:
            path=ROOT/prior['source_pdf_path']
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==prior['source_pdf_sha256']:
                rows.append(prior);continue
        result={'ipo_id':row['ipo_id'],'symbol':row['symbol'],'decision_date':row['decision_date'],
                'financial_period_end':row['financials']['period_end'],'consolidation_basis':row['financials']['basis'],
                'status':'no_matching_source_attachment','source_candidates':[]}
        found=candidates(row,announcements)
        result['source_candidates']=found
        if found:
            notice=found[0]
            try:
                response=session.get(notice['attachment_url'],timeout=20)
                if response.status_code in (401,403,429):
                    raise AccessRestricted(f'HTTP {response.status_code}: source access restricted; collection stopped without bypass')
                response.raise_for_status()
                raw=response.content
                if not raw.startswith(b'%PDF-') or len(raw)>20_000_000:
                    raise ValueError('Source did not return a bounded original PDF')
                digest=hashlib.sha256(raw).hexdigest()
                path=DIRECTORY/f'{digest}.pdf'
                if not path.exists():path.write_bytes(raw)  # Preserve original public source, not an authored document.
                result.update(status='verified_original_pdf',selected_notice=notice,
                    source_url=response.url,source_pdf_sha256=digest,source_pdf_path=path.relative_to(ROOT).as_posix(),
                    retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                    feature_status='original_source_archived_comparatives_not_yet_validated')
            except AccessRestricted as exc:
                result.update(status='source_access_restricted',error=str(exc));stop=str(exc)
            except (requests.RequestException,ValueError) as exc:
                result.update(status='source_unavailable',error=str(exc))
            time.sleep(.25)
        rows.append(result)
        if index%20==0 or index==len(eligible) or stop:
            OUTPUT.write_text('\n'.join(json.dumps(item,ensure_ascii=False) for item in rows)+'\n',encoding='utf-8')
            print(json.dumps({'processed':index,'eligible':len(eligible),'status':pd.Series([r['status'] for r in rows]).value_counts().to_dict()}),flush=True)
        if stop:break
    OUTPUT.write_text('\n'.join(json.dumps(item,ensure_ascii=False) for item in rows)+'\n',encoding='utf-8')
    report={'eligible_financial_cases':len(eligible),'attempted_cases':len(rows),
        'statuses':pd.Series([r['status'] for r in rows]).value_counts().to_dict(),
        'stopped_on_access_restriction':stop,
        'input_sha256':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (FINANCIAL,ANNOUNCEMENTS,Path(__file__))},
        'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'limitations':['Missing matching announcements are incomplete source coverage, not proof that the company published no results.',
           'A downloaded PDF is not yet an accepted comparative feature; dates, columns, units and source checks still must pass.',
           'No prediction is backdated and no live rule or model family is changed.']}
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    run()
