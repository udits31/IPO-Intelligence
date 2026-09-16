"""Archive dated NSE quarterly-result metadata and original XBRL evidence.

Current-period profit/revenue are observations, not invented earnings surprise.
No current vendor ratios, imputed quarters or replacement training IPOs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from ml.calibrate_module2_rules import ROOT
from ml.replay_module2_news_overlay import SOURCES

PAGE='https://www.nseindia.com/companies-listing/corporate-filings-financial-results'
LEGACY='https://www.nseindia.com/api/corporates-financial-results'
INTEGRATED='https://www.nseindia.com/api/integrated-filing-results'
RAW=ROOT/'data/archive/module2_financials'
OUTPUT=ROOT/'data/audit/module2_dated_financial_evidence_20261021.jsonl'
REPORT=ROOT/'output/module2_financial_evidence_collection_20261021.json'
HEADERS={'User-Agent':'Mozilla/5.0 (compatible; IPO-Intelligence-Research/1.0)','Referer':PAGE}


class AccessRestricted(RuntimeError):
    pass


def source_get(session,url,params=None):
    response=session.get(url,params=params,timeout=20)
    if response.status_code in (401,403,429):
        raise AccessRestricted(f'HTTP {response.status_code}: {urlparse(url).netloc}; no access bypass or retry')
    response.raise_for_status()
    if len(response.content)>20_000_000:
        raise ValueError('Unexpectedly large financial source document')
    digest=hashlib.sha256(response.content).hexdigest()
    RAW.mkdir(parents=True,exist_ok=True)
    path=RAW/(digest+'.gz')
    if not path.exists():
        with gzip.open(path,'wb') as stream:
            stream.write(response.content)
    return response,{'source_url':response.url,'raw_sha256':digest,'raw_archive_path':path.relative_to(ROOT).as_posix(),
                    'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}


def publication(item):
    values=[item.get(key) for key in ('exchdisstime','broadCastDate','broadcast_Date','revised_Date') if item.get(key)]
    stamps=[pd.to_datetime(value,format='%d-%b-%Y %H:%M:%S',errors='raise') for value in values]
    return max(stamps) if stamps else None


def eligible_results(items,symbol,cutoff):
    eligible=[]
    for item in items:
        if str(item.get('symbol','')).upper()!=symbol:
            continue
        stamp=publication(item)
        end=pd.to_datetime(item.get('toDate') or item.get('qe_Date'),format='%d-%b-%Y',errors='coerce')
        if stamp is None or stamp>cutoff or pd.isna(end) or end>cutoff:
            continue
        if not item.get('xbrl'):
            continue
        # Quarterly actual results only; no cumulative 9/12-month substitute.
        if item.get('period') and item.get('period')!='Quarterly':
            continue
        if item.get('cumulative') and str(item['cumulative']).casefold()!='non-cumulative':
            continue
        eligible.append({**item,'published_at_ist':stamp.isoformat(),'period_end':end.strftime('%Y-%m-%d')})
    return sorted(eligible,key=lambda r:(r['period_end'],str(r.get('consolidated')).lower()=='consolidated',r['published_at_ist']),reverse=True)


def extract_current_quarter(raw,item,symbol,*,confirmed_current_context=None):
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('External/internal entity declarations are not accepted')
    root=ET.fromstring(raw)
    local=lambda tag:tag.rsplit('}',1)[-1]
    facts=list(root.iter())
    symbols={str(e.text).strip().upper() for e in facts if local(e.tag)=='Symbol' and e.text}
    if symbols!={symbol}:
        raise ValueError('XBRL issuer symbol conflicts with requested security')
    # Old NSE WEB exports sometimes omit the base OneD context itself. The
    # API's explicit current-period dates may identify that one context only;
    # never infer comparative-quarter dates from TwoD/FourD naming conventions.
    period_end=pd.Timestamp(item['period_end'])
    start=pd.to_datetime(item.get('fromDate'),format='%d-%b-%Y',errors='coerce')
    contexts={}
    for context in facts:
        if local(context.tag)!='context':
            continue
        members=[e for e in context.iter() if local(e.tag) in ('explicitMember','typedMember')]
        if members:
            continue
        values={local(e.tag):e.text for e in context.iter()}
        if not values.get('startDate') or not values.get('endDate'):
            continue
        first,last=pd.Timestamp(values['startDate']),pd.Timestamp(values['endDate'])
        if last==period_end and 70<=(last-first).days<=100:
            contexts[context.attrib['id']]=(first,last)
    base_context=next((e for e in facts if local(e.tag)=='context' and e.get('id')=='OneD'),None)
    if not contexts and base_context is None and pd.notna(start) and 70<=(period_end-start).days<=100:
        contexts['OneD']=(start,period_end)
    if not contexts:
        raise ValueError('No unambiguous non-segment quarterly context; no period is fabricated')
    if confirmed_current_context is not None:
        if confirmed_current_context!='OneD' or confirmed_current_context not in contexts:
            raise ValueError('Unsupported or missing independently confirmed current-quarter context')
        contexts={confirmed_current_context:contexts[confirmed_current_context]}
    metrics={}
    for label,concept in [('revenue','RevenueFromOperations'),('profit','ProfitLossForPeriod')]:
        values=[]
        for fact in facts:
            if local(fact.tag)!=concept or fact.get('contextRef') not in contexts:
                continue
            if fact.get('{http://www.w3.org/2001/XMLSchema-instance}nil') in ('true','1'):
                continue
            value=float(fact.text)
            if not math.isfinite(value):
                raise ValueError('Non-finite quarterly financial value')
            unit_id=fact.get('unitRef')
            unit=next((e for e in facts if local(e.tag)=='unit' and e.get('id')==unit_id),None)
            measures=[e.text for e in unit.iter() if local(e.tag)=='measure'] if unit is not None else []
            if not measures or any(str(value).split(':')[-1]!='INR' for value in measures):
                raise ValueError('Financial fact currency is not explicitly INR')
            values.append(value)
        if not values or len(set(values))!=1:
            raise ValueError('Missing or conflicting quarterly '+label)
        metrics[label+'_inr']=values[0]
    if metrics['revenue_inr']<=0:
        raise ValueError('No positive observed operating revenue; margin is not imputed')
    return {**metrics,'profit_margin_pct':metrics['profit_inr']/metrics['revenue_inr']*100,
            'period_end':item['period_end'],'published_at_ist':item['published_at_ist'],
            'basis':item.get('consolidated'),'context_ids':sorted(contexts),
            'metric_note':'Reported quarterly profit/revenue, not earnings surprise, cash flow or forecast growth.'}


def collect_one(session,case,canonical):
    ipo=canonical.loc[case.ipo_id]
    symbol=str(ipo.nse_symbol).strip().upper()
    cutoff=pd.Timestamp(case.decision_date)+pd.Timedelta(hours=15,minutes=30)
    begin=(cutoff-pd.Timedelta(days=365)).strftime('%d-%m-%Y')
    end=cutoff.strftime('%d-%m-%Y')
    row={'ipo_id':int(case.ipo_id),'symbol':symbol,'decision_date':cutoff.date().isoformat(),'cutoff_ist':cutoff.isoformat(),
         'status':'unavailable','metadata_sources':[],'eligible_filings':0,'financials':None,'error':None}
    items=[]
    sources=[LEGACY] if cutoff<pd.Timestamp('2025-04-01') else [LEGACY,INTEGRATED]
    for endpoint in sources:
        params={'symbol':symbol,'from_date':begin,'to_date':end}
        params.update({'index':'equities','period':'Quarterly'} if endpoint==LEGACY else {'type':'Integrated Filing- Financials','page':1,'size':20})
        response,source=source_get(session,endpoint,params)
        row['metadata_sources'].append(source)
        payload=response.json()
        if endpoint==LEGACY:
            if not isinstance(payload,list):
                raise ValueError('Legacy financial-result metadata is not a list')
            items.extend(payload)
        else:
            if not isinstance(payload,dict) or not isinstance(payload.get('data'),list):
                raise ValueError('Integrated financial-result metadata is not a paginated list')
            total=int(payload.get('totalCount',0))
            items.extend(payload['data'])
            for page in range(2,(total+19)//20+1):
                response,source=source_get(session,endpoint,{**params,'page':page})
                row['metadata_sources'].append(source)
                items.extend(response.json()['data'])
    eligible=eligible_results(items,symbol,cutoff)
    row['eligible_filings']=len(eligible)
    if not eligible:
        row['status']='no_eligible_result'
        return row
    # Latest period only. Prefer consolidated, but never substitute an older
    # result or silently switch basis merely because a current XML is malformed.
    chosen=eligible[0]
    row['selected_filing']=chosen
    url=chosen['xbrl']
    if urlparse(url).hostname not in ('nsearchives.nseindia.com','archives.nseindia.com'):
        raise ValueError('Unexpected XBRL source authority')
    response,source=source_get(session,url)
    row['document_source']=source
    try:
        row['financials']=extract_current_quarter(response.content,chosen,symbol)
        row['status']='verified_quarterly_metrics'
    except (ET.ParseError,ValueError,TypeError) as exc:
        row['status']='document_unusable'
        row['error']=str(exc)
    return row


def main(limit=0):
    frames=[pd.read_csv(ledger,parse_dates=['decision_date']) for _,ledger in SOURCES.values()]
    cases=pd.concat(frames).sort_values(['listing_year','ipo_id'])
    if cases.ipo_id.duplicated().any():
        raise ValueError('Duplicate financial-evidence checkpoint')
    if limit:
        cases=cases.head(limit)
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    existing={}
    if OUTPUT.exists():
        existing={r['ipo_id']:r for r in (json.loads(line) for line in OUTPUT.read_text(encoding='utf-8').splitlines())}
    session=requests.Session();session.headers.update(HEADERS)
    landing=session.get(PAGE,timeout=20)
    landing.raise_for_status()
    stop=None
    for index,case in enumerate(cases.itertuples(),1):
        prior=existing.get(int(case.ipo_id))
        if prior and prior['decision_date']==case.decision_date.date().isoformat() and prior['status']!='unavailable':
            continue
        try:
            row=collect_one(session,case,canonical)
        except AccessRestricted as exc:
            stop=str(exc)
            print(stop,flush=True)
            break
        except (requests.RequestException,ValueError,KeyError) as exc:
            row={'ipo_id':int(case.ipo_id),'symbol':str(canonical.loc[case.ipo_id].nse_symbol),
                 'decision_date':case.decision_date.date().isoformat(),'status':'unavailable','error':str(exc)}
        existing[int(case.ipo_id)]=row
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        OUTPUT.write_text('\n'.join(json.dumps(existing[key],ensure_ascii=False) for key in sorted(existing))+'\n',encoding='utf-8')
        if index%20==0 or index==len(cases):
            print(json.dumps({'processed':index,'requested':len(cases),'statuses':pd.Series([r['status'] for r in existing.values()]).value_counts().to_dict()}),flush=True)
        time.sleep(.25)
    records=[existing[int(ipo_id)] for ipo_id in cases.ipo_id if int(ipo_id) in existing]
    report={'source_pages':[PAGE,'https://www.nseindia.com/companies-listing/corporate-integrated-filing'],
        'requested':len(cases),'recorded':len(records),'statuses':pd.Series([r['status'] for r in records]).value_counts().to_dict(),
        'stopped_on_access_restriction':stop,'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest() if OUTPUT.exists() else None,
        'limitations':['Retrieved retrospectively; exchange clocks do not independently prove completeness or archival immutability.',
          'Missing/unusable financials are not zero revenue, zero profit or a no-news signal.',
          'Current-period metrics are not earnings surprise or year-over-year growth.',
          'No live model, paper or original financial values are overwritten.']}
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=0)
    main(parser.parse_args().limit)
