"""Local, resumable extraction across the entire existing 2021-2026 universe.

Each PDF is read in its own process; no LLM, network, OCR or outcome-based
selection. Every issuer gets a record, including explicit source exceptions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ml.extract_dated_ipo_fundamentals import document_date,pure_offer_ratio,number

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/audit/ipo_dated_fundamentals_all_20261024.json'
VERSION='native-rhp-all-v2'
NUM=r'\(?[\-−]?\d[\d,]*(?:\.\d+)?\)?'


def annual_eps(text):
    text=text.replace('−','-')
    text=re.sub(r'(?<=\d)(?:\(\d+\))+','',text)
    starts=list(re.finditer(r'(?i)(?:basic\s+and\s+diluted\s+earnings?|earnings?\s+per\s+(?:equity\s+)?share|basic\s+(?:and\s+)?diluted\s+eps)',text))
    for start in starts:
        block=text[start.start():start.end()+2900]
        end=re.search(r'(?im)^\s*(?:Notes?\s*[:.]|[A-Z]?[.)]?\s*Price\s*/?\s*Earnings?|Return\s+on\s+Net)',block)
        if end and end.start()>150:
            block=block[:end.start()]
        rows=[]
        pattern=rf'(?im)^\s*(?:(?:Financial\s+)?Year\s*(?:ended)?\s*|Fiscal\s*(?:year)?\s*)?(?:(?:March\s+31|31\s+March)\s*,?\s*)?(20\d{{2}})[*#]?\s+({NUM})[*#]?\s+({NUM})[*#]?(?:\s+([123]))?\s*[*#]?\s*$'
        for match in re.finditer(pattern,block):
            rows.append(dict(year=int(match[1]),basic_eps=number(match[2]),diluted_eps=number(match[3]),weight=int(match[4]) if match[4] else None))
        fiscal_dual=rf'(?im)^\s*(?:FY\s*|Fiscal\s*)?(20\d{{2}})[-/](\d{{2}}|20\d{{2}})[*#]?\s+({NUM})[*#]?\s+({NUM})[*#]?(?:\s+([123]))?\s*$'
        for match in re.finditer(fiscal_dual,block):
            start=int(match[1]);end=int(match[2]) if len(match[2])==4 else (start//100)*100+int(match[2])
            if end<start:
                end+=100
            if end==start+1:
                rows.append(dict(year=end,basic_eps=number(match[3]),diluted_eps=number(match[4]),weight=int(match[5]) if match[5] else None))
        # Combined basic-and-diluted EPS column, not a guessed missing column.
        if not rows and re.search(r'(?i)basic\s*(?:and|&|/)\s*diluted',block):
            single=rf'(?im)^\s*(?:March\s+31\s*,?\s*|Fiscal\s*)?(20\d{{2}})[*#]?\s+({NUM})[*#]?(?:\s+([123]))?\s*$'
            if not re.search(r'(?i)basic\s+eps[^\n]*diluted\s+eps',block):
                rows=[dict(year=int(m[1]),basic_eps=None,diluted_eps=number(m[2]),weight=int(m[3]) if m[3] else None)
                      for m in re.finditer(single,block)]
                fiscal=rf'(?im)^\s*(?:FY\s*|Fiscal\s*)?(20\d{{2}})[-/](\d{{2}}|20\d{{2}})[*#]?\s+({NUM})[*#]?\s*$'
                for match in re.finditer(fiscal,block):
                    end=int(match[2]) if len(match[2])==4 else int(str(int(match[1])//100)+match[2])
                    if end==int(match[1])+1:
                        rows.append(dict(year=end,basic_eps=None,diluted_eps=number(match[3]),weight=None))
        rows=sorted(rows,key=lambda row:row['year'],reverse=True)
        if len(rows)>=2 and len({r['year'] for r in rows})==len(rows) and rows[0]['year']-rows[1]['year']==1:
            return rows
    return []


def offer_components(text):
    """Parse explicitly labelled offer amounts/shares; preserve currency units."""
    clean=re.sub(r'\s+',' ',text).replace('−','-')
    clean=re.sub(r'(?<=\d)[*#]','',clean)
    clean=re.sub(r'(?i)(fresh issu(?:e|ance)|offer for sale)\s*(?:\(\d+\))+',r'\1 ',clean)
    result={}
    pure=pure_offer_ratio(clean)
    if pure is not None:
        result['offer_for_sale_ratio']=pure
    for label,prefix in [(r'fresh issu(?:e|ance)','fresh'),('offer for sale','ofs')]:
        amounts=set();shares=set()
        for occurrence in re.finditer(label,clean,re.I):
            segment=clean[occurrence.end():occurrence.end()+650]
            # Don't attribute the next offer component's money to this one.
            stop=re.search(r'(?i)fresh issu(?:e|ance)|offer for sale',segment)
            if stop:
                segment=segment[:stop.start()]
            sentence=re.search(r'(?<=[a-z0-9\)])\.\s+(?=[A-Z])',segment)
            if sentence:
                segment=segment[:sentence.start()+1]
            # Use only this component's FIRST aggregation expression. An RHP
            # placeholder (Rs [bullet]) must not skip ahead to the employee
            # reservation, which is a different row of the offer table.
            aggregation=re.search(r'(?i)aggregating\s+(?:up\s+to\s+|to\s+)?',segment)
            amount=(re.match(rf'(?i)(?:₹|Rs\.?|INR)\s*({NUM})\s*(million|crore|lakh|billion)',
                segment[aggregation.end():]) if aggregation else None)
            if amount and not re.search(r'(?i)pre-?ipo\s+placement|draft\s+red\s+herring',segment[:aggregation.start()]):
                units={'million':1e6,'crore':1e7,'lakh':1e5,'billion':1e9}
                value=number(amount[1])*units[amount[2].lower()]
                if value>0:
                    amounts.add(value)
            share=re.match(rf'(?i)\s*(?:of\s+)?(?:up\s+to\s+)?({NUM})\s+equity\s+shares\b',segment)
            if share and number(share[1])>0:
                shares.add(number(share[1]))
        if len(amounts)==1:
            result[prefix+'_amount_rs']=amounts.pop()
        if len(shares)==1:
            result[prefix+'_shares']=shares.pop()
    return result


def annual_eps_from_tables(page):
    """Fallback for ruled annual-EPS tables whose text columns interleave."""
    result=page.find_tables()
    for table in result.tables:
        cells=table.extract()
        for index,header in enumerate(cells[:4]):
            diluted=next((col for col,value in enumerate(header) if value and
                'diluted' in value.lower() and ('eps' in value.lower() or 'earning' in value.lower())),None)
            if diluted is None:
                continue
            basic=next((col for col,value in enumerate(header) if value and 'basic' in value.lower()),None)
            rows=[]
            for cells_row in cells[index+1:]:
                period=str(cells_row[0] or '')
                if re.search(r'(?i)september|june|december|quarter|months|weighted|average',period):
                    continue
                fiscal=re.search(r'(20\d{2})\s*[-/]\s*(20\d{2}|\d{2})',period)
                years=re.findall(r'20\d{2}',period)
                if fiscal:
                    start=int(fiscal[1]);year=int(fiscal[2]) if len(fiscal[2])==4 else (start//100)*100+int(fiscal[2])
                    if year!=start+1:
                        continue
                elif len(years)==1:
                    year=int(years[0])
                else:
                    continue
                def eps(col):
                    if col is None or col>=len(cells_row) or cells_row[col] is None:
                        return None
                    value=str(cells_row[col]).strip().replace('−','-')
                    value=re.sub(r'(?<=\d)(?:\(\d+\))+','',value)
                    value=re.sub(r'[*#]','',value).strip()
                    return number(value) if re.fullmatch(NUM,value) else None
                value=eps(diluted)
                if value is not None:
                    rows.append(dict(year=year,basic_eps=eps(basic),diluted_eps=value,weight=None))
            rows=sorted(rows,key=lambda row:row['year'],reverse=True)
            if len(rows)>=2 and len({row['year'] for row in rows})==len(rows) and rows[0]['year']-rows[1]['year']==1:
                return rows
    return []


def extract_one(task):
    import pymupdf
    ipo_id,company,path_name=task
    path=Path(path_name)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    base=dict(ipo_id=ipo_id,company_name=company,source_path=path.relative_to(ROOT).as_posix(),
              source_sha256=digest,extractor_version=VERSION)
    try:
        with pymupdf.open(path) as doc:
            cache={}
            def text(index):
                if index not in cache:
                    cache[index]=doc[index].get_text(sort=True)
                return cache[index]
            cover='\n'.join(text(i) for i in range(min(5,len(doc))))
            dated=document_date(cover)
            def identity(value):
                value=re.sub(r'(?i)\bcorp\b','corporation',value)
                value=re.sub(r'(?i)\bco\b','company',value)
                value=re.sub(r'(?i)\b(?:ltd|limited|and|n)\b','',value)
                return re.sub(r'[^a-z0-9]','',value.lower())
            name=identity(company)
            if not dated or name not in identity(cover) or 'PROSPECTUS' not in cover.upper():
                return {**base,'status':'rejected_cover_identity_or_date','facts':{},'evidence':[]}
            toc='\n'.join(text(i) for i in range(min(12,len(doc))))
            offset=None
            for index in range(min(24,len(doc))):
                content=text(index);lines=[line.strip() for line in content.splitlines() if line.strip()]
                printed=[int(value) for value in (lines[0],lines[-1]) if value.isdigit() and 1<=int(value)<=30] if lines else []
                header=re.sub(r'\s+',' ',content[:800]).upper()
                if printed and 'TABLE OF CONTENTS' not in header and 'DEFINITIONS AND ABBREVIATIONS' in header:
                    offset=index-printed[-1]+1;break
            sections={}
            if offset is not None:
                for label,pattern in [('basis',r'BASIS\s+(?:FOR|OF)\s+(?:THE\s+)?(?:OFFER|ISSUE)\s+PRICE'),
                                      ('offer',r'THE\s+(?:OFFER|ISSUE)')]:
                    match=re.search(rf'(?im)^\s*{pattern}[^\n]*?\b(\d{{1,3}})\s*$',toc)
                    if match:
                        begin=int(match[1])-1+offset
                        title=rf'(?im)^\s*{pattern}\s*$'
                        if not (0<=begin<len(doc) and re.search(title,text(begin)[:700])):
                            # Some stored PDFs omit front-matter pages mid-file;
                            # locate the actual heading near the printed-page
                            # estimate rather than trusting a global offset.
                            for distance in range(1,23):
                                candidates=[begin-distance,begin+distance]
                                found=next((index for index in candidates if 0<=index<len(doc)
                                            and re.search(title,text(index)[:700])),None)
                                if found is not None:
                                    begin=found;break
                        sections[label]=list(range(max(0,begin),min(begin+(13 if label=='basis' else 4),len(doc))))
            facts={};evidence=[]
            for page in sections.get('basis',[]):
                annual=annual_eps(text(page))
                pages=[page]
                if not annual and page+1<len(doc):
                    annual=annual_eps(text(page)+'\n'+text(page+1));pages=[page,page+1]
                if annual and pd.Timestamp(f"{annual[0]['year']}-03-31").date()<pd.Timestamp(dated).date():
                    facts.update(annual_diluted_eps=annual[0]['diluted_eps'],
                        previous_annual_diluted_eps=annual[1]['diluted_eps'],financial_period_end=f"{annual[0]['year']}-03-31")
                    combined='\n'.join(text(p) for p in pages)
                    evidence.append(dict(page=pages[0]+1,pages=[p+1 for p in pages],kind='annual_eps',
                        annual_rows=annual,page_text_sha256=hashlib.sha256(combined.encode()).hexdigest()))
                    break
            if 'annual_diluted_eps' not in facts:
                for page in sections.get('basis',[]):
                    content=text(page)
                    if not re.search(r'(?i)earnings?\s+per\s+(?:equity\s+)?share',content):
                        continue
                    annual=annual_eps_from_tables(doc[page])
                    if annual and pd.Timestamp(f"{annual[0]['year']}-03-31").date()<pd.Timestamp(dated).date():
                        facts.update(annual_diluted_eps=annual[0]['diluted_eps'],
                            previous_annual_diluted_eps=annual[1]['diluted_eps'],financial_period_end=f"{annual[0]['year']}-03-31")
                        evidence.append(dict(page=page+1,pages=[page+1],kind='annual_eps_table',
                            annual_rows=annual,page_text_sha256=hashlib.sha256(content.encode()).hexdigest()))
                        break
            offer_pages=sorted(set(range(min(3,len(doc))))|set(sections.get('offer',[])))
            # Pure-offer statements can appear in the summary before THE OFFER.
            offer_pages=sorted(set(offer_pages)|set(range(15,min(36,len(doc)))))
            candidates={}
            for page in offer_pages:
                components=offer_components(text(page))
                for key,value in components.items():
                    candidates.setdefault(key,[]).append((value,page))
            for key,observations in candidates.items():
                if len({value for value,_ in observations})!=1:
                    continue  # Conflicting text must never become a numeric guess.
                value,page=observations[0];facts[key]=value
                evidence.append(dict(page=page+1,pages=[page+1],kind='offer_component',field=key,value=value,
                    page_text_sha256=hashlib.sha256(text(page).encode()).hexdigest()))
            return {**base,'status':'dated_document_facts' if facts else 'no_unambiguous_facts',
                'document_date':dated,'cover_page_count':min(5,len(doc)),'timing_basis':'Prospectus document date, not certified original publication time',
                'facts':facts,'evidence':evidence,'basis_section_found':bool(sections.get('basis'))}
    except Exception as error:
        return {**base,'status':'extraction_error','error':str(error),'facts':{},'evidence':[]}


def refresh_offer_evidence():
    """Reparse only already-cited offer pages; preserve all annual EPS work."""
    import pymupdf
    payload=json.loads(OUT.read_text(encoding='utf-8'))
    backup=OUT.with_name(OUT.stem+'_before_offer_placeholder_fix.json')
    if not backup.exists():
        backup.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    changes=[]
    for record in payload['records']:
        if record['status']!='dated_document_facts':
            continue
        path=ROOT/record['source_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['source_sha256']:
            raise ValueError('Source changed; cannot reuse extracted annual earnings')
        prior=record['facts'].copy()
        offer_evidence=[e for e in record['evidence'] if e['kind'] in ('offer_component','pure_offer_structure')]
        if not offer_evidence:
            continue
        pages=sorted({p for e in offer_evidence for p in e.get('pages',[e['page']])})
        record['evidence']=[e for e in record['evidence'] if e not in offer_evidence]
        for key in ('offer_for_sale_ratio','fresh_amount_rs','ofs_amount_rs','fresh_shares','ofs_shares'):
            record['facts'].pop(key,None)
        candidates={}
        with pymupdf.open(path) as document:
            for page in pages:
                content=document[page-1].get_text(sort=True)
                for key,value in offer_components(content).items():
                    candidates.setdefault(key,[]).append((value,page,content))
            for key,observations in candidates.items():
                if len({value for value,_,_ in observations})!=1:
                    continue
                value,page,content=observations[0]
                record['facts'][key]=value
                record['evidence'].append(dict(page=page,pages=[page],kind='offer_component',field=key,value=value,
                    page_text_sha256=hashlib.sha256(content.encode()).hexdigest()))
        record['offer_repair_scope']='Previously cited native-text offer pages; annual EPS unchanged'
        if prior!=record['facts']:
            changes.append(dict(ipo_id=record['ipo_id'],before=prior,after=record['facts']))
    payload['extractor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload['offer_placeholder_repair']=dict(changed_records=len(changes),changes=changes,
        scope='All dated records with cited offer evidence; no outcome-based selection')
    OUT.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(offer_records_changed=len(changes),annual_eps_reextracted=0)))


def refresh_cover_dates():
    """Check cover dates only; re-extract full facts solely for changed records."""
    import pymupdf
    payload=json.loads(OUT.read_text(encoding='utf-8'))
    backup=OUT.with_name(OUT.stem+'_before_cover_date_fix.json')
    if not backup.exists():
        backup.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    changed=[]
    for index,record in enumerate(payload['records']):
        if not record.get('document_date'):
            continue
        path=ROOT/record['source_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['source_sha256']:
            raise ValueError('Source changed before cover-date audit')
        with pymupdf.open(path) as doc:
            cover='\n'.join(doc[i].get_text(sort=True) for i in range(min(5,len(doc))))
        corrected=document_date(cover)
        if corrected!=record['document_date']:
            replacement=extract_one((record['ipo_id'],record['company_name'],str(path)))
            changed.append(dict(ipo_id=record['ipo_id'],before=record['document_date'],after=corrected))
            payload['records'][index]=replacement
    payload['extractor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload['cover_date_repair']=dict(changed_records=len(changed),changes=changed,
        scope='All dated covers checked; only changed records re-extracted')
    OUT.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cover_dates_changed=len(changed),changes=changed)))


def run(workers=2,repair_missing=False,eps_only=False):
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv')
    years=pd.to_datetime(canonical.listing_date).dt.year
    cohort=canonical[years.between(2021,2026)&~canonical.id.isin([293,307])].sort_values('id')
    old={}
    extractor_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if OUT.exists():
        checkpoint=json.loads(OUT.read_text())
        if checkpoint.get('extractor_sha256')==extractor_hash or repair_missing:
            old={row['ipo_id']:row for row in checkpoint['records']}
    records=[];tasks=[]
    for row in cohort.itertuples():
        paths=list((ROOT/'data/prospectus_pdfs').glob(f'{row.id}_*.pdf'))
        if len(paths)!=1:
            records.append(dict(ipo_id=int(row.id),company_name=row.company_name,status='missing_or_ambiguous_pdf',facts={},evidence=[]))
        elif row.id in old and old[row.id].get('source_sha256')==hashlib.sha256(paths[0].read_bytes()).hexdigest():
            cached=old[row.id];facts=cached.get('facts',{})
            complete_offer=('offer_for_sale_ratio' in facts or
                (any(key in facts for key in ('fresh_amount_rs','fresh_shares')) and any(key in facts for key in ('ofs_amount_rs','ofs_shares'))))
            if not repair_missing or ('annual_diluted_eps' in facts and (complete_offer or eps_only)):
                records.append(cached)
            else:
                tasks.append((int(row.id),row.company_name,str(paths[0])))
        else:
            tasks.append((int(row.id),row.company_name,str(paths[0])))
    def save():
        payload=dict(schema_version=2,extractor_sha256=extractor_hash,selection='Every 2021-2026 issuer in the existing corporate research universe; no outcome selection',
                     expected_issuer_count=len(cohort),processed_count=len(records),records=sorted(records,key=lambda r:r['ipo_id']))
        temporary=OUT.with_suffix('.partial.json')
        temporary.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8');temporary.replace(OUT)
    save()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(extract_one,task) for task in tasks]
        for done,future in enumerate(as_completed(futures),1):
            records.append(future.result())
            if done%10==0 or done==len(tasks):
                save();print(json.dumps(dict(pdf_done=done,pdf_total=len(tasks),total_accounted=len(records),
                    eps=sum('annual_diluted_eps' in r.get('facts',{}) for r in records))),flush=True)
    save()
    print(json.dumps(dict(complete=True,issuers=len(records),
        statuses=pd.Series([r['status'] for r in records]).value_counts().to_dict()),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--repair-missing',action='store_true')
    parser.add_argument('--eps-only',action='store_true')
    parser.add_argument('--refresh-offer-evidence',action='store_true')
    parser.add_argument('--refresh-cover-dates',action='store_true')
    args=parser.parse_args()
    if args.refresh_cover_dates:
        refresh_cover_dates()
    elif args.refresh_offer_evidence:
        refresh_offer_evidence()
    else:
        run(args.workers,args.repair_missing,args.eps_only)
