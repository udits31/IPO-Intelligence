"""Bounded native-text extraction of dated RHP facts; no LLM/PDF API calls.

Ambiguous tables are rejected. These are retrospective document-date facts,
not independently certified historical publication-time observations.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/audit/ipo_dated_fundamentals_20261023.json'
NUMBER=r'\(?-?\d[\d,]*\.\d+\)?'


def number(value):
    return float(value.replace(',','').replace('(','-').replace(')',''))


def eps_rows(text):
    """Accept explicit same-table annual basic/diluted EPS with year weights."""
    sections=re.split(r'(?i)(?:basic\s+and\s+diluted\s+earnings|earnings\s+per\s+(?:equity\s+)?share|basic\s+(?:and\s+)?diluted\s+eps)',text)
    for section in sections[1:]:
        section=section[:2400]
        if not re.search(r'(?i)diluted|basic',section[:900]):
            continue
        rows=[]
        pattern=rf'(?im)^\s*(?:March\s+31\s*,?\s*|Fiscal\s*(?:year)?\s*)?(20\d{{2}})\s+({NUMBER})\s+({NUMBER})\s+([123])\s*$'
        for row in re.finditer(pattern,section):
            rows.append(dict(year=int(row[1]),basic_eps=number(row[2]),diluted_eps=number(row[3]),weight=int(row[4])))
        rows=sorted(rows,key=lambda row:row['year'],reverse=True)
        if len(rows)>=2 and len({row['year'] for row in rows})==len(rows) and rows[0]['year']-rows[1]['year']==1:
            return rows
    return []


def document_date(text):
    months='January|February|March|April|May|June|July|August|September|October|November|December'
    date=rf'((?:{months})\s+\d{{1,2}}\s*,?\s*20\d{{2}}|\d{{1,2}}\s+(?:{months})\s*,?\s*20\d{{2}})'
    # The cover often prints the date without the word "Dated". Prefer that
    # title block; narrative incorporation/partnership dates are not publication.
    heading=re.search(r'(?im)^\s*(?:(?:RED HERRING|DRAFT RED HERRING)\s+)?PROSPECTUS\s*$',text[:3000])
    if heading:
        title_block=text[heading.end():heading.end()+600]
        title_block=re.split(r'(?i)please\s+read|our\s+company',title_block)[0]
        match=re.search(date,title_block,re.I)
        if match:
            return pd.Timestamp(match[1]).date().isoformat()
    match=re.search(rf'(?im)^\s*Dated\s*:?\s*{date}\s*$',text)
    return pd.Timestamp(match[1]).date().isoformat() if match else None


def pure_offer_ratio(text):
    clean=re.sub(r'\s+',' ',text).lower()
    # A mixed offer ALSO says the issuer receives no OFS proceeds. That
    # sentence alone cannot prove the entire offer is OFS.
    sale_only=re.search(r'(?:offer|issue) (?:comprises|consists of|is) (?:entirely|only|solely) (?:of )?(?:an? )?offer for sale',clean)
    no_offer_proceeds=re.search(r'(?:our company|we) (?:will|shall) not receive any (?:of the )?proceeds from (?:the |this )?offer(?! for sale)\b',clean)
    fresh_only=re.search(r'(?:offer|issue) (?:comprises|consists of|is) (?:entirely|only|solely) (?:of )?(?:a )?fresh issue',clean)
    return 1. if sale_only or no_offer_proceeds else 0. if fresh_only else None


def extract(ipo_id,company,path):
    import pymupdf
    with pymupdf.open(path) as doc:
        cover='\n'.join(doc[i].get_text(sort=True) for i in range(min(3,len(doc))))
        dated=document_date(cover)
        expected=re.sub(r'[^a-z0-9]','',re.sub(r'(?i)\b(?:ltd|limited)\b','',company).lower())
        identity=expected in re.sub(r'[^a-z0-9]','',cover.lower())
        if not dated or not identity or 'HERRING PROSPECTUS' not in cover.upper():
            return dict(ipo_id=ipo_id,status='rejected_cover_identity_or_date',source_path=path.relative_to(ROOT).as_posix())
        toc='\n'.join(doc[i].get_text(sort=True) for i in range(min(9,len(doc))))
        page_match=re.search(r'(?im)^\s*BASIS\s+FOR\s+(?:OFFER|ISSUE)\s+PRICE[^\n]*?\b(\d{1,3})\s*$',toc)
        pages=[]
        if page_match:
            printed=int(page_match[1])
            # Determine printed-to-physical offset from the explicit page-1
            # Definitions header, not a guessed file-wide constant.
            offsets=[i for i in range(min(15,len(doc)))
                     if re.search(r'(?m)^\s*1\s*$',doc[i].get_text(sort=True))
                     and 'DEFINITIONS AND ABBREVIATIONS' in doc[i].get_text()]
            for offset in offsets[:1]:
                pages=list(range(printed-1+offset,min(printed+10+offset,len(doc))))
        facts={};evidence=[]
        for page in pages:
            text=doc[page].get_text(sort=True)
            annual=eps_rows(text)
            if annual and pd.Timestamp(f"{annual[0]['year']}-03-31").date()<pd.Timestamp(dated).date():
                facts['annual_diluted_eps']=annual[0]['diluted_eps']
                facts['previous_annual_diluted_eps']=annual[1]['diluted_eps']
                facts['financial_period_end']=f"{annual[0]['year']}-03-31"
                evidence.append(dict(page=page+1,kind='annual_eps',annual_rows=annual,
                                     page_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                break
        # Cover/summary statement establishes a pure OFS or pure fresh offer.
        # Mixed offers are left missing unless amounts are separately verified.
        for page in range(min(40,len(doc))):
            text=doc[page].get_text(sort=True)
            ratio=pure_offer_ratio(text)
            if ratio is not None:
                facts['offer_for_sale_ratio']=ratio
                evidence.append(dict(page=page+1,kind='pure_offer_structure',
                    page_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                break
        return dict(ipo_id=ipo_id,company_name=company,status='dated_document_facts' if facts else 'no_unambiguous_facts',
            source_path=path.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            document_date=dated,timing_basis='RHP document date; publication clock not independently certified',
            facts=facts,evidence=evidence)


def run(per_year=12):
    from ml.return_pipeline import prepare_returns
    frame,_=prepare_returns();records=[]
    for year,group in frame.groupby('year'):
        group=group.sort_values(['listing_date','ipo_id']).reset_index(drop=True)
        # Evenly spaced by listing chronology, never chosen by outcomes.
        positions=sorted(set(round(i*(len(group)-1)/(min(per_year,len(group))-1))
                             for i in range(min(per_year,len(group)))))
        for position in positions:
            row=group.iloc[position]
            paths=list((ROOT/'data/prospectus_pdfs').glob(f'{int(row.ipo_id)}_*.pdf'))
            if len(paths)!=1:
                records.append(dict(ipo_id=int(row.ipo_id),status='missing_or_ambiguous_pdf'));continue
            records.append(extract(int(row.ipo_id),str(row.company_name),paths[0]))
    result=dict(schema_version=1,selection='12 evenly spaced existing early-book IPOs per listing year, without outcomes',
        records=records,limitations=['Document dates are not certified original publication timestamps.',
        'EPS is annual diluted EPS, not revenue or PAT growth. Mixed OFS ratios and ambiguous tables stay missing.'])
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(documents=len(records),with_eps=sum('annual_diluted_eps' in row.get('facts',{}) for row in records),
        with_offer=sum('offer_for_sale_ratio' in row.get('facts',{}) for row in records),
        statuses=pd.Series([row['status'] for row in records]).value_counts().to_dict()),indent=2))
    return result


if __name__=='__main__':
    run()
