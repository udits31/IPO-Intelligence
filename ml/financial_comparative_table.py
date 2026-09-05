"""Conservative spatial extraction of same-quarter financial comparatives.

Accept explicitly labelled quarter columns only. Year-to-date/annual columns,
low-confidence numbers, ambiguous rows, unknown units and mismatched current
XBRL facts are rejected. No missing comparative is imputed or guessed.
"""
from __future__ import annotations
import calendar
from datetime import date
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OCR=ROOT/'data/audit/financial_pdf_ocr/013048e3e504243457231a20bd9b2dd6fae7dc14b9235995709d7a4e6880db0f_page7.json'
FINANCIALS=ROOT/'data/audit/module2_dated_financial_evidence_quarter_verified_20261022.jsonl'
OUTPUT=ROOT/'data/audit/module2_comparative_financial_prototype_20261022.json'


def centre(token):
    box=token['box']
    return sum(point[0] for point in box)/4,sum(point[1] for point in box)/4


def usable_token(token):
    if token.get('source_kind')=='embedded_pdf_text':
        # Exact text objects have no OCR confidence score. Their provenance is
        # replayed from the original PDF by the review gate, not invented as 1.
        return token.get('confidence') is None
    confidence=token.get('confidence')
    return isinstance(confidence,(int,float)) and math.isfinite(confidence) and confidence>=.95


def confidence_label(*tokens):
    values=[token.get('confidence') for token in tokens]
    return min(values) if all(value is not None for value in values) else None


def numeric(token):
    if not usable_token(token):
        return None
    value=token['text'].strip().replace(',','').replace('−','-')
    if not re.fullmatch(r'-?\d+(?:\.\d+)?|\(\d+(?:\.\d+)?\)',value):
        return None
    return -float(value[1:-1]) if value.startswith('(') else float(value)


def assign_column_groups(columns,groups):
    """Map merged header groups using their full dated-column geometry.

    The last of three quarter columns can be nearer to the following annual
    caption than to the centre of its own merged caption. Nearest-caption
    classification is therefore invalid. Require an unambiguous contiguous
    partition whose column centres match all caption centres instead.
    """
    columns=sorted(columns,key=lambda c:c['x'])
    groups=sorted(groups)
    if len(columns)<len(groups) or not columns:
        raise ValueError('Date columns do not cover every explicit period group')
    if any(abs(a['x']-b['x'])<10 for a,b in zip(columns,columns[1:])):
        raise ValueError('Overlapping date labels require manual source review')
    options=[]
    for cuts in itertools.combinations(range(1,len(columns)),len(groups)-1):
        edges=(0,*cuts,len(columns));errors=[]
        for i,(x,_) in enumerate(groups):
            section=columns[edges[i]:edges[i+1]]
            errors.append(abs(sum(c['x'] for c in section)/len(section)-x))
        options.append((sum(e*e for e in errors),max(errors),edges))
    options.sort()
    gaps=[b['x']-a['x'] for a,b in zip(columns,columns[1:])]
    tolerance=min(35,min(gaps)*.35) if gaps else 35
    if not options or options[0][1]>tolerance or (len(options)>1 and options[1][0]-options[0][0]<100):
        raise ValueError('Period-group spans are spatially ambiguous')
    edges=options[0][2]
    for i,(_,kind) in enumerate(groups):
        for column in columns[edges[i]:edges[i+1]]:
            column['kind']=kind
    return columns


def extract(ocr,financial):
    tokens=ocr['tokens']
    if any(token.get('source_kind')=='embedded_pdf_text' for token in tokens) and ocr.get('extraction_method')!='embedded_pdf_text':
        raise ValueError('Native text tokens require explicit native-source provenance')
    compact=lambda text:re.sub(r'\s+','',text).casefold()
    # Filed statements use Arabic/Roman row numbers and lettered revenue
    # subrows; OCR can join the prefix to the label (for example VII.Profit).
    # Strip only an explicit row marker, never a financial word or figure.
    row_text=lambda token:re.sub(
        r'^(?:(?:\d+|[IVX]+)[.)]\s*|(?:\d+|[IVX]+)\s+|[-\u2022]\s*|[a-z][.)]\s*|a(?=Revenue))',
        '',token['text'].strip(),flags=re.I)
    revenue_labels=[t for t in tokens if re.fullmatch(r'(?:\([a-z]\)\s*)?Revenue\s*from\s*Operati[oa]ns(?:\s*\(net\))?',row_text(t),re.I)]
    profit_labels=[t for t in tokens if re.match(r'^(?:net\s+)?(?:profit|loss)\b',row_text(t),re.I) and
        re.search(r'period|year|quarter|after\s+tax',t['text'],re.I) and not re.search(r'before|comprehensive|attributable|\bfrom\b|discontinued',t['text'],re.I)]
    if len(revenue_labels)!=1 or not profit_labels:
        raise ValueError('Current operating-revenue and after-tax profit rows must be unique')
    revenue_y=centre(revenue_labels[0])[1]
    titles=[token for token in tokens if centre(token)[1]<revenue_y and
            re.match(r'^(?:statement|unaudited|un-audited|audited|part\s*[i1l]\b|standalone|consolidated)',token['text'],re.I) and
            re.search(r'financials?results',compact(token['text']))]
    basis=financial['basis'].casefold()
    if basis not in ('consolidated','standalone','non-consolidated'):
        raise ValueError('The source does not explicitly identify a consolidation basis')
    title=([token for token in titles if 'consolidated' in token['text'].casefold() and
              'standalone' not in token['text'].casefold()] if basis=='consolidated' else
              [token for token in titles if 'standalone' in token['text'].casefold()])
    basis_authority='Explicit PDF table title plus the independently verified NSE filing'
    # Some genuinely standalone issuers label their only result statement
    # simply "Financial Results". The verified filing metadata explicitly says
    # Non-Consolidated. Use that named authority, not an invented PDF label,
    # only when the table has no conflicting consolidation title. Numeric
    # quarter parity and full-page visual review remain mandatory downstream.
    if not title and basis in ('standalone','non-consolidated') and len(titles)==1 and not any(
            re.search(r'consolidated|standalone',token['text'],re.I) for token in titles):
        title=titles
        basis_authority='Verified non-consolidated NSE filing metadata; the PDF table title is unlabelled, not explicitly standalone'
    if len(title)!=1:
        raise ValueError('A unique table with the verified consolidation basis is required')
    units=[token for token in tokens if centre(token)[1]<revenue_y and re.search(
        r'\b(?:in\s+|inr\s+|rs\.?\s+)(millions?|lakhs?|lacs?|crores?)\b',token['text'],re.I)]
    if len(units)!=1:
        raise ValueError('Unknown or ambiguous monetary units')
    if not all(usable_token(token) for token in (title[0],units[0])):
        raise ValueError('Table basis or currency-unit label has low OCR confidence')
    unit=re.search(r'\b(millions?|lakhs?|lacs?|crores?)\b',units[0]['text'],re.I).group(1).casefold()
    multiplier=1e6 if unit.startswith('million') else 1e7 if unit.startswith('crore') else 1e5
    header_top=centre(title[0])[1];header_bottom=centre(revenue_labels[0])[1]
    header=[t for t in tokens if header_top<centre(t)[1]<header_bottom]
    groups=[]
    for token in header:
        text=token['text'].casefold()
        if re.fullmatch(r'(?:for\s+)?quarter\s+ended(?:\s+on)?',text):
            groups.append((centre(token)[0],'quarter'))
        elif re.search(r'(?:months?|year|annual|half.year|period).*ended|^(?:nine|six|twelve)\s+months?$',text):
            groups.append((centre(token)[0],'not_quarter'))
    if not any(kind=='quarter' for _,kind in groups):
        raise ValueError('Quarter headings are not explicit')
    columns=[]
    for token in header:
        if not usable_token(token):
            continue
        text=token['text'].strip()
        complete=re.fullmatch(r'\(?\s*(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\s*\)?',text)
        months='|'.join([*calendar.month_name[1:],*calendar.month_abbr[1:]])
        textual=re.fullmatch(rf'({months})\s*(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*((?:19|20)\d{{2}})',text,re.I)
        day_first=re.fullmatch(rf'(\d{{1,2}})(?:st|nd|rd|th)?[\s-]+({months})\s*,?\s*[\s-]*((?:19|20)\d{{2}})',text,re.I)
        if complete or textual or day_first:
            if complete:
                day,month,year=map(int,complete.groups())
            elif textual:
                month_text,day,year=textual.groups();day,year=int(day),int(year)
                month=next(index for index in range(1,13) if month_text.casefold() in (calendar.month_name[index].casefold(),calendar.month_abbr[index].casefold()))
            else:
                day,month_text,year=day_first.groups();day,year=int(day),int(year)
                month=next(index for index in range(1,13) if month_text.casefold() in (calendar.month_name[index].casefold(),calendar.month_abbr[index].casefold()))
            value=date(year,month,day)
            x,_=centre(token)
            columns.append({'date':value,'x':x,'kind':min(groups,key=lambda pair:abs(pair[0]-x))[1],
                            'label_confidence':confidence_label(token)})
            continue
        year_only=re.fullmatch(r'((?:19|20)\d{2})',text)
        if year_only:
            x,y=centre(token)
            above=[(t,re.fullmatch(rf'(\d{{1,2}})(?:st|nd|rd|th)?\s*({months})\s*,?',t['text'].strip(),re.I))
                   for t in header if abs(centre(t)[0]-x)<30 and 0<y-centre(t)[1]<35 and usable_token(t)]
            above=[(t,m) for t,m in above if m]
            if len(above)==1:
                t,m=above[0];day,month_text=m.groups()
                month=next(index for index in range(1,13) if month_text.casefold() in (calendar.month_name[index].casefold(),calendar.month_abbr[index].casefold()))
                columns.append({'date':date(int(text),month,int(day)),'x':x,
                    'kind':min(groups,key=lambda pair:abs(pair[0]-x))[1],
                    'label_confidence':confidence_label(t,token)})
            continue
        match=re.fullmatch(r'(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*((?:19|20)\d{2})',token['text'].strip(),re.I)
        if not match or not usable_token(token):
            continue
        x,y=centre(token)
        month_tokens=[t for t in header if abs(centre(t)[0]-x)<30 and 0<y-centre(t)[1]<35 and
                      any(t['text'].casefold()==month.casefold() for month in calendar.month_name[1:])]
        if len(month_tokens)!=1:
            continue
        month=list(calendar.month_name).index(month_tokens[0]['text'].capitalize())
        value=date(int(match.group(2)),month,int(match.group(1)))
        group=min(groups,key=lambda pair:abs(pair[0]-x))[1]
        columns.append({'date':value,'x':x,'kind':group,'label_confidence':confidence_label(token,month_tokens[0])})
    columns=assign_column_groups(columns,groups)
    current=date.fromisoformat(financial['period_end'])
    prior=current.replace(year=current.year-1)
    now_col=[c for c in columns if c['date']==current and c['kind']=='quarter']
    old_col=[c for c in columns if c['date']==prior and c['kind']=='quarter']
    if len(now_col)!=1 or len(old_col)!=1:
        raise ValueError('Unique same-quarter current/prior-year dates are required; YTD columns are not eligible')
    now_col,old_col=now_col[0],old_col[0]
    gaps=[abs(c['x']-now_col['x']) for c in columns if c is not now_col and abs(c['x']-now_col['x'])>10]
    column_tolerance=min(gaps)/2 if gaps else 35
    results={};proof={}
    def row_values(name,label,expected):
        lx,ly=centre(label)
        expected/=multiplier
        current_tokens=[token for token in tokens if numeric(token) is not None and
            abs(centre(token)[0]-now_col['x'])<column_tolerance and abs(centre(token)[1]-ly)<30 and
            math.isclose(numeric(token),expected,abs_tol=.011)]
        if len(current_tokens)!=1:
            raise ValueError('Current PDF '+name+' does not uniquely match the independent XBRL value')
        anchor=current_tokens[0];ax,ay=centre(anchor)
        slope=(ay-ly)/(ax-lx)
        if abs(slope)>.06:
            raise ValueError('Financial row angle is too ambiguous')
        candidates=[token for token in tokens if numeric(token) is not None and
            abs(centre(token)[0]-old_col['x'])<column_tolerance and
            abs(centre(token)[1]-(ay+slope*(centre(token)[0]-ax)))<8]
        if len(candidates)!=1:
            raise ValueError('Prior-year PDF '+name+' row is ambiguous or low-confidence')
        earlier=candidates[0]
        return (numeric(anchor)*multiplier,numeric(earlier)*multiplier,
                {'current_token':anchor,'prior_year_token':earlier,'row_label':label})
    rev=row_values('revenue',revenue_labels[0],financial['revenue_inr'])
    profit_candidates=[]
    for label in profit_labels:
        try:
            profit_candidates.append(row_values('profit',label,financial['profit_inr']))
        except ValueError:
            continue
    if not profit_candidates:
        raise ValueError('No profit row matches the current XBRL fact and a unique year-earlier quarter value')
    if len({(value[0],value[1]) for value in profit_candidates})!=1:
        raise ValueError('Competing total-profit rows have different comparative values; manual source review is required')
    for name,value in [('revenue',rev),('profit',profit_candidates[0])]:
        results[name+'_current_inr']=value[0]
        results[name+'_prior_year_quarter_inr']=value[1]
        proof[name]=value[2]
    if results['revenue_prior_year_quarter_inr']<=0:
        raise ValueError('Prior-year revenue is missing or non-positive; no growth is imputed')
    old_profit=results['profit_prior_year_quarter_inr']
    return {**results,'current_quarter_end':str(current),'prior_year_quarter_end':str(prior),'basis':financial['basis'],
        'consolidation_basis_authority':basis_authority,'extraction_method':ocr.get('extraction_method','rapidocr'),
        'revenue_yoy_change_pct':(results['revenue_current_inr']/results['revenue_prior_year_quarter_inr']-1)*100,
        'profit_change_scaled_to_abs_prior_pct':(results['profit_current_inr']-old_profit)/abs(old_profit)*100 if old_profit else None,
        'profit_change_note':'Difference divided by absolute prior profit; not a conventional growth rate when prior profit is negative, and not earnings surprise.',
        'unit_multiplier_inr':multiplier,'ocr_proof':proof,'current_quarter_header':{**now_col,'date':str(current)},
        'prior_year_header':{**old_col,'date':str(prior)}}


def run():
    ocr=json.loads(OCR.read_text(encoding='utf-8'))
    if hashlib.sha256((ROOT/ocr['source_pdf_path']).read_bytes()).hexdigest()!=ocr['source_pdf_sha256']:
        raise ValueError('Original financial PDF changed since OCR')
    if hashlib.sha256((ROOT/ocr['page_render_path']).read_bytes()).hexdigest()!=ocr['render_sha256']:
        raise ValueError('Reviewed page image changed since OCR')
    row=next(r for r in (json.loads(line) for line in FINANCIALS.read_text(encoding='utf-8').splitlines()) if r['ipo_id']==313)
    if row['financials']['published_at_ist']>row['decision_date']+'T15:30:00':
        raise ValueError('The financial filing was not available by this historical decision cutoff')
    result=extract(ocr,row['financials'])
    artifact={'ipo_id':313,'symbol':row['symbol'],'decision_date':row['decision_date'],
        'financial_published_at_ist':row['financials']['published_at_ist'],
        'source_attachment_url':'https://nsearchives.nseindia.com/na_attachments/ELIN_NA_11022023125439_1.zip',
        'source_pdf_sha256':ocr['source_pdf_sha256'],'source_pdf_path':ocr['source_pdf_path'],'page_number':ocr['page_number'],
        'raw_ocr_path':OCR.relative_to(ROOT).as_posix(),'raw_ocr_sha256':hashlib.sha256(OCR.read_bytes()).hexdigest(),
        'feature_status':'source_checked_single_case_prototype_not_a_training_or_accuracy_result',
        'visual_review':'Complete original page rendered and inspected; current and year-earlier quarter columns match extracted values.',
        'financial_changes':result,'limitations':['One source-checked example cannot support a model fit or an accuracy claim.',
            'Prior-year figures are as presented in this dated current filing and may have been regrouped/restated.',
            'No consensus expectation, earnings surprise or future stock outcome is inferred.']}
    OUTPUT.write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in artifact.items() if key!='financial_changes'},indent=2))
    print(json.dumps({key:value for key,value in result.items() if key!='ocr_proof'},indent=2))
    return artifact


if __name__=='__main__':
    run()
