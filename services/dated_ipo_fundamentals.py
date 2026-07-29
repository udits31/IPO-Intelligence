"""As-of RHP features. Never copy the old unproven DRHP defaults into a model."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'data/audit/ipo_dated_fundamentals_all_20261024.json'


def load_registry(path=REGISTRY):
    import pymupdf
    from ml.extract_dated_ipo_fundamentals import document_date,eps_rows,pure_offer_ratio
    from ml.extract_all_ipo_fundamentals import annual_eps,annual_eps_from_tables,offer_components
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version')==2 and payload['processed_count']!=payload['expected_issuer_count']:
        raise ValueError('Full-cohort extraction is still incomplete')
    if payload.get('schema_version')==2:
        from ml import extract_all_ipo_fundamentals
        if payload['extractor_sha256']!=hashlib.sha256(Path(extract_all_ipo_fundamentals.__file__).read_bytes()).hexdigest():
            raise ValueError('Extractor changed; regenerate dated facts before training')
        if len(payload['records'])!=payload['processed_count'] or len({r['ipo_id'] for r in payload['records']})!=len(payload['records']):
            raise ValueError('Incomplete or duplicated issuer coverage')
    records=payload['records']
    result={}
    for record in records:
        if record['status']!='dated_document_facts':
            continue
        pdf=(ROOT/record['source_path']).resolve()
        pdf.relative_to((ROOT/'data/prospectus_pdfs').resolve())
        if hashlib.sha256(pdf.read_bytes()).hexdigest()!=record['source_sha256']:
            raise ValueError('Dated RHP source changed; re-extract before using financial facts')
        with pymupdf.open(pdf) as document:
            cover='\n'.join(document[i].get_text(sort=True) for i in range(min(record.get('cover_page_count',3),len(document))))
            if document_date(cover)!=record['document_date']:
                raise ValueError('RHP date no longer reproduces its source')
            facts=record['facts']
            checked=set()
            for evidence in record['evidence']:
                text='\n'.join(document[page-1].get_text(sort=True) for page in evidence.get('pages',[evidence['page']]))
                if hashlib.sha256(text.encode()).hexdigest()!=evidence['page_text_sha256']:
                    raise ValueError('Financial evidence page does not match its source')
                if evidence['kind'] in ('annual_eps','annual_eps_table'):
                    annual=(annual_eps_from_tables(document[evidence['page']-1]) if evidence['kind']=='annual_eps_table' else
                            annual_eps(text) if record.get('extractor_version')=='native-rhp-all-v2' else eps_rows(text))
                    if (annual!=evidence['annual_rows'] or not annual or
                        annual[0]['diluted_eps']!=facts['annual_diluted_eps'] or
                        annual[1]['diluted_eps']!=facts['previous_annual_diluted_eps'] or
                        facts['financial_period_end']!=f"{annual[0]['year']}-03-31"):
                        raise ValueError('Annual EPS does not reproduce the source table')
                    checked.add('annual_diluted_eps')
                elif evidence['kind']=='pure_offer_structure':
                    if pure_offer_ratio(text)!=facts['offer_for_sale_ratio']:
                        raise ValueError('OFS ratio does not reproduce the source statement')
                    checked.add('offer_for_sale_ratio')
                elif evidence['kind']=='offer_component':
                    if offer_components(text).get(evidence['field'])!=evidence['value'] or facts[evidence['field']]!=evidence['value']:
                        raise ValueError('Offer component does not reproduce the source statement')
                    checked.add(evidence['field'])
            if any(key in facts and key not in checked for key in ('annual_diluted_eps','offer_for_sale_ratio',
                'fresh_amount_rs','ofs_amount_rs','fresh_shares','ofs_shares')):
                raise ValueError('Fundamental fact lacks a checked source page')
        ipo_id=int(record['ipo_id'])
        if ipo_id in result:
            raise ValueError('Duplicate dated issuer facts')
        result[ipo_id]=record
    return result


def at_decision(record, decision_date, offer_cap_price):
    missing=dict(earnings_yield_pct=None,eps_growth_asinh=None,loss_making=None,ofs_ratio=None,
                 evidence_status='missing',financial_observed=False,offer_observed=False,
                 offer_evidence_status='missing')
    if record is None:
        return missing
    decision=pd.Timestamp(decision_date)
    published=pd.Timestamp(record['document_date'])
    # A date-only document is not available at 00:00 on its own dated day.
    if pd.isna(decision) or pd.isna(published) or published.date()>=decision.date():
        return {**missing,'evidence_status':'not_available_before_decision'}
    facts=record.get('facts',{})
    result={**missing,'evidence_status':'retrospective_dated_rhp'}
    if 'annual_diluted_eps' in facts:
        period=pd.Timestamp(facts['financial_period_end'])
        if period.date()>=published.date():
            raise ValueError('Future earnings period cannot enter dated RHP facts')
        eps=float(facts['annual_diluted_eps'])
        previous=float(facts['previous_annual_diluted_eps'])
        if not math.isfinite(eps) or not math.isfinite(previous):
            raise ValueError('Invalid EPS observation')
        result['loss_making']=float(eps<0)
        result['financial_observed']=True
        if offer_cap_price is not None and math.isfinite(float(offer_cap_price)) and float(offer_cap_price)>0:
            result['earnings_yield_pct']=100*eps/float(offer_cap_price)
        if previous>0:
            result['eps_growth_asinh']=math.asinh((eps-previous)/previous)
    if 'offer_for_sale_ratio' in facts:
        ofs=float(facts['offer_for_sale_ratio'])
        if not math.isfinite(ofs) or not 0<=ofs<=1:
            raise ValueError('Invalid OFS ratio')
        result['ofs_ratio']=ofs
        result['offer_observed']=True
        result['offer_evidence_status']='dated_explicit_ratio'
    else:
        cap=float(offer_cap_price) if offer_cap_price is not None and pd.notna(offer_cap_price) else None
        fresh=facts.get('fresh_amount_rs')
        sale=facts.get('ofs_amount_rs')
        # A misaligned table can associate reservation/partial-holder proceeds
        # with the whole OFS. Do not choose the convenient conflicting value.
        for prefix in ('fresh','ofs'):
            amount=facts.get(prefix+'_amount_rs');shares=facts.get(prefix+'_shares')
            if amount is not None and shares is not None and cap is not None and cap>0:
                expected=float(shares)*cap
                if expected<=0 or abs(float(amount)/expected-1)>.1:
                    result['offer_evidence_status']='conflicting_amount_and_share_count'
                    return result
        if fresh is None and cap is not None and cap>0 and facts.get('fresh_shares') is not None:
            fresh=float(facts['fresh_shares'])*cap
        if sale is None and cap is not None and cap>0 and facts.get('ofs_shares') is not None:
            sale=float(facts['ofs_shares'])*cap
        if fresh is not None and sale is not None:
            fresh,sale=float(fresh),float(sale)
            if not all(math.isfinite(value) and value>0 for value in (fresh,sale)):
                raise ValueError('Invalid mixed-offer amounts')
            result['ofs_ratio']=sale/(fresh+sale)
            result['offer_observed']=True
            result['offer_evidence_status']='dated_components'
    return result
