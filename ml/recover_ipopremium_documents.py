"""Targeted native-text extraction of short linked anchor/capital documents."""
import hashlib,json,re,time
from datetime import datetime,timezone
import requests
import pymupdf
from services.saved_preclose_service import ROOT
from services.historical_bid_service import name_key

DIR=ROOT/'data/archive/ipopremium_20261026/documents'


def extract(text,company,kind):
    clean=' '.join(text.split());identity=name_key(company) in name_key(clean)
    out=dict(issuer_text_matches=identity,kind=kind,anchor_domestic_mf_pct=None)
    if not identity:return out
    if kind=='Anchor List':
        candidates=[];snippets=[]
        for match in re.finditer(r'domestic\s+mutual\s+funds?',clean,re.I):
            snippet=clean[max(0,match.start()-600):match.end()+100]
            if re.search(r'out\s+of\s+(?:the\s+)?total\s+allocation',snippet,re.I) and re.search(r'allocated',snippet,re.I):
                percents=re.findall(r'(\d+(?:\.\d+)?)\s*%',snippet[:snippet.lower().find('domestic')])
                if percents:
                    value=float(percents[-1])
                    if 0<=value<=100:candidates.append(value);snippets.append(snippet)
        if candidates and len(set(candidates))==1:out.update(anchor_domestic_mf_pct=candidates[0],evidence=snippets[0])
        elif candidates:out.update(candidate_percentages=candidates,review_required='Conflicting summary percentages')
    out['valuation_evidence_snippets']=[clean[max(0,m.start()-80):m.end()+260] for m in re.finditer(r'net asset value|\bNAV\b|earnings per share',clean,re.I)][:12]
    date=re.search(r'\bDate\s*:\s*([A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})',clean[:1600],re.I)
    out['document_date_text']=date[1] if date else None
    return out


def run():
    DIR.mkdir(parents=True,exist_ok=True);session=requests.Session();results=[];restricted=False
    records=[json.loads(line) for line in (ROOT/'data/audit/ipopremium_recovered_20261026.jsonl').read_text(encoding='utf8').splitlines()]
    for row in records:
        if not row['identity_verified']:continue
        for link in row['documents']:
            if link['text'] not in ('Anchor List','Capital Structure'):continue
            key=hashlib.sha256(link['url'].encode()).hexdigest();meta=DIR/(key+'.json');pdf=DIR/(key+'.pdf')
            if meta.exists():results.append(json.loads(meta.read_text(encoding='utf8')));continue
            result=dict(issuer_key=row['issuer_key'],company_name=row['company_name'],source_url=link['url'],kind=link['text'],fetch_timestamp=datetime.now(timezone.utc).isoformat())
            if restricted:result.update(status='deferred_after_source_restriction');results.append(result);continue
            try:
                time.sleep(1);response=session.get(link['url'],timeout=25);result['http_status']=response.status_code
                if response.status_code in (401,403,429):restricted=True;result['status']='access_restricted_or_rate_limited'
                else:
                    response.raise_for_status()
                    if len(response.content)>15000000:raise ValueError('Document exceeds targeted 15 MB limit')
                    pdf.write_bytes(response.content);digest=hashlib.sha256(response.content).hexdigest();document=pymupdf.open(pdf)
                    result.update(pages=len(document),sha256=digest,path=str(pdf.relative_to(ROOT)))
                    if len(document)>30:result['status']='long_document_retained_not_bulk_scanned'
                    else:
                        text='\n'.join(p.get_text() for p in document);(DIR/(key+'.txt')).write_text(text,encoding='utf8')
                        result.update(status='native_text_extracted' if text.strip() else 'image_only_no_ocr_used',**extract(text,row['company_name'],link['text']))
                    document.close()
            except Exception as exc:result.update(status='fetch_or_parse_failed',error=str(exc)[:220])
            meta.write_text(json.dumps(result,indent=2,default=str),encoding='utf8');results.append(result)
            print(json.dumps(dict(company=row['company_name'],kind=link['text'],status=result['status'],mf_pct=result.get('anchor_domestic_mf_pct'))),flush=True)
    (ROOT/'data/audit/ipopremium_documents_recovered_20261026.jsonl').write_text(''.join(json.dumps(r,default=str)+'\n' for r in results),encoding='utf8')
    print(json.dumps(dict(documents=len(results),mf_percentage_recovered=sum(r.get('anchor_domestic_mf_pct') is not None for r in results))),flush=True)
    return results


if __name__=='__main__':run()
