"""Normalize captured public DOM; preserve timing gaps and frozen model inputs."""
import calendar,hashlib,json,math,re,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from services.historical_bid_service import name_key

ARCHIVE=ROOT/'data/archive/ipopremium_20261026'


def number(value):
    text=str(value).replace(',','').replace('₹','').replace('%','').replace('−','-').strip()
    if re.fullmatch(r'\([\d.]+\)',text):text='-'+text[1:-1]
    return float(text) if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',text) else None


def period(value):
    text=value.strip();match=re.fullmatch(r'([A-Za-z]{3})[- ](\d{2}|\d{4})',text)
    if match:
        month=datetime.strptime(match[1].title(),'%b').month;year=int(match[2]);year=year+2000 if year<100 else year
        return pd.Timestamp(year,month,calendar.monthrange(year,month)[1])
    return pd.to_datetime(text,dayfirst=True,errors='coerce') if re.search(r'\d{4}|\d{1,2}-[A-Za-z]{3}-\d{2}',text) else pd.NaT


def amount(text):
    m=re.search(r'(?:₹\s*)?([\d,.]+)\s*(cr(?:ore)?s?|lakhs?)\b',text,re.I)
    value=number(m[1]) if m else None
    return value*(.01 if m[2].lower().startswith('lakh') else 1) if value is not None else None


def component(text):
    shares=re.search(r'(\d[\d,]*)\s*shares',text,re.I)
    return dict(shares=int(shares[1].replace(',','')) if shares else None,cr=amount(text))


def parse(page,reference):
    text=' '.join(page['text'].split());tables=page['tables'];expected=page['expected']
    price=re.search(r'PRICE BAND\s*₹\s*([\d,.]+)(?:\s*[–-]\s*([\d,.]+))?',text,re.I)
    cap=float((price[2] or price[1]).replace(',','')) if price else None
    floor=float(price[1].replace(',','')) if price else None
    listed=re.search(r'Listed on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',text,re.I)
    listing=pd.to_datetime(listed[1],errors='coerce') if listed else pd.NaT
    cap_reference=page.get('_reference_cap',expected.get('issue_price'))
    valid_cap=cap is not None and cap_reference is not None and (abs(cap-float(cap_reference))<.011 or floor<=float(cap_reference)<=cap)
    # TMB's published band is 500-525 while the reference issue price is 510.
    # The issuer/date match is retained; the site incorrectly repeats 525-525.
    official_band_source=None
    if expected['issuer_key']=='332' and cap==525 and expected.get('issue_price')==510:
        valid_cap=True;official_band_source='https://www.nseindia.com/market-data/issue-information?series=EQ&symbol=TMB&type=Active'
    valid_date=pd.notna(listing) and str(listing.date())==str(expected.get('listing_date'))[:10]
    result=dict(issuer_key=expected['issuer_key'],company_name=expected['company_name'],source_url=page['source_url'],fetch_timestamp=page['fetched_at'],
        source_file=page['_path'],sha256=page['_sha256'],identity_verified=bool(valid_cap and valid_date),identity_checks=dict(cap_price=cap,cap_reference=cap_reference,cap_matches=valid_cap,official_band_resolution=official_band_source,listing_date=str(listing.date()) if pd.notna(listing) else None,listing_matches=valid_date),
        daywise=[],offer={},financial={},reservation={},application_counts={},documents=[],lead_managers=[],limitations=[])
    if not result['identity_verified']:return result
    cutoff=pd.Timestamp(reference.get('cutoff_at_ist') or str(expected['listing_date'])+'T00:00:00+05:30')
    year=listing.year
    for label,key in [('Open','open_date'),('Close','close_date'),('Allotment','allotment_date')]:
        m=re.search(r'\b'+label+r'\s+([A-Za-z]{3})\s+(\d{1,2})\s*\(completed\)',text)
        if m:
            date=pd.to_datetime(f'{m[1]} {m[2]} {year}')
            if date>listing:date=date.replace(year=year-1)
            result[key]=str(date.date())
    updated=re.search(r'Last updated on\s+(\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2}:\d{2})',text,re.I)
    result['last_table_update_ist']=pd.to_datetime(updated[1],format='%d-%b-%Y %H:%M:%S').tz_localize('Asia/Kolkata').isoformat() if updated else None
    lot=re.search(r'LOT SIZE\s+([\d,]+)\s+shares',text,re.I)
    if lot:result['lot_size_shares']=int(lot[1].replace(',',''))
    eps_by_period=[];own_nav=None;own_eps=None;financial_rows=[]
    for table in tables:
        rows=[r for r in table['rows'] if r];heading=table['heading'].lower()
        if not rows:continue
        if 'day-wise' in heading:
            days=[int(re.search(r'\d+',v)[0]) for v in rows[0][1:] if re.search(r'day\s*\d+',v,re.I)]
            for index,day in enumerate(days,1):
                values={r[0]:number(r[index]) for r in rows[1:] if len(r)>index}
                result['daywise'].append(dict(day=day,values=values,source_timestamp=None,timing_basis='day_number_only_no_individual_publication_clock'))
        if heading=='share reservation':
            for r in rows[1:]:
                if len(r)>=3:result['reservation'][r[0]]=dict(shares=number(r[1]),pct=number(r[2]))
        if heading=='subscription status' and len(rows[0])>=4:
            if rows[0][1].lower()=='reserved':
                result['application_counts']={r[0]:dict(reserved=number(r[1]),applied=number(r[2]),times=number(r[3]),quality='publisher_approximate_not_validated_allotment_basis') for r in rows[1:] if len(r)>=4}
            elif rows[0][1].lower()=='offered' and 'subscription_shares' not in result:
                result['subscription_shares']={r[0]:dict(offered=number(r[1]),applied=number(r[2]),times=number(r[3])) for r in rows[1:] if len(r)>=4}
        for r in rows:
            if len(r)<2:continue
            label=re.sub(r'[^a-z]','',r[0].lower());value=' '.join(r[1:])
            if label in ('freshissue','offerforsale','totalissuesize','issuesize') and ('detail' in heading or 'company' in heading):
                result['offer'][{'freshissue':'fresh','offerforsale':'ofs','totalissuesize':'total','issuesize':'total'}[label]]=component(value)
            if label in ('shareholdingpreissue','shareholdingpostissue'):
                result['offer'][label]=number(r[1])
        if ('performance indicator' in heading or heading=='key performance indicator') and len(rows)>1:
            if len(rows[0])>2:
                for r in rows[1:]:
                    if re.sub(r'[^a-z]','',r[0].lower()) in ('basiceps','dilutedeps','eps','basicepsrs','epsbasic'):
                        for col in range(1,min(len(r),len(rows[0]))):
                            date=period(rows[0][col]);value=number(r[col])
                            if pd.notna(date) and date<cutoff.tz_localize(None) and (date.month,date.day) in ((3,31),(12,31)) and value is not None:eps_by_period.append((date,value))
            else:
                for r in rows[1:]:
                    label=re.sub(r'[^a-z]','',r[0].lower())
                    if len(r)>1 and label in ('pbv','pricebook','pb'):
                        value=number(r[1])
                        if value is not None and value>0:result['financial']['ipo_price_book_raw']=value
                    if len(r)>1 and label in ('eps','basiceps','dilutedeps','epsrs'):
                        value=number(r[1])
                        if value is not None:result['financial']['ipo_pre_offer_eps_unperiodized']=value;result['financial']['ipo_earnings_yield_pct_unperiodized']=100*value/cap
                    if len(r)>1 and label in ('pe','peratio','priceearning','priceearnings'):
                        value=number(r[1])
                        if value is not None and value>0:result['financial']['ipo_pe_raw_unperiodized']=value
        if ('financial' in heading or 'financial information' in heading) and any(r and r[0].lower().strip() in ('profit after tax','profit','pat') for r in rows):
            header=rows[0]
            for col in range(1,len(header)):
                date=period(header[col])
                if pd.isna(date) or date>=cutoff.tz_localize(None) or (date.month,date.day) not in ((3,31),(12,31)):continue
                values={re.sub(r'[^a-z]','',r[0].lower()):number(r[col]) for r in rows[1:] if len(r)>col}
                financial_rows.append((date,values))
        if 'peer comparison' in heading and 'financial' in heading:
            for r in rows[1:]:
                if len(r)<4:continue
                key=name_key(r[0]);own=name_key(expected['company_name'])
                if key==own or (len(own)>8 and key.startswith(own)):
                    own_nav=number(r[1]);own_eps=number(r[3]);result['financial']['own_peer_row']=r
    offer=result['offer'];fresh=offer.get('fresh',{});ofs=offer.get('ofs',{});total=offer.get('total',{})
    if total.get('shares') and ofs.get('shares') is not None and 0<=ofs['shares']<=total['shares']:
        offer['ofs_ratio']=ofs['shares']/total['shares'];offer['ratio_basis']='offer_component_share_counts'
    elif fresh.get('cr') is not None and ofs.get('cr') is not None and fresh['cr']+ofs['cr']>0:
        offer['ofs_ratio']=ofs['cr']/(fresh['cr']+ofs['cr']);offer['ratio_basis']='offer_component_rupee_amounts'
    elif total.get('shares') and fresh.get('shares')==total['shares']:
        offer['ofs_ratio']=0.;offer['ratio_basis']='all_fresh_share_count'
    if eps_by_period:
        date,eps=max(eps_by_period,key=lambda r:r[0]);result['financial'].update(ipo_pre_offer_eps=eps,eps_annual_period=str(date.date()),ipo_earnings_yield_pct=100*eps/cap,valuation_basis='annual_KPI_EPS_and_matched_offer_price')
    elif own_eps is not None:
        result['financial'].update(ipo_pre_offer_eps_unperiodized=own_eps,ipo_earnings_yield_pct_unperiodized=100*own_eps/cap)
    if own_nav is not None and own_nav>0:result['financial'].update(ipo_price_book_raw=cap/own_nav,nav_basis='own_company_IPO_peer_table_period_not_explicit',nav_per_share=own_nav)
    if result['financial'].get('ipo_price_book_raw',0)>0:result['financial']['log_ipo_price_book']=math.log1p(result['financial']['ipo_price_book_raw'])
    if financial_rows:
        financial_rows.sort(key=lambda r:r[0],reverse=True);date,v=financial_rows[0];result['financial'].update(annual_period=str(date.date()),annual_table=v)
        result['financial']['annual_rows']=[dict(period=str(d.date()),values=values) for d,values in financial_rows]
        revenue=next((v[k] for k in ('revenue','revenuefromoperations','totalincome','netsales') if v.get(k) is not None),None)
        pat=v.get('profitaftertax',v.get('pat'));assets=v.get('assets');worth=v.get('networth');debt=v.get('totalborrowing',v.get('borrowing'))
        if pat is not None:
            result['financial']['annual_loss']=int(pat<0)
            if revenue and revenue>0:result['financial']['profit_margin_pct']=100*pat/revenue
            if assets and assets>0:result['financial']['return_on_assets_pct']=100*pat/assets
        if debt is not None and worth and worth>0:result['financial']['debt_to_networth']=debt/worth
        if len(financial_rows)>1 and (date-financial_rows[1][0]).days in (365,366) and revenue is not None:
            prior=financial_rows[1][1];old=next((prior[k] for k in ('revenue','revenuefromoperations','totalincome','netsales') if prior.get(k) is not None),None)
            if old and old>0:result['financial']['revenue_growth_asinh']=float(np.arcsinh((revenue-old)/old))
    for link in page.get('links',[]):
        if link['text'] in ('DRHP / RHP','Anchor List','Capital Structure'):result['documents'].append(link)
        if '/view/lead-manager/' in link['url']:result['lead_managers'].append(link['text'])
    result['lead_managers']=sorted(set(result['lead_managers']))
    result['limitations']=['Public IPO context recovered later; original publication instants and revision history are not certified.','Day-wise columns have no individual capture clock; a post-close last-update time is not a pre-17:00 observation.','Application counts are explicitly approximate, not a final validated lottery basis.']
    return result


def run():
    frame=pd.read_csv(ROOT/'output/module1_progressive_ablation_20261026_features.csv',dtype={'issuer_key':str})
    reference={r['issuer_key']:r for r in frame.to_dict('records')};records=[];statuses=[]
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    for path in sorted(ARCHIVE.glob('ipo_*.json')):
        page=json.loads(path.read_text(encoding='utf8'));expected=page.get('expected',{})
        if re.search(r'too many requests|^\s*429\b|access denied|forbidden',page.get('text',''),re.I):
            statuses.append(dict(issuer_key=expected.get('issuer_key'),company_name=expected.get('company_name'),status='source_rate_limited_or_restricted'));continue
        if page.get('status')!='captured_public_dom':statuses.append(dict(issuer_key=expected.get('issuer_key'),company_name=expected.get('company_name'),status=page.get('status')));continue
        page['_path']=str(path.relative_to(ROOT));page['_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        key=expected['issuer_key']
        if key.isdigit() and int(key) in canonical.index and pd.notna(canonical.loc[int(key),'price_band_upper']):page['_reference_cap']=float(canonical.loc[int(key),'price_band_upper'])
        record=parse(page,reference.get(expected['issuer_key'],{}));records.append(record)
        statuses.append(dict(issuer_key=record['issuer_key'],company_name=record['company_name'],status='identity_matched' if record['identity_verified'] else 'identity_conflict',daywise_days=len(record['daywise']),source_url=record['source_url']))
    output=ROOT/'data/audit/ipopremium_recovered_20261026.jsonl';output.write_text(''.join(json.dumps(r,default=str,allow_nan=False)+'\n' for r in records),encoding='utf8')
    pd.DataFrame(statuses).to_csv(ROOT/'output/ipopremium_recovery_status_20261026.csv',index=False)
    valid=[r for r in records if r['identity_verified']]
    summary=dict(pages=len(records),identity_matched=len(valid),identity_conflicts=len(records)-len(valid),with_daywise=sum(bool(r['daywise']) for r in valid),with_ofs=sum('ofs_ratio' in r['offer'] for r in valid),with_reservation=sum(bool(r['reservation']) for r in valid),with_annual_eps=sum('ipo_earnings_yield_pct' in r['financial'] for r in valid),with_book=sum('log_ipo_price_book' in r['financial'] for r in valid),with_lead_managers=sum(bool(r['lead_managers']) for r in valid),with_anchor_document=sum(any(l['text']=='Anchor List' for l in r['documents']) for r in valid),retrains_run=0)
    (ROOT/'output/ipopremium_recovery_summary_20261026.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary),flush=True)
    return records


if __name__=='__main__':run()
