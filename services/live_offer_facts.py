"""Live-only offer extraction. Historical research parsers stay frozen."""
import json
import math
import re
from bs4 import BeautifulSoup

OFFER_PARSER_VERSION='offer_facts_v4_20261028'


def number(value):
    try:
        n=float(str(value).replace(',','').strip())
        return n if math.isfinite(n) and n>=0 else None
    except (ValueError,TypeError):return None


def parse_offer(html, cap_price):
    from ml.extract_cached_deal_structure import parse,amount_cr
    result=parse(html,cap_price)
    soup=BeautifulSoup(html,'html.parser');pairs=[]
    for dt in soup.find_all('dt'):
        dd=dt.find_next_sibling('dd')
        if dd:pairs.append((dt.get_text(' ',strip=True),dd.get_text(' ',strip=True)))
    for table in soup.find_all('table'):
        if re.search(r'subscribed|subscription',table.get_text(' ',strip=True),re.I):continue
        for tr in table.find_all('tr'):
            cells=tr.find_all(['th','td'])
            if len(cells)==2:pairs.append(tuple(c.get_text(' ',strip=True) for c in cells))
    for label,value in pairs:
        key=label.casefold().strip()
        amount=amount_cr(value)
        if re.fullmatch(r'issue price|price band',key):
            band=re.search(r'([\d,]+(?:\.\d+)?)\s*(?:[-–]|to)\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)',value,re.I)
            single=re.search(r'([\d,]+(?:\.\d+)?)',value)
            cap=number(band[2] if band else single[1] if single else None)
            if cap is not None:
                if cap_price is not None and abs(cap-float(cap_price))>.01:
                    raise ValueError('Offer page cap price conflicts with issuer record')
                result['offer_cap_price']=cap
        if re.fullmatch(r'(?:total )?issue size|total offer size',key) and amount is not None:
            result['total_issue_cr']=amount
        elif re.fullmatch(r'fresh (?:issue|offer)',key) and amount is not None:
            result['fresh_issue_cr']=amount
        elif re.fullmatch(r'offer for sale(?:\s*\(ofs\))?|ofs',key) and amount is not None:
            result['ofs_issue_cr']=amount
        elif re.fullmatch(r'lot size',key):
            lot=re.fullmatch(r'([\d,]+)\s*shares',value,re.I)
            if lot:result['lot_size_shares']=int(lot[1].replace(',',''))
        elif re.fullmatch(r'retail (?:portion|quota|reservation)',key):
            pct=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*%\s*',value)
            if pct and 0<=float(pct[1])<=100:
                result['retail_quota_pct']=float(pct[1]);result['retail_quota_exact']=True
    # A chart may show 20 QIB + 15 NII + 35 Retail + 30 Anchor. Retail
    # remains 35%; do not renormalise it to 50% or reject it for summing to 70.
    script=soup.find('script',id='ipo-reservation-chart-data')
    if script:
        try:
            chart=json.loads(script.get_text());retail=number(chart.get('retail')) if isinstance(chart,dict) else None
            if retail is not None and retail<=100 and sum(number(v) or 0 for v in chart.values())<=100.5:
                old=result.get('retail_quota_pct')
                if old is not None and abs(old-retail)>.05:raise ValueError('Conflicting retail reservations')
                result.update(retail_quota_exact=True,reservation_chart=chart)
                if old is None:result['retail_quota_pct']=retail
                elif chart.get('emp'):
                    result.update(retail_quota_basis='net_offer',retail_quota_total_pct=retail)
        except json.JSONDecodeError:pass
    total=result.get('total_issue_cr');fresh=result.get('fresh_issue_cr');ofs=result.get('ofs_issue_cr')
    if total is not None and total>0:
        if fresh is not None and ofs is None and 0<=fresh<=total:
            ofs=total-fresh;result.update(ofs_issue_cr=ofs,ofs_derivation='total_issue_minus_fresh_issue')
        elif ofs is not None and fresh is None and 0<=ofs<=total:
            fresh=total-ofs;result.update(fresh_issue_cr=fresh,ofs_derivation='reported_ofs_divided_by_total_issue')
        if fresh is not None and ofs is not None and abs(fresh+ofs-total)>max(.03,total*.001):
            raise ValueError('Offer components conflict with total issue size')
    if fresh is not None and ofs is not None and fresh+ofs>0:
        result['ofs_ratio']=ofs/(fresh+ofs)
    for table in soup.find_all('table'):
        rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])] for tr in table.find_all('tr')]
        if not rows:continue
        headers=[v.casefold() for v in rows[0]]
        net=next((i for i,v in enumerate(headers) if '% of net' in v),None)
        gross=next((i for i,v in enumerate(headers) if '% of total' in v),None)
        for row in rows[1:]:
            if row and row[0].casefold()=='retail' and net is not None and len(row)>net:
                pct=number(row[net].rstrip('%'))
                if pct is not None and pct<=100:
                    result.update(retail_quota_pct=pct,retail_quota_exact=True,retail_quota_basis='net_offer')
                    if gross is not None and len(row)>gross:
                        result['retail_quota_total_pct']=number(row[gross].rstrip('%'))
    objects=soup.find('table',id='ObjectiveIssue')
    if objects and re.search(r'Cr',objects.get_text(' ',strip=True),re.I):
        uses=[];reported_total=None
        for tr in objects.find_all('tr'):
            cells=[c.get_text(' ',strip=True) for c in tr.find_all('td')]
            if len(cells)<3:continue
            amount=number(cells[-1])
            if cells[1].casefold()=='total':reported_total=amount;continue
            uses.append(dict(description=cells[1],amount_cr=amount))
        if (uses and reported_total and fresh and fresh>0
                and abs(sum(u['amount_cr'] or 0 for u in uses)-reported_total)<=max(.05,reported_total*.01)):
            capex=0.;classified=True
            for use in uses:
                if re.search(r'capital expenditure|\bcapex\b|capital equipment|setting.up|establish|plant.{0,20}machinery|purchase.{0,30}equipment|expansion',use['description'],re.I):
                    if use['amount_cr'] is None:classified=False
                    else:capex+=use['amount_cr']
                elif not re.search(r'borrowings|repayment|prepayment|working capital|general corporate|issue expenses|offer expenses',use['description'],re.I):
                    classified=False
            result['proceeds_uses']=uses
            if classified:
                result.update(objects_capex_ratio=capex/fresh,objects_capex_cr=capex,objects_capex_basis='gross_fresh_issue')
    # Keep the published objects even when there is no accounting split into
    # capital expenditure. Technology spending/acquisitions are not assumed to
    # be capitalised, and lending capital is not physical CAPEX.
    if not result.get('proceeds_uses'):
        objectives=soup.select('dd[id^="issue-objective-"]')
        if objectives:
            result['proceeds_uses']=[dict(description=n.get_text(' ',strip=True),
                amount_cr=amount_cr(n.get_text(' ',strip=True))) for n in objectives]
    if result.get('proceeds_uses') and result.get('objects_capex_ratio') is None:
        result['objects_capex_status']='NOT_SEPARATELY_ITEMISED'
    return result


def total_book(raw, family):
    """Use a reported total or sum non-overlapping NSE parent categories."""
    if family=='chittorgarh_investorgain':
        soup=BeautifulSoup(raw,'html.parser')
        for table in soup.find_all('table'):
            cells=[[c.get_text(' ',strip=True) for c in row.find_all(['th','td'])] for row in table.find_all('tr')]
            if not cells or len(cells[0])!=2:continue
            header=[c.casefold() for c in cells[0]]
            if 'category' not in header[0] or 'subscription' not in header[1]:continue
            totals=[row for row in cells[1:] if len(row)==2 and row[0].casefold().strip() in ('total','overall')]
            if len(totals)==1:
                match=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*[x×]?\s*',totals[0][1],re.I)
                if match:return dict(multiple=float(match[1]),method='published_subscription_total')
        return None
    if family!='nse_exchange':return None
    payload=json.loads(raw);rows=payload.get('dataList') or []
    totals=[r for r in rows if str(r.get('category','')).strip().casefold()=='total']
    if len(totals)!=1:return None
    r=totals[0];offered=number(r.get('noOfShareOffered'));bid=number(r.get('noOfSharesBid'))
    reported=number(r.get('noOfTotalMeant'))
    if not offered or bid is None:return None
    derived=bid/offered
    if reported is not None and abs(reported-derived)>max(.0001,derived*.005):
        raise ValueError('Reported total conflicts with bid/offer quantities')
    return dict(multiple=reported if reported is not None else derived,
        offered_shares=offered,bid_shares=bid,method='NSE_total_row',
        publisher_update=payload.get('updateTime'))
