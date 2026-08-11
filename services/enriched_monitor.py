"""Shadow multi-factor monitor. Unknown inputs never imply institutional buying."""
from datetime import datetime,time
import math
import pandas as pd
from services.point_in_time import IST,timestamp,eligible,finite


def evaluate(features):
    f=features;reasons=[];action='HOLD';code='no_override'
    close=finite(f.get('close'));issue=finite(f.get('issue_price'));low=finite(f.get('day1_low'))
    sma10=finite(f.get('sma_10'));sma20=finite(f.get('sma_20'));absorption=finite(f.get('delivery_accumulation_ratio'))
    relative=finite(f.get('rs_vs_nifty_10d'));sentiment=finite(f.get('filing_sentiment_score'));vwap=finite(f.get('day1_vwap'))
    if close is None or issue is None:return dict(action='DATA_PENDING',decision_code='missing_price_basis',reasons=['Verified dated price and issue price are required.'])
    if f.get('hard_governance_red_flag') is True:
        action='REDUCE / EXIT';code='governance_flag';reasons.append('An eligible dated filing triggered the proposed governance-risk rule.')
    elif f.get('unlock_window_bulk_selling_flag') is True and sma10 is not None and close<sma10:
        action='REDUCE / EXIT';code='bulk_sale_and_price_weakness';reasons.append('A dated bulk sale near the verified unlock window coincides with a close below SMA10. Seller identity must be checked separately.')
    elif low is not None and close<min(issue,low):
        action='REDUCE / EXIT';code='listing_floor_break';reasons.append('The close is below both the issue price and listing-day low.')
    elif sma20 is not None and close<sma20 and ((absorption is not None and absorption<.85) or (relative is not None and relative<-.02)):
        action='REDUCE / EXIT';code='weak_structure_and_flow';reasons.append('The close is below SMA20 with weak delivery absorption or 10-session relative strength.')
    elif f.get('in_unlock_danger_zone') is True and absorption is not None and absorption<1.15:
        action='REVIEW / TRIM';code='unlock_without_absorption';reasons.append('A verified unlock window is near and delivery absorption is weak.')
    elif sentiment is not None and sentiment<=-.30:
        action='REVIEW / TRIM';code='negative_filing';reasons.append('Eligible exchange announcements have adverse sentiment.')
    elif sma10 is not None and close<sma10 and f.get('post_listing_rally_observed') is True:
        action='REVIEW / TRIM';code='rally_pullback';reasons.append('The close fell below SMA10 after a post-listing rally.')
    elif (vwap is not None and sma10 is not None and sma20 is not None and close>max(issue,vwap) and close>sma10>sma20
          and ((absorption is not None and absorption>=1.25) or f.get('verified_positive_institutional_bulk_support') is True)
          and f.get('in_unlock_danger_zone') is False and sentiment is not None and sentiment>-.20):
        action='ACCUMULATE';code='multifactor_accumulation';reasons.append('Price structure, verified flow, dated news and unlock checks passed the proposed accumulation rule.')
    else:reasons.append('No proposed exit or accumulation rule passed with the available dated evidence.')
    return dict(action=action,decision_code=code,reasons=reasons,mode='shadow_research',validated_prediction_probability=None)


def assemble(prices,*,decision_timestamp,issue_price,delivery=(),filings=(),deals=(),anchor=None,strict=True,issue_shares=None):
    cutoff=timestamp(decision_timestamp)
    if cutoff is None:raise ValueError('Timezone-aware decision timestamp required')
    frame=prices.copy();frame['date']=pd.to_datetime(frame['date'])
    frame=frame[frame.date.dt.date<=cutoff.date()].sort_values('date')
    if frame.empty:return dict(issue_price=issue_price,close=None,missing=['prices'])
    # A close for the current date is unavailable before that session closes.
    if cutoff.time()<time(15,30):frame=frame[frame.date.dt.date<cutoff.date()]
    if frame.empty:return dict(issue_price=issue_price,close=None,missing=['completed_session'])
    key='close_price_adjusted' if 'close_price_adjusted' in frame else 'close'
    close=pd.to_numeric(frame[key],errors='coerce')
    if finite(close.iloc[-1]) is None:return dict(issue_price=issue_price,close=None,missing=['valid_completed_close'])
    f=dict(issue_price=float(issue_price),close=float(close.iloc[-1]),decision_timestamp=cutoff.isoformat())
    if len(close)>=10:f['sma_10']=float(close.tail(10).mean())
    if len(close)>=20:f['sma_20']=float(close.tail(20).mean())
    for label,col in [('day1_low','low'),('day1_high','high'),('day1_vwap','vwap')]:
        if col in frame and finite(frame.iloc[0][col]) is not None:f[label]=float(frame.iloc[0][col])
    if all(c in frame for c in ('high','low')) and len(frame)>=15:
        tr=pd.concat([frame.high-frame.low,(frame.high-close.shift()).abs(),(frame.low-close.shift()).abs()],axis=1).max(axis=1)
        f['atr_14_pct']=float(tr.tail(14).mean()/close.iloc[-1]*100)
    if len(close)>=11:
        stock=float(close.iloc[-1]/close.iloc[-11]-1)
        for col,out in [('nifty50_close','rs_vs_nifty_10d'),('sector_index_close','rs_vs_sector_10d')]:
            if col in frame and frame[col].tail(11).notna().all():f[out]=stock-float(frame[col].iloc[-1]/frame[col].iloc[-11]-1)
    f['post_listing_rally_observed']=bool(close.max()>max(float(issue_price),float(close.iloc[0])))
    valid_delivery=[]
    for row in delivery:
        observed=pd.Timestamp(row['session_date']).date()
        allowed=eligible(row,decision_timestamp) if strict else observed<cutoff.date()
        if allowed and observed<=cutoff.date() and finite(row.get('delivery_pct')) is not None and finite(row.get('deliverable_qty')) is not None:valid_delivery.append(row)
    by_day={r['session_date']:r for r in valid_delivery};valid_delivery=sorted(by_day.values(),key=lambda r:r['session_date'])
    if len(valid_delivery)>=5:f['delivery_pct_5d_avg']=sum(float(r['delivery_pct']) for r in valid_delivery[-5:])/5
    if len(valid_delivery)>=10:
        temp=frame.set_index(frame.date.dt.date);ups=[];downs=[];changes=close.diff().to_dict()
        for row in valid_delivery[-10:]:
            date=pd.Timestamp(row['session_date']).date()
            if date not in temp.index:continue
            index=frame.index[frame.date.dt.date.eq(date)][0];change=changes.get(index)
            if change is not None and pd.notna(change):(ups if change>0 else downs).append(float(row['deliverable_qty']))
        if ups and downs:f['delivery_accumulation_ratio']=(sum(ups)/len(ups))/max(1.,sum(downs)/len(downs))
    admitted=list({(r.get('broadcast_datetime_ist'),r.get('subject'),r.get('attachment_url')):r for r in filings if eligible(r,decision_timestamp)}.values())
    if admitted:
        f['filing_sentiment_score']=sum(float(r['filing_sentiment_score']) for r in admitted)/len(admitted)
        f['hard_governance_red_flag']=any(r.get('hard_governance_red_flag') is True for r in admitted)
    # No observed filings does not mean a complete clean-news history.
    if anchor and all(anchor.get(k) for k in ('anchor_30d_unlock_date','anchor_90d_unlock_date')) and anchor.get('calendar_verified') is True:
        a=(pd.Timestamp(anchor['anchor_30d_unlock_date']).date()-cutoff.date()).days;b=(pd.Timestamp(anchor['anchor_90d_unlock_date']).date()-cutoff.date()).days
        f.update(days_to_30d_anchor_unlock=a,days_to_90d_anchor_unlock=b,in_unlock_danger_zone=(-5<=a<=5 or -10<=b<=5))
    admitted_deals=list({(r.get('ipo_id'),r.get('trade_date'),r.get('deal_kind'),r.get('client_name'),r.get('side'),r.get('quantity')):r for r in deals if eligible(r,decision_timestamp)}.values())
    if anchor and anchor.get('calendar_verified') is True and anchor.get('anchor_30d_unlock_date'):
        # A supplied exchange calendar is required: weekdays alone miss holidays.
        sessions=sorted({pd.Timestamp(d).date() for d in anchor.get('exchange_sessions',[])})
        unlock=pd.Timestamp(anchor['anchor_30d_unlock_date']).date()
        subsequent=[i for i,d in enumerate(sessions) if d>=unlock]
        if subsequent:
            center=subsequent[0]
            if center>=2 and len(sessions)>center+5:
                window=set(sessions[center-2:center+6])
                if any(pd.Timestamp(r['trade_date']).date() in window and str(r.get('side','')).startswith('S') for r in admitted_deals):
                    f['unlock_window_bulk_selling_flag']=True
                    f['unlock_sale_identity_basis']='Observed bulk/block seller; anchor/PE identity is not inferred.'
    if issue_shares and issue_shares>0:
        institutional=[r for r in admitted_deals if r.get('institutional_identity_verified') is True and pd.Timestamp(r['trade_date']).date()==frame.iloc[0].date.date()]
        if institutional:
            net=sum(float(r['quantity'])*(1 if r['side'].startswith('B') else -1) for r in institutional)
            f.update(day1_net_institutional_bulk_pct_float=100*net/issue_shares,verified_positive_institutional_bulk_support=net>0,bulk_denominator_basis='issue_shares_not_verified_free_float')
    f['coverage']=dict(delivery_admitted=len(valid_delivery),filings_admitted=len(admitted),deals_admitted=len(admitted_deals),strict_point_in_time=strict)
    return f
