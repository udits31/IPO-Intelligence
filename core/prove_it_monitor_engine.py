"""Versioned prove-it policy. Parameters are proposed rules, not trading laws."""
import numpy as np
import pandas as pd
from services.point_in_time import finite,eligible
from core.features import availability

VERSION='prove-it-structural-20261026-v1'


def listing_day_rule(bars,*,continuous_open_timestamp,decision_timestamp,listing_open,issue_price):
    """First 15 completed continuous minutes; never use indicative auction bars."""
    opened=pd.Timestamp(continuous_open_timestamp);cutoff=pd.Timestamp(decision_timestamp)
    if opened.tzinfo is None or cutoff.tzinfo is None:raise ValueError('Timezone-aware session timestamps required')
    expected={opened+pd.Timedelta(minutes=i) for i in range(15)};valid=[]
    for row in bars:
        if row.get('session_phase')!='CONTINUOUS' or not eligible(row,decision_timestamp):continue
        start=pd.Timestamp(row['bar_start_ist']);end=pd.Timestamp(row['bar_end_ist'])
        if start.tzinfo is None or end-start!=pd.Timedelta(minutes=1) or end>cutoff or start<opened:continue
        valid.append({**row,'start':start,'end':end})
    first={r['start']:r for r in valid if r['start'] in expected}
    if set(first)!=expected:return dict(action='DATA_PENDING',reason='Fifteen verified completed continuous-session bars are required; auction indications are excluded.')
    floor=min(float(r['low']) for r in first.values());after=sorted((r for r in valid if r['start']>=opened+pd.Timedelta(minutes=15)),key=lambda r:r['start'])
    breach=next((r for r in after if float(r['close'])<floor),None)
    return dict(action='EXIT / REDUCE' if listing_open<=issue_price and breach else 'WATCH',first_15m_low=floor,
        signal_timestamp=breach['end'].isoformat() if listing_open<=issue_price and breach else None,
        execution_filled=False,reason='Proposed weak-opening/first-15-minute-floor signal. This does not establish a sell fill or validated accuracy.')


def decision(features):
    f=features
    if f.get('sessions',0)<20:return dict(action='WATCH',sell=None,gates={},reasons=['Fewer than 20 completed sessions; the mature-position policy is not applicable.'],policy=VERSION)
    close=finite(f.get('close'));issue=finite(f.get('issue_price'));low=finite(f.get('day1_low'))
    if close is None or issue is None:return dict(action='DATA_PENDING',sell=None,gates={},reasons=['A comparable issue price and completed close are required.'],policy=VERSION)
    gates={
      'issue_and_listing_floor':None if low is None else close>issue*1.01 and close>low,
      'trend':None if any(finite(f.get(k)) is None for k in ('sma_20','sma_10','ret_10d')) else close>f['sma_20'] and f['sma_10']>=.985*f['sma_20'] and f['ret_10d']>=-.02,
      'relative_strength':None if finite(f.get('rs_vs_nifty_20d')) is None else f['rs_vs_nifty_20d']>=-.01,
      'drawdown':None if finite(f.get('drawdown_pct')) is None else f['drawdown_pct']<=max(10.,2.5*f['atr_14_pct']) if finite(f.get('atr_14_pct')) is not None else f['drawdown_pct']<=10.,
      'filings_and_flow':None,
    }
    flow=f.get('delivery_accumulation_ratio');sentiment=finite(f.get('filing_sentiment_score'))
    clean=sentiment is not None and sentiment>-.30 and f.get('hard_governance_red_flag') is not True
    # A missing unlock calendar cannot certify an absence of supply risk.
    unlock=f.get('unabsorbed_unlock_window')
    if sentiment is not None and unlock is not None:
        gates['filings_and_flow']=bool(clean and not unlock and (flow is None or flow>=.90))
    gates={k:bool(v) if v is not None else None for k,v in gates.items()}
    if close<=issue*1.01 or (low is not None and close<=low):
        action='EXIT / REDUCE';reason='The close failed the proposed issue-price or listing-day floor rule.'
    elif all(v is True for v in gates.values()):
        action='ACCUMULATE' if close>f['sma_10'] and f['rs_vs_nifty_20d']>.03 else 'HOLD'
        reason='All five proposed survival gates passed with the available dated evidence.'
    else:
        action='REVIEW / REDUCE';reason='The position has not passed every survival gate; protect gains or review exposure.'
    missing=[k for k,v in gates.items() if v is None];failed=[k for k,v in gates.items() if v is False]
    return dict(action=action,sell=action not in ('HOLD','ACCUMULATE'),gates=gates,missing_gates=missing,failed_gates=failed,
        reasons=[reason]+(['Insufficient evidence: '+', '.join(missing)] if missing else [])+(['Failed checks: '+', '.join(failed)] if failed else []),
        policy=VERSION,validated_probability=None)


def features(prices,*,issue_price,decision_timestamp,delivery=(),filings=(),anchor=None,archive_assumption=False):
    cutoff=pd.Timestamp(decision_timestamp);frame=prices.copy();frame['date']=pd.to_datetime(frame.date)
    frame=frame[frame.date.dt.date<=cutoff.date()].sort_values('date')
    if cutoff.hour<15 or (cutoff.hour==15 and cutoff.minute<30):frame=frame[frame.date.dt.date<cutoff.date()]
    if frame.empty:return dict(sessions=0)
    key='close_price_adjusted' if 'close_price_adjusted' in frame else 'close'
    close=pd.to_numeric(frame[key],errors='coerce');f=dict(sessions=len(frame),close=float(close.iloc[-1]),issue_price=float(issue_price))
    f['drawdown_pct']=100*(1-close.iloc[-1]/close.max())
    if len(close)>=10:f['sma_10']=float(close.tail(10).mean())
    if len(close)>=20:f['sma_20']=float(close.tail(20).mean())
    for lag in (5,10,20):
        if len(close)>lag:f['ret_'+str(lag)+'d']=float(close.iloc[-1]/close.iloc[-lag-1]-1)
    if len(close)>20 and 'nifty50_close' in frame:
        n=frame.nifty50_close
        if pd.notna(n.iloc[-1]) and pd.notna(n.iloc[-21]) and n.iloc[-21]>0:f['rs_vs_nifty_20d']=f['ret_20d']-float(n.iloc[-1]/n.iloc[-21]-1)
    by_date={};assumed=0
    for row in delivery:
        allowed=eligible(row,decision_timestamp)
        if not allowed and archive_assumption:
            admission=availability(kind='EXCHANGE_EOD_TAPE',decision_timestamp=decision_timestamp,trading_date=row['session_date'],allow_historical_assumptions=True)
            allowed=admission['eligible'];assumed+=int(allowed)
        if allowed and pd.Timestamp(row['session_date']).date()<=cutoff.date():by_date[row['session_date']]=row
    first_date=str(frame.iloc[0].date.date());first=by_date.get(first_date,{})
    if finite(first.get('low')) is not None:f['day1_low']=float(first['low'])*float(frame.iloc[0].get('shares_per_listing_share',1.))
    if 'low' in frame and finite(frame.iloc[0]['low']) is not None:f['day1_low']=float(frame.iloc[0]['low'])
    daily=frame.set_index(frame.date.dt.strftime('%Y-%m-%d'));flow_rows=[];tr=[]
    for index,row in enumerate(frame.itertuples()):
        date=str(row.date.date());tape=by_date.get(date,{})
        factor=float(getattr(row,'shares_per_listing_share',1.));prev=float(close.iloc[index-1]) if index else None
        high=finite(tape.get('high'));low=finite(tape.get('low'));qty=finite(tape.get('deliverable_qty'))
        if high is not None and low is not None:
            high*=factor;low*=factor;tr.append(max(high-low,abs(high-prev),abs(low-prev)) if prev is not None else high-low)
        if prev is not None and qty is not None:flow_rows.append((float(close.iloc[index]-prev),qty))
    if len(tr)>=14:f['atr_14_pct']=100*float(np.mean(tr[-14:]))/close.iloc[-1]
    if len(flow_rows)>=10:
        up=sum(q for change,q in flow_rows[-10:] if change>0);down=sum(q for change,q in flow_rows[-10:] if change<0)
        f.update(up_day_delivery_vol_10d=up,down_day_delivery_vol_10d=down,delivery_accumulation_ratio=up/max(1.,down))
    news=list({(r.get('broadcast_datetime_ist'),r.get('subject'),r.get('attachment_url')):r for r in filings if eligible(r,decision_timestamp)}.values())
    if news:
        f['filing_sentiment_score']=float(np.mean([r['filing_sentiment_score'] for r in news]));f['hard_governance_red_flag']=any(r.get('hard_governance_red_flag') is True for r in news)
    # Explicit calendar values may be verified or admitted under the separate
    # historical static-calendar assumption, never inferred from listing date.
    if anchor and anchor.get('allotment_date') and anchor.get('calendar_usable') is True:
        days=(cutoff.date()-pd.Timestamp(anchor['allotment_date']).date()).days
        f['days_since_allotment']=days;f['unabsorbed_unlock_window']=bool((26<=days<=35 or 82<=days<=92) and f.get('ret_5d',0)<0)
    f['coverage']=dict(delivery_sessions=len(by_date),delivery_archive_assumption_rows=assumed,filings=len(news),day1_low_observed='day1_low' in f,anchor_calendar_usable='unabsorbed_unlock_window' in f)
    return f
