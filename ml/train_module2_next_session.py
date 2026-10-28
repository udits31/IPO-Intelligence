"""Chronological next-session research; no promotion or raw-ledger mutation.

Forecast close-to-close separately from an executable next-open-to-close trade.
All fits and policy choices precede the later cohort. Archived news is sparse;
unknown news stays missing. This is retrospective, not prospective evidence.
"""
from pathlib import Path
import gzip
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, roc_auc_score
from services.point_in_time import eligible, timestamp
from services.filing_governance import filing_governance
from services.news_service import NewsService

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/module2_next_session_20261029'
PRICE_PATHS=[ROOT/'data/audit'/f'module2_{part}_ohlcv_{years}.csv'
             for part,years in [('development','2021_2024'),('holdout','2025_2026')]]
OFFICIAL=[ROOT/'data/audit/module2_development_prices_2021_2024_nse_extended_ca_reconciled.csv',
          ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full_ca_reconciled.csv']
NEWS=ROOT/'data/audit/module2_dated_news_issuer_verified_20261021.jsonl'
IPOS=ROOT/'data/canonical_listed_ipos.csv'
PRICE=['ret1','ret5','ret10','ret20','ma10_gap','ma20_gap','vol20','drawdown','age','close_issue']
FLOW=['volume_ratio','range_pct','close_location','gap','nifty1','nifty5','nifty20','sector5','relative5']
TEXT=['news_score','news_count','filing_score','filing_count','filing_hard']
GROUPS={'price':PRICE,'price_volume_market':PRICE+FLOW,'price_volume_market_news':PRICE+FLOW+TEXT}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def native(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,pd.Timestamp):return v.isoformat()
    raise TypeError(type(v).__name__)


def historical_news_at(records,cutoff):
    stamp=pd.Timestamp(cutoff)
    return [(t,s) for t,s in records if stamp-pd.Timedelta(days=7)<=t<stamp.normalize()]


def split_data(data):
    later=data[data.listing_year.ge(2025)].copy();first=later.decision_date.min()
    validation=data[data.listing_year.eq(2024)&data.outcome_date.lt(first)].copy()
    train=data[data.listing_year.le(2023)&data.outcome_date.lt(validation.decision_date.min())].copy()
    final=data[data.listing_year.le(2024)&data.outcome_date.lt(first)].copy()
    for a,b in [(train,validation),(final,later)]:
        if a.empty or b.empty or set(a.ipo_id)&set(b.ipo_id) or a.outcome_date.max()>=b.decision_date.min():
            raise ValueError('Issuer overlap, immature outcome, or empty chronological split')
    return train,validation,final,later


def news_archive():
    """Validate source bytes and use only previously issuer-verified headlines."""
    by_id={}; manifests={};scorer=NewsService()
    if NEWS.exists():
        for r in map(json.loads,NEWS.read_text(encoding='utf8').splitlines()):
            if r.get('status')!='verified_rss_response':continue
            path=ROOT/r['archive_path']
            with gzip.open(path,'rb') as f:raw=f.read()
            if hashlib.sha256(raw).hexdigest()!=r['raw_sha256']:raise ValueError('News archive hash mismatch')
            manifests[str(path.relative_to(ROOT))]=sha(path)
            for a in r.get('articles',[]):
                ts=pd.to_datetime(a.get('published_at'),utc=True,errors='coerce')
                if pd.isna(ts):continue
                # Existing historical archive policy excludes same-day headlines.
                by_id.setdefault(int(r['ipo_id']),{})[(str(ts),a['title'])]=(ts,scorer.analyze_sentiment(a['title']))
    return {k:list(v.values()) for k,v in by_id.items()},manifests


def build():
    ipos=pd.read_csv(IPOS,parse_dates=['listing_date']).set_index('id')
    news,manifest=news_archive();rows=[];rejected={}
    official=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in OFFICIAL]).drop_duplicates(['ipo_id','date'])
    exchange={(int(r.ipo_id),str(r.date.date())):float(r.close_price_adjusted) for r in official.itertuples()}
    seen=set()
    for path in PRICE_PATHS:
        print('Building dated daily examples: '+path.name,flush=True)
        prices=pd.read_csv(path,parse_dates=['date'])
        if prices.duplicated(['ipo_id','date']).any():raise ValueError('Duplicate price session')
        for id,g in prices.groupby('ipo_id'):
            id=int(id)
            if id in seen:raise ValueError('Issuer duplicated between input cohorts')
            seen.add(id)
            if id not in ipos.index or id in (293,307):continue
            g=g.sort_values('date').reset_index(drop=True);ipo=ipos.loc[id]
            if (g.date.iloc[0]-ipo.listing_date).days not in range(8):
                rejected[str(id)]='Listing/session identity mismatch';continue
            cols=['open','high','low','ohlcv_close','volume']
            v=g[cols].apply(pd.to_numeric,errors='coerce')
            valid=(v.notna().all(axis=1)&v[['open','high','low','ohlcv_close']].gt(0).all(axis=1)&v.volume.ge(0)
                   &v.high.ge(v[['open','ohlcv_close','low']].max(axis=1))&v.low.le(v[['open','ohlcv_close']].min(axis=1)))
            reference=np.array([exchange.get((id,str(d.date())),np.nan) for d in g.date])
            # Admit only same-basis OHLCV agreeing with reconciled exchange closes.
            comparable=np.isfinite(reference)&(np.abs(v.ohlcv_close/reference-1)<=.02)
            filing_path=ROOT/f'data/market/corporate_announcements/{id}.jsonl'
            filings=[r for r in map(json.loads,filing_path.read_text(encoding='utf8').splitlines()) if r.get('schema_version')==2] if filing_path.exists() else []
            if filing_path.exists():manifest[str(filing_path.relative_to(ROOT))]=sha(filing_path)
            prepared=[]
            for r in filings:
                t=timestamp(r.get('source_timestamp')) if r.get('source_timestamp_verified') is True else None
                if t is None and r.get('observation_verified') is True:t=timestamp(r.get('available_by_timestamp'))
                broadcast=timestamp(r.get('broadcast_datetime_ist'))
                if t is not None:prepared.append((r,t,broadcast))
            c=v.ohlcv_close;ret=c.pct_change(fill_method=None)*100
            nifty=pd.to_numeric(g.nifty50_close,errors='coerce');sector=pd.to_numeric(g.sector_index_close,errors='coerce')
            for i in range(20,min(len(g)-1,61)):
                # Reject the whole feature/outcome window if price basis is unresolved.
                if not valid.iloc[:i+2].all() or not comparable[:i+2].all():continue
                if not 0<(g.date.iloc[i+1]-g.date.iloc[i]).days<=7:continue
                day=g.date.iloc[i];cutoff=day.strftime('%Y-%m-%d')+'T18:00:00+05:30'
                stamp=pd.Timestamp(cutoff)
                def change(s,n):
                    a,b=s.iloc[i],s.iloc[i-n]
                    return float((a/b-1)*100) if pd.notna(a) and pd.notna(b) and b>0 else np.nan
                eligible_text=historical_news_at(news.get(id,[]),stamp)
                admissible=[r for r,t,b in prepared if t<=stamp]
                recent=[r for r,t,b in prepared if t<=stamp and b is not None and stamp-pd.Timedelta(days=7)<=b<=stamp]
                fs=[float(r['filing_sentiment_score']) for r in recent if pd.notna(r.get('filing_sentiment_score'))]
                governance=filing_governance(admissible,cutoff)
                back=30 if i>=30 else i
                sr=(c.iloc[i]/c.iloc[i-back]-1)*100
                sec=(sector.iloc[i]/sector.iloc[i-back]-1)*100 if pd.notna(sector.iloc[i]) and pd.notna(sector.iloc[i-back]) and sector.iloc[i-back]>0 else np.nan
                cum=(c.iloc[i]/float(ipo.issue_price)-1)*100
                # Exact currently reachable price SELL branch with unverified
                # anchor-sale evidence; WATCH/pullback maps to retaining exposure.
                spread=sr-sec
                sector_valid='broad-market proxy' not in str(g.iloc[i].get('benchmark_source','')).lower()
                rule_sell=bool(round(cum,2)<=-12 or (round(cum,2)<=-5 and sr<=-5 and sector_valid and pd.notna(spread) and spread < -6) or governance['override_sell'])
                volume_prior=v.volume.iloc[i-19:i].mean();band=v.high.iloc[i]-v.low.iloc[i]
                row=dict(ipo_id=id,company_name=ipo.company_name,listing_year=int(ipo.listing_date.year),
                    decision_date=day,decision_timestamp=cutoff,outcome_date=g.date.iloc[i+1],
                    forecast_return_pct=float((c.iloc[i+1]/c.iloc[i]-1)*100),
                    executable_return_pct=float((c.iloc[i+1]/v.open.iloc[i+1]-1)*100),
                    next_open=float(v.open.iloc[i+1]),next_close=float(c.iloc[i+1]),rule_position=int(not rule_sell),
                    ret1=ret.iloc[i],ret5=change(c,5),ret10=change(c,10),ret20=change(c,20),
                    ma10_gap=(c.iloc[i]/c.iloc[i-9:i+1].mean()-1)*100,
                    ma20_gap=(c.iloc[i]/c.iloc[i-19:i+1].mean()-1)*100,
                    vol20=ret.iloc[i-19:i+1].std(),drawdown=(c.iloc[i]/c.iloc[:i+1].max()-1)*100,
                    age=i+1,close_issue=cum,volume_ratio=v.volume.iloc[i]/volume_prior if volume_prior>0 else np.nan,
                    range_pct=100*band/c.iloc[i],close_location=(c.iloc[i]-v.low.iloc[i])/band if band>0 else np.nan,
                    gap=(v.open.iloc[i]/c.iloc[i-1]-1)*100,nifty1=change(nifty,1),nifty5=change(nifty,5),nifty20=change(nifty,20),
                    sector5=change(sector,5),relative5=change(c,5)-change(nifty,5),
                    news_score=np.mean([s for t,s in eligible_text]) if eligible_text else np.nan,
                    news_count=len(eligible_text) if eligible_text else np.nan,
                    filing_score=np.mean(fs) if fs else np.nan,filing_count=len(recent) if recent else np.nan,
                    filing_hard=int(any(r.get('hard_governance_red_flag') for r in recent)) if recent else np.nan)
                rows.append(row)
    data=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    if data.empty:raise ValueError('No eligible OHLCV windows')
    return data,dict(rejected_issuers=rejected,evidence_hashes=manifest)


def learner(kind,alpha):
    model=Ridge(alpha=alpha) if kind=='regression' else LogisticRegression(C=1/alpha,max_iter=2000,random_state=42)
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),model)


def fit_models(data,features,alpha):
    weights=1/data.groupby('ipo_id').ipo_id.transform('size');weights*=len(weights)/weights.sum()
    models={}
    for name,target in [('forecast','forecast_return_pct'),('trade','executable_return_pct'),('direction','forecast_return_pct')]:
        m=learner('classifier' if name=='direction' else 'regression',alpha)
        y=data[target].gt(0).astype(int) if name=='direction' else data[target]
        m.fit(data[features],y,**{m.steps[-1][0]+'__sample_weight':weights});models[name]=m
    return models


def trading(frame,position,fee_bps=35):
    """Long/cash, buy next open and liquidate same close, cost per round trip.

    Equal capital reserved for every eligible issuer, including cash decisions;
    no leverage, overnight fills, shorting or hindsight daily-high execution.
    """
    position=np.asarray(position,dtype=int)
    net=position*(frame.executable_return_pct.to_numpy()-fee_bps/100)
    daily=pd.DataFrame({'date':frame.outcome_date.to_numpy(),'net':net}).groupby('date').net.mean()/100
    wealth=np.r_[1.,np.cumprod(1+daily.to_numpy())];dd=wealth/np.maximum.accumulate(wealth)-1
    return dict(trades=int(position.sum()),cash_decisions=int((1-position).sum()),
        mean_net_return_per_opportunity_pct=float(net.mean()),
        mean_net_return_per_trade_pct=float(net[position==1].mean()) if position.sum() else None,
        profitable_trades_pct=float((net[position==1]>0).mean()*100) if position.sum() else None,
        equal_weight_daily_compound_return_pct=float((wealth[-1]-1)*100),max_drawdown_pct=float(dd.min()*100),
        mean_daily_return_pct=float(daily.mean()*100))


def continuous_trading(frame,position,fee_bps=35):
    """Execute changes at next open; HOLD carries shares across sessions.

    This companion avoids penalising the current HOLD rules with artificial
    daily sell/rebuy costs. Each issuer starts in cash and ends liquidated.
    """
    f=frame.copy();f['position']=np.asarray(position,dtype=int);fee=fee_bps/20000
    returns=[];drawdowns=[];buys=sells=0
    for _,g in f.groupby('ipo_id'):
        g=g.sort_values('outcome_date');cash=1.;shares=0.;wealth=[1.]
        for r in g.itertuples():
            if r.position and not shares:
                shares=cash*(1-fee)/r.next_open;cash=0.;buys+=1
            elif not r.position and shares:
                cash=shares*r.next_open*(1-fee);shares=0.;sells+=1
            wealth.append(cash+shares*r.next_close)
        if shares:cash=shares*g.iloc[-1].next_close*(1-fee);sells+=1
        wealth.append(cash);w=np.asarray(wealth)
        returns.append(100*(cash-1));drawdowns.append(100*np.min(w/np.maximum.accumulate(w)-1))
    return dict(issuers=len(returns),buys=buys,sells=sells,
        mean_issuer_net_window_return_pct=float(np.mean(returns)),median_issuer_net_window_return_pct=float(np.median(returns)),
        mean_issuer_max_drawdown_pct=float(np.mean(drawdowns)),
        convention='One unit per issuer initially in cash, transition at next open, carry overnight on HOLD, liquidate at last eligible close. Windows vary; not annualised.')


def score(frame,models,features,buffer):
    forecast=models['forecast'].predict(frame[features]);trade=models['trade'].predict(frame[features]);p=models['direction'].predict_proba(frame[features])[:,1]
    pos=trade>buffer
    result=dict(n=len(frame),issuers=int(frame.ipo_id.nunique()),forecast_mae_pct=float(mean_absolute_error(frame.forecast_return_pct,forecast)),
        accuracy_pct=float(accuracy_score(frame.forecast_return_pct>0,p>=.5)*100),
        balanced_accuracy_pct=float(balanced_accuracy_score(frame.forecast_return_pct>0,p>=.5)*100),
        auc=float(roc_auc_score(frame.forecast_return_pct>0,p)),
        always_positive_accuracy_pct=float(frame.forecast_return_pct.gt(0).mean()*100),
        always_flat_negative_accuracy_pct=float(frame.forecast_return_pct.le(0).mean()*100),
        current_rule_accuracy_pct=float(accuracy_score(frame.forecast_return_pct>0,frame.rule_position)*100),
        news_rows=int(frame.news_score.notna().sum()),filing_rows=int(frame.filing_score.notna().sum()),
        costs={str(cost):trading(frame,pos,cost) for cost in (15,35,60)},
        baselines={name:trading(frame,position,35) for name,position in {
            'always_cash':np.zeros(len(frame)), 'always_buy_next_open':np.ones(len(frame)),
            'current_rules_next_open':frame.rule_position,'momentum_ret5_positive':frame.ret5>0,
            'reversal_ret1_negative':frame.ret1<0}.items()})
    result['continuous_transition_companion']={name:continuous_trading(frame,position,35) for name,position in {
        'learned_model':pos,'current_rules':frame.rule_position,'always_hold':np.ones(len(frame)),
        'always_cash':np.zeros(len(frame)), 'momentum':frame.ret5>0,'reversal':frame.ret1<0}.items()}
    ledger=frame[['ipo_id','company_name','listing_year','decision_date','decision_timestamp','outcome_date','forecast_return_pct','executable_return_pct','next_open','next_close','rule_position']].copy()
    ledger['predicted_close_return_pct']=forecast;ledger['positive_probability']=p
    ledger['predicted_open_to_close_pct']=trade;ledger['action']=np.where(pos,'BUY NEXT OPEN / EXIT NEXT CLOSE','SELL AT NEXT OPEN / STAY IN CASH')
    ledger['strategy_net_pct']=np.where(pos,frame.executable_return_pct-.35,0)
    return result,ledger


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    data,audit=build();data.to_csv(OUT/'daily_examples.csv',index=False)
    train,validation,final,later=split_data(data)
    candidates=[]
    # Small declared grid; choice uses only 2024 costs, never 2025/26 outcomes.
    for name,features in GROUPS.items():
        for alpha in (10.,100.,1000.):
            models=fit_models(train,features,alpha)
            for buffer in (.35,.60,1.0):
                s,_=score(validation,models,features,buffer)
                candidates.append(dict(group=name,alpha=alpha,trade_threshold_pct=buffer,validation=s))
    selected=max(candidates,key=lambda r:(r['validation']['costs']['35']['mean_net_return_per_opportunity_pct'],-r['validation']['forecast_mae_pct']))
    # Pick one model per feature group on earlier validation for an honest ablation.
    comparisons={};ledgers=[];artifacts={}
    for name,features in GROUPS.items():
        choice=max([c for c in candidates if c['group']==name],key=lambda r:(r['validation']['costs']['35']['mean_net_return_per_opportunity_pct'],-r['validation']['forecast_mae_pct']))
        models=fit_models(final,features,choice['alpha']);s,ledger=score(later,models,features,choice['trade_threshold_pct'])
        s['selected_on_2024']={k:v for k,v in choice.items() if k!='validation'}
        s['by_listing_year']={str(year):score(g,models,features,choice['trade_threshold_pct'])[0] for year,g in later.groupby('listing_year')}
        zero=np.zeros(len(later));s['zero_return_mae_pct']=float(mean_absolute_error(later.forecast_return_pct,zero))
        s['training_mean_mae_pct']=float(mean_absolute_error(later.forecast_return_pct,np.full(len(later),final.forecast_return_pct.mean())))
        comparisons[name]=s;ledger['variant']=name;ledgers.append(ledger);artifacts[name]=models
    champion=selected['group'];summary=comparisons[champion];ledger=next(l for l in ledgers if l.variant.iloc[0]==champion)
    # Paired uncertainty resamples whole issuers; daily rows are not independent IPOs.
    chosen_net=ledger.set_index(later.index).strategy_net_pct
    delta=chosen_net-later.rule_position*(later.executable_return_pct-.35)
    means=pd.DataFrame({'id':later.ipo_id,'delta':delta}).groupby('id').delta.mean().to_numpy()
    rng=np.random.default_rng(42);draws=means[rng.integers(0,len(means),(5000,len(means)))].mean(axis=1)
    report=dict(protocol=dict(decision='18:00 IST after completed session',forecast_target='next observed close / current close - 1',
        trading_target='next observed close / next observed open - 1',execution='Long/cash, entry next open, exit same close, no shorting.',
        round_trip_cost_bps=35,sensitivity_bps=[15,35,60],feature_sessions='21st through 61st observed session',
        selection='2021-2023 train, 2024 validation; final 2021-2024; 2025-2026 test, disjoint issuers; outcomes mature strictly before next split.',
        same_day_eod_assumption='Historical daily bars and market closes assumed available by 18:00; actual archive publication clocks are not universally recorded.',
        headline_policy='Issuer-verified archived headlines, preceding calendar dates only, trailing 7 days; empty coverage remains missing.',
        news_limit='Published timestamp permits retrospective reconstruction; retrieval was later, and archive coverage is sparse and not exhaustive.'),
        splits={k:dict(rows=len(v),issuers=int(v.ipo_id.nunique()),first_decision=str(v.decision_date.min().date()),last_outcome=str(v.outcome_date.max().date())) for k,v in [('train',train),('validation',validation),('final_train',final),('test',later)]},
        selected=selected,comparisons=comparisons,paired_vs_rules_issuer_mean_net_delta_pct=float(means.mean()),
        paired_vs_rules_issuer_bootstrap95_pct=np.quantile(draws,[.025,.975]).tolist(),
        live_promoted=False,reason='Research evidence must establish an advantage after costs; existing inspected test years are not an untouched prospective test.',
        audit=audit,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in PRICE_PATHS+OFFICIAL+[IPOS,NEWS,Path(__file__)]})
    pd.concat(ledgers).to_csv(OUT/'test_predictions.csv',index=False)
    joblib.dump(dict(models=artifacts[champion],features=GROUPS[champion],trade_threshold_pct=selected['trade_threshold_pct'],protocol=report['protocol'],research_only=True),OUT/'next_session_research.joblib')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,default=native)+'\n',encoding='utf8')
    lines=['# Module 2 next-session forecast and trading evaluation','',str(report['protocol']),'',
        '| Model | Test rows / IPOs | Direction accuracy | Close-return MAE | Net per opportunity at 35bps | Trades |','|---|---:|---:|---:|---:|---:|']
    for name,s in comparisons.items():
        t=s['costs']['35'];lines.append(f"| {name} | {s['n']} / {s['issuers']} | {s['accuracy_pct']:.2f}% | {s['forecast_mae_pct']:.3f}% | {t['mean_net_return_per_opportunity_pct']:.4f}% | {t['trades']} |")
    lines+=['',f"Selected using 2024 only: {champion}, alpha {selected['alpha']}, entry threshold {selected['trade_threshold_pct']}%.",'',
        '| Baseline (same test rows) | Net per opportunity at 35bps | Trades |','|---|---:|---:|']
    for name,s in summary['baselines'].items():lines.append(f"| {name} | {s['mean_net_return_per_opportunity_pct']:.4f}% | {s['trades']} |")
    lines+=['','## Continuous holdings companion','',
        'HOLD carries the position overnight; costs apply only to actual position changes. Each issuer starts in cash and ends liquidated. Same test rows, varying issuer windows.',
        '', '| Strategy | Mean issuer window return after costs | Buy fills | Sell fills |','|---|---:|---:|---:|']
    for name,s in summary['continuous_transition_companion'].items():lines.append(f"| {name} | {s['mean_issuer_net_window_return_pct']:.3f}% | {s['buys']} | {s['sells']} |")
    lines+=['',f"Issuer-bootstrap interval for mean net improvement vs rules: {report['paired_vs_rules_issuer_bootstrap95_pct']} percentage points.",
        '', 'SELL means exiting an existing holding or staying in cash; short selling is not modeled. All strategies buy at next open and liquidate at the same close for an identical horizon. The existing rules were designed for longer holding periods; their next-session result is a new comparison.',
        '', 'No live policy promotion. Repeated sessions are correlated, not independent IPOs. Archive coverage and equal-capital availability limit interpretation; fills at circuits, market impact, taxes and overnight execution risk need prospective testing.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(splits=report['splits'],selected={k:v for k,v in selected.items() if k!='validation'},results=comparisons),default=native),flush=True)
    return report


if __name__=='__main__':run()
