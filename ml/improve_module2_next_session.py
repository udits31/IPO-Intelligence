"""Execution-aligned, bounded Module 2 retraining with recovered dated events.

Select policies on 2024, then compare identical 2025/26 issuer sessions.
Daily direction, entry selection and continuous net returns stay distinct.
"""
from pathlib import Path
import gzip,hashlib,json,re
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,mean_absolute_error
from threadpoolctl import threadpool_limits
from ml.train_module2_next_session import ROOT,PRICE_PATHS,OFFICIAL,NEWS,PRICE,FLOW,TEXT,split_data,news_archive,historical_news_at,continuous_trading,sha,native
from ml.recover_module2_daily_news import parse_feed,archived_sources
from services.point_in_time import eligible,timestamp

OUT=ROOT/'output/module2_next_session_improvement_20261029'
SOURCE=ROOT/'output/module2_next_session_20261029/daily_examples.csv'
EXTRA=['intraday1','intraday5','overnight1','overnight5','signed_volume5','relative_volume',
       'normalised_ret5','normalised_ret1','market_sector_spread']
EVENT=['news_positive','news_negative','news_age_days','news_earnings','news_order','news_governance',
       'news_regulatory','news_capital','news_business_score','filing_earnings','filing_order','filing_regulatory',
       'filing_governance','filing_age_days']
GROUPS={'execution_aligned_price':PRICE+FLOW,
        'execution_aligned_flow':PRICE+FLOW+EXTRA,
        'execution_aligned_events':PRICE+FLOW+EXTRA+TEXT+EVENT}
SETTINGS=[('ridge',100.),('ridge',1000.),('boosting',20.)]
POLICIES=[('return_hysteresis',.15,-.15),('return_hysteresis',.35,0.),('rules_with_ml_guard',.35,-.35)]


def evergreen(title):
    return bool(re.search(r'^technical analysis of\b|\b(stock screener|stock price history|share price history)\b',title,re.I))


def events():
    from services.news_service import NewsService
    _,hashes=news_archive();items={};scorer=NewsService();statuses=[]
    for r in map(json.loads,NEWS.read_text(encoding='utf8').splitlines()):
        if r.get('status')=='verified_rss_response':
            items[int(r['ipo_id'])]=[dict(a,score=scorer.analyze_sentiment(a['title']),event_tags=[]) for a in r.get('articles',[]) if not evergreen(a['title'])]
    for r,raw,reference in archived_sources():
        if hashlib.sha256(raw).hexdigest()!=r['raw_sha256']:raise ValueError('Recovered feed source hash mismatch')
        rows,reject,count=parse_feed(raw,r['company_name'],pd.Timestamp(r['start']),pd.Timestamp(r['end']))
        items.setdefault(int(r['ipo_id']),[]).extend(a for a in rows if not evergreen(a['title']))
        hashes[reference]=r['raw_sha256']
        statuses.append(dict(ipo_id=r['ipo_id'],status=r['status'],articles=len(rows),raw_items=count,rejected=reject))
    dedup={}
    for id,rows in items.items():
        # Distinct headlines, not syndication counts, drive event features.
        table={}
        for a in rows:
            key=(str(pd.Timestamp(a['published_at'])),re.sub(r'\W+',' ',a['title'].casefold()))
            if a['title']:table[key]=a
            elif not any(k[0]==key[0] for k in table):table[key]=a
        dedup[id]=list(table.values())
    return dedup,hashes,statuses


def assemble():
    frame=pd.read_csv(SOURCE,parse_dates=['decision_date','outcome_date'])
    all_prices=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in PRICE_PATHS])
    official=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in OFFICIAL]).drop_duplicates(['ipo_id','date'])
    verified={(int(r.ipo_id),r.date):float(r.close_price_adjusted) for r in official.itertuples()}
    by_price={};news,hashes,statuses=events();extra=[];target=[];maturity=[];news_counts=[]
    for id,g in all_prices.groupby('ipo_id'):
        by_price[int(id)]=g.sort_values('date').reset_index(drop=True)
    filing_by={}
    for id in frame.ipo_id.unique():
        p=ROOT/f'data/market/corporate_announcements/{int(id)}.jsonl'
        records=[r for r in map(json.loads,p.read_text(encoding='utf8').splitlines()) if r.get('schema_version')==2] if p.exists() else []
        filing_by[int(id)]=[(r,timestamp(r.get('broadcast_datetime_ist'))) for r in records]
        if p.exists():hashes[str(p.relative_to(ROOT))]=sha(p)
    for r in frame.itertuples():
        g=by_price[int(r.ipo_id)];loc=g.index[g.date.eq(r.decision_date)]
        if len(loc)!=1:raise ValueError('Decision price identity conflict')
        i=int(loc[0]);c=g.ohlcv_close;v=g.volume;h=g.iloc[:i+1]
        # Two future opens are needed to measure the return earned by a
        # next-open position until its next executable rebalance.
        valid=i+2<len(g) and (g.date.iloc[i+2]-g.date.iloc[i+1]).days<=7
        if valid:
            a,b=float(g.open.iloc[i+1]),float(g.open.iloc[i+2])
            reference=verified.get((int(r.ipo_id),g.date.iloc[i+2]),np.nan)
            valid=np.isfinite(a) and np.isfinite(b) and min(a,b)>0 and np.isfinite(reference) and reference>0 and abs(float(g.ohlcv_close.iloc[i+2])/reference-1)<=.02
        target.append(100*(b/a-1) if valid else np.nan)
        maturity.append(g.date.iloc[i+2] if valid else pd.NaT)
        intraday=(h.ohlcv_close/h.open-1)*100
        overnight=(h.open/h.ohlcv_close.shift(1)-1)*100
        vol=h.volume.iloc[-5:];signed=np.sign(h.ohlcv_close.diff().iloc[-5:])
        f=dict(intraday1=intraday.iloc[-1],intraday5=intraday.tail(5).mean(),overnight1=overnight.iloc[-1],
            overnight5=overnight.tail(5).mean(),signed_volume5=float((signed*vol).sum()/vol.sum()) if vol.sum()>0 else np.nan,
            relative_volume=r.volume_ratio/(1+abs(r.ret1)),normalised_ret5=r.ret5/max(.1,r.vol20),
            normalised_ret1=r.ret1/max(.1,r.vol20),market_sector_spread=r.sector5-r.nifty5)
        stamp=pd.Timestamp(r.decision_timestamp)
        known=[a for a in news.get(int(r.ipo_id),[]) if stamp-pd.Timedelta(days=7)<=pd.Timestamp(a['published_at'])<stamp.normalize()]
        scores=[a['score'] for a in known];tags=[tag for a in known for tag in a.get('event_tags',[])]
        f.update(news_score=np.mean(scores) if scores else np.nan,news_count=len(known) if known else np.nan,
            news_positive=np.mean(np.array(scores)>0) if scores else np.nan,
            news_negative=np.mean(np.array(scores)<0) if scores else np.nan,
            news_age_days=(stamp-max(pd.Timestamp(a['published_at']) for a in known)).total_seconds()/86400 if known else np.nan)
        for tag in ('earnings','order','governance','regulatory','capital'):f['news_'+tag]=tags.count(tag) if known else np.nan
        business=[a['score'] for a in known if set(a.get('event_tags',[]))&{'earnings','order','governance','regulatory','capital'}]
        f['news_business_score']=np.mean(business) if business else np.nan
        filing=[(a,t) for a,t in filing_by[int(r.ipo_id)] if t is not None and stamp-pd.Timedelta(days=7)<=t<=stamp and eligible(a,r.decision_timestamp)]
        mapping={'earnings':'is_earnings_release','order':'is_order_win_or_expansion',
            'regulatory':'is_regulatory_or_tax_notice','governance':'is_auditor_or_kmp_resignation'}
        for tag,key in mapping.items():f['filing_'+tag]=sum(a.get(key) is True for a,t in filing) if filing else np.nan
        f['filing_age_days']=(stamp-max(t for a,t in filing)).total_seconds()/86400 if filing else np.nan
        extra.append(f);news_counts.append(len(known))
    added=pd.DataFrame(extra,index=frame.index)
    for col in added:frame[col]=added[col]
    frame['holding_return_pct']=target;frame['trade_outcome_date']=maturity
    coverage=frame.groupby('listing_year').agg(rows=('ipo_id','size'),news_rows=('news_score','count'),filing_rows=('filing_score','count')).reset_index().to_dict('records')
    # Use identical dates/issuers for all candidates and rules; never drop a
    # test case for absent news or a wrong prediction.
    missing=frame.holding_return_pct.isna();removed=frame[missing][['ipo_id','company_name','decision_date']]
    # Terminal next-open targets are absent in some windows. Keep these
    # decisions for validation/test trading; only fitting requires the label.
    frame=frame.replace([np.inf,-np.inf],np.nan).copy()
    if frame.ipo_id.nunique()!=367:raise ValueError('Unexpected issuer coverage change')
    frame.to_csv(OUT/'daily_examples.csv',index=False);removed.to_csv(OUT/'unavailable_next_rebalance_open.csv',index=False)
    gap_rows=[]
    for id,g in frame.groupby('ipo_id'):
        gap_rows.append(dict(ipo_id=int(id),company_name=g.company_name.iloc[0],decisions=len(g),
            missing_headline_decisions=int(g.news_score.isna().sum()),missing_filing_decisions=int(g.filing_score.isna().sum()),
            headline_status='PARTIAL_ARCHIVE' if g.news_score.notna().any() else 'NO_ELIGIBLE_DATED_HEADLINES_RECOVERED',
            note='Missing evidence remains unknown; no zero sentiment or clean-news assumption.'))
    pd.DataFrame(gap_rows).to_csv(OUT/'remaining_dated_event_gaps.csv',index=False)
    return frame,dict(coverage_before_maturity_purge=coverage,source_hashes=hashes,source_status=statuses,missing_training_rebalance_targets=len(removed))


def pipeline(family,alpha,classifier=False):
    if family=='ridge':
        m=LogisticRegression(C=1/alpha,max_iter=2000,random_state=42) if classifier else Ridge(alpha=alpha)
    else:
        cls=HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        m=cls(max_iter=100,max_leaf_nodes=7,min_samples_leaf=60,l2_regularization=alpha,learning_rate=.05,random_state=42)
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),m)


def fit(frame,features,family,alpha):
    w=1/frame.groupby('ipo_id').ipo_id.transform('size');w*=len(w)/w.sum();models={}
    for name,target in [('forecast','forecast_return_pct'),('holding','holding_return_pct'),('direction','forecast_return_pct')]:
        m=pipeline(family,alpha,name=='direction')
        y=frame[target].gt(0).astype(int) if name=='direction' else frame[target]
        m.fit(frame[features],y,**{m.steps[-1][0]+'__sample_weight':w});models[name]=m
    return models


def positions(frame,prediction,policy):
    mode,entry,exit=policy;f=frame.copy();f['signal_prediction']=prediction;result=pd.Series(0,index=frame.index,dtype=int)
    for id,g in f.groupby('ipo_id'):
        held=0
        for r in g.sort_values('decision_date').itertuples():
            if mode=='rules_with_ml_guard':held=1 if r.signal_prediction>entry else 0 if r.signal_prediction<exit else int(r.rule_position)
            else:
                if r.signal_prediction>entry:held=1
                elif r.signal_prediction<exit:held=0
            result.loc[r.Index]=held
    return result


def realised(frame,position,fee_bps=35):
    # Same engine as the original rule baseline, with entry/exit costs only
    # on actual transitions and final liquidation. Return the issuer ledger.
    f=frame.copy();f['position']=np.asarray(position);fee=fee_bps/20000;rows=[]
    for id,g in f.groupby('ipo_id'):
        g=g.sort_values('outcome_date');cash=1.;shares=0.;wealth=[1.];trades=[]
        for r in g.itertuples():
            if r.position and not shares:
                shares=cash*(1-fee)/r.next_open;cash=0.;trades.append(dict(date=str(r.outcome_date.date()),action='BUY',price=r.next_open))
            elif not r.position and shares:
                cash=shares*r.next_open*(1-fee);shares=0.;trades.append(dict(date=str(r.outcome_date.date()),action='SELL',price=r.next_open))
            wealth.append(cash+shares*r.next_close)
        if shares:
            cash=shares*g.iloc[-1].next_close*(1-fee)
            trades.append(dict(date=str(g.iloc[-1].outcome_date.date()),action='SELL_TERMINAL',price=float(g.iloc[-1].next_close)))
        wealth.append(cash);w=np.asarray(wealth)
        rows.append(dict(ipo_id=int(id),company_name=g.company_name.iloc[0],listing_year=int(g.listing_year.iloc[0]),
            net_return_pct=100*(cash-1),max_drawdown_pct=float(100*np.min(w/np.maximum.accumulate(w)-1)),
            buys=sum(t['action']=='BUY' for t in trades),sells=sum(t['action'].startswith('SELL') for t in trades),
            exposure_sessions=int(g.position.sum()),sessions=len(g),trades=trades))
    return rows


def summary(rows):
    f=pd.DataFrame(rows)
    return dict(issuers=len(f),mean_net_return_pct=float(f.net_return_pct.mean()),median_net_return_pct=float(f.net_return_pct.median()),
        mean_max_drawdown_pct=float(f.max_drawdown_pct.mean()),worst_decile_mean_return_pct=float(f.net_return_pct.nsmallest(max(1,int(np.ceil(len(f)*.1)))).mean()),
        buys=int(f.buys.sum()),sells=int(f.sells.sum()),exposure_sessions=int(f.exposure_sessions.sum()))


def paired(a,b):
    a=pd.DataFrame(a).set_index('ipo_id');b=pd.DataFrame(b).set_index('ipo_id').loc[a.index]
    delta=(a.net_return_pct-b.net_return_pct).to_numpy();rng=np.random.default_rng(42)
    draws=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(axis=1)
    return dict(mean_gain_pct=float(delta.mean()),bootstrap95_pct=np.quantile(draws,[.025,.975]).tolist(),
        improved_issuers=int((delta>0).sum()),worsened_issuers=int((delta<0).sum()),ties=int((delta==0).sum()),
        limitation='Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.')


def run():
    OUT.mkdir(parents=True,exist_ok=True);frame,audit=assemble()
    # Purge by the later executable label maturity, not the earlier forecast
    # close; this matters at every train/validation/test boundary.
    ts,vs,fs,ls=split_data(frame)
    val,test=vs.copy(),ls.copy()
    train=ts[ts.trade_outcome_date.notna()&ts.trade_outcome_date.lt(val.decision_date.min())].copy()
    final=fs[fs.trade_outcome_date.notna()&fs.trade_outcome_date.lt(test.decision_date.min())].copy()
    assert not set(final.ipo_id)&set(test.ipo_id)
    assert final.trade_outcome_date.max()<test.decision_date.min()
    candidates=[];validation_rules=realised(val,val.rule_position)
    print(f'Fitting {len(train)} train rows / {train.ipo_id.nunique()} issuers; validation {len(val)} / {val.ipo_id.nunique()}',flush=True)
    for group,features in GROUPS.items():
        for family,alpha in SETTINGS:
            models=fit(train,features,family,alpha);pred=models['holding'].predict(val[features])
            for policy in POLICIES:
                pos=positions(val,pred,policy);rows=realised(val,pos);s=summary(rows)
                candidates.append(dict(group=group,family=family,alpha=alpha,policy=policy,validation=s,
                    validation_gain_vs_rules_pct=s['mean_net_return_pct']-summary(validation_rules)['mean_net_return_pct']))
    # Min 10 independent issuer entries avoids selecting cash based on tiny
    # episodic returns. Selection is fixed before later predictions are read.
    eligible_candidates=[c for c in candidates if c['validation']['buys']>=10]
    if not eligible_candidates:raise ValueError('No candidate has adequate validation trade support')
    selected=max(eligible_candidates,key=lambda c:(c['validation']['mean_net_return_pct'],c['validation']['mean_max_drawdown_pct']))
    (OUT/'selection_locked_before_test.json').write_text(json.dumps(dict(selected=selected,candidates=candidates),indent=2)+'\n',encoding='utf8')
    baselines={name:realised(test,p) for name,p in dict(current_rules=test.rule_position,always_hold=np.ones(len(test)),
        always_cash=np.zeros(len(test)),momentum=test.ret5>0,reversal=test.ret1<0).items()}
    comparisons={};prediction_tables=[];paths={};champion=None
    for group,features in GROUPS.items():
        choices=[c for c in eligible_candidates if c['group']==group]
        choice=max(choices,key=lambda c:(c['validation']['mean_net_return_pct'],c['validation']['mean_max_drawdown_pct']))
        models=fit(final,features,choice['family'],choice['alpha']);p=models['direction'].predict_proba(test[features])[:,1]
        expected=models['holding'].predict(test[features]);pos=positions(test,expected,choice['policy']);rows=realised(test,pos)
        forecast=models['forecast'].predict(test[features]);y=test.forecast_return_pct>0
        s=dict(selection=choice,direction_accuracy_pct=float(accuracy_score(y,p>=.5)*100),
            balanced_accuracy_pct=float(balanced_accuracy_score(y,p>=.5)*100),
            predicted_positive=int((p>=.5).sum()),missed_positive=int(((p<.5)&y).sum()),false_positive=int(((p>=.5)&~y).sum()),
            return_mae_pct=float(mean_absolute_error(test.forecast_return_pct,forecast)),
            strategy=summary(rows),cost_sensitivity={str(cost):summary(realised(test,pos,cost)) for cost in (15,35,60)},
            vs_rules=paired(rows,baselines['current_rules']),vs_hold=paired(rows,baselines['always_hold']),
            by_year={str(year):summary([r for r in rows if r['listing_year']==year]) for year in sorted(test.listing_year.unique())})
        comparisons[group]=s;paths[group]=rows
        ledger=test[['ipo_id','company_name','listing_year','decision_date','outcome_date','forecast_return_pct','holding_return_pct','rule_position']].copy()
        ledger['positive_probability']=p;ledger['predicted_return_pct']=forecast;ledger['expected_next_open_rebalance_return_pct']=expected
        ledger['position']=pos;ledger['variant']=group;ledger['action']='HOLD / CASH'
        for id,g in ledger.groupby('ipo_id'):
            previous=0
            for ix,r in g.sort_values('decision_date').iterrows():
                ledger.loc[ix,'action']='BUY' if r.position and not previous else 'SELL' if not r.position and previous else 'HOLD' if r.position else 'CASH'
                previous=r.position
        prediction_tables.append(ledger)
        if group==selected['group']:champion=dict(models=models,features=features,policy=choice['policy'],selection=choice,research_only=True)
    chosen=comparisons[selected['group']]
    robust=(chosen['vs_rules']['bootstrap95_pct'][0]>0 and chosen['vs_hold']['bootstrap95_pct'][0]>0
        and chosen['strategy']['buys']>=20 and chosen['strategy']['mean_max_drawdown_pct']>=summary(baselines['current_rules'])['mean_max_drawdown_pct'])
    report=dict(protocol=dict(training='2021-2023 listings; 2024 validation; final 2021-2024; test 2025-2026. Disjoint issuer IDs and later-label maturity purges.',
        forecast='After completed session, predict next observed close / current close - 1.',
        execution='Signal at 18:00 IST, execute at next open. Learned trading target: following open / next open - 1. Carry overnight; change positions only on signals. Liquidate at last common test close.',
        news='Verified issuer headlines only, trailing seven days, previous calendar dates only. Exchange events require source publication timestamp <= decision. Current fetch time never supplies a historical clock.',
        costs='35bps per round trip (17.5bps each buy and sell), stress tested at 15/60bps. No shorting or leverage.',
        selection='Maximum 2024 mean net issuer return among candidates with >=10 validation buys. No choice based on later results.',
        caveats=['Previously inspected later years are retrospective development tests, not blind proof.','Historical EOD bars assumed available by 18:00; source publication clocks are not exhaustive.',
            'Windows begin at session 21; early listing-day forecasts are not evaluated.','RSS archive results are incomplete; empty results do not establish absence of news.',
            'Equal initial capital per issuer, varying windows, no annualisation; circuit fills, slippage beyond assumed costs and shared calendar dependence need prospective verification.']),
        splits={name:dict(rows=len(d),issuers=int(d.ipo_id.nunique()),first=str(d.decision_date.min().date()),last_label=str(d.trade_outcome_date.max().date())) for name,d in [('train',train),('validation',val),('final_train',final),('test',test)]},
        majority_direction_accuracy_pct=float(max((test.forecast_return_pct>0).mean(),(test.forecast_return_pct<=0).mean())*100),
        zero_return_mae_pct=float(test.forecast_return_pct.abs().mean()),selected=selected,comparisons=comparisons,
        baselines={k:summary(v) for k,v in baselines.items()},audit=audit,
        coverage={name:dict(rows=len(d),news_rows=int(d.news_score.notna().sum()),filing_rows=int(d.filing_score.notna().sum())) for name,d in [('train',final),('test',test)]},
        promotion_gate_passed=bool(robust),live_promoted=False,
        verdict='Candidate shows retrospective net improvement but still needs prospective verification.' if robust else 'No sufficiently supported advantage over both rules and buy-and-hold; keep challenger in research.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,Path(__file__),NEWS,
            ROOT/'ml/recover_module2_daily_news.py',ROOT/'services/news_service.py',ROOT/'services/news_identity.py']+PRICE_PATHS+OFFICIAL})
    pd.concat(prediction_tables).to_csv(OUT/'predictions_and_actions.csv',index=False)
    (OUT/'issuer_paths.json').write_text(json.dumps(dict(candidates=paths,baselines=baselines),default=native,indent=2)+'\n',encoding='utf8')
    joblib.dump(champion,OUT/'module2_retrained_research.joblib')
    (OUT/'report.json').write_text(json.dumps(report,default=native,indent=2)+'\n',encoding='utf8')
    lines=['# Module 2: dated-news recovery and execution-aligned retraining','',report['verdict'],'',str(report['splits']),'',
        '| Policy | Mean net issuer return | Median | Mean drawdown | Buy fills | Direction accuracy |','|---|---:|---:|---:|---:|---:|']
    for name,s in comparisons.items():
        a=s['strategy'];lines.append(f"| {name} | {a['mean_net_return_pct']:.3f}% | {a['median_net_return_pct']:.3f}% | {a['mean_max_drawdown_pct']:.3f}% | {a['buys']} | {s['direction_accuracy_pct']:.2f}% |")
    for name,a in report['baselines'].items():lines.append(f"| {name} | {a['mean_net_return_pct']:.3f}% | {a['median_net_return_pct']:.3f}% | {a['mean_max_drawdown_pct']:.3f}% | {a['buys']} | — |")
    lines+=['',f"Selected using 2024 only: {selected['group']} / {selected['family']}; policy {selected['policy']}.",
        f"Paired net gain vs current rules: {chosen['vs_rules']}.",f"News/filing coverage: {report['coverage']}.",'',str(report['protocol'])]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(selected=selected,splits=report['splits'],comparisons=comparisons,baselines=report['baselines'],coverage=report['coverage'],verdict=report['verdict']),default=native),flush=True)
    return report


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
