"""Bounded learned challenger and exact policy-change audit. Research only.

No live artifact or manuscript is changed. Extra earlier checkpoints do not
become extra independent test IPOs; each development issuer has total weight 1.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT/'data/audit/module2_development_prices_2021_2024_nse_extended_ca_reconciled.csv'
LATER = ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full_ca_reconciled.csv'
OLD = ROOT/'output/module2_holdout_decision_ledgerpresentation_20261022.csv'
CURRENT = ROOT/'output/module2_holdout_decision_ledger_governance_20261023.csv'
OUT = ROOT/'output/module2_daily_learning_research_20261023.json'
FEATURES = ['stock_5','stock_10','stock_20','stock_30','reversal_5',
            'ma10_distance','ma20_distance','volatility_20','down_fraction_20',
            'drawdown_30','range_position_30','nifty_5','nifty_20','nifty_30',
            'sector_10','sector_30','relative_10','relative_30','age_sessions']


def features_at(group, index):
    """No row after the decision close is accessible to feature extraction."""
    history = group.iloc[:index+1]
    stock = history.close_price_adjusted.astype(float)
    sector = history.sector_index_close.astype(float)
    nifty = history.nifty50_close.astype(float)
    def change(series, lag):
        return (series.iloc[-1]/series.iloc[-1-lag]-1)*100
    value = {f'stock_{lag}': change(stock,lag) for lag in (5,10,20,30)}
    value.update({f'nifty_{lag}': change(nifty,lag) for lag in (5,20,30)})
    value.update({f'sector_{lag}': change(sector,lag) for lag in (10,30)})
    window = stock.iloc[-31:]
    spread = window.max()-window.min()
    value.update(reversal_5=value['stock_5']-((stock.iloc[-6]/stock.iloc[-11]-1)*100),
        ma10_distance=(stock.iloc[-1]/stock.iloc[-10:].mean()-1)*100,
        ma20_distance=(stock.iloc[-1]/stock.iloc[-20:].mean()-1)*100,
        volatility_20=stock.pct_change().iloc[-20:].std()*100,
        down_fraction_20=float(stock.pct_change().iloc[-20:].lt(0).mean()),
        drawdown_30=(stock.iloc[-1]/window.max()-1)*100,
        range_position_30=float((stock.iloc[-1]-window.min())/spread) if spread else .5,
        relative_10=value['stock_10']-value['sector_10'],
        relative_30=value['stock_30']-value['sector_30'], age_sessions=index+1)
    return value


def examples(path, checkpoints, allowed_ids=None, *, require_exchange=True):
    prices = pd.read_csv(path, parse_dates=['date'])
    if prices.duplicated(['ipo_id','date']).any() or (require_exchange and not prices.price_basis.str.startswith('Official NSE').all()):
        raise ValueError('Require unique exchange-based price histories')
    rows = []
    for ipo_id, group in prices.groupby('ipo_id'):
        if allowed_ids is not None and ipo_id not in allowed_ids:
            continue
        group = group.sort_values('date').reset_index(drop=True)
        for index in checkpoints:
            if index+31 >= len(group):
                continue
            future_return = (group.iloc[index+30].close_price_adjusted/group.iloc[index].close_price_adjusted-1)*100
            rows.append(dict(ipo_id=int(ipo_id), decision_date=group.iloc[index].date,
                outcome_date=group.iloc[index+30].date, checkpoint=index,
                future_return_pct=float(future_return), up=int(future_return>0), **features_at(group,index)))
    return pd.DataFrame(rows)


def learned(name):
    if name.startswith('lr'):
        clf = LogisticRegression(C=float(name.split('_')[1]),max_iter=3000,random_state=42)
    elif name == 'shallow_boosting':
        clf = HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=20,
                                             l2_regularization=10,learning_rate=.05,random_state=42)
    elif name == 'shallow_forest':
        clf = RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=10,
                                     random_state=42,n_jobs=1)
    else:
        raise ValueError('Unknown predeclared challenger')
    return Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True)),
                     ('scale',StandardScaler()),('model',clf)])


def fit(name, frame):
    weights = 1/frame.groupby('ipo_id').ipo_id.transform('size')
    # Keep average weight 1 without giving longer-history issuers extra votes.
    weights *= len(weights)/weights.sum()
    return learned(name).fit(frame[FEATURES],frame.up,model__sample_weight=weights)


def score(y,p):
    return dict(n=len(y), correct=int(((p>=.5)==y).sum()),
        accuracy_pct=float(accuracy_score(y,p>=.5)*100),
        balanced_accuracy_pct=float(balanced_accuracy_score(y,p>=.5)*100),
        brier=float(brier_score_loss(y,p)),
        auc=float(roc_auc_score(y,p)) if len(set(y))>1 else None,
        always_hold_accuracy_pct=float(np.mean(y)*100),
        always_sell_accuracy_pct=float((1-np.mean(y))*100))


def paired_gain(candidate_correct, baseline_correct):
    difference=np.asarray(candidate_correct,dtype=int)-np.asarray(baseline_correct,dtype=int)
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(difference),size=(10000,len(difference)))
    return dict(change_pp=float(difference.mean()*100),
        bootstrap_95_pp=(np.quantile(difference[draws].mean(axis=1),[.025,.975])*100).tolist(),
        corrected_errors=int((difference>0).sum()),introduced_errors=int((difference<0).sum()))


def policy_audit():
    old, current = [pd.read_csv(path).set_index('ipo_id') for path in (OLD,CURRENT)]
    if set(old.index)!=set(current.index):
        raise ValueError('A policy comparison cannot change IPOs')
    current = current.loc[old.index]
    if not np.allclose(old.day30_return_pct,current.day30_return_pct):
        raise ValueError('Policy outcomes changed')
    up=current.day30_return_pct.gt(0)
    changed=old.sell.ne(current.sell)
    audit=dict(n=len(current),changed_calls=int(changed.sum()),
        old_correct_on_changed=int(old.sell[changed].ne(up[changed]).sum()),
        new_correct_on_changed=int(current.sell[changed].ne(up[changed]).sum()),
        old_correct=int(old.sell.ne(up).sum()),new_correct=int(current.sell.ne(up).sum()),
        false_sell=int((current.sell & up).sum()),false_hold_proxy=int((~current.sell & ~up).sum()),
        groups=[dict(verdict=name,n=len(group),correct=int(group.sell.ne(group.day30_return_pct.gt(0)).sum()))
                for name,group in current.groupby('verdict')])
    ledger=current.copy()
    ledger['old_verdict']=old.verdict
    ledger['call_changed']=changed
    ledger['old_correct']=old.sell.ne(up)
    ledger['current_binary_proxy_correct']=current.sell.ne(up)
    ledger['miss_type']=np.where(current.sell & up,'Sold before a rise',
                          np.where(~current.sell & ~up,'No exit before a fall/flat','Correct'))
    ledger.to_csv(OUT.with_name(OUT.stem+'_policy_errors.csv'))
    return audit


def run(*, extended_secondary=False):
    global OUT
    if extended_secondary:
        OUT=ROOT/'output/module2_extended_history_learning_20261023.json'
    audit=policy_audit()
    ipos=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv',parse_dates=['listing_date']).set_index('id')
    # Eligible audited issuer IDs, not outcome-selected cases.
    dev_ids=set(pd.read_csv(ROOT/'output/module2_development_decision_ledger_goal_corporate_actions_20261021.csv').ipo_id)
    development_path=ROOT/'data/audit/module2_development_prices_2021_2024.csv' if extended_secondary else DEV
    dev=examples(development_path,tuple(range(30,251,15)) if extended_secondary else (30,40,50,60,90,120),
                 dev_ids,require_exchange=not extended_secondary)
    dev['listing_year']=dev.ipo_id.map(ipos.listing_date.dt.year)
    confirm=dev[dev.listing_year.eq(2024) & dev.checkpoint.eq(30) & dev.outcome_date.le('2024-12-31')]
    train=dev[dev.listing_year.between(2021,2023) & dev.outcome_date.lt(confirm.decision_date.min())]
    final=dev[dev.listing_year.between(2021,2024) & dev.outcome_date.le('2024-12-31')]
    if set(train.ipo_id)&set(confirm.ipo_id) or confirm.empty:
        raise ValueError('Training and confirmation issuers must be disjoint')
    names=('lr_0.05','lr_0.5','shallow_boosting','shallow_forest')
    candidates=[]
    for name in names:
        p=fit(name,train).predict_proba(confirm[FEATURES])[:,1]
        candidates.append(dict(name=name,confirmation=score(confirm.up.to_numpy(),p)))
    selected=max(candidates,key=lambda c:(c['confirmation']['balanced_accuracy_pct'],-c['confirmation']['brier']))
    # Lock the choice before loading later outcomes. Later cases are unchanged.
    expected=pd.read_csv(CURRENT).set_index('ipo_id')
    later=examples(LATER,(30,),set(expected.index)).sort_values('ipo_id')
    if set(later.ipo_id)!=set(expected.index) or not final.outcome_date.lt(later.decision_date.min()).all():
        raise ValueError('Later coverage or training maturity mismatch')
    if not np.allclose(later.future_return_pct,expected.loc[later.ipo_id].day30_return_pct):
        raise ValueError('Challenger must retain the original 30-session target')
    p=fit(selected['name'],final).predict_proba(later[FEATURES])[:,1]
    later['positive_probability']=p
    later['predicted_positive']=p>=.5
    later['correct']=later.predicted_positive.eq(later.up.astype(bool))
    current_correct=expected.loc[later.ipo_id].sell.to_numpy()!=later.up.to_numpy()
    comparisons=dict(vs_current_rule=paired_gain(later.correct.to_numpy(),current_correct),
                     vs_always_sell=paired_gain(later.correct.to_numpy(),~later.up.to_numpy(dtype=bool)))
    result=dict(policy_change=audit,selected=selected['name'],candidates=candidates,
        development_price_source='Original retrospective Yahoo histories; secondary, not fully exchange-reconciled' if extended_secondary else 'Reconciled NSE first-session histories',
        training_examples=len(train),training_issuers=int(train.ipo_id.nunique()),
        final_examples=len(final),final_issuers=int(final.ipo_id.nunique()),
        confirmation_n=len(confirm),later=score(later.up.to_numpy(),p),
        paired_comparisons=comparisons,
        features=FEATURES,live_promoted=False,synthetic_rows=0,
        limitations=['Research after previous inspection of later results; not a blind test.',
            'Feature fitting and candidate choice use only completed earlier IPO outcomes.',
            'Repeated earlier checkpoints are issuer-weighted, not independent samples.',
            'No as-published news/filing forecast or transaction-cost advantage established.',
            'WATCH is not a HOLD forecast; the legacy binary comparison interprets no exit as hold.'],
        input_sha256={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (development_path,LATER,OLD,CURRENT,Path(__file__))})
    later.to_csv(OUT.with_name(OUT.stem+'_predictions.csv'),index=False)
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Module 2: policy drop and learned daily-history challenger','',
        f"Same {audit['n']} later IPOs. Changed {audit['changed_calls']} old SELL calls to review alerts; "
        f"old calls were correct {audit['old_correct_on_changed']} times, the no-exit proxy {audit['new_correct_on_changed']} times.",
        '', '| Current call | IPOs | Correct under binary proxy |','|---|---:|---:|']
    lines += [f"| {row['verdict']} | {row['n']} | {row['correct']} |" for row in audit['groups']]
    lines += ['',f"Challenger selected on earlier confirmation: {selected['name']}.",
        f"Training: {len(final)} real checkpoints from {final.ipo_id.nunique()} earlier issuers, each issuer total weight equal.",
        f"Confirmation n={len(confirm)}: {selected['confirmation']['accuracy_pct']:.2f}% accuracy; "
        f"later n={len(later)}: {result['later']['accuracy_pct']:.2f}% accuracy, "
        f"{result['later']['balanced_accuracy_pct']:.2f}% balanced accuracy.",
        f"Always-sell later baseline: {result['later']['always_sell_accuracy_pct']:.2f}%.",
        '', 'No live model, monitor policy or paper was changed.', '',*['- '+s for s in result['limitations']]]
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','features')},indent=2))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--extended-secondary',action='store_true')
    run(extended_secondary=parser.parse_args().extended_secondary)
