"""Earlier-data calibration of the existing monitor by observed market regime.

No learned classifier is introduced. The hypothesis is fixed before the later
replay: the same loss/trailing rules may require tighter settings in a weak
market and wider settings in a stronger one. Only matured 2021–2024 IPO
outcomes select settings, and all 132 matured later IPOs remain in evaluation.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import DEFAULT,assemble,decisions,metrics,paired_interval,ROOT
from ml.replay_module2_news_overlay import SOURCES

PRESETS={'default':DEFAULT,'tighter':{'loss_pct':8.,'lag_loss_pct':0.,'trail_pct':4.},
         'wider':{'loss_pct':18.,'lag_loss_pct':5.,'trail_pct':12.}}
BASE={'regime_threshold_nifty30_pct':None,'weak_market':'default','strong_market':'default'}
GRID=[BASE]+[dict(zip(('regime_threshold_nifty30_pct','weak_market','strong_market'),values))
    for values in itertools.product((-3.,0.),tuple(PRESETS),tuple(PRESETS))
    if values[1:]!=('default','default')]
REPORT=ROOT/'output/module2_market_rule_calibration_20261021.json'


def market_calls(frame,parameters):
    result=decisions(frame,DEFAULT).copy()
    threshold=parameters['regime_threshold_nifty30_pct']
    if threshold is None:
        return result
    observed=frame.nifty_return_30d_pct.notna()
    weak=observed & frame.nifty_return_30d_pct.lt(threshold)
    strong=observed & ~weak
    for mask,name in [(weak,'weak_market'),(strong,'strong_market')]:
        result[mask.to_numpy()]=decisions(frame.loc[mask],PRESETS[parameters[name]])
    return result


def select(known,freeze):
    if (known.empty or not known.listing_year.between(2021,2024).all()
            or not known.outcome_date.le(pd.Timestamp(freeze)).all()):
        raise ValueError('Market rule selection requires matured 2021–2024 IPO outcomes only')
    base=decisions(known,DEFAULT)
    candidates=[]
    for parameters in GRID:
        calls=market_calls(known,parameters)
        candidates.append({'parameters':parameters,'metrics':metrics(known,calls),
                           'changed_calls':int((base!=calls).sum())})
    winner=max(candidates,key=lambda r:(r['metrics']['balanced_accuracy_pct'],
                                      r['metrics']['accuracy_pct'],-r['changed_calls']))
    return winner,candidates


def run():
    dev=assemble(*SOURCES['development'])
    later=assemble(*SOURCES['holdout'])
    confirmation=dev[dev.listing_year.eq(2024)]
    first=confirmation.decision_date.min()
    early=dev[dev.listing_year.isin([2022,2023]) & dev.outcome_date.lt(first)]
    early_choice,early_candidates=select(early,first-pd.Timedelta(days=1))
    known=dev[dev.outcome_date.le('2024-12-31')]
    choice,candidates=select(known,'2024-12-31')
    calls=market_calls(later,choice['parameters'])
    score=metrics(later,calls)
    ledger=later.copy()
    ledger['market_rule_sell']=calls
    ledger['market_rule_correct']=calls!=later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem+'_ledger.csv'),index=False)
    report={'model_family':'Existing deterministic loss/relative-lag/trailing rules, regime-conditioned thresholds only',
        'core_live_rules_changed':False,'presets':PRESETS,'candidate_count':len(GRID),
        'protocol':'One 31st-close checkpoint and 30-session target; weak/strong uses only observed prior-30-session Nifty return.',
        'coverage':{'development':len(dev),'early_selection':len(early),'2024_confirmation':len(confirmation),
                    'known_by_2024_end':len(known),'later':len(later),
                    'later_missing_nifty_kept_on_default':int(later.nifty_return_30d_pct.isna().sum())},
        'early_choice':early_choice,'final_choice':choice,'early_candidates':early_candidates,'final_candidates':candidates,
        'confirmation_2024':{'default':metrics(confirmation,decisions(confirmation,DEFAULT)),
                            'market_rules':metrics(confirmation,market_calls(confirmation,early_choice['parameters']))},
        'later':{'default':metrics(later,decisions(later,DEFAULT)),'market_rules':score,
                 'paired_change':paired_interval(later,calls)},
        'later_by_year':{str(y):metrics(g,market_calls(g,choice['parameters'])) for y,g in later.groupby('listing_year')},
        'requested_accuracy_established':bool(score['accuracy_pct']>=75 and score['balanced_accuracy_pct']>=70
           and score['accuracy_pct']>max(score['always_hold_accuracy_pct'],score['always_sell_accuracy_pct'])),
        'input_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (*SOURCES['development'],*SOURCES['holdout'],ROOT/'core/monitor_engine.py')},
        'limitations':['Market co-movement and nonstationarity limit transfer from earlier years.',
           'Later outcomes were inspected in previous experiments; not a blind holdout.',
           'No news, transaction-cost execution guarantee or cash-dividend total return is inferred.',
           'This rule experiment is not a replacement learned Module 2 classifier or a paper-result update.']}
    REPORT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    lines=['# Module 2 — Observed market-regime rule calibration','',
           'The existing loss/relative-lag/trailing rules are retained. The experiment tests a small fixed set '
           'of tighter/default/wider settings conditioned only on the observed Nifty return, not the listing year '
           'or future market direction. Settings are selected on matured 2021–2024 IPO outcomes.', '',
           '| Same later cases | 30-session accuracy |','|---|---:|',
           f"| Current rules | {report['later']['default']['accuracy_pct']:.2f}% |",
           f"| Earlier-data-selected market settings | {score['accuracy_pct']:.2f}% |",
           f"| Always sell | {score['always_sell_accuracy_pct']:.2f}% |",'',
           f"Cases: {len(later)}; selected settings: {choice['parameters']}. "
           f"Balanced accuracy: {score['balanced_accuracy_pct']:.2f}%. "
           f"75–80% target established: {report['requested_accuracy_established']}.", '',
           'No live policy or paper was changed. The goal remains incomplete unless both full requirements are met.', '',
           '## Limitations','']+['- '+x for x in report['limitations']]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('coverage','final_choice','confirmation_2024','later','later_by_year',
                                          'requested_accuracy_established')},indent=2))
    return report


if __name__=='__main__':
    run()
