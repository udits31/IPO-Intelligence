"""Test volatility-scaled thresholds in the existing Module 2 rule family.

Hypothesis: a fixed 8% trailing threshold is not the same risk for a quiet IPO
and a volatile one. Scale only existing loss/lag/trail thresholds using the
20 observed daily returns available at the unchanged 31st-close checkpoint.
No replacement classifier, synthetic prices, changed horizon or subset score.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import DEFAULT, ROOT, assemble, metrics, paired_interval
from ml.calibrate_module2_market_rules import PRESETS
from ml.replay_module2_news_overlay import SOURCES

REPORT=ROOT/'output/module2_volatility_rule_calibration_20261021.json'
GRID=[{'preset':name,'exponent':0.,'lower_scale':1.,'upper_scale':1.} for name in PRESETS]+[
    dict(zip(('preset','exponent','lower_scale','upper_scale'),(name,power,*bounds)))
    for name,power,bounds in itertools.product(tuple(PRESETS),(.5,1.),((.5,2.),(.75,1.5)))]


def observed_volatility(prices_path: Path, ledger_path: Path):
    cases=assemble(prices_path,ledger_path)
    prices=pd.read_csv(prices_path,parse_dates=['date'])
    values=[]
    for case in cases.itertuples():
        history=prices[prices.ipo_id.eq(case.ipo_id) & prices.date.le(case.decision_date)].sort_values('date')
        if len(history)!=31 or history.date.duplicated().any():
            raise ValueError('Volatility window must stop at the unchanged 31st observed close')
        close=history.close_price_adjusted.to_numpy(dtype=float)
        returns=(close[1:]/close[:-1]-1)*100
        sigma=float(np.std(returns[-20:],ddof=1))
        if not np.isfinite(sigma) or sigma<=0:
            raise ValueError('Observed volatility unavailable; no synthetic replacement is allowed')
        values.append(sigma)
    cases['observed_daily_volatility_20_pct']=values
    return cases


def scaled_calls(frame,parameters,reference):
    if not np.isfinite(reference) or reference<=0:
        raise ValueError('Volatility reference must be a finite observed training statistic')
    scale=np.clip((frame.observed_daily_volatility_20_pct.to_numpy()/reference)**parameters['exponent'],
                  parameters['lower_scale'],parameters['upper_scale'])
    chosen=PRESETS[parameters['preset']]
    loss,lag_loss,trail=(chosen[key]*scale for key in DEFAULT)
    cumulative=frame.cumulative_gain_pct.to_numpy()
    current,issue,peak=(frame[key].to_numpy() for key in ('decision_close','issue_price','peak_close'))
    lag=(frame.attribution_category.eq('STOCK_SPECIFIC_LAG') & frame.spread_vs_sector_30d_pct.le(-6)).to_numpy()
    return ((cumulative<=-loss) | (current<issue*(1-loss/100)) |
            ((cumulative<=-lag_loss) & lag) | ((current>=issue) & (current<=peak*(1-trail/100))))


def select(known,freeze):
    if known.empty or not known.listing_year.between(2021,2024).all() or not known.outcome_date.le(pd.Timestamp(freeze)).all():
        raise ValueError('Only matured 2021–2024 outcomes may select settings or reference volatility')
    reference=float(known.observed_daily_volatility_20_pct.median())
    base=scaled_calls(known,GRID[0],reference)
    candidates=[]
    for parameters in GRID:
        calls=scaled_calls(known,parameters,reference)
        candidates.append({'parameters':parameters,'metrics':metrics(known,calls),
                           'changed_calls':int((base!=calls).sum())})
    winner=max(candidates,key=lambda r:(r['metrics']['balanced_accuracy_pct'],r['metrics']['accuracy_pct'],
                                      -r['changed_calls'],-r['parameters']['exponent']))
    return {**winner,'observed_training_volatility_median_pct':reference},candidates


def run():
    dev=observed_volatility(*SOURCES['development'])
    later=observed_volatility(*SOURCES['holdout'])
    confirmation=dev[dev.listing_year.eq(2024)]
    first=confirmation.decision_date.min()
    early=dev[dev.listing_year.isin([2022,2023]) & dev.outcome_date.lt(first)]
    early_choice,early_candidates=select(early,first-pd.Timedelta(days=1))
    final_known=dev[dev.outcome_date.le('2024-12-31')]
    chosen,candidates=select(final_known,'2024-12-31')
    if not later.decision_date.gt('2024-12-31').all():
        raise ValueError('Later evaluation overlaps the threshold/reference freeze')
    reference=chosen['observed_training_volatility_median_pct']
    calls=scaled_calls(later,chosen['parameters'],reference)
    score=metrics(later,calls)
    ledger=later.copy()
    ledger['volatility_rule_sell']=calls
    ledger['volatility_rule_correct']=calls!=later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem+'_ledger.csv'),index=False)
    report={
        'model_family':'Existing Module 2 deterministic loss/relative-lag/trailing rules; observed-volatility scaling only',
        'core_live_rules_changed':False,'synthetic_rows':0,
        'hypothesis':'Constant percentage stops impose different risk budgets across IPO volatilities.',
        'protocol':'Unchanged 31st-close checkpoint, 30 later sessions, all eligible IPOs; daily volatility from 20 past close-to-close returns only.',
        'coverage':{'development':len(dev),'early_selection':len(early),'confirmation_2024':len(confirmation),
                    'known_at_2024_freeze':len(final_known),'later':len(later)},
        'candidate_count':len(GRID),'early_choice':early_choice,'final_choice':chosen,
        'early_candidates':early_candidates,'final_candidates':candidates,
        'confirmation_2024':{'default':metrics(confirmation,scaled_calls(confirmation,GRID[0],early_choice['observed_training_volatility_median_pct'])),
            'volatility_rules':metrics(confirmation,scaled_calls(confirmation,early_choice['parameters'],
               early_choice['observed_training_volatility_median_pct']))},
        'later':{'default':metrics(later,scaled_calls(later,GRID[0],reference)),
                 'volatility_rules':score,'paired_change':paired_interval(later,calls)},
        'later_by_year':{str(year):metrics(group,scaled_calls(group,chosen['parameters'],reference))
                         for year,group in later.groupby('listing_year')},
        'requested_accuracy_established':bool(score['accuracy_pct']>=75 and score['balanced_accuracy_pct']>=70
          and score['accuracy_pct']>max(score['always_sell_accuracy_pct'],score['always_hold_accuracy_pct'])),
        'input_sha256':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (*SOURCES['development'],*SOURCES['holdout'],Path(__file__),ROOT/'core/monitor_engine.py')},
        'limitations':['2025–2026 outcomes have been inspected previously; this is not a blind holdout.',
          'Only thresholds are scaled; no claim that volatility alone predicts future direction.',
          'Earlier outcomes, including their future 30-session windows, must mature before selection.',
          'Cash dividends, execution slippage and complete issuer news/disclosure coverage are not supplied.',
          'Later results are not a promotion instruction and cannot justify repeated tuning on those outcomes.'],
    }
    REPORT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    text=['# Module 2 — Volatility-aware calibration of the existing rules','',report['protocol'],'',
          '| Same later IPOs | 30-session direction accuracy |','|---|---:|',
          f"| Default monitor | {report['later']['default']['accuracy_pct']:.2f}% |",
          f"| Earlier-data-selected volatility thresholds | {score['accuracy_pct']:.2f}% |",
          f"| Always sell | {score['always_sell_accuracy_pct']:.2f}% |",'',
          f"n={len(later)}; balanced accuracy {score['balanced_accuracy_pct']:.2f}%. "
          f"Requested target established: {report['requested_accuracy_established']}.",'',
          f"Reference volatility: {reference:.4f}% per day, frozen using known pre-2025 outcomes. "
          f"Selected: {chosen['parameters']}.",'',
          'No live rules, paper, original prices or frozen decisions were changed.','',
          '## Limitations','']+['- '+item for item in report['limitations']]
    REPORT.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('coverage','final_choice','confirmation_2024','later','later_by_year',
                                               'requested_accuracy_established')},indent=2))
    return report


if __name__=='__main__':
    run()
