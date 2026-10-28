"""Dated-headline replay of the same rule-based Module 2 monitor.

The default news governance is executed, not assumed predictive. A small,
predeclared grid of news/price rule overlays is selected only from matured
2021–2024 outcomes. Later cases are never dropped for having empty news.
No new classifier family or live rule is promoted by this experiment.
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.monitor_r0_frozen_20261026 import HoldSellMonitor
from ml.calibrate_module2_rules import assemble, decisions, DEFAULT, metrics, paired_interval
from ml.collect_module2_dated_news import parse_news
from ml.reparse_module2_news import OUTPUT as NEWS

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'output/module2_dated_news_rule_replay_issuer_verified_20261021.json'
SOURCES={name:(ROOT/f'data/audit/module2_{"development_prices_2021_2024_nse_extended" if name=="development" else "holdout_prices_2025_2026_nse_full"}_ca_reconciled.csv',
               ROOT/f'output/module2_{name}_decision_ledger_goal_corporate_actions_20261021.csv')
         for name in ('development','holdout')}
NO_OVERLAY={'mode':'unchanged','threshold':None,'min_articles':None,'lookback_days':None}
GRID=[NO_OVERLAY]+[dict(zip(('mode','threshold','min_articles','lookback_days'),values))
    for values in itertools.product(('negative_only','positive_only','both'),(.30,.60),(1,2),(3,7))]


def load_evidence():
    rows=[json.loads(line) for line in NEWS.read_text(encoding='utf-8').splitlines()]
    if len({r['ipo_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate historical news checkpoint')
    for row in rows:
        if row['status']!='verified_rss_response':
            continue
        path=(ROOT/row['archive_path']).resolve()
        path.relative_to((ROOT/'data/archive/module2_dated_news').resolve())
        with gzip.open(path,'rb') as stream:
            raw=stream.read()
        if hashlib.sha256(raw).hexdigest()!=row['raw_sha256']:
            raise ValueError('Raw dated RSS archive changed')
        articles,_,_=parse_news(raw,row['company_name'],pd.Timestamp(row['cutoff_ist']).to_pydatetime())
        if articles!=row['articles']:
            raise ValueError('Historical headlines differ from the original archived RSS parser')
    return {r['ipo_id']:r for r in rows}


def assemble_news(name,evidence):
    prices,ledger=SOURCES[name]
    cases=assemble(prices,ledger)
    frozen_price_calls=decisions(cases,DEFAULT)
    if not np.array_equal(frozen_price_calls,cases.sell.to_numpy(dtype=bool)):
        raise ValueError('Frozen 12%/5%/8% price formula no longer matches its original ledger')
    monitor=HoldSellMonitor(prices_path=str(prices))
    scores,counts,ages,returns,current_sells,statuses=[],[],[],[],[],[]
    for row in cases.itertuples():
        record=evidence.get(row.ipo_id)
        if not record or record['decision_date']!=row.decision_date.strftime('%Y-%m-%d'):
            raise ValueError(f'Missing exact-decision news attempt for IPO {row.ipo_id}; finish collection first')
        result=monitor.evaluate_holding(row.ipo_id,as_of_date=record['decision_date'],disclosures=[],
                                        fetch_news=False,historical_news=record)
        if 'verdict' not in result:
            raise ValueError('Historical monitor failed')
        # This older experiment is bound to its original numeric price policy,
        # not to the later UI change that reclassified 8% exits as WATCH alerts.
        baseline=bool(row.sell)
        score=result['news_sentiment']['score']
        scores.append(score)
        counts.append(result['news_sentiment']['article_count'])
        statuses.append(result['news_sentiment']['status'])
        articles=result['recent_news']
        cutoff=pd.Timestamp(record['cutoff_ist'])
        age=(cutoff-min(pd.Timestamp(a['published_at']) for a in articles)).total_seconds()/86400 if articles else None
        ages.append(age)  # Require all contributing articles to fit the selected window.
        returns.append((result.get('attribution_10d') or {}).get('stock_return_pct'))
        # Its headline governance only blocked accumulation; HOLD/WATCH both
        # retain in this binary protocol. New filing-based R1 is audited apart.
        current_sells.append(baseline)
    cases['news_score']=scores
    cases['news_article_count']=counts
    cases['news_max_age_days']=ages
    cases['stock_return_10d_pct']=returns
    cases['current_news_governance_sell']=current_sells
    cases['news_status']=statuses
    return cases


def overlay(frame,parameters):
    base=decisions(frame,DEFAULT).copy()
    if parameters['mode']=='unchanged':
        return base
    observed=(frame.news_score.notna() & frame.news_article_count.ge(parameters['min_articles'])
              & frame.news_max_age_days.le(parameters['lookback_days']))
    if parameters['mode'] in ('negative_only','both'):
        # Bearish news can tighten a weak-price holding, not invent an exit
        # when there is no observed headline or price confirmation.
        negative=(observed & frame.news_score.le(-parameters['threshold'])
                  & frame.stock_return_10d_pct.le(0))
        base[negative.to_numpy()]=True
    if parameters['mode'] in ('positive_only','both'):
        # Positive news can defer a profit-taking trail, never waive an
        # underwater loss/structural stop or absent price observations.
        positive=(observed & frame.news_score.ge(parameters['threshold'])
                  & frame.stock_return_10d_pct.ge(0) & frame.decision_close.ge(frame.issue_price))
        base[positive.to_numpy()]=False
    return base


def select(known,freeze):
    if (known.empty or not known.listing_year.between(2021,2024).all()
            or not known.outcome_date.le(pd.Timestamp(freeze)).all()):
        raise ValueError('News rule selection may use only matured 2021–2024 IPO outcomes')
    candidates=[]
    baseline=decisions(known,DEFAULT)
    for parameters in GRID:
        call=overlay(known,parameters)
        candidates.append({'parameters':parameters,'metrics':metrics(known,call),
                           'changed_calls':int((call!=baseline).sum())})
    chosen=max(candidates,key=lambda r:(r['metrics']['balanced_accuracy_pct'],
                                        r['metrics']['accuracy_pct'],-r['changed_calls']))
    return chosen,candidates


def coverage(frame):
    base=decisions(frame,DEFAULT)
    truth=frame.day30_return_pct.gt(0).to_numpy()
    observed=frame.news_score.notna().to_numpy()
    correct=base!=truth
    return {'n':len(frame),'cases_with_eligible_headlines':int(observed.sum()),
            'eligible_headlines':int(frame.news_article_count.sum()),
            'source_status':frame.news_status.value_counts().to_dict(),
            'observed_news_only_oracle_accuracy_ceiling_pct':float((correct|observed).mean()*100),
            'ceiling_definition':'Generous diagnostic upper bound: assume every observed-news case becomes correct, unchanged calls elsewhere; not achievable measured model accuracy.'}


def run():
    evidence=load_evidence()
    dev=assemble_news('development',evidence)
    later=assemble_news('holdout',evidence)
    first=dev.loc[dev.listing_year.eq(2024),'decision_date'].min()
    early=dev[dev.listing_year.isin([2022,2023]) & dev.outcome_date.lt(first)]
    chosen_early,early_grid=select(early,first-pd.Timedelta(days=1))
    known=dev[dev.outcome_date.le('2024-12-31')]
    chosen_final,final_grid=select(known,'2024-12-31')
    confirmation=dev[dev.listing_year.eq(2024)]
    later_call=overlay(later,chosen_final['parameters'])
    score=metrics(later,later_call)
    report={'model_family':'Same deterministic price rules and existing keyword news scorer; bounded rule overlay research only',
            'default_live_rules_promoted_or_replaced':False,'synthetic_training_rows':0,
            'protocol':'One 31st-close checkpoint, 30-session target, all matured later cases; decisions and source windows unchanged.',
            'coverage':{'development':coverage(dev),'later':coverage(later),
                        'early_matured_n':len(early),'final_known_n':len(known),
                        'unmatured_development_labels_not_fitted':int((dev.outcome_date>'2024-12-31').sum())},
            'current_policy':{'price_only':metrics(later,decisions(later,DEFAULT)),
                              'with_actual_dated_news':metrics(later,later.current_news_governance_sell.to_numpy(dtype=bool)),
                              'news_changed_binary_calls':int((later.sell.to_numpy(dtype=bool)!=later.current_news_governance_sell.to_numpy(dtype=bool)).sum())},
            'selected_on_earlier_data':chosen_early,'selected_on_2024_end_known_outcomes':chosen_final,
            'confirmation_2024':{'default':metrics(confirmation,decisions(confirmation,DEFAULT)),
                                  'overlay':metrics(confirmation,overlay(confirmation,chosen_early['parameters']))},
            'later':{'overlay':score,'paired_change':paired_interval(later,later_call),
                     'changed_calls':int((later_call!=decisions(later,DEFAULT)).sum())},
            'candidate_count':len(GRID),'early_candidates':early_grid,'final_candidates':final_grid,
            'requested_accuracy_target_met_on_this_replay':bool(score['accuracy_pct']>=75 and score['balanced_accuracy_pct']>=70
                 and score['accuracy_pct']>max(score['always_hold_accuracy_pct'],score['always_sell_accuracy_pct'])),
            'input_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (NEWS,*SOURCES['development'],*SOURCES['holdout'],ROOT/'core/monitor_engine.py',ROOT/'services/news_service.py',
                          ROOT/'services/news_identity.py',ROOT/'data/news_issuer_identity_master.csv')},
            'limitations':['Retrospective secondary RSS index; incomplete news coverage and no immutable original-publisher text.',
                'Same-day items excluded because their original publisher clocks were not independently verified.',
                'The existing keyword scorer is heuristic sentiment, not a validated causal event model.',
                'Current NSE issuer master is used for entity disambiguation only; not a historical as-published security master.',
                'Later outcomes were previously inspected; no blind or prospective accuracy claim.',
                'Cash dividends and executable profit are not measured; prices use dated bonus/split adjustments.']}
    ledger=later.copy()
    ledger['selected_overlay_sell']=later_call
    ledger['selected_overlay_correct']=later_call!=later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem+'_ledger.csv'),index=False)
    REPORT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    lines=['# Module 2 — Dated news and the unchanged monitor', '',
        'This experiment uses actual archived headlines, not generated news or fabricated sentiment labels. '
        'The original price/news policy is executed before testing a bounded rule overlay; no live rule is promoted.', '',
        f"Later coverage: {len(later)} IPOs, {report['coverage']['later']['cases_with_eligible_headlines']} with eligible "
        f"preceding-date headlines. Generous observed-news-only accuracy ceiling: "
        f"{report['coverage']['later']['observed_news_only_oracle_accuracy_ceiling_pct']:.2f}% "
        '(assumes every news-covered case becomes correct; not measured predictive accuracy).', '',
        '| Policy, same later IPOs | 30-session direction accuracy |', '|---|---:|',
        f"| Price-only current rules | {report['current_policy']['price_only']['accuracy_pct']:.2f}% |",
        f"| Current rules plus dated news | {report['current_policy']['with_actual_dated_news']['accuracy_pct']:.2f}% |",
        f"| Earlier-data-selected news overlay | {score['accuracy_pct']:.2f}% |",
        f"| Always sell | {score['always_sell_accuracy_pct']:.2f}% |",'',
        f"Current news governance changed {report['current_policy']['news_changed_binary_calls']} binary calls. "
        f"Selected overlay changed {report['later']['changed_calls']}. "
        f"Selected parameters: {chosen_final['parameters']}.", '',
        f"Selection saw {len(known)} matured 2021–2024 outcomes; "
        f"{report['coverage']['unmatured_development_labels_not_fitted']} later outcomes of 2024 IPOs were not yet known. "
        '2025–2026 outcomes did not choose the rule.', '',
        'The goal is not complete unless the full timing-correct requirements are actually met. '
        'No report of final-book Module 1 accuracy is substituted for this Module 2 result.', '', '## Limits','']
    lines+=['- '+s for s in report['limitations']]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('coverage','current_policy','selected_on_2024_end_known_outcomes','later',
                      'requested_accuracy_target_met_on_this_replay')},indent=2))
    return report


if __name__=='__main__':
    run()
