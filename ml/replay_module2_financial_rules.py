"""Earlier-only selection of financial-conditioned existing Module 2 rules.

Quarterly profit margin is observed profitability, NOT earnings surprise.
All 132 later IPOs retain the same decision date and 30-session target; missing
financial evidence leaves the original rule in force, rather than disappearing.
"""
from __future__ import annotations
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT,DEFAULT,assemble,decisions,metrics,paired_interval
from ml.replay_module2_news_overlay import SOURCES
from ml.reparse_module2_financials import OUTPUT as EVIDENCE,verify_detail
from ml.collect_module2_financial_evidence import extract_current_quarter,eligible_results

REPORT=ROOT/'output/module2_financial_rule_replay_20261022.json'
TIGHTER={'loss_pct':8.,'lag_loss_pct':0.,'trail_pct':4.}
PROFIT_TRAIL={'loss_pct':12.,'lag_loss_pct':5.,'trail_pct':12.}
BASE={'mode':'unchanged','max_publication_age_days':None,'profitable_margin_pct':None}
GRID=[BASE]+[{'mode':'loss_only','max_publication_age_days':age,'profitable_margin_pct':None} for age in (30,90)]+[
    dict(zip(('mode','max_publication_age_days','profitable_margin_pct'),values))
    for values in itertools.product(('profit_only','both'),(30,90),(10.,20.))]


def archive_bytes(source):
    path=(ROOT/source['raw_archive_path']).resolve()
    path.relative_to((ROOT/'data/archive/module2_financials').resolve())
    with gzip.open(path,'rb') as stream:
        raw=stream.read()
    if hashlib.sha256(raw).hexdigest()!=source['raw_sha256']:
        raise ValueError('Financial evidence archive hash mismatch')
    return raw


def load_evidence():
    rows=[json.loads(line) for line in EVIDENCE.read_text(encoding='utf-8').splitlines()]
    if len({r['ipo_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate financial evidence checkpoint')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    for row in rows:
        if row['symbol']!=canonical.loc[row['ipo_id']].nse_symbol:
            raise ValueError('Financial issuer identity conflicts with canonical exchange symbol')
        if row['status']!='verified_quarterly_metrics':
            continue
        item=row['selected_filing']
        cutoff=pd.Timestamp(row['cutoff_ist'])
        if not eligible_results([item],row['symbol'],cutoff):
            raise ValueError('Financial publication or period is after the decision cutoff')
        contained=[]
        for source in row['metadata_sources']:
            payload=json.loads(archive_bytes(source))
            contained.extend(payload if isinstance(payload,list) else payload['data'])
        if not any(candidate.get('xbrl')==item['xbrl'] and
                   str(candidate.get('seqNumber') or candidate.get('seq_Id'))==str(item.get('seqNumber') or item.get('seq_Id'))
                   for candidate in contained):
            raise ValueError('Selected financial filing is absent from the archived metadata')
        metrics_=extract_current_quarter(archive_bytes(row['document_source']),item,row['symbol'],
            confirmed_current_context='OneD' if row.get('quarter_crosscheck') else None)
        if row.get('quarter_crosscheck'):
            verify_detail(json.loads(archive_bytes(row['quarter_crosscheck']['source'])),item,metrics_)
        if metrics_!=row['financials']:
            raise ValueError('Saved financial values differ from their actual source document')
    return {row['ipo_id']:row for row in rows}


def assemble_financial(name,evidence):
    data=assemble(*SOURCES[name])
    margin=[];age=[];status=[]
    for case in data.itertuples():
        row=evidence.get(case.ipo_id)
        if row is None or pd.Timestamp(row['decision_date'])!=case.decision_date:
            raise ValueError('Financial checkpoint differs from the unchanged price-rule case')
        status.append(row['status'])
        if row['status']=='verified_quarterly_metrics':
            value=row['financials']
            day_age=(pd.Timestamp(row['cutoff_ist'])-pd.Timestamp(value['published_at_ist'])).total_seconds()/86400
            period_age=(case.decision_date-pd.Timestamp(value['period_end'])).days
            if day_age<0 or period_age<0:
                raise ValueError('Future financial evidence reached a decision')
            margin.append(value['profit_margin_pct'] if period_age<=180 else np.nan)
            age.append(day_age)
        else:
            margin.append(np.nan);age.append(np.nan)
    data['quarter_profit_margin_pct']=margin
    data['financial_publication_age_days']=age
    data['financial_status']=status
    return data


def financial_calls(data,parameters):
    base=decisions(data,DEFAULT).copy()
    if parameters['mode']=='unchanged':
        return base
    known=data.quarter_profit_margin_pct.notna() & data.financial_publication_age_days.le(parameters['max_publication_age_days'])
    if parameters['mode'] in ('loss_only','both'):
        mask=(known & data.quarter_profit_margin_pct.lt(0)).to_numpy()
        base[mask]=decisions(data.loc[mask],TIGHTER)
    if parameters['mode'] in ('profit_only','both'):
        mask=(known & data.quarter_profit_margin_pct.ge(parameters['profitable_margin_pct']) & data.decision_close.ge(data.issue_price)).to_numpy()
        # Profits can widen a profitable position's trail, but cannot waive
        # the existing structural loss/relative-lag thresholds.
        base[mask]=decisions(data.loc[mask],PROFIT_TRAIL)
    return base


def select(known,freeze):
    if known.empty or not known.listing_year.between(2021,2024).all() or not known.outcome_date.le(pd.Timestamp(freeze)).all():
        raise ValueError('Only matured 2021–2024 outcomes may select financial-rule settings')
    baseline=decisions(known,DEFAULT)
    candidates=[]
    for parameters in GRID:
        calls=financial_calls(known,parameters)
        candidates.append({'parameters':parameters,'metrics':metrics(known,calls),'changed_calls':int((calls!=baseline).sum())})
    selected=max(candidates,key=lambda r:(r['metrics']['balanced_accuracy_pct'],r['metrics']['accuracy_pct'],-r['changed_calls']))
    return selected,candidates


def coverage(data):
    known=data.quarter_profit_margin_pct.notna().to_numpy()
    truth=data.day30_return_pct.gt(0).to_numpy()
    correct=decisions(data,DEFAULT)!=truth
    return {'n':len(data),'verified_financial_cases':int(known.sum()),
        'recent_30_day_publications':int((known & data.financial_publication_age_days.le(30).to_numpy()).sum()),
        'loss_making_quarters':int(data.quarter_profit_margin_pct.lt(0).sum()),
        'source_status':data.financial_status.value_counts().to_dict(),
        'financial_cases_only_oracle_accuracy_ceiling_pct':float((correct|known).mean()*100),
        'oracle_note':'Diagnostic bound only: assume every financial-covered call becomes correct; not measured or achievable model accuracy.'}


def run():
    evidence=load_evidence()
    dev=assemble_financial('development',evidence);later=assemble_financial('holdout',evidence)
    confirmation=dev[dev.listing_year.eq(2024)]
    first=confirmation.decision_date.min()
    early=dev[dev.listing_year.isin([2022,2023]) & dev.outcome_date.lt(first)]
    early_choice,early_candidates=select(early,first-pd.Timedelta(days=1))
    known=dev[dev.outcome_date.le('2024-12-31')]
    choice,final_candidates=select(known,'2024-12-31')
    calls=financial_calls(later,choice['parameters'])
    score=metrics(later,calls)
    ledger=later.copy();ledger['financial_rule_sell']=calls
    ledger['financial_rule_correct']=calls!=later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem+'_ledger.csv'),index=False)
    report={'model_family':'Existing deterministic loss/relative-lag/trailing rules, conditional thresholds only',
        'live_policy_changed':False,'synthetic_rows':0,
        'primary_target':'Same 31st-close checkpoint and subsequent 30-session sign; all 132 later cases retained.',
        'data':'Real NSE quarterly-result metadata, raw INR XBRL facts and exact current-quarter details for conflicting WEB exports.',
        'coverage':{'development':coverage(dev),'later':coverage(later),'matured_before_2025_selection':len(known)},
        'candidate_count':len(GRID),'early_choice':early_choice,'final_choice':choice,
        'early_candidates':early_candidates,'final_candidates':final_candidates,
        'confirmation_2024':{'default':metrics(confirmation,decisions(confirmation,DEFAULT)),
                             'financial_rules':metrics(confirmation,financial_calls(confirmation,early_choice['parameters']))},
        'later':{'default':metrics(later,decisions(later,DEFAULT)),'financial_rules':score,'paired_change':paired_interval(later,calls)},
        'requested_accuracy_established':bool(score['accuracy_pct']>=75 and score['balanced_accuracy_pct']>=70
            and score['accuracy_pct']>max(score['always_sell_accuracy_pct'],score['always_hold_accuracy_pct'])),
        'input_sha256':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (EVIDENCE,*SOURCES['development'],*SOURCES['holdout'],Path(__file__),
                                    ROOT/'ml/collect_module2_financial_evidence.py',ROOT/'ml/reparse_module2_financials.py')},
        'limitations':['Historical retrieval is not independent proof of immutable as-published source versions.',
          'Profit margin is not an earnings surprise, cash-flow measure or forecast of growth.',
          'Margins differ across sectors; no cross-sector profitability rank is inferred.',
          'Unavailable evidence remains missing and retains the default rule, not a fabricated zero-profit observation.',
          'Later outcomes have already been inspected. This is exploratory, not blind or prospective validation.',
          'No live model or paper result is promoted from this experiment.']}
    REPORT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    text=['# Module 2 — Dated quarterly-financial evidence and the unchanged rule family','',report['primary_target'],'',
          '| Same later cases | 30-session direction accuracy |','|---|---:|',
          f"| Original monitor | {report['later']['default']['accuracy_pct']:.2f}% |",
          f"| Earlier-data-selected financial thresholds | {score['accuracy_pct']:.2f}% |",
          f"| Always sell | {score['always_sell_accuracy_pct']:.2f}% |",'',
          f"Later verified financial cases: {report['coverage']['later']['verified_financial_cases']} / {len(later)}. "
          f"Selected settings: {choice['parameters']}. Goal established: {report['requested_accuracy_established']}.",'',
          'Missing financials do not remove an IPO or become zero profit. No underlying model family, live policy or paper was changed.','',
          '## Limitations','']+['- '+item for item in report['limitations']]
    REPORT.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('coverage','final_choice','confirmation_2024','later','requested_accuracy_established')},indent=2))
    return report


if __name__=='__main__':
    run()
