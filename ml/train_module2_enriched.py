"""Same-132-issuer shadow rule and grouped chronological classifier benchmark."""
import json,hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from services.enriched_monitor import assemble,evaluate
from services.point_in_time import IST
from services.saved_preclose_service import ROOT
from services.enrichment_runtime import anchor_metadata

FEATURES=['close_to_issue','sma10_gap','sma20_gap','atr_14_pct','rs_vs_nifty_10d','rs_vs_sector_10d',
          'delivery_pct_5d_avg','delivery_accumulation_ratio','days_to_30d_anchor_unlock','days_to_90d_anchor_unlock',
          'filing_sentiment_score','hard_governance_red_flag','unlock_window_bulk_selling_flag','day1_net_institutional_bulk_pct_float']


def records(kind,id):
    path=ROOT/'data/market'/kind/(str(int(id))+'.jsonl')
    rows=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()] if path.exists() else []
    return [r for r in rows if kind!='corporate_announcements' or r.get('schema_version')==2]


def build(ledger,prices,ipos):
    output=[]
    for row in ledger.to_dict('records'):
        id=int(row['ipo_id']);cutoff=str(row['decision_date'])+'T15:30:00+05:30'
        group=prices[prices.ipo_id.eq(id)].sort_values('date')
        f=assemble(group,decision_timestamp=cutoff,issue_price=float(ipos.loc[id].issue_price),
            delivery=records('delivery_series',id),filings=records('corporate_announcements',id),deals=records('bulk_block_deals',id),anchor=anchor_metadata(id,cutoff),strict=True)
        decision=evaluate(f);values={k:f.get(k,np.nan) for k in FEATURES}
        values.update(close_to_issue=f['close']/f['issue_price']-1 if f.get('close') else np.nan,
            sma10_gap=f['close']/f['sma_10']-1 if f.get('sma_10') else np.nan,
            sma20_gap=f['close']/f['sma_20']-1 if f.get('sma_20') else np.nan)
        y=float(row['day30_return_pct'])>0
        sell=decision['action']=='REDUCE / EXIT'
        output.append(dict(ipo_id=id,year=int(row['listing_year']),decision_date=row['decision_date'],outcome_date=row['outcome_date'],
            up=int(y),old_sell=bool(row['sell']),enriched_action=decision['action'],enriched_sell=sell,enriched_correct=(not sell)==y,
            old_correct=(not bool(row['sell']))==y,coverage=f.get('coverage'),features=values))
    return output


def run():
    ipos=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    pairs=[('development','output/module2_development_decision_ledger_goal_corporate_actions_20261021.csv','data/audit/module2_development_prices_2021_2024_nse_extended_ca_reconciled.csv'),
           ('holdout','output/module2_holdout_decision_ledger_governance_20261023.csv','data/audit/module2_holdout_prices_2025_2026_nse_full_ca_reconciled.csv')]
    data={}
    for group,ledger,prices in pairs:data[group]=build(pd.read_csv(ROOT/ledger),pd.read_csv(ROOT/prices),ipos)
    dev=pd.DataFrame([{**{k:v for k,v in r.items() if k not in ('features','coverage')},**r['features']} for r in data['development']])
    later=pd.DataFrame([{**{k:v for k,v in r.items() if k not in ('features','coverage')},**r['features']} for r in data['holdout']])
    if len(later)!=132 or later.ipo_id.nunique()!=132:raise ValueError('Fixed later cohort changed')
    first_test=pd.to_datetime(later.decision_date).min()
    validation=dev[dev.year.eq(2024)&(pd.to_datetime(dev.outcome_date)<first_test)]
    train=dev[(dev.year<=2023)&(pd.to_datetime(dev.outcome_date)<pd.to_datetime(validation.decision_date).min())]
    if train.empty or validation.empty:raise ValueError('Matured pre-test training and validation are required')
    settings=[]
    for c in (.05,.5,5.):
        model=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=42))
        model.fit(train[FEATURES],train.up);p=model.predict_proba(validation[FEATURES])[:,1]
        settings.append(dict(C=c,brier=float(brier_score_loss(validation.up,p))))
    selected=min(settings,key=lambda r:r['brier'])
    earlier=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=selected['C'],max_iter=3000,random_state=42))
    earlier.fit(train[FEATURES],train.up)
    calibration=IsotonicRegression(out_of_bounds='clip').fit(earlier.predict_proba(validation[FEATURES])[:,1],validation.up)
    fitting=dev[(dev.year<=2024)&(pd.to_datetime(dev.outcome_date)<first_test)]
    assert not set(fitting.ipo_id)&set(later.ipo_id)
    model=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=selected['C'],max_iter=3000,random_state=42))
    model.fit(fitting[FEATURES],fitting.up);raw=model.predict_proba(later[FEATURES])[:,1];p=calibration.predict(raw)
    later['enriched_ml_raw_probability']=raw
    later['enriched_ml_probability']=p;later['enriched_ml_correct']=(p>=.5)==later.up
    result=dict(protocol='One case per issuer; earlier outcomes mature before validation/test decisions; select C on 2024 Brier only; 132 fixed 2025-2026 cases.',
        holdout_n=len(later),old_rule_correct=int(later.old_correct.sum()),old_rule_accuracy_pct=100*later.old_correct.mean(),
        always_sell_correct=int((later.up==0).sum()),always_sell_accuracy_pct=100*(later.up==0).mean(),
        enriched_rule_correct=int(later.enriched_correct.sum()),enriched_rule_accuracy_pct=100*later.enriched_correct.mean(),
        enriched_ml_correct=int(later.enriched_ml_correct.sum()),enriched_ml_accuracy_pct=100*later.enriched_ml_correct.mean(),
        enriched_ml_raw_correct=int(((raw>=.5)==later.up).sum()),enriched_ml_raw_accuracy_pct=100*float(((raw>=.5)==later.up).mean()),
        calibration='Isotonic on 2024 predictions from pre-2024 matured training only; final classifier refit on pre-test matured 2021-2024 outcomes.',
        binary_rule_mapping='REDUCE / EXIT means sell; HOLD, ACCUMULATE and REVIEW / TRIM mean retain for this fixed direction test.',
        training_issuers=len(fitting),training_ids=list(map(int,fitting.ipo_id)),test_ids=list(map(int,later.ipo_id)),training_test_overlap=0,
        calibration_issuers=len(validation),calibration_latest_outcome=str(pd.to_datetime(validation.outcome_date).max().date()),first_test_decision=str(first_test.date()),
        validation=settings,selected=selected,live_promoted=False,
        coverage=dict(strict_delivery_cases=sum(r['coverage']['delivery_admitted']>0 for r in data['holdout']),filing_cases=sum(r['coverage']['filings_admitted']>0 for r in data['holdout']),bulk_cases=sum(r['coverage']['deals_admitted']>0 for r in data['holdout'])),
        limitations=['Strict delivery/bulk publication clocks remain missing; those inputs are excluded, not zero.','Missing anchor calendars cannot imply safe unlock status.','Exchange-summary snippets are not full attachments.','These later cases were inspected previously; research is not an untouched prospective test.'])
    (ROOT/'output/module2_enriched_benchmark_20261025.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf8')
    later.to_csv(ROOT/'output/module2_enriched_benchmark_20261025.csv',index=False)
    joblib.dump(dict(model=model,calibration=calibration,features=FEATURES,research_only=True,training_ids=result['training_ids']),ROOT/'ml/artifacts/module2_enriched_research_20261025.joblib')
    print(json.dumps({k:v for k,v in result.items() if k not in ('training_ids','test_ids','validation','limitations')}),flush=True)
    return result

if __name__=='__main__':run()
