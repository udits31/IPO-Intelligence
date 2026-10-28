"""Prior-year-only candidate selection, followed by fixed 23-issuer diagnosis."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from services.saved_preclose_service import ROOT
from ml.research_module1_forward_update import BASE,MACRO,FINANCIAL


def run():
    frame=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv')
    columns=BASE+MACRO+FINANCIAL
    train=frame[frame.year.le(2024)];validation=frame[frame.year.eq(2025)]
    def x(rows):
        values=rows.reindex(columns=columns).astype(float).copy()
        values['gmp_pct']=np.sign(values.gmp_pct)*np.log1p(abs(values.gmp_pct))
        return values
    configs=[('logistic_regularized',make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=42))),
             ('logistic',make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=3000,random_state=42))),
             ('shallow_boosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=4,min_samples_leaf=25,l2_regularization=5,random_state=42))]
    scores=[]
    for name,model in configs:
        model.fit(x(train),train.target_binary)
        p=model.predict_proba(x(validation))[:,1]
        scores.append(dict(name=name,brier=brier_score_loss(validation.target_binary,p),accuracy=float(((p>=.5)==validation.target_binary).mean())))
    selected=min(scores,key=lambda r:r['brier'])['name']
    model=next(model for name,model in configs if name==selected)
    fitting=frame[frame.year.le(2025)]
    model.fit(x(fitting),fitting.target_binary)
    replay=json.loads((ROOT/'output/listed_closing_day_backtest.json').read_text());records=replay['records']
    features=pd.DataFrame([r['features'] for r in records]);probabilities=model.predict_proba(x(features))[:,1]
    ledger=[]
    for row,p in zip(records,probabilities):
        actual=row['outcome']
        ledger.append(dict(company=row['company_name'],probability=float(p),correct=bool((p>=.5)==actual['positive']) if actual else None))
    resolved=[r for r in ledger if r['correct'] is not None]
    result=dict(selection='2025 Brier score only; fit 2021-2024 for selection, refit through 2025; no 2026 labels in training or selection',
        retrospective_limitation='Candidate design followed inspection of recent misses; these 23 are a diagnostic set, not a fresh holdout.',
        validation=scores,selected=selected,training_ids=sorted(map(int,fitting.ipo_id)),
        correct=sum(r['correct'] for r in resolved),evaluated=len(resolved),accuracy_pct=100*sum(r['correct'] for r in resolved)/len(resolved),records=ledger)
    (ROOT/'output/listed_input_fix_candidate.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    joblib.dump(dict(model=model,columns=columns,transform='signed_log_gmp',training_ids=result['training_ids'],research_only=True),ROOT/'ml/artifacts/listed_input_fix_research.joblib')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','training_ids')}))

if __name__=='__main__': run()
