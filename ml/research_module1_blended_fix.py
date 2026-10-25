"""Small, predeclared mixture selected only on earlier validation."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from ml.research_module1_downside_fix import ROOT,THRESHOLDS
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain

PREFIX=ROOT/'output/module1_blended_fix_20261024'


def select(base,trajectory):
    if not base.year.isin([2022,2023,2024]).all() or not trajectory.year.isin([2022,2023,2024]).all():
        raise ValueError('Later-year labels cannot select blend')
    if set(base.ipo_id)!=set(trajectory.ipo_id):
        raise ValueError('Validation issuer mismatch')
    b=base.set_index('ipo_id').sort_index(); t=trajectory.set_index('ipo_id').loc[b.index]
    if not np.array_equal(b.target_binary,t.target_binary):
        raise ValueError('Validation outcomes differ')
    y=b.target_binary.to_numpy(); settings=[]
    for weight in (.25,.5,.75):
        p=(1-weight)*b.candidate_02.to_numpy()+weight*t.candidate_19.to_numpy()
        for threshold in THRESHOLDS:
            call=p>=threshold;negative_recall=float((~call[y==0]).mean())
            if negative_recall<.5:
                continue
            settings.append(dict(weight_trajectory=weight,threshold=float(threshold),n=len(y),
                accuracy=float((call==y).mean()),negative_recall=negative_recall,
                balanced_accuracy=(negative_recall+float(call[y==1].mean()))/2,
                brier=float(np.mean((p-y)**2))))
    if not settings:
        raise ValueError('No blend met validation negative recall')
    settings.sort(key=lambda r:(-r['accuracy'],-r['balanced_accuracy'],r['brier'],abs(r['weight_trajectory']-.5),abs(r['threshold']-.5)))
    return settings[0],settings


def run():
    base_path=ROOT/'output/module1_downside_fix_20261024_validation.csv'
    trajectory_path=ROOT/'output/module1_trajectory_fix_20261024_validation.csv'
    choice,settings=select(pd.read_csv(base_path),pd.read_csv(trajectory_path))
    selection_path=PREFIX.with_name(PREFIX.name+'_selection.json')
    selection_path.write_text(json.dumps(dict(selected=choice,settings=settings,test_used_for_selection=False),indent=2)+'\n',encoding='utf8')
    base_artifact=ROOT/'ml/artifacts/module1_combined_train_2021_2024_20261024.joblib'
    trajectory_artifact=ROOT/'ml/artifacts/module1_trajectory_fix_research_20261024.joblib'
    base=joblib.load(base_artifact);trajectory=joblib.load(trajectory_artifact)
    if set(base['training_ids'])!=set(trajectory['training_ids']):
        raise ValueError('Model training membership differs')
    artifact=ROOT/'ml/artifacts/module1_blended_fix_research_20261024.joblib'
    joblib.dump(dict(base=base,trajectory=trajectory,selection=choice,research_only=True),artifact)
    test_path=ROOT/'output/module1_trajectory_fix_20261024_predictions.csv'
    test=pd.read_csv(test_path)
    if set(test.ipo_id)&set(base['training_ids']):
        raise ValueError('Training overlap')
    p_base=base['model'].predict_proba(test[base['features']])[:,1]
    p_trajectory=trajectory['model'].predict_proba(test[trajectory['features']])[:,1]
    if not np.allclose(p_base,test.p_combined) or not np.allclose(p_trajectory,test.p_trajectory_fix):
        raise ValueError('Artifact replay differs')
    weight=choice['weight_trajectory']; test['p_blended_fix']=(1-weight)*p_base+weight*p_trajectory
    test['blended_call']=test.p_blended_fix.ge(choice['threshold']).astype(int)
    test['blended_correct']=test.blended_call.eq(test.target_binary)
    test.to_csv(PREFIX.with_name(PREFIX.name+'_predictions.csv'),index=False)
    test[~test.blended_correct].to_csv(PREFIX.with_name(PREFIX.name+'_errors.csv'),index=False)
    fp=test.target_binary.eq(0)&test.p_combined.ge(.5)
    report=dict(choice=choice,results=summarize(test.target_binary,test.p_blended_fix,choice['threshold']),
        paired_change=paired_gain(test.blended_correct,test.p_combined.ge(.5).eq(test.target_binary)),
        corrected_original_36=int((fp&test.blended_call.eq(0)).sum()),
        newly_missed_positive=int((test.target_binary.eq(1)&test.p_combined.ge(.5)&test.blended_call.eq(0)).sum()),
        training_ids=base['training_ids'],test_ids=sorted(map(int,test.ipo_id)),training_years=[2021,2022,2023,2024],
        by_year={str(int(y)):summarize(g.target_binary,g.p_blended_fix,choice['threshold']) for y,g in test.groupby('year')},
        artifact=str(artifact.relative_to(ROOT)),live_promoted=False,
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (base_path,trajectory_path,base_artifact,trajectory_artifact,test_path,selection_path,artifact,
             ROOT/'ml/research_module1_blended_fix.py',ROOT/'output/module1_blended_fix_protocol_20261024.md')})
    PREFIX.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('choice','results','paired_change','corrected_original_36','newly_missed_positive')},indent=2))


if __name__=='__main__':
    run()
