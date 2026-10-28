"""Explain frozen six-IPO misses and test adding 2025 without changing live fits."""
import gzip
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ml.model_pipeline import prepare_dataset, fit_model, score_model, summarize, FEATURE_COLS
from ml.audit_module1_listing_opens import verified_outcomes
from ml.research_monitor_daily_learning import paired_gain
from services.snapshot_service import DB_PATH, get_prelisting_outcome_scorecard
from services.listing_outcome_service import verified_listing_outcomes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/module1_six_ipo_and_2025_training_audit_20261023.json'


def explain(bundle,qib,retail,hni):
    transformed=np.log1p(np.maximum(np.asarray([qib,retail,hni],dtype=float),.01))
    terms=(transformed-np.asarray(bundle['scaler_mean']))/np.asarray(bundle['scaler_scale'])*np.asarray(bundle['coefficients'])
    linear=float(bundle['intercept'])+float(terms.sum())
    return dict(probability=1/(1+math.exp(-linear)),linear_score=linear,
        base_log_odds=float(bundle['intercept']),
        contributions_log_odds=dict(zip(('QIB','Retail','HNI'),map(float,terms))))


def calibration(y,p):
    y,p=np.asarray(y),np.asarray(p)
    rows=[]
    for low,high in ((0,.5),(.5,.8),(.8,.9),(.9,1.0000001)):
        selected=(p>=low)&(p<high)
        if not selected.any():
            continue
        n=int(selected.sum()); rate=float(y[selected].mean()); z=1.96
        centre=(rate+z*z/(2*n))/(1+z*z/n)
        margin=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        rows.append(dict(score_band=f'{low*100:.0f}–{min(high,1)*100:.0f}%',n=n,
            mean_predicted_positive_pct=float(p[selected].mean()*100),
            actual_positive_pct=rate*100,positive_count=int(y[selected].sum()),
            actual_rate_wilson_95_pct=[(centre-margin)*100,(centre+margin)*100]))
    return rows


def six_audit(bundle):
    saved=get_prelisting_outcome_scorecard()['records']
    outcomes=verified_listing_outcomes()
    rows=[]
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory=sqlite3.Row
        for record in saved:
            capture=dict(connection.execute('SELECT * FROM subscription_capture_v2 WHERE id=?',
                                            (record['capture_id'],)).fetchone())
            raw=gzip.decompress(capture['raw_html_gzip'])
            if hashlib.sha256(raw).hexdigest()!=record['evidence_sha256']:
                raise ValueError('Saved bid capture hash mismatch')
            for key in ('qib_x','nii_x','retail_x'):
                if not math.isclose(float(record[key]),float(capture[key]),abs_tol=1e-10):
                    raise ValueError('Saved model inputs differ from capture')
            explanation=explain(bundle,record['qib_x'],record['retail_x'],record['nii_x'])
            if not math.isclose(explanation['probability'],record['positive_listing_probability'],abs_tol=1e-10):
                raise ValueError('Cannot reproduce the frozen prediction')
            outcome=outcomes[record['ipo_id']]
            rows.append(dict(ipo_id=record['ipo_id'],company_name=record['company_name'],
                probability_pct=record['positive_listing_probability']*100,
                qib_x=record['qib_x'],retail_x=record['retail_x'],hni_x=record['nii_x'],
                issue_price=outcome['issue_price'],listing_open=outcome['nse_open_price'],
                actual_return_pct=record['actual_open_return_pct'],correct=record['direction_correct'],
                predicted_at=record['recorded_at_ist'],bid_capture_at=record['captured_at_ist'],
                source_family=record['source_family'],source_phase=record['capture_phase'],
                closing_date=capture['closing_date'],snapshot_stage=capture['snapshot_stage'],
                source_hash_verified=True,explanation=explanation))
    return rows


def study_2025(frame,audit):
    base=frame[frame.year.between(2021,2026) & ~frame.id.isin([293,307])].copy()
    # New 2025 trusts are not corporate IPOs; exclude by instrument identity,
    # not by whether they made or lost money. Original 252 fit rows are retained.
    noncorporate=base.year.eq(2025)&base.company_name.str.contains('Trust|REIT',case=False,regex=True)
    excluded=base[noncorporate][['id','company_name']].to_dict('records')
    base=base[~noncorporate]
    verified=base.merge(audit[['ipo_id','exchange_open']],left_on='id',right_on='ipo_id',validate='one_to_one')
    test=verified[verified.year.eq(2026)].copy()
    y=test.exchange_open.gt(test.issue_price).to_numpy(dtype=int)
    result=[];predictions=test[['id','company_name','issue_price','exchange_open']].copy()
    for labels in ('canonical','exchange_verified'):
        pool=base.copy() if labels=='canonical' else verified.copy()
        if labels=='exchange_verified':
            pool['target_binary']=pool.exchange_open.gt(pool.issue_price).astype(int)
        probabilities={}
        for end in (2024,2025):
            train=pool[pool.year.between(2021,end)]
            if not train.listing_date.lt(test.listing_date.min()).all() or set(train.id)&set(test.id):
                raise ValueError('Training crossed into the test period')
            scaler,model=fit_model(train)
            p=score_model(scaler,model,test)
            probabilities[end]=p
            name=f'{labels}_through_{end}'
            predictions[name]=p
            result.append(dict(label_source=labels,trained_through=end,training_n=len(train),
                training_ids_sha256=hashlib.sha256(','.join(map(str,sorted(train.id))).encode()).hexdigest(),
                result=summarize(y,p,.5),
                candidate_parameters=dict(scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                    coefficients=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))))
        result[-1]['paired_change_from_2024_fit']=paired_gain((probabilities[2025]>=.5)==y,(probabilities[2024]>=.5)==y)
    predictions.to_csv(OUT.with_name(OUT.stem+'_2026_comparison.csv'),index=False)
    missing=base[base.year.eq(2026)&~base.id.isin(test.id)][['id','company_name']].to_dict('records')
    return dict(test_year=2026,test_n=len(test),fixed_direction_threshold=.5,
        excluded_2025_noncorporate=excluded,unverified_2026_openings=missing,comparisons=result,
        live_model_changed=False,
        protocol='Same fixed three log-bid inputs, StandardScaler and LogisticRegression(C=0.5). '
                 'No 2026 tuning; canonical-vs-canonical isolates adding 2025, verified-vs-verified is a separate label-source comparison.')


def run():
    bundle_path=ROOT/'ml/model_bundle.json'
    before=hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    bundle=json.loads(bundle_path.read_text(encoding='utf-8'))
    frame=prepare_dataset();audit=verified_outcomes()
    six=six_audit(bundle)
    later=frame[frame.year.isin([2025,2026])].merge(audit[['ipo_id','exchange_open']],
                                                    left_on='id',right_on='ipo_id',validate='one_to_one')
    p=np.array([explain(bundle,row.qib_subscription,row.retail_subscription,row.hni_subscription)['probability']
                for row in later.itertuples()])
    report=dict(six_ipos=six,calibration_2025_2026=calibration(later.exchange_open.gt(later.issue_price),p),
        adding_2025=study_2025(frame,audit),live_model_changed=False,
        limitations=['Six cases are too few to establish calibration or select new parameters.',
            'All six saved forecasts were created after bidding; this is not live pre-close accuracy.',
            'Final-book training does not validate interim subscription probabilities.',
            '2026 outcomes have been inspected before: this is retrospective research, not untouched prospective validation.',
            'Missing features are plausible omissions, not proven causes of an individual price move.'],
        input_sha256={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (bundle_path,ROOT/'data/canonical_listed_ipos.csv',ROOT/'data/subscription_observations.csv',
                ROOT/'data/audit/module1_exchange_listing_open_audit.csv',Path(__file__))})
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest()!=before:
        raise ValueError('Research must not modify the live bundle')
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Why Bonfiglioli missed, six-IPO comparison, and training through 2025','',
        '## Frozen six-IPO calls','',
        '| IPO | QIB | Retail | HNI | Saved positive probability | Actual opening return | Direction correct? |',
        '|---|---:|---:|---:|---:|---:|---|']
    lines += [f"| {row['company_name']} | {row['qib_x']:.4f}x | {row['retail_x']:.2f}x | {row['hni_x']:.2f}x | "
              f"{row['probability_pct']:.2f}% | {row['actual_return_pct']:+.2f}% | {'Yes' if row['correct'] else 'No'} |" for row in six]
    bonfig=next(row for row in six if row['ipo_id']=='chitto_2861')
    exp=bonfig['explanation']
    lines += ['', '## Bonfiglioli: exact score, not a guess about its price move','',
        f"The archived bid response hash and saved inputs match. The exact saved probability replays to {bonfig['probability_pct']:.2f}%.",
        f"Linear score = {exp['base_log_odds']:.6f} intercept " +
        ' '.join(f"{value:+.6f} {name}" for name,value in exp['contributions_log_odds'].items())+'.',
        'The intercept is the model score at mean transformed training features, not the unconditional positive-listing rate.',
        'All three bid coefficients are positive. Higher demand mechanically increases the probability; it does not measure whether the issue price is attractive.',
        'GMP, valuation, profitability, OFS/fresh-capital allocation, issue size and market conditions are NOT inputs to this deployed classifier.',
        'A 91% positive probability is not a 91% expected gain. One miss does not prove miscalibration, but the miss must count.',
        '', '## Timing problems across the six','',
        'HD Fire and Fusion use 14 October interim snapshots, although their offers closed later. The six-call comparison mixes interim and post-close book stages. '
        'The final-book model is not validated on interim observations. Do not rewrite those saved calls with later numbers.',
        'The other four captures are post-close. Bonfiglioli and Tonbo both have very strong subscriptions, yet opened flat and +65.71% respectively. '
        'A blanket penalty based solely on strong bids or OFS would not resolve this distinction.',
        'Current webpage GMP or other facts retrieved after listing are not substituted as decision-time evidence.',
        '', '## Calibration diagnostic on 179 later historical IPOs','',
        '| Score band | IPOs | Mean predicted positive | Actually positive |','|---|---:|---:|---:|']
    lines += [f"| {row['score_band']} | {row['n']} | {row['mean_predicted_positive_pct']:.1f}% | {row['actual_positive_pct']:.1f}% |"
              for row in report['calibration_2025_2026']]
    lines += ['', '## Teacher-requested test: add 2025, evaluate only 2026','',
        'No model settings or thresholds were selected on 2026. Same 75 exchange-verified 2026 outcomes in every row.',
        '', '| Training label source | Trained through | Training cases | 2026 direction accuracy |','|---|---:|---:|---:|']
    lines += [f"| {row['label_source']} | {row['trained_through']} | {row['training_n']} | {row['result']['classifier_accuracy_pct']:.1f}% |"
              for row in report['adding_2025']['comparisons']]
    lines += ['', 'Four newly added 2025 trust instruments are excluded because this is a corporate IPO model. '
        'Six 2026 records lack verified opens (including trust instruments) and are listed explicitly in the JSON coverage report.',
        'The 75-case 2026 score must not replace or be confused with the 179-case 2025–2026 score.',
        '', '## Fix direction','',
        '1. Separate interim-book forecasts from final-book forecasts, and validate each stage using dated observations.',
        '2. Select probability calibration using chronological earlier-fold predictions, not these six answers.',
        '3. Add valuation, earnings and offer-structure features only when their pre-decision publication evidence is available. Test incremental benefit against bids-only and GMP-only on identical IPOs.',
        '4. Keep all misses, retain immutable saved probabilities, and require paired evidence before promoting a new model.',
        '', 'No live bundle, saved forecast, manuscript or performance headline was changed.', '', *['- '+line for line in report['limitations']]]
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(calibration=report['calibration_2025_2026'],adding_2025=report['adding_2025']),indent=2))
    return report


if __name__=='__main__':
    run()
