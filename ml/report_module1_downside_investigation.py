"""One readable summary of completed downside investigations, no new tuning."""
import json

import joblib
import pandas as pd

from ml.research_module1_downside_fix import ROOT

PREFIX=ROOT/'output/module1_downside_investigation_20261024'


def run():
    def report(name):
        return json.loads((ROOT/f'output/module1_{name}_20261024.json').read_text())
    old=report('2025_2026_paired_holdout')
    first=report('downside_fix'); financial=report('financial_fix')
    temporal=report('trajectory_fix'); blend=report('blended_fix')
    original=pd.read_csv(ROOT/'output/module1_downside_fix_20261024_predictions.csv')
    final=pd.read_csv(ROOT/'output/module1_blended_fix_20261024_predictions.csv')
    cross=pd.read_csv(ROOT/'data/audit/module1_later_bid_crosscheck_20261024.csv')
    if set(original.ipo_id)!=set(final.ipo_id) or set(cross.ipo_id)!=set(final.ipo_id):
        raise ValueError('Report cohort mismatch')
    fields=['ipo_id','p_blended_fix','blended_call','blended_correct']
    review=original.merge(final[fields],on='ipo_id',validate='one_to_one')
    review=review.merge(cross[['ipo_id','status','source_url']],on='ipo_id',validate='one_to_one',suffixes=('','_crosscheck'))
    wrong=review[review.target_binary.eq(0)&review.p_combined.ge(.5)].copy()
    wrong['actual_outcome']=wrong.target_return.apply(lambda v:'Flat' if abs(v)<1e-9 else 'Negative')
    wrong['corrected_by_new_model']=wrong.blended_call.eq(0)
    artifact=joblib.load(ROOT/'ml/artifacts/module1_combined_train_2021_2024_20261024.joblib')
    wrong['old_logodds_training_intercept']=float(artifact['model'].named_steps['model'].intercept_[0])
    wrong['largest_old_positive_component']=wrong[[c for c in wrong if c.startswith('old_logodds_')]].idxmax(axis=1).str.replace('old_logodds_','',regex=False)
    wrong.to_csv(PREFIX.with_name(PREFIX.name+'_36_error_review.csv'),index=False)
    rows=[('Previous combined model',old['results']['p_combined']),
          ('GMP sign baseline',old['results']['p_gmp_sign']),
          ('Fitted GMP-only baseline',old['results']['p_gmp_fitted']),
          ('Earlier-validation-selected bid/GMP fix: rejected',first['results']['selected_fix']),
          ('Financial extension: rejected',financial['results']),
          ('Demand-trajectory model',temporal['results']),
          ('Original + trajectory combination',blend['results'])]
    flat=int(wrong.actual_outcome.eq('Flat').sum());weak=int(wrong.gmp_pct.le(5).sum())
    positive_gmp=int(wrong.gmp_pct.gt(0).sum())
    lines=['# Module 1: why 36 positive predictions were wrong, and what changed','',
        '**Result: the new combination gets 118/155 right (76.1%). It has not reached 80%.**','',
        'All fits and model/cutoff choices use 2021–2024 only. All results below use the exact same 155 IPOs from 2025–2026. No losses or flat openings were dropped, no final bids were substituted, and no saved original forecasts were rewritten.',
        '', '## What the 36 misses show','',
        f'- {flat} were flat openings; {36-flat} opened below issue price. Flat is correctly counted as a missed positive call.',
        f'- {positive_gmp}/36 had positive unofficial GMP, so positive GMP was not sufficient evidence of an opening gain.',
        f'- {weak}/36 had GMP of 5% or less. The earlier logistic model often let QIB/HNI evidence outweigh that weak sentiment.',
        '- Training had 77.6% positive openings; this test had 68.4%. The model learned an optimistic base rate that did not transfer fully.',
        '- Independent source checks found nine material bid disagreements. Scoda had 646x retail on one page versus 6.99x on another. These are source conflicts, not a licence to invent decimal corrections. Consistent whole-cohort source substitution did not improve the previous model’s 74.8% accuracy.',
        '', '## Actual tests','',
        '| Experiment | Correct / 155 | Accuracy | Flat/negative caught | Positive openings caught |',
        '|---|---:|---:|---:|---:|']
    for name,item in rows:
        lines.append(f'| {name} | {item["tn"]+item["tp"]}/155 | {item["classifier_accuracy_pct"]:.1f}% | {item["tn"]}/49 | {item["tp"]}/106 |')
    lines += ['', 'The rejected results remain in their reports. The final combination was selected from three fixed mixture weights using only earlier-year chronological validation; its later-year score was not used to choose its weight or cutoff.',
        '', '## The final candidate','',
        '- Adds how QIB/HNI/retail demand changed from the first bidding day to the decision date, plus earlier GMP movement from the same publisher. It never uses the final day’s bids.',
        '- Combines the original logistic score with the balanced demand-trajectory forest. Earlier-year validation selected 75% trajectory weight and a 47.5% decision-score cutoff. This is a learned call policy; its score is not a certified calibrated probability.',
        f'- Fixes {blend["corrected_original_36"]} of the original 36 wrong positive calls, leaving 17 wrong positive calls. It newly misses {blend["newly_missed_positive"]} previously correct positive openings. Total correct rises by two.',
        f'- Paired accuracy change versus previous: {blend["paired_change"]["change_pp"]:+.2f} percentage points; 95% paired bootstrap interval {blend["paired_change"]["bootstrap_95_pp"]}. This does not establish a reliable overall accuracy improvement.',
        '- Negative recall rises from 26.5% to 65.3%; positive recall falls from 97.2% to 81.1%. Precision among positive calls is 83.5%, which must not be reported as 83.5% overall accuracy.',
        '', '## Remaining gaps','',
        '- Training contains only 170 usable matched interim-book/GMP IPOs; just 38 are nonpositive. The expanding validation cohort contains only 28 nonpositive IPOs. Additional model complexity did not solve that information limit.',
        '- Trajectory fields are missing for 45 earlier-year training IPOs. Financial valuations are observed for 61/170 training IPOs versus 138/155 test IPOs. That unequal coverage limits how well these features can be learned.',
        '- The 2025–2026 outcomes have been examined repeatedly. These are exploratory retrospective results, even though no later-year label entered model fitting or selection. A new prospective test is needed before claiming that a small improvement will persist.',
        '- Public historical pages provide dated rows and fiscal periods, but do not certify original publication times. Original saved pre-close forecasts remain distinct.',
        '- No candidate was deployed automatically; the existing live model, paper and Module 2 remain unchanged.',
        '', '## Files','',
        f'- Per-IPO review of all 36 original false positives: `{PREFIX.name}_36_error_review.csv`.',
        '- All 155 original/new calls and actual exchange opens: `module1_blended_fix_20261024_predictions.csv`.',
        '- Reproducible research model: `ml/artifacts/module1_blended_fix_research_20261024.joblib`.',
        '- Model-selection traces, source hashes and every rejected experiment are retained alongside this report.']
    PREFIX.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(flat_false_positives=flat,weak_gmp_false_positives=weak,positive_gmp_false_positives=positive_gmp,
                          training_nonpositive=int(pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv').target_binary.eq(0).sum())),indent=2))


if __name__=='__main__':
    run()
