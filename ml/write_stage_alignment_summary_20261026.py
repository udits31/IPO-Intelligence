"""Readable results, source manifest, and exact remaining category gaps."""
import hashlib,json
from pathlib import Path
import pandas as pd
from services.saved_preclose_service import ROOT
from ml.run_full_enrichment_audit import integrity

def run():
 m1=json.loads((ROOT/'output/module1_controlled_stage_ablation_20261026.json').read_text())
 m2=json.loads((ROOT/'output/module2_rule_ablation_20261026.json').read_text())
 stage=json.loads((ROOT/'output/module1_stage_alignment_bridge_20261026.json').read_text())
 audit=pd.read_csv(ROOT/'output/module1_day2_recovery_audit_20261026.csv')
 missing=audit[audit.cohort.eq('historical') & audit.year.le(2025) & ~audit.day2_categories_recovered]
 missing.to_csv(ROOT/'output/module1_remaining_day2_category_gaps_20261026.csv',index=False)
 lines=['# Stage-aligned IPO model tests — 26 October 2026','',
 'Training remains 2021–2025 (290 original IPOs); primary comparisons remain the same 69 earlier 2026 IPOs and 23 verified recent listing opens. Five experts and the C=0.5 log-odds stack are preserved. No probability calibration, new live model, ledger rewrite or git push.','',
 '## Module 1: actual comparison','',
 '| Candidate | Earlier 2026 correct | False positives among 23 losers | Recent correct | Promotion gate |',
 '|---|---:|---:|---:|---|']
 for name,r in m1['results'].items():
  if 'test_69' not in r:continue
  a=r['test_69'];b=r['recent_23']
  lines.append(f"| {name} | {a['correct']}/69 ({a['accuracy_pct']:.1f}%) | {a['wrong_positive_calls']}/23 | {b['correct']}/23 ({b['accuracy_pct']:.1f}%) | {'Pass' if r['passes_user_gate'] else 'Fail'} |")
 d=m1['results']['M1_DAY2_TRACK'];a=d['metrics']['test_69'];b=d['metrics']['recent_23']
 lines += [f"| Separate Day-2 track ({d['training_n']} train IPOs) | {a['correct']}/69 ({a['accuracy_pct']:.1f}%) | {a['wrong_positive_calls']}/23 | {b['correct']}/23 ({b['accuracy_pct']:.1f}%) | Fail |",'',
 'Best eligible research result: the frozen weights with recovered financial/category inputs. These are retrospective recomputations, not new calls saved while bidding was open. Neither the retrained quota/OFS/regime variants nor the Day-2 candidate passed all gates. The existing deployed artifact and saved calls are retained.','',
 '### Why Option B cannot honestly be fitted yet','',
 'The 161 earlier matched-total rows do not provide 161 distinct Day-2-to-Day-3 pre-close pairs. Of those ratios, 159 equal 1.0: primarily Day-2 totals divided by the same Day-2 totals. The source audit found zero training pairs meeting the requested distinct-day, category-matched, closing-day-before-17:00 criteria. A final-book inflation offset likewise needs paired pre-close and post-close books, which are not present. The regularised forward/inverse bridge is implemented and input-guarded; no synthetic readings or arbitrary discounts were inserted.','',
 'Recovered Day-2 categories: 273/290 training IPOs, 69/69 earlier 2026 IPOs and 24/24 recent IPOs. The 22 recent total-ratio gaps were not 22 missing category-growth rows. The new replay computes growth from the current category book versus the recovered actual Day-2 categories.','',
 'The dedicated Day-2 evening model was fitted on 273 training issuers with a recoverable dated Day-2 table. Its macro data is restricted to the Day-2 date; same-day GMP with no publication clock is withheld. It is a research candidate and has not been routed into live predictions because its false positives increased.','',
 '### 24th issuer: separate supplied-outcome sensitivity','',
 'The task supplies NSE/BSE issue price Rs1,785 and opening Rs1,800 (+0.84%). There is no bound primary BSE opening record in the local outcome ledger. The JSON includes a separate 24-case sensitivity using those supplied numbers; it does not silently change the verified 23-case denominator.','',
 '### Capital-allocation results','',
 'The JSON includes APPLY precision, SKIP recall and allotment-weighted retail rupee scenarios for every eligible candidate. Reciprocal subscription multiples are approximations, not exact SEBI allotment probabilities. Original lot costs are used, not hard-coded 14/68-lot applications. Exact sNII/bNII outcomes remain unreported where dated split books and validated allotment basis/application counts are absent. Conditional allocation and full-allotment loss dependence remain unverified.','',
 '## Module 2: deterministic rules only','',
 '| Rules | Correct /132 | Winner sensitivity /55 | Loser specificity /77 | Mean net return, original protocol |',
 '|---|---:|---:|---:|---:|']
 for name,r in m2['results'].items():
  lines.append(f"| {name} | {r['correct']}/132 ({r['accuracy_pct']:.2f}%) | {r['winner_sensitivity_n']}/55 | {r['loser_specificity_n']}/77 | {r['conditional_strategy_net_mean_pct']:.3f}% |")
 lines += ['',
 'The full five-gate policy catches more losers but retains only 13 of 55 winners. It does not beat always-sell on accuracy or mean return. The learned classifier is not used by this evaluation or by the existing core monitor; the underperforming candidate is kept separate from the active deterministic monitor. Every gate exposes failed versus unknown evidence.','',
 'Protocol: decision at the 31st observed close; execution at the next observed close; direction measured over the next 30 sessions; original selling-cost treatment retained. The JSON also provides consistent exit costs on both sell and retained positions. Exchange trading dates determine retrospective tape eligibility; unclocked same-day EOD data is not presumed published by 15:30. Filings require eligible broadcast times; archived allotment dates remain an explicit assumption.','',
 '## Remaining data required for the bridge','',
 '1. For training issuers: category-level Day-2 readings paired with a distinct closing-day 14:00–before-17:00 reading, each with its own source time. Existing Day-2/Day-2 ratios cannot train this mapping.',
 '2. For final-book deflation: pre-close and post-close category books for the same training issuer; no test issuer may fit the offset.',
 f"3. Recover Day-2 categories for the {len(missing)} training IPOs in module1_remaining_day2_category_gaps_20261026.csv. All 24 recent IPOs already have category readings.",
 '4. Bind the supplied NSE opening to a primary BSE archive before including it in verified accuracy.',
 '5. Exact HNI rupee evaluation needs dated sNII/bNII books and a validated basis of allotment, not approximate application counts.','',
 'These inspected cohorts have been used for repeated development comparisons. Any selected improvement needs a new prospective cohort before claiming independent generalisation.','']
 (ROOT/'output/STAGE_ALIGNED_MODEL_RESULTS_20261026.md').write_text('\n'.join(lines),encoding='utf8')
 paths=['ml/stage_alignment_bridge.py','ml/run_module1_ablation_20261026.py','ml/run_module2_rule_ablation_20261026.py','core/prove_it_monitor_engine.py','data/audit/module1_inputs_after_ipopremium_20261026.csv','data/audit/module1_stage_aligned_inputs_20261026.csv','output/module1_controlled_stage_ablation_20261026.json','output/module2_rule_ablation_20261026.json']
 (ROOT/'output/stage_alignment_manifest_20261026.json').write_text(json.dumps(dict(files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},integrity=integrity()),indent=2))
 print(json.dumps(dict(selected=m1['selected'],remaining_day2_training=len(missing),frozen_integrity=integrity()['all_preserved'])))
if __name__=='__main__':run()
