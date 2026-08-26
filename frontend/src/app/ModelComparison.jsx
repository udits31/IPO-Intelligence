"use client";
import {useEffect,useState} from 'react';
import {api} from '../lib/api';

const score=m=>`${m.correct}/${m.sample_size} · ${m.classifier_accuracy_pct.toFixed(1)}%`;
const names={p_legacy_original252:'Original subscription model · 252 training IPOs',
  p_legacy_matched186:'Three-input model · matched early-book training',
  p_five_expert_strict2024:'Five-expert model · strict 2024 refit',
  p_gmp_only:'GMP-only model',p_always_positive:'Always positive',
  p_deployed_five_expert:'Current five-expert model',p_stage_logistic_through2025:'Three-input model · same 290 training IPOs'};

export default function ModelComparison({data:supplied}={}){
  const [data,setData]=useState(supplied||null);const [error,setError]=useState(null);
  useEffect(()=>{if(supplied){setData(supplied);return;}api.modelComparison().then(setData).catch(e=>setError(e.message));},[supplied]);
  if(error)return <section className="panel"><h3>Model comparison unavailable</h3><p>{error}</p></section>;
  if(!data)return <section className="panel"><p>Loading matched model comparison…</p></section>;
  if(data.status!=='ready')return <section className="panel"><h3>Model comparison</h3><p>{data.message}</p></section>;
  const strict=data.matched_results.clean_day2_163;
  const deployed=data.actual_deployed_2026_only.all69;
  return <section className="panel"><div className="section-heading"><div><span className="eyebrow">MODEL LAB</span><h3>Which model actually performs better?</h3></div></div>
    <p>Current model comparison: train through 2025, test on the same 69 earlier 2026 IPOs. Every model is scored against the same listing opens and bid snapshots.</p>
    <div className="table-wrap"><table><thead><tr><th>Model</th><th>Direction correct</th></tr></thead><tbody>{['p_deployed_five_expert','p_stage_logistic_through2025'].map(k=><tr key={k}><td>{names[k]}</td><td>{score(deployed.direction[k])}</td></tr>)}{data.period_stability?.current_app_gmp_comparison_2026&&<tr><td>App GMP-only baseline · fixed 62.5% rule</td><td>{score(data.period_stability.current_app_gmp_comparison_2026.same69.gmp_only)}</td></tr>}</tbody></table></div>
    <p className="fine-print">The current model leads the three-input model by four IPOs. Its lead over the app’s GMP-only baseline is one IPO. These small retrospective differences do not establish a reliable future advantage.</p>
    {data.period_stability&&<details className="research-history" open><summary>Consistency across years versus GMP-only</summary>
      <p>Fixed 50% rule for both learned models below; the app’s separately locked GMP baseline is shown above. Each year uses earlier training outcomes only. The primary comparison includes the same IPOs with verified Day-2 evidence and dated GMP in every column.</p>
      <div className="table-wrap"><table><thead><tr><th>Test year</th><th>IPOs</th><th>Our model</th><th>GMP-only model</th><th>GMP above zero</th><th>Always positive</th></tr></thead><tbody>{Object.entries(data.period_stability.by_year).map(([year,r])=><tr key={year}><td>{year}</td><td>{r.model.sample_size}</td><td>{score(r.model)}</td><td>{score(r.gmp_only)}</td><td>{score(r.raw_gmp_positive_rule)}</td><td>{score(r.always_positive)}</td></tr>)}</tbody></table></div>
      <p>We do not beat GMP-only in every period. These historical comparisons guide validation; future scores are saved automatically before bidding closes and measured after the listing open.</p>
      <details><summary>Quarterly results and coverage</summary><div className="table-wrap"><table><thead><tr><th>Quarter</th><th>IPOs</th><th>Our model</th><th>GMP-only model</th><th>GMP above zero</th></tr></thead><tbody>{Object.entries(data.period_stability.by_quarter).map(([q,r])=><tr key={q}><td>{q}</td><td>{r.model.sample_size}</td><td>{score(r.model)}</td><td>{score(r.gmp_only)}</td><td>{score(r.raw_gmp_positive_rule)}</td></tr>)}</tbody></table></div><p className="fine-print">Small quarters are descriptive. Intraday and closing-day forecasts have different input stages and are evaluated prospectively; Day-2 backtest accuracy does not transfer automatically.</p></details>
    </details>}
    {data.stage_matched&&<details className="research-history" open><summary>Matching the prediction time: Day 1 and Day 2</summary>
      <p>Separate five-expert refits use clocked subscription books and rebuild GMP, growth and market inputs at the same stage. Training ends in 2025; both tests contain the same 69 IPOs from 2026.</p>
      <div className="table-wrap"><table><thead><tr><th>Input stage</th><th>Training IPOs</th><th>Stage model · 50% rule</th><th>GMP-only · fixed 62.5% rule</th><th>Observed GMP</th></tr></thead><tbody>{Object.entries(data.stage_matched.stages).filter(([,r])=>r.status==='evaluated_research').map(([stage,r])=><tr key={stage}><td>{stage==='DAY1_EOD'?'Day 1 close':'Day 2 close'}</td><td>{r.training_n}</td><td>{score(r.all_test.model)}</td><td>{score(r.all_test.gmp_fixed625)}</td><td>{r.observed_gmp_n}/{r.test_n}</td></tr>)}</tbody></table></div>
      <p>These stage-specific candidates remain in research because they did not outperform the stronger GMP control. The active frozen model is unchanged.</p>
      <details><summary>Stage coverage and validation details</summary><p>Earlier-year checks use 2024 and 2025 only. Final closing-day tables stamped 5 pm or later were excluded from the pre-close cohort; no eligible pre-2026 closing-day training cohort was recovered. Missing GMP uses training-only imputation in the all-69 learned-model comparison. Raw-GMP comparisons use only observed quotes.</p><div className="table-wrap"><table><thead><tr><th>Stage</th><th>Validation year</th><th>Model</th><th>GMP-only</th></tr></thead><tbody>{Object.entries(data.stage_matched.stages).filter(([,r])=>r.status==='evaluated_research').flatMap(([s,r])=>Object.entries(r.validation).map(([year,v])=><tr key={s+year}><td>{s==='DAY1_EOD'?'Day 1 close':'Day 2 close'}</td><td>{year}</td><td>{score(v.all.model)}</td><td>{score(v.all.gmp_fixed625)}</td></tr>))}</tbody></table></div></details>
    </details>}
    <details className="research-history"><summary>Strict 2021–2024 training → 2025–2026 test</summary>
      <p>All weights and preprocessing are fitted before 2025. The primary comparison uses {strict.combined.n} IPOs with confirmed Day-2 evidence; both matched refits train on 186 issuers.</p>
      <div className="table-wrap"><table><thead><tr><th>Model</th><th>2025</th><th>2026</th><th>Combined</th></tr></thead><tbody>{['p_legacy_original252','p_legacy_matched186','p_five_expert_strict2024','p_gmp_only','p_always_positive'].map(k=><tr key={k}><td>{names[k]}</td><td>{score(strict['2025'].direction[k])}</td><td>{score(strict['2026'].direction[k])}</td><td>{score(strict.combined.direction[k])}</td></tr>)}</tbody></table></div>
      <p>The original model was trained on final books; applying it to early books changes its input stage. The matched three-input refit controls for that difference. Neither matched architecture demonstrates an 80% direction result under this strict split.</p>
      <p>Original final-book diagnostic on all 189 original test IPOs: {score(data.original_final_book_combined)}. That is a different input stage and is not a pre-close accuracy claim.</p>
      <p className="fine-print">The full-table GMP-only model imputes missing quotes. On the same {strict.combined.dated_gmp_direction_comparison.n} IPOs with actual dated GMP: five-expert {score(strict.combined.dated_gmp_direction_comparison.models.p_five_expert_strict2024)}, GMP-only {score(strict.combined.dated_gmp_direction_comparison.models.p_gmp_only)}.</p>
      <p className="fine-print">Coverage: {data.test_coverage.original_n} original test IPOs; {data.test_coverage.matched_n} richer-input matches; {data.test_coverage.verified_day2_n} confirmed Day-2 cases. {data.test_coverage.no_richer_input_row} lack matched dated inputs and {data.test_coverage.unresolved_stage} have unresolved stages.</p>
    </details>
    <details className="research-history"><summary>Return estimates and ranking versus GMP</summary>
      <p>These fixed return regressors are research companions, separate from the deployed direction model. Smaller return error is better.</p>
      <div className="table-wrap"><table><thead><tr><th>Return model</th><th>Mean absolute error</th><th>Ranking correlation</th><th>Mean weekly selected listing return</th></tr></thead><tbody>{Object.entries(strict.combined.dated_gmp_return_comparison.models).map(([k,v])=><tr key={k}><td>{k==='r_five_view_ridge'?'Five-view return companion':k==='r_gmp_only'?'Dated GMP':'Three-input return companion'}</td><td>{v.mae_pp.toFixed(2)} percentage points</td><td>{v.spearman?.toFixed(3)??'—'}</td><td>{v.weekly_ranking.mean_selected_listing_return_pct.toFixed(2)}%</td></tr>)}</tbody></table></div>
      <p className="fine-print">Compared on the same {strict.combined.dated_gmp_return_comparison.n} IPOs with dated GMP. Weekly rankings exclude allotment and execution costs; the report is not realised portfolio profit.</p>
    </details>
    <p className="fine-print">Retrospective development comparison. These cohorts have been inspected before. No live weights or saved predictions were replaced.</p>
  </section>;
}
