"use client";
import {useEffect,useState} from 'react';
import {api} from '../lib/api';
const pct=n=>`${Number(n).toFixed(2)}%`;
const names={execution_aligned_price:'Prices & market · shallow boosting',execution_aligned_flow:'Prices, market & volume features',execution_aligned_events:'Prices, volume & dated events'};
export default function NextSessionImprovement(){
 const [data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{api.nextSessionImprovement().then(setData).catch(e=>setError(e.message));},[]);
 if(error)return <p role="alert">Updated next-day research could not load: {error}</p>;
 if(!data?.comparisons)return null;
 const chosen=data.comparisons[data.selected.group],coverage=data.coverage.test;
 return <div className="perf-surface"><span className="eyebrow">UPDATED NEXT-DAY ML · DATED NEWS RECOVERED</span><h3>Better than the earlier model. Still behind the current rules.</h3>
  <p className="perf-caption">Same {data.splits.test.rows.toLocaleString()} decisions across {data.splits.test.issuers} IPOs in 2025–2026. All policies execute at the next open, carry positions overnight and pay costs on actual buys and sells.</p>
  <div className="perf-table-scroll"><table><thead><tr><th>Policy</th><th>Direction accuracy</th><th>Mean net return per IPO window</th><th>Mean drawdown</th><th>Buy fills</th></tr></thead><tbody>
   <tr><td>Current price & filing rules</td><td>—</td><td>{pct(data.baselines.current_rules.mean_net_return_pct)}</td><td>{pct(data.baselines.current_rules.mean_max_drawdown_pct)}</td><td>{data.baselines.current_rules.buys}</td></tr>
   {Object.entries(data.comparisons).map(([k,s])=><tr key={k}><td>{names[k]}{k===data.selected.group?' · selected on 2024':''}</td><td>{pct(s.direction_accuracy_pct)}</td><td>{pct(s.strategy.mean_net_return_pct)}</td><td>{pct(s.strategy.mean_max_drawdown_pct)}</td><td>{s.strategy.buys}</td></tr>)}
   <tr><td>Buy and hold</td><td>—</td><td>{pct(data.baselines.always_hold.mean_net_return_pct)}</td><td>{pct(data.baselines.always_hold.mean_max_drawdown_pct)}</td><td>{data.baselines.always_hold.buys}</td></tr>
  </tbody></table></div>
  <p className="perf-caption">Headline coverage increased from 180 to {coverage.news_rows.toLocaleString()} decisions ({pct(100*coverage.news_rows/coverage.rows)}). Dated filing scores cover {coverage.filing_rows.toLocaleString()} decisions. The retrained policy trades more often and misses parts of strong rallies. It remains a research challenger.</p>
  <details className="perf-details"><summary>Selection, uncertainty and costs</summary>
   <p>Fit on {data.splits.final_train.issuers} IPOs from 2021–2024. Candidate and entry/exit settings selected on 2024 before the later comparison. Terminal next-open labels without an observed price are excluded from fitting; test decisions remain in every policy comparison.</p>
   <p>Selected model gain versus rules: {pct(chosen.vs_rules.mean_gain_pct)}, with an issuer-bootstrap 95% interval of {pct(chosen.vs_rules.bootstrap95_pct[0])} to {pct(chosen.vs_rules.bootstrap95_pct[1])}. Majority-direction baseline: {pct(data.majority_direction_accuracy_pct)}.</p>
   <p>Costs are 0.175% on every buy and every sell, including terminal liquidation. Returns are per available IPO window, not annualised. News search coverage is incomplete; absent evidence is never treated as clean news. These later years have been inspected during development.</p>
   <table><thead><tr><th>Round-trip cost</th><th>Selected model net return</th></tr></thead><tbody>{Object.entries(chosen.cost_sensitivity).map(([cost,s])=><tr key={cost}><td>{cost} basis points</td><td>{pct(s.mean_net_return_pct)}</td></tr>)}</tbody></table>
  </details>
 </div>;
}
