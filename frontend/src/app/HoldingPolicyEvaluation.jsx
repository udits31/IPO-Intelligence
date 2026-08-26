"use client";
import {useEffect,useState} from 'react';
import {api} from '../lib/api';
const p=n=>`${Number(n).toFixed(2)}%`;
const names={tuned:'Earlier-selected holding policy',original_ml:'Previous ML entry / exit policy',current_rules:'Current price & filing rules',always_hold:'Buy and hold'};
export default function HoldingPolicyEvaluation(){
 const [data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{api.holdingPolicyEvaluation().then(setData).catch(e=>setError(e.message));},[]);
 if(error)return <p role="alert">Holding policy evaluation could not load: {error}</p>;
 if(!data?.comparison)return null;
 const s=data.selected_policy,c=data.comparison,gain=data.paired_comparisons.current_rules;
 return <div className="perf-surface"><span className="eyebrow">HOLDING & EXIT STABILITY · EARLIER-ONLY SELECTION</span>
 <h3>{c.tuned.mean_net_return_pct>c.current_rules.mean_net_return_pct?'Fewer trades. Higher retrospective net returns.':'Fewer trades. Current rules still lead.'}</h3>
 <p className="perf-caption">Identical {data.n_decisions.toLocaleString()} decisions across {data.n_issuers} IPOs. Model weights stay fixed; 2023 and 2024 determine the holding policy. Forecast accuracy is unchanged by these execution rules.</p>
 <div className="perf-table-scroll"><table><thead><tr><th>Policy</th><th>Net return per IPO window</th><th>Mean drawdown</th><th>Buys</th><th>Sells, including final exit</th></tr></thead><tbody>{Object.entries(c).map(([k,r])=><tr key={k}><td>{names[k]||k}</td><td>{p(r.mean_net_return_pct)}</td><td>{p(r.mean_max_drawdown_pct)}</td><td>{r.buys}</td><td>{r.sells}</td></tr>)}</tbody></table></div>
 <p className="perf-caption">{s.rule_led_entry?'Entry follows the current risk rules. ':''}Exits use a {s.ema_span}-session exponential smoothing span, {s.exit_confirmation} consecutive weak readings, a minimum {s.min_hold} completed holding sessions and a {s.cooldown}-session re-entry cooldown. {s.require_price_deterioration?'An additional ML exit requires the price below SMA-20 and negative five-session momentum. ':''}Existing risk exits always take priority.</p>
 <details className="perf-details"><summary>Earlier-period checks, uncertainty and costs</summary>
 <p>Cleared both earlier return/drawdown/churn gates: {data.qualified_on_both_earlier_folds?'Yes':'No'}. Later paired gain versus rules: {p(gain.mean_gain_pct)}; issuer-bootstrap 95% interval {p(gain.bootstrap95_pct[0])} to {p(gain.bootstrap95_pct[1])}. Live policy replaced: {data.live_promoted?'Yes':'No'}.</p>
 <table><thead><tr><th>Period</th><th>Tuned policy net return</th><th>Current rules</th></tr></thead><tbody>{Object.entries(data.by_year).map(([year,r])=><tr key={year}><td>{year}</td><td>{p(r.tuned.mean_net_return_pct)}</td><td>{p(r.current_rules.mean_net_return_pct)}</td></tr>)}</tbody></table>
 <p>{data.execution} {data.interpretation}</p>
 <table><thead><tr><th>Round-trip costs</th><th>Tuned policy</th><th>Current rules</th></tr></thead><tbody>{Object.entries(data.costs).map(([cost,r])=><tr key={cost}><td>{cost} bps</td><td>{p(r.tuned.mean_net_return_pct)}</td><td>{p(r.current_rules.mean_net_return_pct)}</td></tr>)}</tbody></table>
 </details></div>;
}
