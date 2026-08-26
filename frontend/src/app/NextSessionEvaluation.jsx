"use client";
import {useEffect,useState} from 'react';
import {api} from '../lib/api';
import NextSessionImprovement from './NextSessionImprovement';
import HoldingPolicyEvaluation from './HoldingPolicyEvaluation';
import MultiHorizonEvaluation from './MultiHorizonEvaluation';
const p=n=>`${Number(n).toFixed(2)}%`;
const names={price:'Prices',price_volume_market:'Prices, volume & market',price_volume_market_news:'Prices, volume, market & news'};
export default function NextSessionEvaluation(){
 const [data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{api.nextSessionEvaluation().then(setData).catch(e=>setError(e.message));},[]);
 if(error)return <p role="alert">Next-session experiment could not load: {error}</p>;
 if(!data||!data.comparisons)return null;
 const selected=data.comparisons[data.selected.group],continuous=selected.continuous_transition_companion;
 return <><MultiHorizonEvaluation/><details className="perf-details"><summary>Earlier holding-policy comparison</summary><HoldingPolicyEvaluation/></details><details className="perf-details"><summary>Dated-news recovery and model retraining</summary><NextSessionImprovement/></details><details className="perf-details"><summary>Earlier next-session experiment</summary><div className="perf-surface perf-experiment"><span className="eyebrow">NEXT-SESSION ML · RESEARCH</span><h3>Can we forecast the next trading day?</h3>
  <p className="perf-caption">Trained through 2024; settings selected on 2024. Tested on {selected.n.toLocaleString()} daily decisions from {selected.issuers} different IPOs in 2025–2026. Sessions 21–61 after listing; these are correlated daily observations.</p>
  <div className="perf-table-scroll"><table><thead><tr><th>Inputs</th><th>Next-day direction</th><th>Average return error</th><th>Buy trades</th><th>Net per trade</th></tr></thead><tbody>{Object.entries(data.comparisons).map(([k,s])=><tr key={k}><td>{names[k]}{k===data.selected.group?' · selected on 2024':''}</td><td>{p(s.accuracy_pct)}</td><td>{p(s.forecast_mae_pct)}</td><td>{s.costs['35'].trades}</td><td>{p(s.costs['35'].mean_net_return_per_trade_pct)}</td></tr>)}</tbody></table></div>
  <p className="perf-caption">Always flat/negative: {p(selected.always_flat_negative_accuracy_pct)}. A forecast of zero return has {p(selected.zero_return_mae_pct)} average error. Daily trades enter at the next open and exit at the same close, with 0.35% total costs.</p>
  {continuous&&<details className="perf-details"><summary>Compare actual holdings and transaction costs</summary><p>HOLD carries shares overnight. Costs apply only to entries, exits and final liquidation. Mean return per issuer over its available test window:</p><table><thead><tr><th>Policy</th><th>Net window return</th><th>Buy / sell fills</th></tr></thead><tbody>{Object.entries(continuous).map(([k,s])=><tr key={k}><td>{k.replaceAll('_',' ')}</td><td>{p(s.mean_issuer_net_window_return_pct)}</td><td>{s.buys} / {s.sells}</td></tr>)}</tbody></table></details>}
  <p className="perf-caption">The selected model has not established a profitable trading advantage and is not active in the live monitor. Published headlines cover {selected.news_rows} test decisions; dated filing scores cover {selected.filing_rows}. Missing evidence is kept missing. Test years were inspected in earlier research.</p>
 </div></details></>;
}
