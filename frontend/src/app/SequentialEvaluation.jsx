"use client";
import {useEffect,useState} from 'react';
import {api} from '../lib/api';
const pct=n=>n==null?'—':`${Number(n).toFixed(3)}%`;
const names={M2_R0_CONTINUOUS:'Original rules · daily exits',M2_R1_CONTINUOUS:'Filing-aware monitor · daily exits',M2_R1_STATIC_31:'Single checkpoint call',ALWAYS_HOLD_31_TO_61:'Always hold',ALWAYS_SELL_AT_31:'Always exit at checkpoint'};
export default function SequentialEvaluation(){
 const [data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{api.sequentialEvaluation().then(setData).catch(e=>setError(e.message));},[]);
 if(error)return <section className="panel">{error}</section>;
 if(!data?.windows)return <section className="panel">Daily-path evaluation is being built.</section>;
 const w=data.windows.LOCKED_EXECUTION_ALIGNED,f=w.held_at_31_fates;
 return <section className="panel"><span className="eyebrow">MODULE 2 · DAILY EXECUTION REPLAY</span><h2>Did later warnings improve the exit?</h2><p>The monitor is checked every session. A sell call executes at the next observed close, then stays in cash.</p><div className="table-wrap"><table><thead><tr><th>Policy</th><th>Mean net return</th><th>Median</th><th>Worst-decile drawdown</th></tr></thead><tbody>{Object.entries(w.policies).filter(([k])=>!k.includes('ALL_SUCCESSIVE')).map(([name,p])=><tr key={name}><th>{names[name]||name}</th><td>{pct(p.B.mean_net_return_pct)}</td><td>{pct(p.B.median_net_return_pct)}</td><td>{pct(p.B.worst_decile_mean_max_drawdown_pct)}</td></tr>)}</tbody></table></div><p>Of the {f.held_eventual_losers.n} eventual losers initially held, {f.held_eventual_losers.midwindow_exits} received a later exit. Of the {f.held_eventual_winners.n} initially held winners, {f.held_eventual_winners.midwindow_exits} also received a later exit.</p><p className="fine-print">Every row includes the same 0.30% terminal liquidation cost. The locked static result remains {data.locked_static_reference?.correct}/132 and {pct(data.locked_static_reference?.mean_net_return_A_pct)} under its original cost convention.</p><details><summary>Timing and cost conventions</summary><p>{data.index_note}</p><p>The table uses consistent terminal liquidation costs. Both cost conventions, the literal Session-31-to-61 window and the optional re-entry strategy are in the exported report.</p><p>Daily decisions and trade executions are retrospective. They do not establish a prospective trading edge.</p></details></section>;
}
