"use client";
import {useEffect,useMemo,useState} from 'react';
import {Check,Clock3,RefreshCw,X} from 'lucide-react';
import {api} from '../lib/api';
const money=v=>v==null?'—':`₹${Number(v).toLocaleString('en-IN',{maximumFractionDigits:2})}`;
const percent=v=>v==null?'—':`${Number(v)>0?'+':''}${Number(v).toFixed(2)}%`;
const day=v=>v?new Date(v).toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short'}):'Pending';
const stamp=v=>new Date(v).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short',hour:'numeric',minute:'2-digit'})+' IST';
const statusLabel=r=>r.matched===true?'Matched':r.matched===false?'Missed':r.status==='pending'?'Pending':'Evidence check';
const statusTone=r=>r.matched===true?'positive':r.matched===false?'negative':'warning';

export default function CallOutcomeChart({ipoId,evaluationDate,compact=false}){
 const [horizon,setHorizon]=useState(1),[data,setData]=useState(null),[error,setError]=useState(''),[selectedId,setSelectedId]=useState(null),[retry,setRetry]=useState(0),[expanded,setExpanded]=useState(false),[refreshing,setRefreshing]=useState(false);
 useEffect(()=>{
  if(!ipoId)return;
  let disposed=false;if(data?.ipo_id!==ipoId||data?.horizon_sessions!==horizon)setData(null);setError('');setRefreshing(true);
  api.module2CallOutcomes(ipoId,horizon).then(r=>{if(disposed)return;setData(r);setSelectedId(previous=>r.calls.some(c=>c.id===previous)?previous:(r.calls.find(c=>c.matched!=null)||r.calls[0])?.id??null);}).catch(e=>{if(!disposed)setError(e.message);}).finally(()=>{if(!disposed)setRefreshing(false);});
  return()=>{disposed=true;};
 },[ipoId,evaluationDate,horizon,retry]);
 useEffect(()=>{const timer=setInterval(()=>{if(!document.hidden)setRetry(n=>n+1);},60000);return()=>clearInterval(timer);},[]);
 const selected=data?.calls.find(r=>r.id===selectedId);
 const chart=useMemo(()=>{
  if(!data)return null;
  const start=data.calls.length?data.calls.reduce((a,b)=>a<b.reference_date?a:b.reference_date,data.calls[0].reference_date):data.prices.slice(-30)[0]?.date;
  const prices=data.prices.filter(p=>!start||p.date>=start);
  const dates=[...new Set([...prices.map(p=>p.date),...data.calls.map(c=>c.call_date)])].sort();
  const values=[...prices.map(p=>p.price),...data.calls.map(c=>c.reference_price)];
  if(!dates.length||!values.length)return null;
  const low=Math.min(...values),high=Math.max(...values),padding=Math.max((high-low)*.12,high*.005,1),min=low-padding,max=high+padding;
  const x=date=>70+(dates.indexOf(date)/Math.max(1,dates.length-1))*820;
  const y=price=>260-((price-min)/(max-min))*225;
  return {prices,dates,min,max,x,y,path:prices.map((p,i)=>`${i?'L':'M'}${x(p.date)},${y(p.price)}`).join(' ')};
 },[data]);
 if(!ipoId)return null;
 return <section className="panel call-outcome-panel" aria-label="Module 2 saved calls versus actual prices">
  <div className="section-heading"><div><span className="eyebrow">SAVED CALL → ACTUAL OUTCOME</span><h3>Our saved call vs the actual move</h3><p>Choose a call on the chart or in the table to cross-check it against later prices.</p></div><label className="perf-select">Check after<select aria-label="Call outcome horizon" value={horizon} onChange={e=>setHorizon(Number(e.target.value))}><option value={1}>Next completed market close</option><option value={5}>5 trading sessions</option><option value={30}>30 trading sessions</option></select></label></div>
  <div className="call-refresh-row"><p className="call-source-label">Recorded daily assessments · prices checked automatically</p><button className="icon-button" aria-label="Refresh call outcomes" disabled={refreshing} onClick={()=>setRetry(n=>n+1)}><RefreshCw size={16} className={refreshing?'spin':''}/></button></div>
  {error&&<p role="alert">Could not check the saved outcomes: {error} <button className="secondary-button" onClick={()=>setRetry(n=>n+1)}>Retry</button></p>}
  {!data&&!error&&<p role="status">Checking saved calls against dated closing prices…</p>}
  {data&&!data.calls.length&&<p className="no-news">No daily monitor call has been saved for this stock yet. New calls will appear here automatically; a call made today will not be presented as an earlier prediction.</p>}
  {data&&!!data.calls.length&&<>
   <div className="call-score-strip"><span><strong>{data.summary.matched} / {data.summary.evaluated}</strong> directions matched</span><span><strong>{data.summary.missed}</strong> missed</span><span><strong>{data.summary.saved-data.summary.evaluated}</strong> pending / evidence checks</span></div>
   {chart&&<><div className="call-chart-legend"><span><i className="call-line-key"/> Closing price</span><span><i className="call-dot-key"/> Saved call · select to inspect</span><span><i className="call-outcome-key"/> Selected outcome</span></div><div className="call-chart"><svg viewBox="0 0 960 310" role="img" aria-label={`Saved hold and sell calls compared with actual closing prices after ${horizon} trading sessions`}>
    {[0,.5,1].map(f=>{const value=chart.min+(chart.max-chart.min)*f,y=chart.y(value);return <g key={f}><line x1="70" x2="890" y1={y} y2={y} stroke="currentColor" opacity=".1"/><text x="60" y={y+4} textAnchor="end">{money(value)}</text></g>;})}
    <path d={chart.path} fill="none" stroke="var(--accent)" strokeWidth="2.5"/>
    {selected?.actual_price!=null&&<g><line x1={chart.x(selected.call_date)} y1={chart.y(selected.reference_price)} x2={chart.x(selected.outcome_date)} y2={chart.y(selected.actual_price)} stroke={selected.matched?'var(--green)':'var(--red)'} strokeDasharray="5 5" strokeWidth="1.5"/><path d={`M${chart.x(selected.outcome_date)},${chart.y(selected.actual_price)-7} l7,7 l-7,7 l-7,-7 Z`} fill={selected.matched?'var(--green)':'var(--red)'}/></g>}
    {data.calls.map(r=><g key={r.id} role="button" tabIndex="0" aria-label={`Inspect ${r.call}, saved ${stamp(r.called_at_ist)}, ${statusLabel(r)}`} onClick={()=>setSelectedId(r.id)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setSelectedId(r.id);}}}><title>{r.call} · {stamp(r.called_at_ist)} · {statusLabel(r)}</title><circle cx={chart.x(r.call_date)} cy={chart.y(r.reference_price)} r={selectedId===r.id?8:5} fill={r.expected_direction==='Flat / down'?'var(--amber)':'var(--accent)'} stroke="var(--surface)" strokeWidth="2"/></g>)}
    {[...new Set([0,Math.floor((chart.dates.length-1)/2),chart.dates.length-1])].map(i=><text x={chart.x(chart.dates[i])} y="293" textAnchor="middle" key={i}>{day(chart.dates[i])}</text>)}
   </svg></div></>}
   {selected&&<div className={`call-selected call-selected-${statusTone(selected)}`} aria-live="polite">
    <div><span className="eyebrow">OUR SAVED CALL</span><strong>{selected.call}</strong><small>{stamp(selected.called_at_ist)}</small><small>Close before call: {money(selected.reference_price)} · {day(selected.reference_date)}</small>{selected.stale_decision_input&&<small className="text-warning">The saved call used an older input: {money(selected.decision_reference_price)} · {day(selected.decision_reference_date)}. The result uses the latest preceding market close.</small>}</div>
    <div><span className="eyebrow">WHAT HAPPENED {horizon===1?'NEXT':`AFTER ${horizon} SESSIONS`}</span><strong>{selected.actual_direction?`${selected.actual_direction} · ${percent(selected.actual_return_pct)}`:'Outcome pending'}</strong><small>{selected.actual_price!=null?`${money(selected.actual_price)} close · ${day(selected.outcome_date)}`:selected.status==='pending'?`${selected.observed_future_sessions} of ${horizon} later session closes available`:'A price-source or share-basis check is required before grading.'}</small></div>
    <div className={`call-result text-${statusTone(selected)}`}>{selected.matched===true?<Check size={24}/>:selected.matched===false?<X size={24}/>:<Clock3 size={24}/>}<strong>{statusLabel(selected)}</strong><small>{selected.matched==null?'Not counted as correct or wrong':selected.expected_direction==='Flat / down'?selected.matched?'The defensive call matched a flat / falling price.':'Price rose after the defensive call.':selected.matched?'The hold / buy-side call matched a rising price.':'Price did not rise after the hold / buy-side call.'}</small></div>
   </div>}
   <details className="m2-call-table" open={compact?undefined:true}><summary>All {data.calls.length} saved calls and outcomes</summary><div className="perf-table-scroll"><table className="call-outcome-table"><thead><tr><th>Call saved</th><th>Our call</th><th>Close before call</th><th>Actual later close</th><th>Move</th><th>Result</th></tr></thead><tbody>{data.calls.slice(0,expanded?90:8).map(r=><tr key={r.id} className={selectedId===r.id?'call-row-selected':''}><td>{stamp(r.called_at_ist)}</td><td><button onClick={()=>setSelectedId(r.id)} aria-pressed={selectedId===r.id}>{r.call}</button></td><td>{money(r.reference_price)}<small>{day(r.reference_date)}</small></td><td>{r.actual_price==null?'Pending':money(r.actual_price)}<small>{r.outcome_date?day(r.outcome_date):''}</small></td><td className={r.actual_return_pct>0?'text-positive':r.actual_return_pct<0?'text-negative':''}>{percent(r.actual_return_pct)}</td><td className={`text-${statusTone(r)}`}>{statusLabel(r)}</td></tr>)}</tbody></table></div>
   {data.calls.length>8&&<button className="secondary-button" onClick={()=>setExpanded(v=>!v)}>{expanded?'Show fewer calls':`Show all ${data.calls.length} saved daily calls`}</button>}</details>
   <details className="perf-details"><summary>How the comparison is checked</summary><p>{data.method}</p><p>Price source: {data.price_source.source}. Verified source extends through {day(data.price_source.known_through)}. Call dots sit at the close known when the call was saved; they are not an intraday execution quote. The separate learned model’s research forecasts remain on the performance page.</p></details>
  </>}
 </section>;
}
