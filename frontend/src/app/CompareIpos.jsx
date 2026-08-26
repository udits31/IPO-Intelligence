"use client";
import {useState} from 'react';
import {ArrowRight,Check,Layers3,LoaderCircle,Plus,Search,ShieldCheck} from 'lucide-react';
import {forecastView,applicationScreenView} from '../lib/forecast.mjs';

const rupees=v=>v==null?'—':`₹${Number(v).toLocaleString('en-IN',{maximumFractionDigits:0})}`;
const percent=v=>v==null?'—':`${Number(v).toFixed(1)}%`;
const stamp=v=>v?new Date(v).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short',hour:'numeric',minute:'2-digit'})+' IST':'No saved time';
const basis={saved_preclose:'Saved before closing',current_estimate:'Current estimate',historical_backtest:'Historical backtest',retrospective_estimate:'Reconstructed estimate',historical_reconstruction:'Reconstructed estimate'};

export default function CompareIpos({ipos,targets,setTargets,results,running,onRun,onResearch}){
 const [query,setQuery]=useState('');
 const candidates=ipos.filter(i=>['OPEN','UPCOMING','CLOSED'].includes(i.category)&&i.company_name.toLowerCase().includes(query.toLowerCase()));
 return <div className="page-stack narrow-page">
  <section className="page-intro"><div><span className="eyebrow">COMPARE OPPORTUNITIES</span><h1>A forecast is only the first check.</h1><p>Compare opening direction alongside the application screen: demand, offer structure and a retail payoff scenario after assumed costs.</p></div></section>
  <section className="panel compare-selection"><div className="section-heading"><div><h3>Build your shortlist</h3><p>Select two or three IPOs. Results use the same forecast and application policy as IPO research.</p></div><div className="small-search"><Search size={16}/><input aria-label="Find IPOs to compare" placeholder="Find an IPO" value={query} onChange={e=>setQuery(e.target.value)}/></div></div>
   <div className="compare-picker">{candidates.map(ipo=>{const selected=targets.some(t=>t.company_name===ipo.company_name);return <button key={ipo.url||ipo.company_name} aria-pressed={selected} disabled={running||!selected&&targets.length>=3} className={selected?'selected':''} onClick={()=>setTargets(p=>selected?p.filter(t=>t.company_name!==ipo.company_name):[...p,ipo])}><span>{selected?<Check size={14}/>:<Plus size={14}/>}</span><span>{ipo.company_name}<small>{ipo.category==='CLOSED'?'Bidding closed':ipo.category==='UPCOMING'?'Upcoming':'Open for bidding'}</small></span></button>;})}</div>
   {!candidates.length&&<p className="muted">No matching offers. Clear the search to see the current calendar.</p>}
   <div className="compare-run"><span>{targets.length} of 3 selected</span><button className="primary-button" disabled={targets.length<2||running} onClick={onRun}>{running?<LoaderCircle size={17} className="spin"/>:<Layers3 size={17}/>} {running?'Checking both layers…':'Compare selected'}</button></div>
  </section>
  {running&&<div className="loading-block" role="status"><LoaderCircle size={20} className="spin"/><span>Checking saved model calls and application screens…</span></div>}
  {!!results.length&&<><div className="compare-contract"><ShieldCheck size={18}/><span><strong>Two separate decisions.</strong> A positive model forecast can still be a Skip. The application screen includes the payoff scenario and structural checks.</span></div><section className="comparison-grid" aria-label="IPO comparison results">{results.map((item,index)=>{
   const d=item.scraped_data||{},v=forecastView(item),s=applicationScreenView(item);
   return <article className={`comparison-card compare-policy-${s.tone}`} key={d.company_name||index}>
    <span className="eyebrow">{d.category==='CLOSED'||d.category==='LISTED'?'BIDDING CLOSED':'APPLICATION SCREEN'}</span><h2>{d.company_name}</h2>
    {item.error?<p role="alert" className="text-negative">{item.error}</p>:<>
    <div className={`verdict verdict-${s.tone}`} data-testid="compare-application-screen">{s.historical?'Saved screen · ':''}{s.label}</div>
    <ul className="compare-reasons">{s.reasons.map((r,i)=><li key={i}>{r}</li>)}</ul>
    <div className="compare-direction"><span className="eyebrow">MODEL DIRECTION · SEPARATE FROM APPLY / SKIP</span><div><strong className={v?.probability>=.5?'text-positive':''}>{v?`${percent(v.probability*100)} score`:'Not scored'}</strong><span>{v?v.probability>=.5?'Positive opening':'Flat / negative opening':'No eligible forecast'}</span></div><small>{v?.modelLabel||'Forecast unavailable'} · {basis[v?.basis]||'Evidence pending'}</small></div>
    <dl className="compare-values"><div><dt>One retail lot</dt><dd>{rupees(s.lotCost)}</dd></div><div><dt>Net payoff scenario / application</dt><dd className={s.utility!=null&&s.utility<=0?'text-negative':''}>{rupees(s.utility)}</dd></div><div><dt>QIB demand</dt><dd>{v?.inputs.qib_subscription==null?'—':`${Number(v.inputs.qib_subscription).toFixed(2)}×`}</dd></div><div><dt>GMP in this forecast</dt><dd>{percent(v?.inputs.gmp_pct)}</dd></div></dl>
    <p className="fine-print">{s.historical?'This is a historical application screen; bidding has ended.':s.pending?'Application decision is pending evidence.':'Payoff is a scenario using approximate allotment odds and assumed costs; it is not realised profit.'}</p>
    <details className="compare-evidence"><summary>Model, inputs and policy</summary><p>{v?.modelLabel||'No forecast'}{v?.modelSha?` · ${v.modelSha.slice(0,12)}…`:''}</p><p>Bid evidence: {stamp(v?.asOf)}. Forecast saved: {stamp(v?.recordedAt||v?.forecast?.recorded_at_ist)}.</p><p>Application policy: {s.policy||'Not recorded with this call'}. {v?.warnings.length?`${v.warnings.length} model inputs use frozen missing-data handling.`:'See full research for individual sources.'}</p>{s.plan?.flags&&<ul>{Object.entries(s.plan.flags).map(([flag,value])=><li key={flag}>{flag.replaceAll('_',' ')}: {value===null?'unknown':value?'flagged':'passed'}</li>)}</ul>}<p>Raw model direction uses the fixed 50% cutoff. The application filter does not alter the model score or its accuracy.</p></details>
    <button className="card-action" onClick={()=>onResearch(d.company_name)}>Full research <ArrowRight size={16}/></button>
    </>}
   </article>;
  })}</section></>}
  {!results.length&&!running&&<div className="empty-state"><Layers3 size={24}/><h3>Choose the offers you want to compare</h3><p>See the direction forecast, application decision and reasons together.</p></div>}
 </div>;
}
