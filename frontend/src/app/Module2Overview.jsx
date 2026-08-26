"use client";
import {useState} from 'react';
import {ArrowDownRight,ArrowUpRight,ChevronDown,Clock3,ShieldCheck} from 'lucide-react';
import CallOutcomeChart from './CallOutcomeChart';
import IntradayMonitor from './IntradayMonitor';

const money=v=>v==null?'—':`₹${Number(v).toLocaleString('en-IN',{maximumFractionDigits:2})}`;
const date=v=>v?new Date(v).toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short',year:'numeric'}):'Date not verified';
const pct=v=>v==null?'—':`${v>0?'+':''}${Number(v).toFixed(1)}%`;
const reasons={issue_and_listing_floor:'The listing-day low has not been verified.',trend:'This check needs 20 completed sessions with sufficient price history.',relative_strength:'Matching dated stock and index history is needed.',drawdown:'A verified price peak is needed.',filings_and_flow:'Dated filing, delivery or unlock evidence is incomplete.'};

export default function Module2Overview({result,events=[]}){
 const [minutes,setMinutes]=useState(false);
 if(!result||result.loading)return null;
 const id=result.ipo_id,waterfall=result.price_waterfall||{},decision=result.decision||{};
 const verdict=result.verdict||decision.verdict||'';
 const defensive=/SELL|EXIT|REDUCE|TRIM|TAKE PROFIT/i.test(verdict);
 const adding=/BUY|ACCUMULATE/i.test(verdict);
 const watching=/WATCH|REVIEW/i.test(verdict);
 const usable=!!verdict&&!['DATA_STALE','PRICE_BASIS_UNVERIFIED','PRICE_UNAVAILABLE','DATA_UNAVAILABLE','DATA_IDENTITY_AMBIGUOUS','PRE_LISTING'].includes(result.status);
 const action=defensive?'Reduce / exit':adding?'Consider adding':watching?'Review position':'Hold existing position';
 const tone=defensive||watching?'defensive':adding?'adding':'holding';
 const current=result.current_price_adjusted??waterfall.current_price;
 const gain=waterfall.cumulative_gain_pct??waterfall.cumulative_return_pct;
 const drawdown=waterfall.drawdown_from_peak_pct??waterfall.peak_drawdown_pct;
 const why=result.decision_reason?.summary||result.key_reasons?.at(-1)||result.message||'The source did not provide a supported assessment.';
 const news=result.recent_news||result.qualitative_news||[];
 const playbook=result.actionable_playbook||decision.playbook||{};
 return <div className="m2-workspace">
  <section className="panel m2-assessment">
   <div className="m2-assessment-head"><div><span className="eyebrow">POST-LISTING MONITOR</span><h2>{result.company_name||'Stock analysis'}</h2><p>{result.trendlyne_data?.industry||result.sector||'Listed equity'}{result.listing_date?` · Listed ${date(result.listing_date)}`:''}</p></div><div className="m2-price"><span>Latest completed close</span><strong>{money(current)}</strong><small>{date(result.evaluation_date)}</small></div></div>
   {usable?<div className={`m2-action m2-action-${tone}`}><div className="m2-action-title">{defensive?<ArrowDownRight size={25}/>:adding?<ArrowUpRight size={25}/>:<ShieldCheck size={25}/>}<div><span>Current assessment</span><h3>{action}</h3></div></div><p>{why}</p></div>:<div className="m2-action m2-action-pending"><Clock3 size={22}/><div><h3>{result.status==='PRE_LISTING'?'Not listed yet':result.status==='DATA_IDENTITY_AMBIGUOUS'?'Choose the exact listing':'Current assessment paused'}</h3><p>{result.error||result.message||'A verified market price is required. Saved calls can still be reviewed below.'}</p></div></div>}
   <div className="m2-facts"><div><span>{waterfall.issue_price_is_share_adjusted?'Adjusted issue price':'Issue price'}</span><strong>{money(waterfall.issue_price)}</strong></div><div><span>Since issue</span><strong className={gain>0?'text-positive':gain<0?'text-negative':''}>{pct(gain)}</strong></div><div><span>From closing peak</span><strong>{pct(drawdown)}</strong></div></div>
   <p className="m2-method">Daily price and dated-news rules · {result.price_source||'Source verification pending'}. This assessment is separate from the learned model’s research forecasts.</p>
  </section>
  {id&&<CallOutcomeChart key={id} ipoId={id} evaluationDate={result.evaluation_date} compact/>}
  {!id&&<section className="panel m2-empty-history"><h3>No saved history linked to this selection</h3><p>{result.message||'Select an exact listed IPO from the search suggestions to link its recorded calls.'}</p></section>}
  <details className="panel m2-disclosure" onToggle={e=>setMinutes(e.currentTarget.open)}><summary><div><span className="eyebrow">OPTIONAL LIVE VIEW</span><strong>Intraday prices and alerts</strong><small>Minute candles are separate from the daily assessment above.</small></div><ChevronDown size={18}/></summary>{minutes&&<IntradayMonitor ipoId={id} company={result.company_name}/>}</details>
  <details className="panel m2-disclosure"><summary><div><span className="eyebrow">SUPPORTING EVIDENCE</span><strong>News, risk levels and data coverage</strong><small>Inspect the source details behind this assessment.</small></div><ChevronDown size={18}/></summary>
   <div className="m2-evidence-body">
    {playbook.stop_loss_basis&&<div className="m2-level"><span>Rule-based risk reference</span><strong>{playbook.stop_loss_basis}</strong><p>{result.decision_reason?.method||'A fixed risk reference, not a guaranteed fill or a validated return forecast.'}</p></div>}
    {result.decision_reason?.history_note&&<p className="fine-print">{result.decision_reason.history_note}</p>}
    {!!result.structural_diagnostics?.length&&<div className="m2-checks">{result.structural_diagnostics.map(check=><div key={check.key}><strong>{check.label}</strong><span>{check.passed===true?'Supported by available evidence':check.passed===false?'Needs attention':reasons[check.key]||'Additional verified evidence is needed.'}</span></div>)}</div>}
    {result.benchmark_source&&<p className="fine-print">Reference benchmark used: {result.benchmark_source}.</p>}
    <h3>Recent news</h3>{news.length?<div className="m2-news">{news.slice(0,4).map((item,i)=><a href={item.url||item.link} target="_blank" rel="noreferrer" key={i}><strong>{item.title}</strong><small>{item.publisher||item.source} · {item.published_date||item.published_at}</small><ArrowUpRight size={15}/></a>)}</div>:<p className="fine-print">No recent verified articles were returned. This does not mean that no news exists.</p>}
    <p className="fine-print">News changed this call: {result.news_changed_decision===true?'Yes':result.news_changed_decision===false?'No':'Not recorded'}.</p>
    {!!events.length&&<details><summary>Recorded assessment changes ({events.length})</summary><div className="m2-checks">{events.slice(0,8).map(item=><div key={item.id}><strong>{item.decision?.final_verdict||'Assessment recorded'}</strong><span>{date(item.observed_at_utc)} · price close {date(item.price_date)}</span></div>)}</div></details>}
   </div>
  </details>
 </div>;
}
