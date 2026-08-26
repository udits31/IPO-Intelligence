"use client";
import {useEffect,useState,useRef} from 'react';
import {api} from '../lib/api';
const money=n=>n==null?'—':`₹${Number(n).toLocaleString('en-IN',{maximumFractionDigits:2})}`;
const clock=s=>s?new Date(s).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',dateStyle:'medium',timeStyle:'short'})+' IST':'—';
const signals={REVIEW_EXIT:'Review exit · support broken',VWAP_LOST:'Price lost VWAP support',VWAP_RECLAIMED:'Price recovered VWAP support',ABOVE_VWAP:'Trading above VWAP',BELOW_VWAP:'Trading below VWAP',DATA_PENDING:'Waiting for minute prices'};
function MinuteChart({candles=[],company}){
 const chartRef=useRef(null),[width,setWidth]=useState(800);
 useEffect(()=>{if(!chartRef.current)return;const observer=new ResizeObserver(([entry])=>setWidth(Math.max(300,entry.contentRect.width)));observer.observe(chartRef.current);return()=>observer.disconnect();},[candles.length>1]);
 if(candles.length<2)return <p className="fine-print">The chart will appear once two completed minute candles are available.</p>;
 let volume=0,notional=0;
 const points=candles.map(r=>{volume+=r.volume;notional+=(r.high+r.low+r.close)/3*r.volume;return {...r,t:Date.parse(r.bar_start_utc),vwap:volume>0?notional/volume:null};});
 const values=points.flatMap(r=>[r.close,r.vwap]).filter(Number.isFinite),lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.08,1);
 const min=lo-pad,max=hi+pad,x=t=>62+(t-points[0].t)/(points.at(-1).t-points[0].t)*(width-82),y=p=>18+(max-p)/(max-min)*175;
 const path=key=>points.map((r,i)=>Number.isFinite(r[key])?`${i===0||!Number.isFinite(points[i-1][key])||r.t-points[i-1].t>90000?'M':'L'}${x(r.t).toFixed(1)},${y(r[key]).toFixed(1)}`:'').join(' ');
 const label=t=>new Date(t).toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit'});
 return <div ref={chartRef} style={{marginTop:24}}><div className="fine-print" style={{display:'flex',gap:20}}><span style={{color:'#96a8ff'}}>● Minute close</span><span style={{color:'#e8b35a'}}>● Bar-based VWAP</span></div><svg viewBox={`0 0 ${width} 225`} role="img" aria-label={`${company} intraday price and VWAP chart, ${points.length} completed minutes`} style={{display:'block',width:'100%',height:225}}>{[min,(min+max)/2,max].map(v=><g key={v}><line x1="62" x2={width-20} y1={y(v)} y2={y(v)} stroke="var(--line)"/><text x="54" y={y(v)+4} textAnchor="end" fill="var(--muted)" fontSize="11">{money(v)}</text></g>)}<path d={path('close')} fill="none" stroke="#96a8ff" strokeWidth="2.3"/><path d={path('vwap')} fill="none" stroke="#e8b35a" strokeWidth="1.6" strokeDasharray="5 3"/><text x="62" y="216" fill="var(--muted)" fontSize="11">{label(points[0].t)}</text><text x={width-20} y="216" textAnchor="end" fill="var(--muted)" fontSize="11">{label(points.at(-1).t)} IST</text></svg></div>;
}
export default function IntradayMonitor({ipoId,company}){
 const [data,setData]=useState(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);
 useEffect(()=>{
   if(!ipoId&&!company)return;
   let live=true,inflight=false;
   setData(null);setError('');
   const refresh=async()=>{
     if(inflight)return;inflight=true;
     try{const d=await api.module2Intraday(ipoId,true,company);if(live){setData(d);setError('');}}
     catch(e){if(live)setError(e.message);}finally{inflight=false;}
   };
   refresh();const timer=setInterval(refresh,60000);
   return()=>{live=false;clearInterval(timer);};
 },[ipoId,company]);
 if(!ipoId&&!company)return null;
 const usable=data?.status==='FRESH'&&!error,quote=data?.latest_price!=null?data:data?.last_successful_observation;
 const status=error?'Price refresh failed':({FRESH:'Recent minute data · indicative',DELAYED:'Delayed price data',MARKET_CLOSED:'Latest completed session',SOURCE_UNAVAILABLE:'Minute feed unavailable',NOT_COLLECTED:'Starting collection',IDENTITY_UNAVAILABLE:'Market symbol needs verification',UNAVAILABLE:'Waiting for today’s candles'}[data?.status]||'Connecting to minute feed');
 return <section className="panel intraday-monitor" aria-label="Intraday monitor">
   <div className="section-heading"><div><span className="eyebrow">INTRADAY MONITOR · 1-MINUTE CANDLES</span><h3>{usable?signals[data.signal]||'Minute price update':status}</h3></div><button className="secondary-button" disabled={busy} onClick={async()=>{setBusy(true);try{setData(await api.module2Intraday(ipoId,true,company));setError('');}catch(e){setError(e.message);}finally{setBusy(false);}}}>{busy?'Checking…':'Refresh'}</button></div>
   <p>{status}. {usable?data.reason:data?.reason||'Fetching completed minute candles.'}</p>
   {error&&<p role="alert">Update failed: {error}. Previously shown prices may be stale.</p>}
   <div className="return-metrics"><div><span>Last completed minute</span><strong>{money(quote?.latest_price)}</strong><small>{clock(quote?.bar_end_utc)}</small></div><div><span>Bar-based VWAP</span><strong>{money(quote?.bar_based_vwap)}</strong><small>Typical-price × volume approximation</small></div><div><span>Opening 15-minute range</span><strong>{quote?.opening_range_complete?`${money(quote.opening_range_low)} – ${money(quote.opening_range_high)}`:'Not confirmed'}</strong></div><div><span>Session range</span><strong>{quote?`${money(quote.session_low)} – ${money(quote.session_high)}`:'—'}</strong></div></div>
   <MinuteChart candles={data?.candles||[]} company={company||data?.company_name||'Stock'} />
   {data?.chart_is_historical&&<p className="fine-print">Saved session: {data.chart_session_date}. The market is closed; these prices and alerts remain available for review.</p>}
   <p className="fine-print">Yahoo Finance · indicative feed, not a guaranteed real-time quote. Updates every minute while this panel is open; recent IPOs are also collected by the backend. Intraday alerts are separate from the daily hold/sell call and its historical accuracy. No automatic orders; an alert cannot ensure an exit at a circuit limit.</p>
   <p className="fine-print">Last-minute volume: {quote?.latest_minute_volume?.toLocaleString('en-IN')??'—'} shares · {quote?.volume_vs_previous_20_minute_median!=null?`${quote.volume_vs_previous_20_minute_median.toFixed(2)}× the preceding 20-minute median`:'Volume comparison needs 20 preceding minutes'}. Exchange circuit bands are not supplied by this feed.</p>
   <details><summary>Saved intraday alerts ({data?.alerts?.length||0} recent)</summary>{data?.alerts?.length?<ul>{data.alerts.map(a=><li key={`${a.bar_start_utc}-${a.signal}`}><strong>{signals[a.signal]||a.signal}</strong> · {clock(a.observed_at_utc)} · observed {money(a.latest_price)}<p>{a.reason}</p></li>)}</ul>:<p>No eligible intraday alert has been saved for this IPO yet.</p>}</details>
 </section>;
}
