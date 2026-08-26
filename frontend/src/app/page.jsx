"use client";

import React, { useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  Activity, AlertTriangle, ArrowRight, ArrowUpRight, BarChart3, Bell,
  BriefcaseBusiness, CalendarDays, Check, Clock3, Eye, FileText, Gauge,
  Layers3, LoaderCircle, Menu, MessageSquare, Plus, RefreshCw, Search, ShieldCheck,
  Target, Trash2, TrendingUp, X,
} from 'lucide-react';
import { api } from '../lib/api';
import { assistantContext, forecastView } from '../lib/forecast.mjs';
import { holdingValue } from '../lib/holding.mjs';
import { marketStatus } from '../lib/marketStatus.mjs';
import { ReturnEstimate, PriorityBoard, ReturnBenchmark } from './ReturnResearch';
const PerformanceReview = dynamic(() => import('./PerformanceReview'));
const PredictionResults = dynamic(() => import('./PredictionResults'));
const PerformanceDashboard = dynamic(() => import('./PerformanceDashboard'), {loading: () => <LoadingBlock label="Loading model evidence…"/>});
const CompareIpos = dynamic(() => import('./CompareIpos'));
const Module2Overview = dynamic(() => import('./Module2Overview'));

const nav = [
  { id: 'today', label: 'Market today', icon: Activity },
  { id: 'research', label: 'Module 1 · IPO research', icon: Search },
  { id: 'performance', label: 'Model performance', icon: BarChart3 },
  { id: 'compare', label: 'Compare', icon: Layers3 },
  { id: 'portfolio', label: 'Module 2 · Hold / Sell', icon: BriefcaseBusiness },
];
const money = (v, d = 0) => v == null || Number.isNaN(Number(v)) ? '—' : `₹${Number(v).toLocaleString('en-IN', { maximumFractionDigits: d })}`;
const pct = (v, signed = false) => v == null || Number.isNaN(Number(v)) ? '—' : `${signed && Number(v) > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;
const num = (v, d = 1) => v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d);
const bid = v => v == null || Number.isNaN(Number(v)) ? '—' : `${Number(v).toFixed(Number(v) > 0 && Number(v) < 0.01 ? 4 : 2)}x`;
const intradayBand = p => p == null || Number.isNaN(Number(p)) ? null : Number(p) >= 0.6 ? 'HIGH' : Number(p) <= 0.4 ? 'LOW' : 'MIXED';
const historicalSourceName = source => ({ 'economictimes.indiatimes.com': 'Economic Times',
  'moneycontrol.com': 'Moneycontrol', 'theipolist.in': 'The IPO List',
  'niftytrader.in': 'NiftyTrader', 'ipomilega.in': 'IPO Milega', 'choiceindia.com': 'Choice',
  '5paisa.com': '5paisa', 'rajkotcityguide.com': 'Rajkot City Guide' }[source.publisher] || source.publisher);
const historicalSourceCount = row => `${row.sources.length} ${row.sources.length > 1 ? 'matching sources' : row.source_reported_at_ist ? 'timestamped source' : 'historical source'}`;
const compact = (v) => v == null ? '—' : `₹${Number(v).toLocaleString('en-IN')} Cr`;
const cleanCompany = (name = '') => name.replace(/\s+(Limited|Ltd\.)$/i, '');
const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const formatDateParts = ({ year, month, day }, includeYear = true) => `${day} ${monthNames[month - 1]}${includeYear ? ` ${year}` : ''}`;
const parseIsoDate = (value = '') => {
  const match = String(value).trim().match(/^(\d{4})-(\d{2})-(\d{2})$/);
  return match ? { year: Number(match[1]), month: Number(match[2]), day: Number(match[3]) } : null;
};
const formatDate = (value) => {
  const parsed = parseIsoDate(value);
  return parsed ? formatDateParts(parsed) : (value || '—');
};
const formatIpoTimestamp = (value) => {
  if (!value) return 'time unknown';
  const local = String(value).match(/^(\d{1,2})-(\d{1,2})-(\d{4})\s+(\d{1,2}):(\d{2})/);
  if (local) {
    const hour = Number(local[4]);
    return `${Number(local[1])} ${monthNames[Number(local[2]) - 1]} ${local[3]}, ${hour % 12 || 12}:${local[5]} ${hour >= 12 ? 'pm' : 'am'} IST`;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) :
    `${parsed.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', dateStyle: 'medium', timeStyle: 'short' })} IST`;
};
const formatIpoDate = (value) => {
  if (!value) return 'Dates pending';
  const raw = String(value).trim();
  const isoRange = raw.match(/^(\d{4}-\d{2}-\d{2})\s+(?:to|–|—|-)\s+(\d{4}-\d{2}-\d{2})$/i);
  if (isoRange) {
    const start = parseIsoDate(isoRange[1]); const end = parseIsoDate(isoRange[2]);
    if (start.year === end.year && start.month === end.month) return `${start.day}–${end.day} ${monthNames[end.month - 1]} ${end.year}`;
    if (start.year === end.year) return `${start.day} ${monthNames[start.month - 1]}–${end.day} ${monthNames[end.month - 1]} ${end.year}`;
    return `${formatDateParts(start)}–${formatDateParts(end)}`;
  }
  const singleIso = parseIsoDate(raw);
  if (singleIso) return formatDateParts(singleIso);
  const shortRange = raw.match(/^(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+([A-Za-z]{3})(?:\s+(\d{4}))?$/);
  if (shortRange) return `${Number(shortRange[1])}–${Number(shortRange[2])} ${shortRange[3]} ${shortRange[4] || new Date().getFullYear()}`;
  const listed = raw.match(/^Listed\s*\(([A-Za-z]{3})\s+(\d{1,2}),\s*(\d{4})\)$/i);
  if (listed) return `${Number(listed[2])} ${listed[1]} ${listed[3]}`;
  const sameMonthBidding = raw.match(/^Active Bidding\s*\(([A-Za-z]{3})\s+(\d{1,2})\s*[-–—]\s*(\d{1,2}),\s*(\d{4})\)$/i);
  if (sameMonthBidding) return `${Number(sameMonthBidding[2])}–${Number(sameMonthBidding[3])} ${sameMonthBidding[1]} ${sameMonthBidding[4]}`;
  const crossMonth = raw.match(/^Active Bidding\s*\(([A-Za-z]{3})\s+(\d{1,2})\s*[-–—]\s*([A-Za-z]{3})\s+(\d{1,2}),\s*(\d{4})\)$/i);
  if (crossMonth) return `${Number(crossMonth[2])} ${crossMonth[1]}–${Number(crossMonth[4])} ${crossMonth[3]} ${crossMonth[5]}`;
  return raw.replace(/\s+-\s+/g, '–');
};
const toneForVerdict = (v = '') => {
  const text = v.toUpperCase();
  if (text.includes('BIDDING IN PROGRESS') || text.includes('WAIT FOR DATA')) return 'info';
  if (text.includes('FAVORABLE') || text.includes('APPLY') || text.includes('BUY') || text.includes('ACCUMULATE')) return 'positive';
  if (text.includes('HIGH-RISK') || text.includes('ELEVATED RISK') || text.includes('SKIP') || text.includes('SELL') || text.includes('EXIT')) return 'negative';
  return 'warning';
};

function Pill({ children, tone = 'neutral', dot = false }) {
  return <span className={`pill pill-${tone}`}>{dot && <span className="pill-dot" />}{children}</span>;
}
function Metric({ label, value, note, tone = 'default' }) {
  return <div className="metric"><span className="metric-label">{label}</span><strong className={`metric-value metric-${tone}`}>{value}</strong>{note && <span className="metric-note">{note}</span>}</div>;
}
function LoadingBlock({ label = 'Loading market intelligence…' }) {
  return <div className="loading-block"><LoaderCircle className="spin" size={20} /><span>{label}</span></div>;
}
function EmptyState({ icon: Icon = Search, title, copy, action }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={22} /></div><h3>{title}</h3><p>{copy}</p>{action}</div>;
}
function Freshness({ health, macro, polling = false }) {
  const fallback = macro?.is_fallback || macro?.source === 'fallback';
  const updated = macro?.timestamp;
  const checked = fallback ? macro?.requested_at : macro?.retrieved_at_ist;
  const checkedLabel = checked && Number.isFinite(Date.parse(checked)) ? new Intl.DateTimeFormat('en-IN', {timeZone:'Asia/Kolkata',day:'numeric',month:'short',hour:'numeric',minute:'2-digit',second:'2-digit',hour12:true}).format(new Date(checked)) + ' IST' : null;
  const [checkedNow,setCheckedNow]=useState(false);
  const pollState=useRef({running:false,startedWith:null});
  useEffect(()=>{
    if(polling&&!pollState.current.running)pollState.current.startedWith=checked;
    const completed=!polling&&pollState.current.running;
    pollState.current.running=polling;
    if(completed&&!fallback&&checked&&checked!==pollState.current.startedWith){
      setCheckedNow(true);const timer=setTimeout(()=>setCheckedNow(false),4000);return()=>clearTimeout(timer);
    }
  },[polling,checked,fallback]);
  const status=marketStatus(macro);
  return <div title={`Nifty 50 and India VIX. ${checkedLabel?`${fallback?'Last attempted':'Last checked'}: ${checkedLabel}. `:''}${updated?`Source reading: ${formatIpoTimestamp(updated)}.`:''}`} className={`freshness ${status.warning?'freshness-warning':''} ${status.closed?'freshness-closed':''}`}><span className="live-dot" /><div><strong>{polling?'Checking NSE…':status.title}</strong><span role="status">{checkedNow?'Checked just now':status.detail}</span></div></div>;
}

function SearchDropdown({ value, onChange, items, onSelect, placeholder, listedOnly = false }) {
  const [open, setOpen] = useState(false);
  const suggestions = useMemo(() => {
    const query = value.trim().toLowerCase();
    return items.filter(item => {
      if (listedOnly && item.category && item.category !== 'LISTED') return false;
      const haystack = `${item.company_name || ''} ${item.nse_symbol || ''} ${item.bse_scrip_code || item.bse_code || ''} ${item.sector || ''}`.toLowerCase();
      return !query || haystack.includes(query);
    }).slice(0, 8);
  }, [items, value, listedOnly]);
  return <div className="suggest-wrap">
    <input value={value} onChange={e => { onChange(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 120)} placeholder={placeholder} autoComplete="off" aria-label={placeholder} />
    {open && <div className="suggest-menu">
      <div className="suggest-caption">Select a verified record</div>
      {suggestions.length ? suggestions.map((item, index) => <button type="button" key={`${item.id || item.url || item.company_name}-${index}`} onMouseDown={e => e.preventDefault()} onClick={() => { onSelect(item); setOpen(false); }}>
        <span className="suggest-logo">{(item.company_name || '?').charAt(0)}</span>
        <span className="suggest-copy"><strong>{item.company_name}</strong><small>{item.nse_symbol ? `NSE: ${item.nse_symbol}` : item.category || 'IPO'} · {item.sector || 'Sector pending'}</small></span>
        <span className="suggest-meta"><strong>{money(item.issue_price)}</strong><small>{formatIpoDate(item.listing_date || item.date_range)}</small></span>
      </button>) : <div className="suggest-empty">No matching verified record. You can still paste a source URL.</div>}
    </div>}
  </div>;
}

function IpoCard({ ipo, onResearch, watching, onWatch }) {
  const category = ipo.category || ipo.status || 'UPCOMING';
  const statusTone = category === 'OPEN' ? 'positive' : category === 'CLOSED' ? 'warning' : category === 'LISTED' ? 'blue' : 'neutral';
  const gmp = ipo.gmp_pct == null || ipo.gmp_pct === '' ? null : Number(ipo.gmp_pct);
  const target = Number.isFinite(gmp) && ipo.issue_price ? Number(ipo.issue_price) * (1 + gmp / 100) : null;
  return <article className="ipo-card">
    <div className="ipo-card-top"><Pill tone={statusTone} dot={category === 'OPEN'}>{category}</Pill><button className={`icon-button ${watching ? 'active' : ''}`} onClick={() => onWatch(ipo)} aria-label="Toggle watchlist"><Eye size={17} /></button></div>
    <div><h3>{cleanCompany(ipo.company_name)}</h3><p className="muted">{ipo.sector || 'Sector unavailable'} · {formatIpoDate(ipo.date_range || ipo.listing_date || ipo.status)}</p></div>
    <div className="ipo-stat-row"><div><span>Price</span><strong>{money(ipo.issue_price)}</strong></div><div><span>Issue</span><strong>{compact(ipo.issue_size_cr)}</strong></div><div><span>GMP</span><strong className={gmp > 0 ? 'text-positive' : ''}>{Number.isFinite(gmp) ? `${pct(gmp, true)}${ipo.gmp_rs ? ` (₹${ipo.gmp_rs})` : ''}` : 'Unavailable'}</strong></div></div>
    {target && <div className="estimate-line"><span>GMP-implied price</span><strong>{money(target)}</strong></div>}
    <button className="card-action" onClick={() => onResearch(ipo)}>Open research <ArrowRight size={16} /></button>
  </article>;
}

function ProspectiveForecasts() {
  const [report,setReport] = useState(null);
  const [error,setError] = useState(null);
  const refresh = () => api.prospectiveResults().then(value => {setReport(value);setError(null);}).catch(value => setError(value.message));
  useEffect(() => {refresh(); const timer=setInterval(refresh,60000); return () => clearInterval(timer);},[]);
  const summary=report?.summary || {};
  return <section className="panel"><div className="section-heading"><div><span className="eyebrow">FROZEN MODEL · PROSPECTIVE LOG</span><h3>Saved predictions versus actual openings</h3></div><button className="secondary-button" onClick={refresh}>Refresh log</button></div>
    <p className="fine-print">Calls are saved before the application cutoff. Upcoming estimates with missing bids are provisional. After listing, the same saved call and GMP-only baseline are compared with the verified exchange opening. Positive-call thresholds are frozen at 50% for the model and 62.5% for the GMP-only baseline.</p>
    {error && <p className="text-negative">{error}</p>}
    <p>{summary.ipos_forecast ?? 0} IPOs forecast · {summary.openings_verified ?? 0} openings verified · Model accuracy: {summary.direction_accuracy_pct == null ? 'awaiting listings' : `${num(summary.direction_accuracy_pct,1)}%`} · GMP-only: {summary.gmp_accuracy_pct == null ? 'awaiting listings' : `${num(summary.gmp_accuracy_pct,1)}%`}</p>
    {summary.without_saved_forecast > 0 && <p className="fine-print">{summary.without_saved_forecast} IPO calendar(s) need verification before a pre-close call can be recorded. These gaps remain visible.</p>}
    <div className="table-wrap"><table><thead><tr><th>IPO</th><th>Saved</th><th>Prediction</th><th>GMP-only</th><th>Actual open</th><th>Model outcome</th><th>Application screen</th><th>Screen result</th></tr></thead><tbody>{(report?.records || []).map(row => <tr key={row.ipo_id}><td>{cleanCompany(row.company_name)}<small className="fine-print">{row.quality === 'complete' ? 'Observed inputs' : 'Provisional · missing inputs'}</small></td><td>{formatIpoTimestamp(row.recorded_at_ist)}</td><td>{num(row.probability*100,1)}% · {row.predicted_positive ? 'Positive' : 'Flat / negative'}</td><td>{num(row.gmp_probability*100,1)}% · {row.gmp_positive ? 'Positive' : 'Flat / negative'}</td><td>{row.outcome ? `${money(row.outcome.actual_open,2)} (${pct(row.outcome.return_pct,true)})` : 'Pending listing verification'}</td><td>{row.direction_correct == null ? 'Pending' : row.direction_correct ? 'Right' : 'Missed'}</td><td>{row.application_screen||'Not recorded'}</td><td>{row.screen_result||'Pending'}</td></tr>)}</tbody></table></div>
  </section>;
}

function ForecastLog({historical,children}) {
  return historical ? <details className="panel research-history"><summary>Original live forecast log</summary>{children}</details> : children;
}

function ResearchSection({ collapsed, children, label='Historical closing-book research' }) {
  return collapsed ? <details className="panel research-history"><summary>{label}</summary><div className="page-stack">{children}</div></details> : <>{children}</>;
}

function HistoricalBacktestCard({ report, data, onTrack, onOpenMonitor }) {
  const scored = report.probability != null;
  const positive = Number(report.probability) >= .5;
  const actual = report.outcome;
  const summary = report.summary;
  const inputTime = report.bid?.time_basis === 'nominal_eod_stage_exact_publisher_clock_unverified' ? formatDate(report.bid.observed_at_ist?.slice(0,10)) : formatIpoTimestamp(report.bid?.observed_at_ist);
  return <section className="decision-card decision-card--outcome" data-testid="historical-backtest-card"><div className="decision-main">
    <div className="decision-kicker"><span className="eyebrow">{report.mode==='historical_benchmark'?'MODEL PERFORMANCE · HISTORICAL TEST':'LISTING REVIEW'}</span><span>{report.input_stage||'Historical analysis'}</span></div>
    <h2>{cleanCompany(data.company_name)}</h2>
    <div className="opening-comparison">
      <div><span className="eyebrow">Historical model prediction</span><strong className={scored ? positive ? 'text-positive' : 'text-negative' : ''}>{scored ? positive ? 'Positive opening' : 'Flat / negative opening' : 'Awaiting data'}</strong><p>{scored ? `${num(report.probability*100,2)}% positive-opening estimate` : 'A dated input is needed.'}</p>{report.bid && <small>{report.input_stage||'Inputs'} · {inputTime}</small>}</div>
      <div><span className="eyebrow">Actual opening</span><strong>{actual ? money(actual.actual_open,2) : 'Not verified'}</strong>{actual && <><p className={actual.positive ? 'text-positive' : Number(actual.return_pct) < 0 ? 'text-negative' : ''}>{pct(actual.return_pct,true)} versus issue price</p><small>Issue price {money(actual.issue_price,2)}</small></>}</div>
    </div>
    <div className={`opening-comparison-result ${report.correct == null ? '' : report.correct ? 'text-positive' : 'text-negative'}`}>{report.correct == null ? 'Opening verification pending' : report.correct ? 'Model Outcome: Right' : 'Model Outcome: Missed'}</div>
    <p className="fine-print">{report.mode==='historical_benchmark'?'The same frozen-model result shown on Model Performance, compared with its verified exchange opening. Historical test—not a call recorded live during bidding.':'Historical replay using recovered pre-close data.'}</p>
    <details><summary>Compare all IPOs</summary><div className="table-wrap"><table><thead><tr><th>IPO</th><th>Model estimate</th><th>Actual open</th><th>Opening return</th><th>Model outcome</th><th>Application screen</th><th>Screen result</th></tr></thead><tbody>{report.comparison_records.map(row => <tr key={row.company_name}><td>{cleanCompany(row.company_name)}</td><td>{row.probability == null ? 'Awaiting data' : `${row.predicted_positive ? 'Positive' : 'Flat / negative'} · ${num(row.probability*100,1)}%`}</td><td>{row.outcome ? money(row.outcome.actual_open,2) : 'Unverified'}</td><td>{row.outcome ? pct(row.outcome.return_pct,true) : '—'}</td><td className={row.correct == null ? '' : row.correct ? 'text-positive' : 'text-negative'}>{row.correct == null ? 'Unverified' : row.correct ? 'Right' : 'Missed'}</td><td>{row.application_screen||'Not evaluated'}</td><td>{row.screen_result||'Pending'}</td></tr>)}</tbody></table></div></details>
    <details className="fine-print"><summary>Sources & model details</summary>
      <p>Historical model: {summary.correct}/{summary.evaluated} correct ({num(summary.accuracy_pct,1)}%). Always-positive baseline: {summary.always_positive_correct}/{summary.evaluated}. GMP-only model: {summary.gmp_correct}/{summary.gmp_paired_count} on matched dated quotes. Combined model on those same quotes: {summary.model_correct_on_gmp_pairs}/{summary.gmp_paired_count}. {summary.pending_openings || 0} openings remain unverified.</p>
      <p>{report.limitation}</p>
      {report.bid?.time_basis==='nominal_eod_stage_exact_publisher_clock_unverified'&&<p>Day-2 values are publisher-reported dated tables recovered later. The evaluation uses the end-of-day stage; an exact historical publication clock has not been verified.</p>}
      {report.bid && <p>QIB {bid(report.bid.qib)} · HNI {bid(report.bid.hni)} · Retail {bid(report.bid.retail)} · <a href={report.bid.source} target="_blank" rel="noreferrer">Subscription source</a></p>}
      {report.sources?.gmp && <p>GMP ₹{num(report.sources.gmp.gmp_rs,2)} · {report.sources.gmp.gmp_asof_ist ? formatIpoTimestamp(report.sources.gmp.gmp_asof_ist) : report.sources.gmp.gmp_date} · <a href={report.sources.gmp.url || report.sources.gmp.source_url} target="_blank" rel="noreferrer">GMP source</a> · unofficial market quote</p>}
      {report.missing_features?.length > 0 && <p>Training imputation used for: {report.missing_features.join(', ')}. Probabilities with missing inputs are not calibrated for this replay.</p>}
      <p>Benchmark recorded {formatIpoTimestamp(report.computed_at_ist)}. This is not a prediction recorded before bidding closed.</p>
    </details>
    {report.features && <details className="fine-print"><summary>Model input snapshot</summary><p>{report.input_stage||'Historical inputs'} · {formatIpoTimestamp(report.bid?.observed_at_ist)}. Missing fields, if any, use the frozen training pipeline.</p><div className="snapshot-grid">{Object.entries(report.features).map(([name,value])=><Metric key={name} label={name.replaceAll('_',' ')} value={num(value,3)}/>)}</div></details>}
    <div className="decision-actions"><button className="primary-button" onClick={onTrack}><Plus size={17}/> Add to watchlist</button><button className="secondary-button" onClick={() => onOpenMonitor(data.company_name)}>Open Module 2</button></div>
  </div></section>;
}

function FrozenForecastCard({ forecast, data, onTrack, onOpenMonitor }) {
  if (!forecast) return null;
  if (forecast.historical_backtest && !forecast.saved_call_found) return <HistoricalBacktestCard report={forecast.historical_backtest} data={data} onTrack={onTrack} onOpenMonitor={onOpenMonitor}/>;
  const saved = forecast.mode === 'saved_pre_close' || forecast.mode === 'legacy_saved_pre_close';
  const noSavedCall = forecast.mode === 'no_saved_pre_close';
  const p = Number(forecast.probability);
  const outcome = forecast.outcome;
  const missing = forecast.inputs?.evidence?.missing_features || [];
  const matched = outcome ? (p >= .5) === Boolean(outcome.positive) : null;
  return <section className={`decision-card${outcome || noSavedCall ? ' decision-card--outcome' : ''}`} data-testid="frozen-forecast-card">
    <div className="decision-main">
      <div className="decision-kicker"><span className="eyebrow">{saved ? 'SAVED BEFORE BIDDING CLOSED' : noSavedCall ? 'PRE-CLOSE RECORD CHECK' : forecast.mode === 'retrospective_estimate' ? 'RECONSTRUCTED MODEL ESTIMATE' : 'CURRENT MODEL ESTIMATE'}</span><span>{noSavedCall ? 'No post-listing score substituted' : forecast.legacy_model ? 'Earlier bid model · original saved call' : 'Frozen five-view model'}</span></div>
      <h2>{cleanCompany(data.company_name)}</h2>
      {outcome || noSavedCall ? <>
        <div className="opening-comparison" data-testid="prediction-opening-comparison">
          <div><span className="eyebrow">Our saved pre-close prediction</span><strong className={noSavedCall ? '' : p >= .5 ? 'text-positive' : 'text-negative'}>{noSavedCall ? 'No saved call found' : p >= .5 ? 'Positive opening' : 'Flat / negative opening'}</strong><p>{noSavedCall ? 'No probability was recorded before the cutoff.' : `${num(p*100,1)}% chance of opening above issue price`}</p><small>{saved ? formatIpoTimestamp(forecast.recorded_at_ist) : noSavedCall && forecast.cutoff_at_ist ? `Cutoff: ${formatIpoTimestamp(forecast.cutoff_at_ist)}` : 'Calculated now, using available pre-close evidence'}</small></div>
          <div><span className="eyebrow">Actual listing open</span>{outcome ? <><strong className={outcome.positive ? 'text-positive' : Number(outcome.return_pct) < 0 ? 'text-negative' : ''}>{outcome.positive ? 'Positive opening' : Number(outcome.return_pct) < 0 ? 'Negative opening' : 'Flat opening'}</strong><p><b>{money(outcome.actual_open,2)}</b> · {pct(outcome.return_pct,true)} versus issue price</p><small>Issue price {money(outcome.issue_price,2)} · {formatDate(outcome.listing_date || data.listing_date)}</small></> : <><strong>Verification pending</strong><p>The official opening has not been verified yet.</p></>}</div>
        </div>
        <div className={`opening-comparison-result ${noSavedCall ? '' : matched ? 'text-positive' : 'text-negative'}`} role="status">{noSavedCall ? 'Not scored · no pre-close prediction was recorded' : `${matched ? '✓' : '✕'} ${saved ? matched ? 'Prediction matched the actual opening' : 'Prediction missed the actual opening' : matched ? 'Reconstructed estimate matches the actual direction' : 'Reconstructed estimate misses the actual direction'}`}</div>
      </> : <div className="decision-verdict-row"><div className={`verdict verdict-${p >= .5 ? 'positive' : 'negative'}`}>{p >= .5 ? 'POSITIVE OPENING' : 'FLAT / NEGATIVE OPENING'}</div><span>{forecast.quality !== 'complete' ? 'Provisional · some inputs unavailable' : 'Observed inputs'}</span></div>}
      <p>{noSavedCall ? forecast.explanation : saved ? `Saved ${formatIpoTimestamp(forecast.recorded_at_ist)}. This call is preserved for comparison with the listing open.` : forecast.mode === 'retrospective_estimate' ? 'Calculated now using available pre-close evidence and frozen missing-data handling. This was not recorded before applications closed and is excluded from prospective accuracy.' : 'Calculated from available evidence. The offer calendar must be confirmed before this can enter the pre-close forecast log.'}</p>
      {!noSavedCall && <details className="fine-print"><summary>What this score means</summary><p>This model scores whether the listing opens above its issue price, including a small gain. It does not estimate the size of that gain or long-term business quality. The score is uncalibrated; compare scores only from the same model and input stage. {forecast.legacy_model ? 'This saved call used the earlier subscription model.' : 'This call used the frozen five-expert model.'} The application screen below makes a separate Apply / Skip decision.</p></details>}
      {forecast.subscription_history?.length > 0 && <details className="fine-print"><summary>Subscription readings before the 5 pm cutoff ({forecast.subscription_history.length})</summary><div className="table-wrap"><table><thead><tr><th>Reading · IST</th><th>QIB</th><th>HNI</th><th>Retail</th><th>Evidence</th></tr></thead><tbody>{forecast.subscription_history.map((row,index) => <tr key={index}><td>{row.observed_at_ist ? formatIpoTimestamp(row.observed_at_ist) : `${formatDate(row.observed_on)} · time unverified`}</td><td>{bid(row.qib)}</td><td>{bid(row.hni)}</td><td>{bid(row.retail)}</td><td>{row.source ? <a href={row.source} target="_blank" rel="noreferrer">Source</a> : row.evidence} · {row.evidence === 'captured_before_close' ? 'Captured live' : 'Recovered later; publisher-reported time'}</td></tr>)}</tbody></table></div></details>}
      {missing.length > 0 && <details className="fine-print"><summary>Inputs estimated from training data ({missing.length})</summary><p>{missing.join(', ')}. Missing readings are not observed zero bids. Historical 81.2% accuracy does not validate this provisional estimate.</p></details>}
      {forecast.after_close_archives?.length > 0 && <details className="fine-print"><summary>Other saved scores · recorded after bidding closed</summary><p>These are genuine archived scores, but were saved too late to count as pre-close predictions. Their original values and dates are preserved.</p><div className="table-wrap"><table><thead><tr><th>Saved probability</th><th>Actually recorded · IST</th><th>Bid reading · IST</th></tr></thead><tbody>{forecast.after_close_archives.map((row,index) => <tr key={index}><td>{num(row.positive_listing_probability*100,1)}% positive</td><td>{formatIpoTimestamp(row.recorded_at_ist)}</td><td>{formatIpoTimestamp(row.captured_at_ist)}</td></tr>)}</tbody></table></div></details>}
      <div className="decision-actions"><button className="primary-button" onClick={onTrack}><Plus size={17}/> Add to watchlist</button>{data.category === 'LISTED' && <button className="secondary-button" onClick={() => onOpenMonitor(data.company_name)}>Open Module 2</button>}</div>
    </div>
    {!outcome && !noSavedCall && <aside className="decision-score"><span>{saved ? 'Saved positive-opening estimate' : 'Positive-opening estimate'}</span><strong>{num(p*100,1)}%</strong><div className="probability-track"><i style={{width:`${p*100}%`}}/></div>
      <small>{forecast.gmp_probability == null ? 'No GMP baseline was saved with this earlier model' : `GMP-only model: ${Math.round(Number(forecast.gmp_probability)*100)}%${missing.includes('gmp_pct') ? ' · quote missing, training imputation' : ''}`}</small>
      {saved && forecast.cutoff_at_ist && <small>Locks {formatIpoTimestamp(forecast.cutoff_at_ist)}</small>}
      {outcome ? <><span>Actual listing open</span><strong style={{fontSize:'2rem'}}>{money(outcome.actual_open,2)}</strong><small>{pct(outcome.return_pct,true)} versus issue price</small><b className={saved ? matched ? 'text-positive' : 'text-negative' : 'text-muted'}>{saved ? matched ? 'Prediction matched' : 'Prediction missed' : 'Reconstruction · not a saved forecast'}</b></> : <small>{data.category === 'LISTED' ? 'Exchange opening verification pending' : 'Actual opening will appear here after listing'}</small>}
    </aside>}
  </section>;
}

function CapitalFilter({ plan, historical = false, actualPositive = null }) {
  if (!plan) return null;
  const labels={undersubscribed_category:'One investor category is below 1× subscription',high_ofs_weak_gmp:'High OFS with weak institutional demand and GMP',retail_heavy_quota_weak_demand:'Large retail allocation with weak demand',low_conviction_book:'Low total demand and GMP',score_below_0_50:'Model score is below 50%',nonpositive_retail_utility_scenario:'Estimated retail utility is nonpositive'};
  const screenResult=actualPositive==null?'Pending opening':plan.tier==null?'Not evaluated':plan.tier===3?(actualPositive?'Winner Skipped (Opportunity Cost)':'Loser Avoided'):(actualPositive?'Winner Selected':'Loser Admitted');
  const title=plan.verdict==='DATA_PENDING'?'Capital screen awaiting inputs':plan.tier===3?'Skip under the capital rules':plan.tier===2?'One retail lot only':'Higher-conviction application';
  return <section className="panel"><div className="section-heading"><div><span className="eyebrow">{historical?'HISTORICAL CAPITAL SCREEN':'APPLICATION SCREEN'}</span><h3>{title}</h3></div><Pill tone={plan.tier===3?'negative':plan.tier===1?'positive':'warning'}>{plan.tier===3?'Skip':plan.tier===1?'Tier 1':plan.tier===2?'Tier 2':'Pending'}</Pill></div><p>{historical?'Replay of the application rules using recovered pre-close inputs. Applications for this offer have closed.':'The model predicts opening direction. This separate screen checks demand, offer structure and application cost.'}</p><p><strong>Application Screen: {plan.tier===1?'Apply Tier 1':plan.tier===2?'Apply Tier 2':plan.tier===3?'Skip':'Data pending'}</strong></p><p>Screen Result: {screenResult}</p>{plan.reasons?.length>0&&<ul>{plan.reasons.map(reason=><li key={reason}>{labels[reason]||reason}</li>)}</ul>}{plan.rii_lot_rupees!=null&&<div className="return-metrics"><div><span>One retail lot</span><strong>{money(plan.rii_lot_rupees,0)}</strong></div><div><span>Net utility scenario</span><strong>{money(plan.eu_rupees_rii,0)}</strong><small>Includes assumed allotment odds and costs</small></div></div>}<details><summary>Assumptions and financing</summary><p>Allotment odds are approximations. The return scenario uses the model score, GMP, a 6.5% loss assumption and assumed costs.</p>{plan.nbfc_interest_rupees!=null&&<p>Illustrative funding interest: {money(plan.nbfc_interest_rupees,0)} at {plan.financing.apr_pct}% for {plan.financing.days} days. {plan.nbfc_breakeven_open_pct==null?'A dated bNII book is required to estimate funding breakeven.':`Estimated funding breakeven: ${pct(plan.nbfc_breakeven_open_pct,true)}.`}</p>}</details></section>;
}

function DecisionPanel({ result, history, investment, setInvestment, onTrack, onOpenMonitor }) {
  const evaluation = result?.evaluation; const data = result?.scraped_data;
  const [liveFrozen,setLiveFrozen] = useState(null);
  useEffect(() => {
    setLiveFrozen(null);
    if (!data?.company_name) return;
    let cancelled=false;
    const timer=setInterval(() => api.prospectiveResults().then(report => {
      const saved=report.records.find(row => row.company_name.toLowerCase()===data.company_name.toLowerCase());
      if (!cancelled && saved) setLiveFrozen({...saved,mode:'saved_pre_close'});
    }).catch(() => {}),60000);
    return () => {cancelled=true;clearInterval(timer);};
  },[data?.company_name]);
  const frozenForecast = liveFrozen?.company_name === data?.company_name ? liveFrozen : result?.frozen_forecast;
  if (!evaluation || !data) return null;
  const historicalTest = !frozenForecast?.saved_call_found && frozenForecast?.historical_backtest?.probability != null ? frozenForecast.historical_backtest : null;
  const presented = forecastView({...result,frozen_forecast:frozenForecast});
  const scores = evaluation.hybrid_ml_scores || {};
  const issuedForecast = history?.application_forecast || result.application_forecast || (history?.decisions || []).find(row =>
    row.decision_kind === 'live_bid_forecast' && row.positive_listing_probability != null);
  const reconstruction = !issuedForecast ? history?.historical_bid_forecast || result.historical_bid_forecast : null;
  const reconstructionWarnings = reconstruction?.inputs?.evidence_warnings || [];
  const savedForecast = issuedForecast || reconstruction;
  const savedReturnForecast = issuedForecast?.inputs?.return_forecast;
  const returnForecast = savedReturnForecast?.outside_training_features?.length ? null : savedReturnForecast;
  const returnProbability = returnForecast?.outside_training_features?.length ? null : returnForecast?.positive_listing_probability;
  const historyUnavailable = Boolean(history?.load_error);
  const outcomeForecast = history?.prelisting_outcome_forecast;
  const replay = outcomeForecast || history?.prelisting_replay;
  const replaySourceTime = outcomeForecast ? replay?.captured_at_ist : replay?.observed_at_ist;
  const replayProbability = replay ? Math.round(Number(replay.positive_listing_probability) * 100) : null;
  const replaySignal = replay ? intradayBand(replay.positive_listing_probability) : null;
  const savedProbability = savedForecast
    ? Math.round(Number(returnProbability ?? savedForecast.positive_listing_probability) * 100) : null;
  // Descriptive bands for the saved experimental score, not validated
  // investment-action thresholds or a prediction of return magnitude.
  const savedSignal = savedForecast ? intradayBand(returnProbability ?? savedForecast.positive_listing_probability) : null;
  const savedForecastTime = reconstruction ? formatIpoTimestamp(reconstruction.computed_at_ist) : savedForecast?.recorded_at_ist
    ? formatIpoTimestamp(savedForecast.recorded_at_ist) : null;
  const savedSourceTime = reconstruction ? reconstruction.source_reported_at_ist
    ? formatIpoTimestamp(reconstruction.source_reported_at_ist) : formatDate(reconstruction.observed_on) : savedForecast?.source_asof_ist
    ? formatIpoTimestamp(savedForecast.source_asof_ist) : 'source time unknown';
  const isPreliminary = Boolean(evaluation.is_preliminary || evaluation.verdict === 'BIDDING IN PROGRESS');
  const empMedian = scores.expected_return_median_pct != null ? Number(scores.expected_return_median_pct) : null;
  const empLowPct = scores.expected_range_50_low != null ? Number(scores.expected_range_50_low) : null;
  const empHighPct = scores.expected_range_50_high != null ? Number(scores.expected_range_50_high) : null;
  const downside = scores.downside_p10 != null ? Number(scores.downside_p10) : null;
  const diagnostics = evaluation.conviction_diagnostics || [];
  const isListed = data.category === 'LISTED' || data.is_listed === true || /^LISTED\b/i.test(data.status || '');
  const cutoffTime = result.application_forecast_window?.cutoff_at_ist || savedForecast?.inputs?.application_cutoff_at_ist;
  const isClosed = data.category === 'CLOSED' || isListed || data.bidding_ended === true
    || (cutoffTime && Date.now() >= Date.parse(cutoffTime));
  const forecastLocked = isClosed || savedForecast?.is_locked === true;
  const complete = isClosed;
  const probability = scores.hybrid_positive_listing_prob != null ? Math.min(100, Number(scores.hybrid_positive_listing_prob)) : null;
  const actualOpenReturn = isListed && Number(data.issue_price) > 0 && Number(data.listing_open_price) > 0
    ? (Number(data.listing_open_price) / Number(data.issue_price) - 1) * 100 : null;
  const sensitivityCrossesEven = scores.sampling_sensitivity_low_pct != null && scores.sampling_sensitivity_high_pct != null
    && Number(scores.sampling_sensitivity_low_pct) < 50 && Number(scores.sampling_sensitivity_high_pct) > 50;
  const summary = (evaluation.reasoned_explanation || '').replace(/^[A-Z /]+\s*\(Confidence:\s*[^)]+\)\s*-\s*/i, '');
  const dealBadge = isListed
    ? { label: 'EXCHANGE LISTED', tone: 'blue' }
    : (data.category === 'CLOSED'
      ? { label: 'OFFERING CLOSED', tone: 'neutral' }
      : (data.category === 'OPEN'
        ? { label: 'ACTIVE BIDDING', tone: 'positive' }
        : { label: 'UPCOMING IPO', tone: 'warning' }));

  const demandValues = [data.qib_subscription, data.hni_subscription, data.retail_subscription];
  const partialBook = isPreliminary && !data.subscription_observed &&
    demandValues.some(value => value != null) && demandValues.some(value => value == null);
  const lastPublishedBook = data.subscription_display_status === 'last_published';
  const lateReconstruction = Boolean(data.subscription_reconstructed_later);
  const postCloseReading = data.subscription_capture_phase === 'post_close_reconciliation';
  const unscoredClosed = isClosed && isPreliminary && !data.subscription_is_final;
  const missingClosingBook = unscoredClosed && !postCloseReading;
  const demandTime = data.subscription_source_displayed_at || data.subscription_observed_at;
  const demandTimeText = formatIpoTimestamp(demandTime);
  const demandObservedNote = demandTime
    ? `${postCloseReading ? 'After close' : lastPublishedBook ? 'Last published' : data.subscription_is_final ? 'Closing-day book observed' : 'Observed'} ${demandTimeText} · ${postCloseReading ? 'not available before the application cutoff' : lastPublishedBook ? 'not a current or closing-day model input' : isPreliminary ? 'interim, closing book pending' : 'source reading'}`
    : 'No dated subscription reading';
  const snapshotQib = historicalTest ? presented?.inputs.qib_subscription : savedForecast ? savedForecast.inputs.qib_subscription : data.qib_subscription;
  const snapshotHni = historicalTest ? presented?.inputs.hni_subscription : savedForecast ? savedForecast.inputs.hni_subscription : data.hni_subscription;
  const snapshotRetail = historicalTest ? presented?.inputs.retail_subscription : savedForecast ? savedForecast.inputs.retail_subscription : data.retail_subscription;
  const snapshotNote = historicalTest ? `${historicalTest.input_stage||'Historical snapshot'} used in the benchmark · ${formatDate(historicalTest.bid?.observed_at_ist?.slice(0,10))}` : reconstruction ? `Historical bid reading · ${savedSourceTime} · recovered later` : savedForecast ? `Bids used in the saved forecast · ${savedSourceTime}` : demandObservedNote;
  const snapshotTotal = historicalTest ? historicalTest.total_subscription : data.total_subscription;
  const snapshotOfs = historicalTest ? historicalTest.offer_terms?.ofs_ratio : data.promoter_ofs_ratio;
  const snapshotRetailQuota = historicalTest ? historicalTest.offer_terms?.retail_quota_pct : data.retail_quota_pct;
  const replayGmp = !frozenForecast?.saved_call_found && frozenForecast?.historical_backtest?.sources?.gmp;
  const snapshotGmp = replayGmp ? frozenForecast.historical_backtest.features.gmp_pct : data.gmp_pct;
  const snapshotGmpRs = replayGmp ? replayGmp.gmp_rs : data.gmp_rs;
  const gmpNote = replayGmp ? `Unofficial historical quote · ${replayGmp.gmp_asof_ist ? formatIpoTimestamp(replayGmp.gmp_asof_ist) : formatDate(replayGmp.gmp_date)}` : data.gmp_basis === 'saved_pre_close'
    ? `Saved pre-close observation · ${formatIpoTimestamp(data.gmp_observed_at)}. The saved model call keeps its original inputs.` : data.gmp_source === 'post_listing_quote_excluded'
    ? 'Quote fetched after listing; excluded from pre-listing context'
    : data.gmp_pct == null
      ? 'No unofficial quote available'
      : `Unofficial ${isListed ? 'pre-listing ' : ''}GMP · Chittorgarh/InvestorGain source family · quote time unverified${data.gmp_observed_at ? ` · fetched ${data.gmp_observed_at}` : ''}`;
  const missingDemandNote = partialBook
    ? `Not published in the ${lastPublishedBook ? 'last available' : 'latest captured'} table; not zero bids`
    : 'No verified subscription reading';
  const displayDiagnostics = diagnostics.map(item => {
    if (frozenForecast) {
      const key = /QIB/i.test(item.pillar) ? 'log_qib' : /Retail/i.test(item.pillar) ? 'log_retail' : /HNI/i.test(item.pillar) ? 'log_hni' : /Grey market/i.test(item.pillar) ? 'gmp_pct' : /volatility/i.test(item.pillar) ? 'macro_vix' : null;
      if (key) {
        if (frozenForecast.legacy_model || frozenForecast.mode === 'legacy_saved_pre_close') {
          const fields = {log_qib:'qib_subscription',log_retail:'retail_subscription',log_hni:'hni_subscription'};
          const value = frozenForecast.inputs?.[fields[key]];
          if (fields[key]) return {...item,pillar:item.pillar.replace(/^\d+\. Final QIB demand/,'1. QIB demand at forecast'),status:'CONTEXT',role:'SAVED MODEL INPUT',metric:value==null?'Unavailable':bid(value),explanation:value==null?'This category was not present in the saved inputs.':`${bid(value)} recorded in the original pre-close prediction.`};
          return {...item,status:'CONTEXT',role:'ADDITIONAL CONTEXT',metric:key==='gmp_pct'&&snapshotGmp!=null?pct(snapshotGmp,true):'Not used',explanation:key==='gmp_pct'&&data.gmp_basis==='saved_pre_close'?gmpNote:'This saved subscription model used QIB, HNI and retail demand. This field did not determine its saved score.'};
        }
        if (presented==null) {
          const context={log_qib:data.qib_subscription,log_hni:data.hni_subscription,log_retail:data.retail_subscription,gmp_pct:snapshotGmp,macro_vix:data.india_vix}[key];
          return {...item,status:'CONTEXT',role:'PUBLISHED CONTEXT',metric:context==null?'Unavailable':key.startsWith('log_')?bid(context):num(context,2),explanation:'This value is available as context. No scored historical input snapshot has been bound to this IPO; it is not being described as an imputed model input.'};
        }
        const value = (historicalTest?.features || frozenForecast.inputs?.features)?.[key];
        return {...item,pillar:historicalTest?item.pillar.replace('Final QIB demand','QIB demand at input snapshot'):item.pillar,status:'CONTEXT',role:value == null ? 'MISSING · TRAINING IMPUTATION' : 'FROZEN MODEL INPUT',metric:value == null ? 'Estimated' : key.startsWith('log_') ? bid(Math.expm1(value)) : num(value,2),explanation:value == null ? 'This input was unavailable. The frozen model uses its training-time missing-data handling; no zero reading was invented.' : 'This dated observation was included in the displayed frozen-model estimate.'};
      }
    }
    if (returnForecast?.track === 'bids_plus_gmp' && /Grey market premium/i.test(item.pillar)) {
      return {...item, status: 'CONTEXT', role: 'RECORDED RETURN-MODEL INPUT',
        metric: pct(returnForecast.input_features.gmp_pct, true),
        explanation: `Captured GMP ${pct(returnForecast.input_features.gmp_pct, true)}, observed ${formatIpoTimestamp(returnForecast.gmp_observed_at)}, is an input to this experimental return model.`};
    }
    if (savedForecast && /QIB|Retail|HNI/i.test(item.pillar)) {
      const key = /QIB/i.test(item.pillar) ? 'qib_subscription' : /Retail/i.test(item.pillar) ? 'retail_subscription' : 'hni_subscription';
      const value = savedForecast.inputs?.[key];
      return { ...item, status: 'CONTEXT', role: reconstruction ? 'RECONSTRUCTED FORECAST INPUT' : 'RECORDED FORECAST INPUT',
        metric: bid(value), explanation: `${bid(value)} from the bid reading dated ${savedSourceTime}, used in this model estimate.` };
    }
    if (!isPreliminary || !/QIB|Retail|HNI/i.test(item.pillar)) return item;
    const value = item.pillar.includes('QIB') ? data.qib_subscription
      : item.pillar.includes('Retail') ? data.retail_subscription
      : item.pillar.includes('HNI') ? data.hni_subscription : null;
    if (value == null && item.pillar.includes('QIB') && data.subscription_source) return {
      ...item, role: 'BID OBSERVATION · NOT A MODEL SCORE', metric: 'Not published',
      explanation: `QIB is absent from the ${lastPublishedBook ? 'last available' : 'latest captured'} subscription table. That is missing data, not 0x demand.`,
    };
    return value == null ? item : {
      ...item, role: 'BID OBSERVATION · NOT A MODEL SCORE', metric: `${bid(value)} · ${postCloseReading ? 'after close' : lastPublishedBook ? 'last published' : 'interim'}`,
      explanation: `${postCloseReading ? 'Post-close' : lastPublishedBook ? 'Last published' : 'Observed interim'} demand is ${bid(value)}. This reading has not been scored by an investor-time model.`,
    };
  });

  const historicalValue = isPreliminary ? (unscoredClosed ? 'Not scored' : 'Interim model pending') : empLowPct == null || empHighPct == null ? 'Insufficient history' : `${pct(empLowPct, true)} to ${pct(empHighPct, true)}`;
  const historicalNote = empMedian == null ? 'No return estimate available' : `Earlier-year score band · n=${scores.historical_sample_size} · median ${pct(empMedian, true)}; not a price forecast`;

  return <div className="decision-stack">
    {historyUnavailable && <div className="warning-banner"><AlertTriangle size={18}/><div><strong>Forecast log could not be checked</strong><span>The model result is available, but saved forecasts and replays could not be loaded. Retry the log before concluding that no prediction was recorded.</span></div></div>}
    {isListed && !frozenForecast && <div className="warning-banner"><AlertTriangle size={18} /><div><strong>Historical replay, not a live IPO application call</strong><span>{actualOpenReturn == null ? 'The listing opening price is unavailable here.' : `Recorded listing open: ${money(data.listing_open_price, 2)} (${pct(actualOpenReturn, true)} versus the ${money(data.issue_price, 2)} issue price; source: ${result.source || 'IPO record'}).`} {replay ? 'The separate model score below was locked before listing, but after applications closed.' : 'The displayed closing-book model score was recomputed after listing; it is not a stored pre-listing forecast.'}</span></div></div>}
    {result.forecast_save_error && <div className="warning-banner"><AlertTriangle size={18}/><div><strong>Prediction could not be saved</strong><span>The forecast history could not be updated. Retry Run analysis to record this assessment.</span></div></div>}
    {isPreliminary && !isClosed ? (
      <div className="warning-banner" style={{ background: 'rgba(56, 189, 248, 0.08)', borderColor: 'rgba(56, 189, 248, 0.3)', color: '#38bdf8' }}>
        <Clock3 size={18} />
        <div>
          <strong>{data.category === 'UPCOMING' ? 'Offer has not opened yet' : 'Live bidding evidence · decide before your broker cutoff'}</strong>
          <span>
            {evaluation.pre_bidding_scorecard?.broker_upi_cutoff_note ||
              "Check the exchange issue schedule and your broker's application and UPI-mandate cut-offs. Your broker may close applications before the exchange's final bid-position update."}
          </span>
        </div>
      </div>
    ) : (!complete && (
      <div className="warning-banner">
        <AlertTriangle size={18} />
        <div>
          <strong>Subscription data is not yet available.</strong>
          <span>Treat this verdict as preliminary until QIB, HNI, and retail bids are published.</span>
        </div>
      </div>
    ))}
    <FrozenForecastCard forecast={frozenForecast} data={data} onTrack={onTrack} onOpenMonitor={onOpenMonitor}/>
    {frozenForecast?.saved_call_found && frozenForecast?.benchmark_comparison && <details className="panel research-history"><summary>Current-model historical test · {num(frozenForecast.benchmark_comparison.probability*100,1)}% score · same as Model Performance</summary><HistoricalBacktestCard report={frozenForecast.benchmark_comparison} data={data} onTrack={onTrack} onOpenMonitor={onOpenMonitor}/></details>}
    <CapitalFilter plan={frozenForecast?.historical_backtest?.capital_plan || frozenForecast?.inputs?.capital_plan} historical={data.category==='LISTED'||data.category==='CLOSED'} actualPositive={(frozenForecast?.historical_backtest?.outcome||frozenForecast?.outcome)?.positive??null}/>
    {!frozenForecast && <section className="decision-card">
      <div className="decision-main">
        <div className="decision-kicker"><span className="eyebrow">{reconstruction ? 'RECONSTRUCTED PRE-CLOSE FORECAST' : savedForecast ? forecastLocked ? 'FINAL PRE-CLOSE PREDICTION' : 'PRE-CLOSE PREDICTION' : isClosed ? 'APPLICATION WINDOW ENDED' : 'PRE-CLOSE PREDICTION'}</span><span>{reconstruction ? 'Bids from ' + savedSourceTime : savedForecast ? 'Recorded ' + savedForecastTime : isClosed ? 'Pre-close forecast history' : data.category === 'UPCOMING' ? 'Bidding has not opened' : 'Awaiting a complete subscription reading'}</span></div>
        <h2>{cleanCompany(data.company_name)}</h2>
        <div className="decision-verdict-row"><div className={`verdict verdict-${savedSignal === 'HIGH' ? 'positive' : savedSignal === 'LOW' ? 'negative' : savedForecast ? 'warning' : 'info'}`}>{savedForecast ? savedSignal : isClosed ? 'NO PRE-CLOSE CALL SAVED' : 'AWAITING BID DATA'}</div><span>{reconstruction ? reconstructionWarnings.length ? 'Exploratory replay · source check required' : 'Historical bid data · model replay' : savedForecast ? forecastLocked ? 'Locked at the offer cutoff' : 'Updates as published bids change' : historyUnavailable ? 'Forecast history unavailable' : ''}</span></div>
        {reconstructionWarnings.length > 0 && <p role="status" className="text-warning">This recovered source has timing conflicts or lacks independent confirmation. This score is not included in the clean-data accuracy comparison.</p>}
        <p>{reconstruction ? `Using bids reported as of ${savedSourceTime}, the current model estimates a ${savedProbability}% chance of opening above the issue price. This is the latest recovered reading before the 17:00 IST cutoff on ${formatDate(reconstruction.inputs.application_closing_date)}. Reconstructed ${savedForecastTime}, not a prediction issued live at that earlier time.` : savedForecast ? `${savedProbability}% estimated chance of opening above the issue price, using bids from ${savedSourceTime}. ${forecastLocked ? 'The pre-close call is preserved for comparison with the listing opening.' : 'Each analysis and scheduled update records the estimate. The latest call is locked at 17:00 IST on the closing date.'}` : historyUnavailable ? 'The forecast history could not be loaded. Retry the history to check the last saved call.' : isClosed ? 'No prediction was saved while this offer was accepting applications. Later bid totals remain available in the research history below.' : data.category === 'UPCOMING' ? 'The bid-based prediction starts when this offer opens and subscription figures are published.' : 'A complete QIB, HNI and retail reading is needed. When a refresh is empty, the last complete published reading is retained for the prediction.'}</p>
        <div className="decision-actions">
          {isListed && <button className="primary-button" onClick={() => onOpenMonitor(data.company_name)}><Activity size={17} /> Open Module 2 <ArrowUpRight size={15} /></button>}
          <button className={isListed ? 'secondary-button' : 'primary-button'} onClick={onTrack}><Plus size={17} /> Add to watchlist</button>
          {data.rhp_pdf_url && <a className="secondary-button quiet-button" href={data.rhp_pdf_url} target="_blank" rel="noreferrer"><FileText size={17} /> RHP</a>}
        </div>
      </div>
      <aside className="decision-score">
        <span>{reconstruction ? 'Reconstructed positive-opening estimate' : savedForecast ? 'Chance of a positive listing opening' : 'Pre-close prediction'}</span>
        {savedForecast ? (
          <>
            <strong>{savedProbability}%</strong>
            <div className="probability-track"><i style={{ width: savedProbability + '%' }} /></div>
            <small>{reconstruction ? historicalSourceCount(reconstruction) : savedForecast.source_family === 'nse_exchange' ? 'NSE bid table' : 'Chittorgarh captured table'} · {savedSourceTime}</small>
            <small>QIB {bid(savedForecast.inputs?.qib_subscription)} · HNI {bid(savedForecast.inputs?.hni_subscription)} · Retail {bid(savedForecast.inputs?.retail_subscription)}</small>
            <small>Provisional estimate. Performance on interim bids is being measured.</small>
            {reconstruction && <small>{reconstruction.sources.map((source, index) => <React.Fragment key={source.url}>{index > 0 && ' · '}<a href={source.url} target="_blank" rel="noreferrer">{historicalSourceName(source)}</a></React.Fragment>)}</small>}
            {actualOpenReturn != null && <small>Exchange opening: {pct(actualOpenReturn, true)} versus issue price · {(actualOpenReturn > 0) === (savedProbability >= 50) ? 'direction matched' : 'direction missed'}.</small>}
          </>
        ) : (
          <>
            <strong>—</strong>
            <small>{isClosed ? 'No saved application-time prediction' : data.category === 'UPCOMING' ? 'Awaiting the opening of bidding' : 'Awaiting a complete published bid table'}</small>
          </>
        )}
      </aside>
    </section>}
    {reconstruction && <details className="panel research-history"><summary>Subscription history & sources</summary><div className="section-heading"><div><span className="eyebrow">RECOVERED BEFORE-CLOSE DATA</span><h3>Latest recovered bids for each day</h3></div></div><p className="fine-print">Historical subscription readings recovered later. Times are publisher-reported IST and are not independently verified archive timestamps. This table is subscription evidence, not a log of predictions issued before closing.</p><div className="table-wrap"><table><thead><tr><th>Bids as of · IST</th><th>QIB</th><th>HNI</th><th>Retail</th><th>Sources</th></tr></thead><tbody>{reconstruction.history.map(row => <tr key={row.observed_on}><td>{row.source_reported_at_ist ? formatIpoTimestamp(row.source_reported_at_ist) : formatDate(row.observed_on)}</td><td>{bid(row.inputs.qib_subscription)}</td><td>{bid(row.inputs.hni_subscription)}</td><td>{bid(row.inputs.retail_subscription)}</td><td>{historicalSourceCount(row)}</td></tr>)}</tbody></table></div></details>}
    <section className="panel deal-snapshot"><div className="section-heading"><div><span className="eyebrow">DEAL SNAPSHOT</span><h3>Bid readings and offer terms</h3></div><Pill tone={dealBadge.tone}>{dealBadge.label}</Pill></div><div className="snapshot-grid">
      <Metric label="Issue price" value={money(data.issue_price, 2)} />
      <Metric label="Issue size" value={compact(data.issue_size_cr)} />
      <Metric label={historicalTest?'QIB demand · benchmark input':'QIB demand · published'} value={snapshotQib == null ? 'Not published yet' : bid(snapshotQib)} note={snapshotQib == null ? missingDemandNote : snapshotNote} />
      <Metric label={historicalTest?'HNI demand · benchmark input':'HNI demand · published'} value={snapshotHni == null ? 'Not published yet' : bid(snapshotHni)} note={snapshotHni == null ? missingDemandNote : snapshotNote} />
      <Metric label={historicalTest?'Retail demand · benchmark input':'Retail demand · published'} value={snapshotRetail == null ? 'Not published yet' : bid(snapshotRetail)} note={snapshotRetail == null ? missingDemandNote : snapshotNote} />
      <Metric label={data.gmp_basis === 'saved_pre_close' && !replayGmp ? 'GMP · saved pre-close' : frozenForecast ? 'GMP · model input' : returnForecast?.track === 'bids_plus_gmp' ? 'GMP · return model input' : 'GMP · context only'} value={snapshotGmp != null ? `${pct(snapshotGmp, true)}${snapshotGmpRs != null ? ` (₹${snapshotGmpRs})` : ''}` : 'Unavailable'} note={gmpNote} tone={snapshotGmp != null && Number(snapshotGmp) > 0 ? 'positive' : 'neutral'} />
      <Metric label="OFS share of offer" value={snapshotOfs == null ? 'Unavailable' : pct(Number(snapshotOfs) * 100)} note={data.offer_facts?.source_url ? <a href={data.offer_facts.source_url} target="_blank" rel="noreferrer">Offer source</a> : null} />
      <Metric label="Retail reservation" value={snapshotRetailQuota == null ? 'Unavailable' : pct(snapshotRetailQuota)} note={data.retail_quota_basis==='net_offer'?`Share of net offer${data.retail_quota_total_pct!=null?` · ${pct(data.retail_quota_total_pct)} of total offer`:''}`:'Share of total offer reserved for retail'} />
      <Metric label={historicalTest?'Total demand · historical snapshot':data.total_subscription_status==='saved_pre_close'?'Total demand · saved book':'Total subscription · published'} value={data.total_subscription_status==='not_started' ? 'Bidding not started' : snapshotTotal==null ? isClosed?'Historical total unavailable':'Not published yet' : bid(snapshotTotal)} note={historicalTest?snapshotNote:data.total_subscription_provenance ? `${data.total_subscription_provenance.publisher_update || formatIpoTimestamp(data.total_subscription_provenance.fetched_at_ist)} · ${data.total_subscription_provenance.source_family==='nse_exchange'?'NSE bid positions':'Published subscription total'}` : null} />
      {(!historicalTest || data.objects_capex_ratio != null) && <Metric label="Fresh capex · context" value={data.objects_capex_ratio == null ? data.offer_facts && Number(data.promoter_ofs_ratio) === 1 ? 'No fresh issue' : data.objects_capex_status === 'NOT_SEPARATELY_ITEMISED' ? 'Not separately itemised' : 'Unavailable' : pct(Number(data.objects_capex_ratio) * 100)} note={data.proceeds_uses?.length ? 'See use of proceeds below' : null} />}
    </div>
      {!!data.proceeds_uses?.length && <details className="research-history"><summary>Use of proceeds</summary><p className="fine-print">Published allocation of fresh-issue funds. These uses are not all capital expenditure; a separate CAPEX percentage is shown only when supported by the source.</p>{data.proceeds_uses.map((use,i)=><p key={i}>{use.description}{use.amount_cr != null ? ` · ₹${num(use.amount_cr,2)} Cr` : ''}</p>)}{(data.proceeds_provenance || [data.offer_facts]).filter(Boolean).map((source,i)=><p className="fine-print" key={i}><a href={source.source_url} target="_blank" rel="noreferrer">Offer allocation source {i+1}</a></p>)}</details>}
      {!reconstruction && data.subscription_source && <p className="fine-print">{lastPublishedBook ? 'Last available demand source' : 'Demand source'}: {data.subscription_source_family === 'nse_exchange' ? 'NSE bid-position table' : 'Chittorgarh subscription page'} · {data.subscription_source_displayed_at ? `publisher updated ${formatIpoTimestamp(data.subscription_source_displayed_at)}` : `publisher update time unavailable; fetched ${formatIpoTimestamp(data.subscription_observed_at)}`}{data.subscription_url && <> · <a href={data.subscription_url} target="_blank" rel="noreferrer">View source</a></>}. {postCloseReading ? 'Captured after the application window; not investor-time evidence. ' : ''}{lastPublishedBook ? 'This is an older observation, not a fresh or certified final book.' : 'Exchange bid positions are not certified final subscriptions.'}</p>}
      {!reconstruction && data.subscription_cross_check?.values && <p className="fine-print">Secondary-source check (Chittorgarh/InvestorGain family): HNI {bid(data.subscription_cross_check.values.hni_subscription)}, Retail {bid(data.subscription_cross_check.values.retail_subscription)}; fetched {data.subscription_cross_check.fetched_at_ist}. Its publisher update time is unknown, so this is not a synchronized comparison.</p>}
      {partialBook && <p className="fine-print">{lastPublishedBook ? 'The last available table' : 'The published table'} contains only some demand categories. These are interim observations, not final model inputs.</p>}
    </section>
    <ReturnEstimate forecast={returnForecast} />
    <ResearchSection collapsed={Boolean(isClosed)} label={historicalTest?'Historical model inputs and evidence':'Historical closing-book research'}>
    <div className="split-grid"><section className="panel"><div className="section-heading"><div><span className="eyebrow">SIGNALS EXPLAINED</span><h3>{historicalTest ? 'Inputs behind the historical prediction' : savedForecast ? 'Inputs behind the saved prediction' : 'What is known at this point'}</h3></div></div><p className="fine-print">{historicalTest ? "These are the same dated inputs used for this IPO on Model Performance. The model score and opening outcome are historical test evidence; no live pre-close call is being invented. The input stage is identified above." : frozenForecast?.legacy_model ? "This saved subscription model used QIB, HNI and retail demand. The original recorded inputs and score are preserved; historical GMP and offer facts are additional context." : frozenForecast ? "The frozen model combines published bids, GMP, offer details, market history and available financial facts. Missing inputs use frozen training imputation and are identified below. Predictions are saved before the cutoff and compared with actual exchange openings." : returnForecast ? `The experimental return model uses QIB, HNI, retail demand, issue price and size${returnForecast.track === 'bids_plus_gmp' ? ', plus the captured GMP quote' : ''}. OFS, macro conditions and other document facts remain context. Its performance is tested separately from the legacy direction model.` : savedForecast ? 'QIB, HNI and retail demand determine this provisional direction estimate. GMP, offer terms and market conditions are additional context. Interim performance is measured from the saved calls and later opening prices.' : 'The published demand readings and offer terms are shown with their sources. A complete dated bid table is required for the pre-close prediction.'}</p><div className="diagnostic-list">{displayDiagnostics.map(item => <div className="diagnostic" key={item.pillar}><span className={`status-icon status-${item.status?.toLowerCase()}`}>{item.status === 'PASS' ? <Check size={14} /> : item.status === 'CONTEXT' ? <Activity size={14} /> : <AlertTriangle size={14} />}</span><div><strong>{item.pillar}</strong><span className="diagnostic-role">{item.role}</span><p>{item.explanation}</p></div><span className="diagnostic-metric">{item.metric}</span></div>)}</div></section>
      <section className="panel calculator-panel"><div className="section-heading"><div><span className="eyebrow">CAPITAL PLAN</span><h3>What this means for your money</h3></div></div><label className="field-label">Planned investment</label><div className="money-input"><span>₹</span><input type="number" min="0" step="5000" value={investment} onChange={e => setInvestment(e.target.value)} /></div>
        {returnForecast ? (
          <div className="scenario-list"><div><span>Central return, if allotted</span><strong>{money(Number(investment || 0) * returnForecast.central_return_pct / 100)}</strong></div><div><span>Lower residual scenario</span><strong className="text-negative">{money(Number(investment || 0) * returnForecast.lower_return_pct / 100)}</strong></div><div><span>Upper residual scenario</span><strong>{money(Number(investment || 0) * returnForecast.upper_return_pct / 100)}</strong></div></div>
        ) : savedReturnForecast?.outside_training_features?.length ? (
          <p className="fine-print">No reliable return range is available for this IPO. Its saved opening-direction prediction is shown above.</p>
        ) : isPreliminary ? (
          <div style={{ padding: '16px 0', color: 'var(--muted)', fontSize: '11px', lineHeight: '1.6' }}>
            A return scenario is not estimated from today’s bids: the early-stage model has not passed validation.
          </div>
        ) : empMedian == null || downside == null ? (
          <p className="fine-print">There are too few comparable historical cases for a capital scenario.</p>
        ) : (
          <div className="scenario-list"><div><span>Historical median ({pct(empMedian, true)})</span><strong className={empMedian >= 0 ? "text-positive" : "text-negative"}>{money(Number(investment || 0) * empMedian / 100)}</strong></div><div><span>Flat listing</span><strong>{money(0)}</strong></div><div><span>Historical lower tail ({pct(downside, true)})</span><strong className="text-negative">{money(Number(investment || 0) * downside / 100)}</strong></div></div>
        )}
        <p className="fine-print">{returnForecast ? 'Experimental scenarios assume the full entered amount is allotted; actual allotment and returns can differ. The band is not a guaranteed loss limit.' : 'Group history is illustrative, not a return forecast.'} Values exclude taxes, brokerage, funding costs, and allotment probability.</p></section></div>
    {!isPreliminary && <section className="panel"><div className="section-heading"><div><span className="eyebrow">COMPARABLES</span><h3>Closest historical IPOs</h3></div><span className="muted">Distance-weighted analog engine</span></div><div className="table-wrap"><table><thead><tr><th>Company</th><th>QIB</th><th>Retail</th><th>Listing return</th><th>Similarity</th></tr></thead><tbody>{(evaluation.similarity_baseline?.top_analogs || []).map(item => <tr key={item.id}><td>{item.company_name}</td><td>{num(item.qib_subscription)}x</td><td>{num(item.retail_subscription)}x</td><td className={item.listing_return_pct >= 0 ? 'text-positive' : 'text-negative'}>{pct(item.listing_return_pct, true)}</td><td>{num(item.distance, 2)}</td></tr>)}</tbody></table></div></section>}
    {!isPreliminary && <div className="split-grid memo-layout"><section className="panel"><div className="section-heading"><div><span className="eyebrow">INVESTMENT MEMO</span><h3>Catalysts and risks</h3></div></div><div className="memo-columns"><div><strong>Bull case</strong>{(result.ai_investment_memo?.bull_catalysts || []).map((item, i) => <p key={i}><Check size={13}/>{item}</p>)}</div><div><strong>Bear case</strong>{(result.ai_investment_memo?.bear_risks || []).map((item, i) => <p key={i}><AlertTriangle size={13}/>{item}</p>)}</div></div></section><section className="panel sector-record-panel"><div className="section-heading"><div><span className="eyebrow">SECTOR RECORD</span><h3>{evaluation.sector_track_record?.sector_name || data.sector}</h3></div><span className="muted">Historical listing outcomes for comparable sector IPOs</span></div><div className="sector-record-body"><div className="sector-summary"><Metric label="Historical IPOs" value={evaluation.sector_track_record?.total_ipos_in_sector || '—'} /><Metric label="Win rate" value={pct(evaluation.sector_track_record?.win_rate_pct)} tone="positive" /><Metric label="Average listing" value={pct(evaluation.sector_track_record?.avg_listing_return_pct, true)} /></div><div className="sector-peers">{(evaluation.sector_track_record?.top_sector_peers || []).slice(0,4).map(peer => <div key={peer.company_name}><span>{peer.company_name}</span><strong className={peer.listing_return_pct >= 0 ? 'text-positive' : 'text-negative'}>{pct(peer.listing_return_pct, true)}</strong></div>)}</div></div></section></div>}
    {!isPreliminary && <section className="panel"><h3>Historical closing-book statistics</h3><p className="fine-print">These group statistics use the later closing book. The pre-close prediction above keeps its own recorded inputs.</p><div className="metric-grid four"><Metric label="Historical return band" value={historicalValue} note={historicalNote} /><Metric label="Historical 10th percentile" value={pct(downside, true)} tone="negative" /><Metric label="Historical signal confidence" value={evaluation.conviction || '—'} note={sensitivityCrossesEven ? 'Training refits cross 50%' : evaluation.bHNI_sizing_tier} /><Metric label="Historical analogs" value={evaluation.similarity_baseline?.analog_win_rate_pct == null ? 'Unavailable' : `${num(evaluation.similarity_baseline.analog_win_rate_pct, 0)}% win`} note={`${evaluation.similarity_baseline?.total_analogs || 0} comparable IPOs`} /></div></section>}
    </ResearchSection>
  </div>;
}

function Copilot({ result }) {
  const [question, setQuestion] = useState(''); const [messages, setMessages] = useState([]); const [loading, setLoading] = useState(false);
  if (!result?.evaluation) return null;
  const ask = async preset => { const text = preset || question.trim(); if (!text) return; setMessages(p => [...p, { role: 'user', text }]); setQuestion(''); setLoading(true); try { const response = await api.copilot(result.scraped_data.company_name, text, assistantContext(result)); setMessages(p => [...p, { role: 'ai', text: response.answer }]); } catch (e) { setMessages(p => [...p, { role: 'ai', text: `I could not answer that: ${e.message}` }]); } finally { setLoading(false); } };
  return <section className="panel copilot-panel"><div className="section-heading"><div><span className="eyebrow">RESEARCH ASSISTANT</span><h3><MessageSquare size={18} /> Investment Research Q&amp;A</h3></div></div><div className="prompt-chips">{['Explain this simply', 'What are the biggest risks?', 'How much should I apply?', 'What could change the verdict?'].map(q => <button key={q} onClick={() => ask(q)}>{q}</button>)}</div>{messages.length > 0 && <div className="chat-log">{messages.map((m, i) => <div className={`chat chat-${m.role}`} key={i}>{m.text}</div>)}{loading && <div className="chat chat-ai"><LoaderCircle className="spin" size={15} /> Thinking from the current analysis…</div>}</div>}<div className="chat-input"><input value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => e.key === 'Enter' && ask()} placeholder="Ask about risk, sizing, QIB demand, or exit strategy…" /><button onClick={() => ask()} disabled={loading}><ArrowUpRight size={18} /></button></div></section>;
}

function PriceChart({ points = [] }) {
  if (points.length < 2) return <div className="chart-empty">Price trajectory is not available for this security.</div>;
  const values = points.map(p => Number(p.price)); const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
  const coords = points.map((p, i) => `${(i / (points.length - 1)) * 100},${86 - ((Number(p.price) - min) / range) * 72}`).join(' ');
  const positive = values[values.length - 1] >= values[0];
  return <div className="price-chart"><svg viewBox="0 0 100 100" preserveAspectRatio="none"><defs><linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={positive ? '#53e393' : '#ff776d'} stopOpacity=".28"/><stop offset="1" stopColor={positive ? '#53e393' : '#ff776d'} stopOpacity="0"/></linearGradient></defs><polygon points={`0,100 ${coords} 100,100`} fill="url(#chartFill)"/><polyline points={coords} fill="none" stroke={positive ? '#53e393' : '#ff776d'} strokeWidth="2" vectorEffect="non-scaling-stroke"/></svg><div className="chart-labels"><span>{points[0]?.date}</span><strong>{money(values[values.length - 1], 2)}</strong><span>{points[points.length - 1]?.date}</span></div></div>;
}

function IntradayForecastPanel({ scorecard, onRefresh }) {
  const summary = scorecard?.summary || {};
  const records = scorecard?.daily_records || [];
  return <section className="panel">
    <div className="section-heading"><div><span className="eyebrow">SAVED BEFORE THE CUTOFF</span><h3>Live-bid forecast audit</h3></div><button className="secondary-button" onClick={onRefresh}><RefreshCw size={15} /> Refresh results</button></div>
    <p className="fine-print">The latest saved prediction per IPO is compared with its actual exchange opening. The snapshot model is experimental: final-book backtest accuracy does not apply to these intraday forecasts. Missing listing opens remain pending.</p>
    <div className="metric-grid four"><Metric label="IPOs forecast" value={summary.ipos_with_forecast ?? '—'} /><Metric label="Opening verified" value={summary.ipos_with_verified_open ?? '—'} /><Metric label="Direction matched" value={pct(summary.direction_accuracy_pct)} note="Latest saved forecast per IPO" /><Metric label="Brier score" value={num(summary.brier_score, 3)} note="Lower is better; no claim with a tiny sample" /></div>
    {records.length ? <div className="table-wrap benchmark-table"><table><thead><tr><th>IPO</th><th>Forecast date</th><th>Source time</th><th>Saved probability</th><th>Actual open</th><th>Model outcome</th></tr></thead><tbody>{records.slice(0, 30).map((row, index) => <tr key={row.ipo_id + '-' + row.forecast_date_ist + '-' + index}><td>{cleanCompany(row.company_name)}</td><td>{formatDate(row.forecast_date_ist)}</td><td>{formatIpoTimestamp(row.source_asof_ist)}</td><td>{pct(Number(row.positive_listing_probability) * 100)}</td><td>{row.actual_open_return_pct == null ? 'Pending listing' : pct(row.actual_open_return_pct, true)}</td><td>{row.direction_correct == null ? 'Pending' : row.direction_correct ? 'Right' : 'Missed'}</td></tr>)}</tbody></table></div> : <p className="fine-print">No application-time forecast has been saved yet. The running backend will record one at the next eligible live-bid slot when all three categories are available.</p>}
  </section>;
}

function PrelistingOutcomePanel({ scorecard }) {
  const summary = scorecard?.summary || {};
  const records = scorecard?.records || [];
  return <section className="panel">
    <div className="section-heading"><div><span className="eyebrow">FROZEN BEFORE LISTING · AFTER BIDDING</span><h3>Listing-outcome prediction audit</h3></div><span className="muted">Not an application-time score</span></div>
    <p className="fine-print">These probabilities were stored before listing, after the application window. New runs use only bids captured before 17:00 on the closing day; older post-close captures remain labelled separately. Listing outcomes are checked against the NSE opening, never used to revise the saved call. This is not an application-time accuracy test.</p>
    <div className="metric-grid four"><Metric label="Predictions frozen" value={summary.forecasts_locked ?? '—'} /><Metric label="Openings verified" value={summary.openings_verified ?? '—'} /><Metric label="Direction matched" value={pct(summary.direction_accuracy_pct)} note="Pending until exchange opens" /><Metric label="Sources" value={`${summary.nse_captures ?? 0} NSE / ${summary.third_party_captures ?? 0} other`} /></div>
    {records.length > 0 && <div className="table-wrap benchmark-table"><table><thead><tr><th>IPO</th><th>Prediction locked</th><th>Bid capture</th><th>Model call</th><th>Actual open</th><th>Model outcome</th></tr></thead><tbody>{records.map(row => <tr key={row.ipo_id}><td><strong>{cleanCompany(row.company_name)}</strong><small>Lists {formatDate(row.listing_date)}</small></td><td>{formatIpoTimestamp(row.recorded_at_ist)}</td><td>{formatIpoTimestamp(row.captured_at_ist)}<small>{row.source_family === 'nse_exchange' ? 'NSE' : 'Chittorgarh/InvestorGain'}</small></td><td>{intradayBand(row.positive_listing_probability)} · {Math.round(Number(row.positive_listing_probability) * 100)}%</td><td>{row.actual_open_return_pct == null ? 'Pending listing' : pct(row.actual_open_return_pct, true)}</td><td>{row.direction_correct == null ? 'Pending' : row.direction_correct ? 'Right' : 'Missed'}</td></tr>)}</tbody></table></div>}
  </section>;
}

function PrelistingReplayPanel({ scorecard }) {
  const summary = scorecard?.summary || {};
  const records = scorecard?.records || [];
  return <section className="panel">
    <div className="section-heading"><div><span className="eyebrow">LOCKED AFTER BIDDING · BEFORE LISTING</span><h3>Pre-listing replay scorecard</h3></div><span className="muted">Separate from application-time forecasts</span></div>
    <p className="fine-print">The 2021–24 model scored earlier bid readings after their IPOs closed, while listing outcomes were still unknown. App captures and later-recovered third-party excerpts are separate evidence groups. These replays cannot establish that an investor received a prediction while bidding was open.</p>
    <div className="metric-grid four"><Metric label="Replays locked" value={summary.replays_locked ?? '—'} /><Metric label="Openings verified" value={summary.openings_verified ?? '—'} /><Metric label="App-captured inputs" value={summary.app_captured_inputs ?? '—'} /><Metric label="Recovered · unverified" value={summary.unverified_recovered_inputs ?? '—'} /></div>
    {records.length > 0 && <div className="table-wrap benchmark-table"><table><thead><tr><th>IPO</th><th>Reading · IST</th><th>Evidence</th><th>Locked score</th><th>Actual open</th><th>Match</th></tr></thead><tbody>{records.map(row => <tr key={row.company_name}><td><strong>{cleanCompany(row.company_name)}</strong><small>Locked {formatIpoTimestamp(row.recorded_at_ist)}</small></td><td>{row.observed_at_ist ? formatIpoTimestamp(row.observed_at_ist) : `${formatDate(row.observed_on)} · time unverified`}</td><td>{row.evidence_status === 'app_capture_verified' ? 'App capture' : 'Recovered · unverified'}<small>{row.source_family}</small></td><td>{intradayBand(row.positive_listing_probability)} · {Math.round(Number(row.positive_listing_probability) * 100)}%</td><td>{row.actual_open_return_pct == null ? 'Pending listing' : pct(row.actual_open_return_pct, true)}</td><td>{row.direction_correct == null ? 'Pending' : row.direction_correct ? 'Right' : 'Missed'}</td></tr>)}</tbody></table></div>}
    {summary.openings_verified > 0 && <p className="fine-print">Direction matched: app-captured {pct(summary.direction_accuracy_app_captured_pct)}; recovered/unverified {pct(summary.direction_accuracy_unverified_pct)}. These are separate small-sample checks, not investor-time accuracy.</p>}
  </section>;
}

function BenchmarkPanel({ benchmark }) {
  if (!benchmark) return <LoadingBlock label="Loading the saved chronological model audit…" />;
  const summary = benchmark.summary || {}; const later = benchmark.later_year_summary || {};
  return <div className="benchmark-stack">
    <section className="benchmark-hero panel">
      <div><Pill tone="blue"><ShieldCheck size={13}/> CHRONOLOGICAL AUDIT</Pill><h2>Can the model flag a weak opening?</h2><p>{benchmark.methodology} The active model uses final QIB, retail, and HNI bids. Threshold: {pct(Number(benchmark.decision_threshold) * 100)}.</p></div>
      <div className="benchmark-score"><strong>{pct(summary.negative_recall_pct)}</strong><span>flat or discount listings flagged in 2023–24</span></div>
    </section>
    <div className="metric-grid four"><Metric label="IPOs checked" value={summary.sample_size ?? '—'} note={`${summary.nonpositive_listings ?? '—'} flat or discount opens`}/><Metric label="Apply precision" value={pct(summary.apply_precision_pct)} note="Positive opens among APPLY calls"/><Metric label="Warnings confirmed" value={pct(summary.warning_precision_pct)} note="Flat or discount opens among warnings"/><Metric label="Balanced accuracy" value={pct(summary.balanced_accuracy_pct)} note={`AUC ${num(summary.roc_auc, 3)}`} /></div>
    <section className="panel audit-note"><AlertTriangle size={18}/><div><strong>The trade-off is visible.</strong><p>{benchmark.diagnosis} In 2023–24 the model flagged {summary.tn} of {summary.nonpositive_listings} flat or discount openings, and also warned on {summary.fn} positive openings. Raw accuracy was {pct(summary.classifier_accuracy_pct)} versus {pct(summary.always_positive_baseline_pct)} for always saying positive.</p></div></section>
    <section className="panel audit-note"><AlertTriangle size={18}/><div><strong>Later-year retrospective check</strong><p>After fitting on 2021–24, the model flagged {later.tn} of {later.nonpositive_listings} flat or discount opens in 2025–26 ({pct(later.negative_recall_pct)}). Apply precision was {pct(later.apply_precision_pct)}. These years were examined during development, so this is not a blind or prospective test.</p></div></section>
    <section className="panel audit-note"><AlertTriangle size={18}/><div><strong>No made-up GMP in this test.</strong><p>{benchmark.gmp_note}</p></div></section>
    <section className="panel audit-note"><AlertTriangle size={18}/><div><strong>Data provenance limit</strong><p>{benchmark.data_quality?.note}</p></div></section>
    <section className="panel"><div className="section-heading"><div><span className="eyebrow">YEAR-BY-YEAR AUDIT</span><h3>Earlier years train each replay</h3></div><span className="muted">2022–23 set the threshold; later years are retrospective</span></div><div className="split-cards">{(benchmark.splits || []).map(split => <article key={split.test_year}><span>Test year {split.test_year}</span><strong>{pct(split.negative_recall_pct)} downside caught</strong><p>Trained through {split.trained_through} · {split.test_samples} IPOs</p><small>Apply precision {pct(split.apply_precision_pct)} · AUC {num(split.roc_auc, 3)} · {split.role?.replaceAll('_', ' ')}</small></article>)}</div></section>
    <section className="panel"><div className="section-heading"><div><span className="eyebrow">PREDICTION LEDGER</span><h3>Score versus exchange opening</h3></div><span className="muted">A warning is not a certain loss</span></div><div className="table-wrap benchmark-table"><table><thead><tr><th>IPO</th><th>Listed</th><th>Issue</th><th>Actual open</th><th>Actual return</th><th>Model score</th><th>Risk call</th><th>Model outcome</th></tr></thead><tbody>{(benchmark.records || []).map(row => <tr key={row.ipo_id}><td><strong>{cleanCompany(row.company_name)}</strong><small>{row.nse_symbol || `IPO #${row.ipo_id}`}</small></td><td>{formatDate(row.listing_date)}</td><td>{money(row.issue_price, 2)}</td><td>{money(row.actual_open_price, 2)}</td><td className={row.actual_open_return_pct > 0 ? 'text-positive' : 'text-negative'}>{pct(row.actual_open_return_pct, true)}</td><td>{pct(row.positive_probability_pct)}</td><td><Pill tone={row.risk_call === 'APPLY' ? 'positive' : 'warning'}>{row.risk_call}</Pill></td><td><Pill tone={row.direction_correct ? 'positive' : 'negative'}>{row.actual_result} · {row.direction_correct ? 'matched' : 'missed'}</Pill></td></tr>)}</tbody></table></div></section>
  </div>;
}

function AutomaticDecisionHistory({ history, onRefresh, offerEnded = false }) {
  if (!history) return null;
  if (history.load_error) return <section className="panel"><div className="warning-banner"><AlertTriangle size={18}/><div><strong>Forecast log unavailable</strong><span>Saved predictions could not be checked. The model card must not be interpreted as proof that none were recorded.</span></div></div><button className="secondary-button" onClick={onRefresh}><RefreshCw size={15}/> Retry forecast log</button></section>;
  const rows = history.decisions || [];
  const firstPartyCaptures = history.pre_deadline_captures || [];
  const capturedBidReplays = history.captured_bid_replays || [];
  const recovered = history.recovered_observations || [];
  const replay = history.prelisting_replay;
  const outcomeForecast = history.prelisting_outcome_forecast;
  const hasSavedForecast = rows.some(row => row.decision_kind === 'live_bid_forecast');
  const forecastRows = history.primary_forecast_history || rows.filter(row => row.decision_kind === 'live_bid_forecast');
  const HistoryContainer = history.historical_bid_forecast && !hasSavedForecast ? 'details' : 'section';
  return <HistoryContainer className="panel research-history">
    {HistoryContainer === 'details' && <summary>Original app collection history</summary>}
    <div className="section-heading"><div><span className="eyebrow">PREDICTION HISTORY</span><h3>Calls recorded before applications closed</h3></div><button className="secondary-button" onClick={onRefresh}><RefreshCw size={15}/> Refresh log</button></div>
    <p className="fine-print">Scheduled polls and Run analysis save each prediction while the offer is open. The latest call before 17:00 IST on the closing date is preserved and checked against the listing opening.</p>
    {forecastRows.length ? <div className="table-wrap"><table><thead><tr><th>Prediction recorded</th><th>Bids as of</th><th>QIB</th><th>HNI</th><th>Retail</th><th>Positive-opening estimate</th><th>Central return</th></tr></thead><tbody>{forecastRows.map((row, index) => {
      const view = forecastView({application_forecast: row});
      const rf = row.inputs?.return_forecast;
      return <tr key={`${row.slot_key}-${index}`}><td>{formatIpoTimestamp(row.recorded_at_ist)}</td><td>{formatIpoTimestamp(row.source_asof_ist)}</td><td>{bid(row.inputs?.qib_subscription)}</td><td>{bid(row.inputs?.hni_subscription)}</td><td>{bid(row.inputs?.retail_subscription)}</td><td>{view?.band} · {view ? Math.round(view.probability * 100) + '%' : '—'}<small>{rf && !rf.outside_training_features?.length ? 'Early-book challenger' : 'Legacy direction model'}</small></td><td>{rf?.outside_training_features?.length ? 'Outside model range' : rf ? pct(rf.central_return_pct, true) : 'Not recorded'}{rf && <small>Experimental{rf.outside_training_features?.length ? ' · outside training support' : ''}</small>}</td></tr>;
    })}</tbody></table></div> : <p className="fine-print">{offerEnded ? 'This offer has no saved application-time calls.' : 'The frozen-model estimate is shown above. Its provisional and complete updates are stored in the separate frozen forecast ledger.'}</p>}
    <details className="research-history"><summary>Archived bid captures and later research</summary>
    {outcomeForecast && <div className="warning-banner"><AlertTriangle size={18}/><div><strong>Listing-outcome prediction frozen {formatIpoTimestamp(outcomeForecast.recorded_at_ist)} · {intradayBand(outcomeForecast.positive_listing_probability)} ({Math.round(Number(outcomeForecast.positive_listing_probability) * 100)}%)</strong><span>Archived {outcomeForecast.source_family === 'nse_exchange' ? 'NSE' : 'Chittorgarh/InvestorGain'} capture {formatIpoTimestamp(outcomeForecast.captured_at_ist)} · QIB {bid(outcomeForecast.qib_x)}, HNI {bid(outcomeForecast.nii_x)}, retail {bid(outcomeForecast.retail_x)} · evidence {outcomeForecast.evidence_sha256?.slice(0, 12)} · model {outcomeForecast.model_version}. Frozen before listing, after applications closed. This can be checked against the future opening, but does not measure application-time accuracy.</span></div></div>}
    {replay && <div className="warning-banner"><AlertTriangle size={18}/><div><strong>Pre-listing replay locked {formatIpoTimestamp(replay.recorded_at_ist)} · {intradayBand(replay.positive_listing_probability)} ({Math.round(Number(replay.positive_listing_probability) * 100)}%)</strong><span>Inputs: {bid(replay.qib_x)} QIB, {bid(replay.nii_x)} HNI, {bid(replay.retail_x)} retail · reported {formatIpoTimestamp(replay.observed_at_ist)} · {replay.source_family} · model {replay.model_version}. {replay.evidence_status === 'app_capture_verified' ? `App-captured source evidence ${replay.evidence_sha256?.slice(0, 12) || 'hash unavailable'}.` : `Third-party excerpt found ${formatIpoTimestamp(replay.evidence_found_at_ist)}; original reading unverified.`} This was scored before listing, but after applications closed. It does not count as an investor-time prediction.</span></div></div>}
    {offerEnded && !hasSavedForecast && firstPartyCaptures.length === 0 && recovered.length === 0 && <div className="warning-banner"><AlertTriangle size={18}/><div><strong>No pre-deadline evidence or saved forecast for this IPO</strong><span>Later bid totals cannot establish what an investor could see before applications closed. This IPO is excluded from prospective forecast accuracy.</span></div></div>}
    {firstPartyCaptures.length > 0 && <div className="table-wrap"><p className="fine-print">Actual pre-deadline captures made by this app. If there is no saved forecast above, these bids were recorded but the forecast was not run at that time; a score calculated now would be retrospective.</p><table><thead><tr><th>App captured · IST</th><th>Source</th><th>QIB</th><th>HNI</th><th>Retail</th></tr></thead><tbody>{firstPartyCaptures.slice(0, 5).map((item, index) => <tr key={`${item.slot_key}-${index}`}><td>{formatIpoTimestamp(item.fetched_at_ist || item.attempted_at_ist)}</td><td>{item.source_family}<small>Raw evidence {item.raw_sha256 ? item.raw_sha256.slice(0, 12) : 'hash unavailable'}</small></td><td>{bid(item.qib_x)}</td><td>{bid(item.nii_x)}</td><td>{bid(item.retail_x)}</td></tr>)}</tbody></table></div>}
    {capturedBidReplays.length > 0 && <div className="table-wrap"><p className="fine-print">Daily backfill from the app’s archived pre-17:00 bid captures. Recalculated {formatIpoTimestamp(capturedBidReplays[0].computed_at_ist)}, not issued on the capture dates; excluded from investor-time accuracy.</p><table><thead><tr><th>Bid date</th><th>Observed bids</th><th>Reconstructed score</th><th>Source</th></tr></thead><tbody>{capturedBidReplays.map(row => <tr key={row.capture_day_ist}><td>{formatDate(row.capture_day_ist)}<small>Captured {formatIpoTimestamp(row.captured_at_ist)}</small></td><td>QIB {bid(row.qib_x)} · HNI {bid(row.nii_x)} · retail {bid(row.retail_x)}</td><td>{intradayBand(row.positive_listing_probability)} · {Math.round(Number(row.positive_listing_probability) * 100)}%</td><td>{row.source_family}<small>Evidence {row.evidence_sha256?.slice(0, 12)}</small></td></tr>)}</tbody></table></div>}
    {recovered.length > 0 && <div className="warning-banner"><AlertTriangle size={18}/><div>
      <strong>Earlier bid readings found later — not saved predictions</strong>
      <span>Third-party search-index reports contain these source-reported times, but our app recovered them after the offer closed. The original pages have since changed. They are unverified research context and are excluded from prospective accuracy.</span>
      <div className="table-wrap"><table><thead><tr><th>Source-reported · IST</th><th>Source</th><th>QIB</th><th>HNI</th><th>Retail</th></tr></thead><tbody>{recovered.map((item, index) => <tr key={`${item.source_name}-${index}`}><td>{formatIpoTimestamp(item.source_reported_at_ist)}</td><td><a href={item.source_url} target="_blank" rel="noreferrer">{item.source_name}</a><small>Discovered {formatIpoTimestamp(item.discovered_at_ist)}</small></td><td>{bid(item.qib_x)}</td><td>{bid(item.nii_x)}</td><td>{bid(item.retail_x)}</td></tr>)}</tbody></table></div>
    </div></div>}
    {rows.length ? <div className="table-wrap"><table><thead><tr><th>Capture slot · IST</th><th>Source updated</th><th>QIB</th><th>HNI</th><th>Retail</th><th>Assessment</th></tr></thead><tbody>{rows.slice(0, 12).map((row, index) => <tr key={`${row.slot_key}-${index}`}><td>{row.slot_key?.replace('_', ' ')}</td><td>{row.source_asof_ist ? new Date(row.source_asof_ist).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) : 'Not provided'}<small>{row.source_family || 'No verified source'}</small></td><td>{bid(row.inputs?.qib_subscription)}</td><td>{bid(row.inputs?.hni_subscription)}</td><td>{bid(row.inputs?.retail_subscription)}</td><td>{row.decision_kind === 'live_bid_forecast' ? intradayBand(row.positive_listing_probability) + ' · ' + Math.round(Number(row.positive_listing_probability) * 100) + '% saved' : row.decision_kind === 'reconciliation_only' ? 'Post-close check only' : row.error ? 'Assessment failed' : row.verdict || 'Awaiting inputs'}<small>{row.decision_kind === 'live_bid_forecast' ? 'Experimental · ' + row.model_version : row.decision_kind === 'provisional_scorecard' ? 'Context only' : 'Not investor-time evidence'}</small></td></tr>)}</tbody></table></div> : <p className="fine-print">{offerEnded ? 'No scheduled assessment was saved for this IPO while its offer was open.' : 'No scheduled assessment has been captured for this IPO yet. The collector records one per available polling slot while the backend runs.'}</p>}
    </details>
  </HistoryContainer>;
}

function Module2Detail({result,events}) {
  return <Module2Overview result={result} events={events}/>;
}

export default function Dashboard() {
  const [active, setActive] = useState('today'); const [mobileNav, setMobileNav] = useState(false);
  const [intradayScorecard, setIntradayScorecard] = useState(null);
  const [prelistingScorecard, setPrelistingScorecard] = useState(null);
  const [prelistingOutcomeScorecard, setPrelistingOutcomeScorecard] = useState(null);
  const [priorities, setPriorities] = useState(null);
  const [returnReport, setReturnReport] = useState(null);
  const [performanceEvidence, setPerformanceEvidence] = useState(null);
  const [health, setHealth] = useState(null); const [macro, setMacro] = useState(null); const [calendar, setCalendar] = useState({ calendar: [], recently_listed: [] }); const [canonical, setCanonical] = useState([]); const [benchmark, setBenchmark] = useState(null);
  const [loading, setLoading] = useState(true); const [refreshing, setRefreshing] = useState(false); const [error, setError] = useState(''); const [filter, setFilter] = useState('ALL'); const [marketSearch, setMarketSearch] = useState('');
  const [marketLimit,setMarketLimit]=useState(12);
  const [researchQuery, setResearchQuery] = useState(''); const [evaluation, setEvaluation] = useState(null); const [evaluating, setEvaluating] = useState(false); const [decisionHistory, setDecisionHistory] = useState(null); const [investment, setInvestment] = useState(200000); const [watchlist, setWatchlist] = useState([]);
  const [compareTargets, setCompareTargets] = useState([]); const [compareResults, setCompareResults] = useState([]); const [comparing, setComparing] = useState(false);
  const [portfolio, setPortfolio] = useState([]); const [portfolioResults, setPortfolioResults] = useState({}); const [holdingForm, setHoldingForm] = useState({ company: '', quantity: '', price: '' });
  const monitorSelection = useRef(null);
  const [monitorQuery, setMonitorQuery] = useState(''); const [monitorResult, setMonitorResult] = useState(null); const [monitorLoading, setMonitorLoading] = useState(false);
  const [monitorEvents, setMonitorEvents] = useState([]);
  const marketRequest = useRef(0);
  const marketInFlight = useRef(false);
  const researchRequest = useRef(0);
  const monitorRequest = useRef(0);
  const loadMarket = async (force = false, background = false) => {
    if (marketInFlight.current) return;
    marketInFlight.current = true;
    const requestId = ++marketRequest.current;
    if (!background) force ? setRefreshing(true) : setLoading(true);
    setError('');
    const failures = [];
    const tasks = [
      ['service status', api.health, setHealth],
      ['market context', api.macro, setMacro],
      ['IPO calendar', () => api.calendar(force), value => { setCalendar(value); setLoading(false); }],
      ['company search', api.ipos, value => setCanonical(value.ipos || [])],
    ];
    await Promise.allSettled(tasks.map(async ([label, fetchData, apply]) => {
      try {
        const value = await fetchData();
        if (marketRequest.current === requestId) apply(value);
      } catch (_) { failures.push(label); }
    }));
    marketInFlight.current = false;
    if (marketRequest.current !== requestId) return;
    if (failures.length) setError('Could not refresh: ' + failures.join(', ') + '. Previously loaded data is retained.');
    setLoading(false); setRefreshing(false);
  };
  useEffect(() => {
    if (calendar.source_refresh?.status !== 'refreshing') return;
    let cancelled=false,inflight=false,attempts=0;
    const timer=setInterval(async()=>{
      if(inflight||document.hidden)return;
      inflight=true;
      try {
        const next=await api.calendar();
        if(!cancelled)setCalendar(next);
        if(next.source_refresh?.status!=='refreshing'||++attempts>=40)clearInterval(timer);
      } catch (_) { if(++attempts>=40)clearInterval(timer); }
      finally {inflight=false;}
    },1500);
    return()=>{cancelled=true;clearInterval(timer);};
  },[calendar.source_refresh?.status]);
  useEffect(() => {
    const refresh = () => { if (!document.hidden) loadMarket(false, true); };
    const timer = setInterval(refresh, 120000);
    window.addEventListener('focus', refresh);
    return () => { clearInterval(timer); window.removeEventListener('focus', refresh); };
  }, []);
  useEffect(() => {
    if (active === 'performance') api.benchmark().then(setBenchmark).catch(() => {});
    if (active === 'performance') api.performanceEvidence().then(setPerformanceEvidence).catch(error => setPerformanceEvidence({status:'unavailable',message:error.message}));
    if (active === 'performance') api.returnBenchmark().then(setReturnReport).catch(() => {});
  }, [active]);
  useEffect(() => { loadMarket(); try { setWatchlist(JSON.parse(localStorage.getItem('ipo-watchlist') || '[]')); setPortfolio(JSON.parse(localStorage.getItem('ipo-portfolio') || '[]')); } catch (_) {} }, []);
  useEffect(() => {
    const company=evaluation?.scraped_data?.company_name;
    if(active!=='research'||!company)return;
    let disposed=false,inflight=false;
    const refresh=async()=>{
      if(document.hidden||inflight)return;
      const requestId=researchRequest.current;inflight=true;
      try{const next=await api.evaluate(company,false,false);if(!disposed&&requestId===researchRequest.current)setEvaluation(next);}catch(_){}finally{inflight=false;}
    };
    const timer=setInterval(refresh,60000);
    window.addEventListener('focus',refresh);
    return()=>{disposed=true;clearInterval(timer);window.removeEventListener('focus',refresh);};
  },[active,evaluation?.scraped_data?.company_name]);
  useEffect(() => {
    const company=monitorResult?.company_name;
    if(active!=='portfolio'||!company)return;
    let disposed=false,inflight=false;
    const refresh=async()=>{
      if(document.hidden||inflight)return;
      const requestId=monitorRequest.current;inflight=true;
      try{const next=await api.holding(company);if(!disposed&&requestId===monitorRequest.current)setMonitorResult(next);}catch(_){}finally{inflight=false;}
    };
    const timer=setInterval(refresh,300000);
    window.addEventListener('focus',refresh);
    return()=>{disposed=true;clearInterval(timer);window.removeEventListener('focus',refresh);};
  },[active,monitorResult?.company_name]);
  useEffect(() => { if (active === 'performance') api.intradayForecastScorecard().then(setIntradayScorecard).catch(() => setIntradayScorecard(null)); }, [active]);
  useEffect(() => { if (active === 'performance') api.prelistingReplayScorecard().then(setPrelistingScorecard).catch(() => setPrelistingScorecard(null)); }, [active]);
  useEffect(() => { if (active === 'performance') api.prelistingOutcomeScorecard().then(setPrelistingOutcomeScorecard).catch(() => setPrelistingOutcomeScorecard(null)); }, [active]);
  useEffect(() => {
    const company = evaluation?.scraped_data?.company_name;
    if (active !== 'research' || !company) return;
    let disposed = false;
    const timer = setInterval(() => {
      api.automaticDecisions(company).then(history => { if (!disposed) setDecisionHistory(history); }).catch(() => {});
    }, 60000);
    return () => { disposed = true; clearInterval(timer); };
  }, [active, evaluation?.scraped_data?.company_name]);
  useEffect(() => { window.scrollTo(0, 0); }, [active]);
  useEffect(() => { localStorage.setItem('ipo-watchlist', JSON.stringify(watchlist)); }, [watchlist]); useEffect(() => { localStorage.setItem('ipo-portfolio', JSON.stringify(portfolio)); }, [portfolio]);
  const allIpos = useMemo(() => { const map = new Map(); [...(calendar.calendar || []), ...(calendar.recently_listed || [])].forEach(item => { const key = item.url || item.company_name; if (!map.has(key)) map.set(key, item); }); return [...map.values()]; }, [calendar]);
  const searchUniverse = useMemo(() => { const map = new Map(); [...allIpos, ...canonical].forEach(item => { const key = item.nse_symbol || item.url || item.company_name; if (!map.has(key)) map.set(key, item); }); return [...map.values()]; }, [allIpos, canonical]);
  const visibleIpos = useMemo(() => allIpos.filter(item => (filter === 'ALL' || item.category === filter) && `${item.company_name} ${item.sector}`.toLowerCase().includes(marketSearch.toLowerCase())), [allIpos, filter, marketSearch]);
  const counts = useMemo(() => ({ OPEN: allIpos.filter(i => i.category === 'OPEN').length, UPCOMING: allIpos.filter(i => i.category === 'UPCOMING').length, CLOSED: allIpos.filter(i => i.category === 'CLOSED').length, LISTED: allIpos.filter(i => i.category === 'LISTED').length }), [allIpos]);
  const evaluate = async target => {
    const value = typeof target === 'string' ? target.trim() : target.url || target.company_name;
    if (!value) return;
    const requestId = ++researchRequest.current;
    setResearchQuery(typeof target === 'string' ? target : target.company_name);
    setActive('research'); setEvaluating(true); setEvaluation(null); setDecisionHistory(null); setError('');
    try {
      const result = await api.evaluate(value, true);
      if (requestId !== researchRequest.current) return;
      setEvaluation(result); setEvaluating(false);
      const history = result.scraped_data?.company_name
        ? await api.automaticDecisions(result.scraped_data.company_name).catch(() => ({ load_error: true }))
        : { load_error: true };
      if (requestId === researchRequest.current) setDecisionHistory(history);
    } catch (e) { if (requestId === researchRequest.current) setError(e.message); }
    finally { if (requestId === researchRequest.current) setEvaluating(false); }
  };
  const toggleWatch = ipo => { const name = ipo.company_name || ipo.scraped_data?.company_name; setWatchlist(p => p.some(x => x.company_name === name) ? p.filter(x => x.company_name !== name) : [...p, { company_name: name, url: ipo.url, category: ipo.category, issue_price: ipo.issue_price }]); };
  const runComparison = async () => {
    if (compareTargets.length < 2) return;
    setComparing(true); setCompareResults([]);
    const results = await Promise.all(compareTargets.map(async t => {
      try { return await api.evaluate(t.url || t.company_name, false); }
      catch (e) { return { error: e.message, scraped_data: { company_name: t.company_name } }; }
    }));
    setCompareResults(results);
    setComparing(false);
  };
  const addHolding = async e => { e.preventDefault(); if (!holdingForm.company || !holdingForm.quantity || !holdingForm.price) return; const h = { ...holdingForm, id: Date.now(), quantity: Number(holdingForm.quantity), price: Number(holdingForm.price) }; setPortfolio(p => [...p, h]); setHoldingForm({ company: '', quantity: '', price: '' }); try { const result = await api.holding(h.company); setPortfolioResults(p => ({ ...p, [h.id]: result })); } catch (_) {} };
  const refreshHolding = async h => { setPortfolioResults(p => ({ ...p, [h.id]: { loading: true } })); try { const result = await api.holding(h.company); setPortfolioResults(p => ({ ...p, [h.id]: result })); } catch (e) { setPortfolioResults(p => ({ ...p, [h.id]: { error: e.message } })); } };
  const runMonitor = async target => {
    const chosen=target || monitorSelection.current?.id || monitorSelection.current?.ipo_id || monitorSelection.current?.url || monitorQuery;
    const value = String(chosen || '').trim(); if (!value) return;
    const requestId = ++monitorRequest.current;
    setMonitorLoading(true); setMonitorResult(null); setMonitorEvents([]);
    try {
      const result = await api.holding(value);
      if (requestId !== monitorRequest.current) return;
      setMonitorResult(result);
      if (result?.ipo_id) {
        const history = await api.module2Events(result.ipo_id).catch(() => ({ events: [] }));
        if (requestId === monitorRequest.current) setMonitorEvents(history.events || []);
      }
    } catch (e) { if (requestId === monitorRequest.current) setMonitorResult({ error: e.message }); }
    finally { if (requestId === monitorRequest.current) setMonitorLoading(false); }
  };

  return <div className="app-shell">
    <header className="topbar"><div className="brand" onClick={() => setActive('today')}><div className="brand-mark"><TrendingUp size={20} /></div><div><strong>OpenBell</strong><span>IPO intelligence</span></div></div><nav className={mobileNav ? 'nav-open' : ''}>{nav.map(item => { const Icon = item.icon; return <button key={item.id} className={active === item.id ? 'active' : ''} onClick={() => { setActive(item.id); setMobileNav(false); }}><Icon size={17} />{item.label}</button>; })}</nav><div className="topbar-actions"><Freshness health={health} macro={macro} polling={refreshing} /><button className="icon-button" aria-label="Refresh market data" title="Refresh market data" onClick={() => loadMarket(true)} disabled={refreshing}><RefreshCw size={17} className={refreshing ? 'spin' : ''} /></button><button className="menu-button" aria-label="Navigation menu" aria-expanded={mobileNav} onClick={() => setMobileNav(!mobileNav)}><Menu size={21} /></button></div></header>
    <main>{error && <div className="global-alert"><AlertTriangle size={17} />{error}<button onClick={() => setError('')}><X size={15} /></button></div>}
      {health?.data_maintenance?.status==='updates_delayed'&&<div className="global-alert" role="status"><Clock3 size={17}/><span>Some source data is delayed. Updates are retrying automatically; previously verified observations are retained.</span></div>}
      {active === 'today' && <div className="page-stack">
        <section className="hero-grid"><div className="hero-copy"><Pill tone="blue"><TrendingUp size={13} /> Indian primary markets</Pill><h1>Know what deserves<br /><span>your capital today.</span></h1><p>Live IPO signals, transparent model evidence, and post-listing risk monitoring in one disciplined workflow.</p><div className="hero-search"><Search size={19} /><SearchDropdown value={researchQuery} onChange={setResearchQuery} items={searchUniverse} onSelect={item => setResearchQuery(item.company_name)} placeholder="Search company, NSE symbol, BSE code, or source URL" /><button onClick={() => evaluate(researchQuery)}>Research <ArrowRight size={17} /></button></div><p className="search-hint">Select a verified record from the dropdown to confirm the company, exchange symbol, sector, and issue price.</p></div>
          <div className="market-card"><div className="market-card-head"><div><span className="eyebrow">MARKET REGIME</span><h2>{macro?.regime_label || 'Connecting…'}</h2></div><div className="regime-icon"><Gauge size={27} /></div></div>
            <div className="market-metrics"><Metric label="India VIX" value={num(macro?.india_vix, 2)} note={macro?.is_fallback || macro?.is_market_session_current === false ? `Last NSE close · ${formatIpoTimestamp(macro?.timestamp)}` : `NSE spot · ${pct(macro?.vix_change_pct, true)} this session`} /><Metric label="Nifty 10D" value={pct(macro?.nifty_10d_return, true)} note={macro?.history_source === 'yahoo_finance' ? 'Yahoo Finance daily history' : macro?.history_source === 'last_known_cache' ? 'Last recorded daily history' : 'Daily history unavailable'} tone={macro?.nifty_10d_return == null ? 'neutral' : Number(macro.nifty_10d_return) >= 0 ? 'positive' : 'negative'} /></div>
            {macro?.is_fallback && <div className="source-warning"><AlertTriangle size={15} /> NSE could not be reached. Showing a last recorded reading, not a live value.</div>}<div className="market-footer"><ShieldCheck size={16} /><span>{macro?.source === 'nse_india' ? 'NSE spot values; 10-day history sourced separately' : `${health?.live_data_refresh?.items_synced || calendar.cached_total || 0} IPO records checked at startup`}</span></div></div></section>
        <section className="module-grid"><button onClick={() => setActive('research')}><span className="module-number">MODULE 1</span><div className="module-icon"><Search size={22} /></div><div><h3>Pre-IPO Research & Apply</h3><p>Check offer terms, dated bid evidence, model status, and capital risk before bidding.</p></div><ArrowRight size={19} /></button><button onClick={() => setActive('portfolio')}><span className="module-number">MODULE 2</span><div className="module-icon module-icon-blue"><BriefcaseBusiness size={22} /></div><div><h3>Post-Listing Hold / Sell Monitor</h3><p>Track your entry, live P&amp;L, sector-relative strength, lock-ins, and exit signals.</p></div><ArrowRight size={19} /></button></section>
        <section className="pulse-grid"><div><span className="pulse-icon green"><Bell size={18} /></span><strong>{counts.OPEN}</strong><p>Open for bidding</p></div><div><span className="pulse-icon blue"><CalendarDays size={18} /></span><strong>{counts.UPCOMING}</strong><p>Upcoming IPOs</p></div><div><span className="pulse-icon amber"><Clock3 size={18} /></span><strong>{counts.CLOSED}</strong><p>Awaiting listing</p></div><div><span className="pulse-icon violet"><BarChart3 size={18} /></span><strong>{counts.LISTED}</strong><p>Recently listed</p></div></section>
        <section><div className="section-heading market-heading"><div><span className="eyebrow">OPPORTUNITY BOARD</span><h2>Primary market pipeline</h2><p>Discover, filter, and investigate before committing capital.</p></div><div className="small-search"><Search size={16} /><input value={marketSearch} onChange={e => {setMarketSearch(e.target.value);setMarketLimit(12);}} placeholder="Filter IPOs" /></div></div><div className="filter-row">{['ALL', 'OPEN', 'UPCOMING', 'CLOSED', 'LISTED'].map(item => <button className={filter === item ? 'active' : ''} onClick={() => {setFilter(item);setMarketLimit(12);}} key={item}>{item === 'ALL' ? 'Latest IPOs' : item}<span>{item === 'ALL' ? allIpos.length : counts[item]}</span></button>)}</div>{loading ? <LoadingBlock /> : visibleIpos.length ? <div className="ipo-grid">{visibleIpos.slice(0, marketLimit).map(ipo => <IpoCard key={ipo.url || ipo.company_name} ipo={ipo} onResearch={evaluate} watching={watchlist.some(w => w.company_name === ipo.company_name)} onWatch={toggleWatch} />)}</div> : <EmptyState title="No IPOs match this view" copy="Try a different status or search term." />}{!loading&&visibleIpos.length>marketLimit&&<div className="market-more"><button className="secondary-button" onClick={()=>setMarketLimit(n=>n+12)}>Show more IPOs · {marketLimit} of {visibleIpos.length}</button></div>}{calendar.source_refresh?.status==='refreshing'&&<p className="fine-print" role="status">Showing saved observations while sources refresh in the background.</p>}</section>
        <section className="watch-strip"><div><span className="eyebrow">YOUR WATCHLIST</span><h3>{watchlist.length ? `${watchlist.length} IPO${watchlist.length > 1 ? 's' : ''} under watch` : 'Build your shortlist'}</h3><p>{watchlist.length ? 'Your list is stored privately in this browser.' : 'Save interesting IPOs and return without searching again.'}</p></div><div className="watch-items">{watchlist.slice(0, 4).map(item => <button key={item.company_name} onClick={() => evaluate(item.url || item.company_name)}>{cleanCompany(item.company_name)}<ArrowUpRight size={15} /></button>)}{!watchlist.length && <span className="muted">Use the eye icon on any IPO card.</span>}</div></section>
      </div>}
      {active === 'research' && <div className="page-stack narrow-page"><section className="page-intro"><div><Pill tone="blue">MODULE 1 · PRE-IPO DECISION LAB</Pill><h1>Research before you apply.</h1><p>Select a verified IPO record, then review dated bid evidence, offer terms, model status, and capital risk.</p></div></section><form className="research-bar" onSubmit={e => { e.preventDefault(); evaluate(researchQuery); }}><Search size={20} /><SearchDropdown value={researchQuery} onChange={setResearchQuery} items={searchUniverse} onSelect={item => setResearchQuery(item.company_name)} placeholder="Search company, NSE symbol, BSE code, or source URL" /><button disabled={evaluating || !researchQuery.trim()}>{evaluating ? <LoaderCircle className="spin" size={18} /> : <BarChart3 size={18} />} Run analysis</button></form>{!evaluation && !evaluating && <div className="research-starters"><span>Research an active IPO:</span>{allIpos.filter(i => i.category === 'OPEN').slice(0, 4).map(i => <button key={i.company_name} onClick={() => evaluate(i)}>{cleanCompany(i.company_name)}</button>)}</div>}{evaluating && <LoadingBlock label="Resolving sources and checking the model’s evidence stage…" />}{!evaluation && !evaluating && <EmptyState icon={Target} title="Start with a company you are considering" copy="Choose from the verified dropdown so you can confirm its symbol, sector, issue price, and listing date before analysis." />}<DecisionPanel result={evaluation} history={decisionHistory} investment={investment} setInvestment={setInvestment} onTrack={() => toggleWatch(evaluation.scraped_data)} onOpenMonitor={company => { setMonitorQuery(company); setActive('portfolio'); runMonitor(company); }} /><ForecastLog historical={evaluation?.frozen_forecast?.historical_backtest?.mode==='historical_benchmark'&&!evaluation?.frozen_forecast?.saved_call_found}><AutomaticDecisionHistory history={decisionHistory} offerEnded={['CLOSED', 'LISTED'].includes(evaluation?.scraped_data?.category) || evaluation?.scraped_data?.bidding_ended === true} onRefresh={() => evaluation?.scraped_data?.company_name && api.automaticDecisions(evaluation.scraped_data.company_name).then(setDecisionHistory).catch(() => setDecisionHistory({ load_error: true }))} /></ForecastLog><Copilot result={evaluation} /></div>}
      {active === 'performance' && <div className="narrow-page"><PerformanceDashboard><details className="research-history"><summary>Legacy models and archived benchmark reports</summary><details><summary>Legacy subscription-model saved calls</summary><PredictionResults/></details><PerformanceReview evidence={performanceEvidence} returnReport={returnReport} onRetry={() => api.performanceEvidence().then(setPerformanceEvidence).catch(error => setPerformanceEvidence({status:'unavailable',message:error.message}))}><IntradayForecastPanel scorecard={intradayScorecard} onRefresh={() => api.intradayForecastScorecard().then(setIntradayScorecard).catch(() => setIntradayScorecard(null))} /><PrelistingOutcomePanel scorecard={prelistingOutcomeScorecard} /><PrelistingReplayPanel scorecard={prelistingScorecard} /><BenchmarkPanel benchmark={benchmark}/></PerformanceReview></details></PerformanceDashboard></div>}
      {active === 'compare' && <CompareIpos ipos={allIpos} targets={compareTargets} setTargets={setCompareTargets} results={compareResults} running={comparing} onRun={runComparison} onResearch={evaluate}/>}
      {active === 'portfolio' && <div className="page-stack narrow-page"><section className="page-intro"><div><Pill tone="blue">MODULE 2 · HOLD / SELL MONITOR</Pill><h1>Your post-listing decision desk.</h1><p>See the current assessment, understand the reason, and check our saved calls against actual prices.</p></div></section>
        <form className="research-bar monitor-search" onSubmit={e => { e.preventDefault(); runMonitor(); }}><Search size={20}/><SearchDropdown value={monitorQuery} onChange={value=>{monitorSelection.current=null;setMonitorQuery(value);}} items={searchUniverse} listedOnly onSelect={item => { monitorSelection.current=item;setMonitorQuery(item.company_name);runMonitor(String(item.id||item.ipo_id||item.url||item.company_name)); }} placeholder="Search listed company, NSE symbol, or BSE code"/><button disabled={monitorLoading || !monitorQuery.trim()}>{monitorLoading ? <LoaderCircle className="spin" size={18}/> : <Activity size={18}/>} Analyze stock</button></form>
        {monitorLoading && <LoadingBlock label="Loading current price, anchor unlocks, and relative performance…"/>}<Module2Detail result={monitorResult} events={monitorEvents}/>
        <details className="panel m2-disclosure"><summary><div><span className="eyebrow">OPTIONAL</span><strong>Track my position and P&amp;L</strong></div></summary><div className="m2-evidence-body"><div className="portfolio-divider"><div><span className="eyebrow">YOUR POSITION</span><h2>Track personal P&amp;L</h2><p>Optional: save quantity and average price privately in this browser.</p></div></div>
        <form className="holding-form panel" onSubmit={addHolding}><div><label>Listed company or ticker</label><SearchDropdown value={holdingForm.company} onChange={company => setHoldingForm({ ...holdingForm, company })} items={searchUniverse} listedOnly onSelect={item => setHoldingForm({ ...holdingForm, company: item.company_name })} placeholder="Search company, NSE symbol, or BSE code" /></div><div><label>Quantity</label><input type="number" min="1" value={holdingForm.quantity} onChange={e => setHoldingForm({ ...holdingForm, quantity: e.target.value })} placeholder="100" /></div><div><label>Average price</label><input type="number" min="0" step="0.01" value={holdingForm.price} onChange={e => setHoldingForm({ ...holdingForm, price: e.target.value })} placeholder="119.00" /></div><button className="primary-button"><Plus size={17} /> Add &amp; analyze</button></form>{portfolio.length ? <div className="portfolio-list">{portfolio.map(h => { const result = portfolioResults[h.id]; const current = result?.current_price_adjusted; const valuation = holdingValue(h, result); const pnl = valuation.pnl; const ret = valuation.returnPct; return <article className="holding-card" key={h.id}><div className="holding-head"><div><h3>{h.company}</h3><p>{h.quantity} shares · Avg. {money(h.price, 2)}</p></div><div className="holding-actions"><button onClick={() => refreshHolding(h)}><RefreshCw size={16} className={result?.loading ? 'spin' : ''} /></button><button onClick={() => setPortfolio(p => p.filter(x => x.id !== h.id))}><Trash2 size={16} /></button></div></div>{!result && <button className="load-holding" onClick={() => refreshHolding(h)}>Load live position analysis <ArrowRight size={15} /></button>}{result?.error && <p className="text-negative">{result.error}</p>}{result?.status === "PRICE_BASIS_UNVERIFIED" && <p className="text-warning">{result.message}</p>}{current && <>{valuation.needsConfirmation && <div className="warning-banner"><div><strong>Confirm your split/bonus-adjusted holding</strong><p>The share count changed by {valuation.factor}× since this IPO. Update your quantity and average price if needed, then confirm both already include those changes. P&amp;L is paused to avoid a false loss.</p><button className="secondary-button" onClick={() => setPortfolio(p => p.map(x => x.id === h.id ? { ...x, confirmedShareFactor: valuation.factor } : x))}>My quantity and average already include these changes</button></div></div>}<div className="holding-metrics"><Metric label="Current price" value={money(current, 2)} /><Metric label="Unrealized P&L" value={money(pnl)} note={pct(ret, true)} tone={pnl == null ? undefined : pnl >= 0 ? 'positive' : 'negative'} /><Metric label="Vs sector · 30D" value={pct(result.attribution_30d?.spread_vs_sector_pct, true)} tone={result.attribution_30d?.spread_vs_sector_pct >= 0 ? 'positive' : 'negative'} /><Metric label="Risk" value={result.risk_level || '—'} note="Rule-based risk, not predictive confidence" /></div><div className="holding-verdict"><Pill tone={toneForVerdict(result.verdict)}>{result.verdict}</Pill><strong>{result.actionable_playbook?.action}</strong><span>{result.key_reasons?.[0]}</span></div><button className="open-full-monitor" onClick={() => { setMonitorQuery(h.company); setMonitorResult(result); window.scrollTo({ top: 0, behavior: 'smooth' }); }}>Open full Module 2 analysis <ArrowUpRight size={14}/></button></>}</article>; })}</div> : <EmptyState icon={BriefcaseBusiness} title="No saved positions yet" copy="You can use the full Module 2 monitor above without adding a holding, or save a position here for P&amp;L tracking." />}</div></details></div>}
    </main>
    <footer><div><strong>OpenBell</strong><span>Decision support for Indian IPO investors</span></div><p>Model outputs are estimates, not investment advice. Verify offer documents and consult a SEBI-registered adviser before investing.</p></footer>
  </div>;
}
