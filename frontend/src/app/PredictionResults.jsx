"use client";

import {useEffect, useState} from 'react';
import {api} from '../lib/api';

const money = value => value == null ? '—' : `₹${Number(value).toLocaleString('en-IN', {maximumFractionDigits: 2})}`;
const date = value => value ? new Date(value).toLocaleString('en-IN', {timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'}) : 'Not recorded';

function ResultsTable({rows, historical = false, prospective = false}) {
  return <div className="table-wrap prediction-results-table" tabIndex={0} aria-label="Scrollable prediction results">
    <table>
      <caption>{prospective ? 'New IPOs excluded from training · latest saved pre-close prediction' : historical ? 'Historical test: saved model replay, not live predictions' : 'Other saved forecasts'} · opening compared with issue price</caption>
      <thead><tr><th scope="col">IPO</th>{prospective && <th scope="col">Used in training?</th>}<th scope="col">Our saved prediction</th><th scope="col">Issue price</th><th scope="col">Actual listing open</th><th scope="col">Opening gain / loss</th><th scope="col">Were we correct?</th><th scope="col">Prediction timing</th></tr></thead>
      <tbody>{rows.map((row, index) => <tr key={`${row.company_name}-${index}`}>
        <th scope="row">{row.company_name}{row.listing_date && <small>Listed / scheduled {row.listing_date}</small>}</th>
        {prospective && <td>No<small>Training membership checked</small></td>}
        <td>{row.predicted_direction}<small>{row.positive_probability_pct.toFixed(2)}% chance of positive opening</small></td>
        <td>{money(row.issue_price)}</td>
        <td>{money(row.listing_open_price)}<small>{row.actual_direction === 'Pending' ? 'Verified opening not yet recorded' : row.actual_direction}</small></td>
        <td>{row.actual_return_pct == null ? '—' : `${row.actual_return_pct > 0 ? '+' : ''}${row.actual_return_pct.toFixed(2)}%`}</td>
        <td><span className={`prediction-result prediction-result-${row.result.toLowerCase()}`}>{row.result === 'Correct' ? 'Yes · Correct' : row.result === 'Incorrect' ? 'No · Incorrect' : 'Pending'}</span></td>
        <td>{row.evidence_kind}{!historical && <><small>Saved {date(row.predicted_at_ist)} IST</small><small>Input as of {date(row.source_asof_ist)} IST</small>{prospective && <small>Bidding cutoff {date(row.application_cutoff_at_ist)} IST</small>}</>}</td>
      </tr>)}</tbody>
    </table>
    {!rows.length && <p>No matching IPOs in this group.</p>}
  </div>;
}

function Summary({title, value}) {
  return <div><span>{title}</span><strong>{value.accuracy_pct == null ? 'Pending' : `${value.accuracy_pct.toFixed(1)}% correct`}</strong><small>{value.correct} correct / {value.verified_openings} verified openings · {value.pending} pending</small></div>;
}

export default function PredictionResults() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  async function refresh() {
    setLoading(true); setError('');
    try {setData(await api.predictionResults());} catch (err) {setError(err.message || 'Unable to load saved results.');}
    finally {setLoading(false);}
  }
  useEffect(() => {refresh();}, []);
  const matches = row => row.company_name.toLowerCase().includes(query.trim().toLowerCase());
  return <section className="panel prediction-results">
    <div className="section-heading"><div><span className="eyebrow">NEW IPOs · NOT USED IN TRAINING</span><h1>Our pre-close predictions vs actual openings</h1></div><button className="secondary-button" onClick={refresh} disabled={loading}>{loading ? 'Checking…' : 'Refresh results'}</button></div>
    <p>Did the model work on a new IPO? Each company below was excluded from training, and its prediction was saved before bidding closed. Positive means its actual listing open was above the issue price.</p>
    {error && <p role="alert">{error} {data ? 'Previously loaded results remain below; refresh failed.' : 'Retry to load the results.'}</p>}
    {!data && !error && <p role="status">Loading saved predictions and verified openings…</p>}
    {data && <>
      <div className="return-metrics">
        <div><span>New IPOs with eligible predictions</span><strong>{data.prospective_summary.total}</strong><small>Not used in training · saved before cutoff</small></div>
        <Summary title="Verified listing outcomes" value={data.prospective_summary}/>
        <div><span>Awaiting verified listing open</span><strong>{data.prospective_summary.pending}</strong><small>Pending is not a failed prediction or 0% accuracy</small></div>
      </div>
      {data.training_membership_status !== 'verified' && <p role="alert">Training membership could not be verified. No IPO is labelled unseen until that check passes.</p>}
      {data.training_membership && <p className="fine-print">Membership checked against the saved model’s {data.training_membership.training_n} training IPOs. Being present elsewhere in the application’s database does not mean a company trained the model.</p>}
      <p className="fine-print">A score of 50% or above predicts a positive opening; below 50% predicts flat or negative. This checks direction, not the size of the gain or the APPLY recommendation. Pending openings do not count towards accuracy.</p>
      <label className="prediction-search">Find an IPO<input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search company name"/></label>
      <ResultsTable rows={data.prospective_records.filter(matches)} prospective/>
      {!data.prospective_summary.verified_openings && <p className="fine-print">No eligible IPO has a verified listing opening recorded yet. This live test is awaiting outcomes; historical or after-close replays are not substituted as live successes.</p>}
      <details className="research-history"><summary>Other saved forecasts and after-close replays · not part of the main score</summary>
      <p className="fine-print">These were not verified as saved before the application cutoff. A correct result here is not evidence that we made a timely application call.</p>
      <ResultsTable rows={data.recent_records.filter(row => !data.prospective_records.some(main => main.company_name === row.company_name) && matches(row))}/>
      </details>
      <details className="research-history">
        <summary>All {data.historical_records.length} historical test IPOs · 2025–2026</summary>
        <p>The model was trained on 2021–2024 records. These later-year probabilities are a retrospective final-subscription-book replay, not forecasts saved while investors could still apply. Training records are not included in this test.</p>
        {data.historical_status === 'verified_saved_model_replay' ? <ResultsTable rows={data.historical_records.filter(matches)} historical/> : <p role="status">Historical results need re-verification against the current model and source files. They are not counted here.</p>}
      </details>
    </>}
  </section>;
}
