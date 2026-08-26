"use client";

const percent = value => value == null ? '—' : `${value > 0 ? '+' : ''}${Number(value).toFixed(1)}%`;
const stamp = value => value ? new Date(value).toLocaleString('en-IN', {timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'}) + ' IST' : 'Unknown';

export function ReturnEstimate({forecast}) {
  if (!forecast) return null;
  return <section className="panel return-estimate">
    <div className="section-heading"><div><span className="eyebrow">LISTING RETURN FORECAST</span><h3>How much could it gain or lose?</h3></div><span className="return-status">Experimental</span></div>
    <div className="return-metrics">
      <div><span>Central opening-return estimate</span><strong>{percent(forecast.central_return_pct)}</strong></div>
      <div><span>Lower residual scenario</span><strong className={forecast.lower_return_pct < 0 ? 'text-negative' : ''}>{percent(forecast.lower_return_pct)}</strong></div>
      <div><span>Upper residual scenario</span><strong>{percent(forecast.upper_return_pct)}</strong></div>
      <div><span>Risk-adjusted research score</span><strong>{Number(forecast.priority_score_pp).toFixed(1)} pp</strong></div>
    </div>
    <p className="fine-print">Frozen using bids dated {stamp(forecast.source_asof_ist)}. {forecast.track === 'bids_plus_gmp' ? 'Uses bids, offer terms and a captured GMP quote.' : 'Uses bids and offer terms; no usable dated GMP input.'} {forecast.outside_training_features?.length ? 'Its early-book probability is withheld outside training support; the direction-only estimate above is from the legacy classifier.' : `The early-book classifier estimates a ${(forecast.positive_listing_probability*100).toFixed(1)}% chance of opening above issue price.`}</p>
    <p className="fine-print">The band is not a guaranteed minimum/maximum or a validated 80% live interval. {forecast.stage_warning} A positive score does not mean you will receive an allotment or make a profit.</p>
    {forecast.outside_training_features?.length > 0 && <p className="text-warning">Inputs exceed the historical training range. Excluded from priority ranking.</p>}
    {forecast.source_timestamp_kind === 'capture_only' && <p className="fine-print">Only our capture clock is known for this book; the publisher update time is unverified. Estimate retained, but excluded from fresh priority ranking.</p>}
  </section>;
}

export function PriorityBoard({data, onResearch}) {
  if (!data) return null;
  return <section className="panel priority-board">
    <div className="section-heading"><div><span className="eyebrow">BEFORE YOU APPLY</span><h3>Open IPO research priorities</h3></div><span className="return-status">Experimental ranking</span></div>
    <p>{data.top_priority ? `Highest current research score: ${data.top_priority}.` : data.ranked?.length ? 'None clears the current downside-and-cost hurdle.' : 'No fresh eligible return forecasts in this comparison yet.'}</p>
    {data.gmp_comparable_candidates > 0 && <p className="fine-print">GMP-only would prioritise {data.gmp_only_top_priority_on_paired_candidates || 'none'} among the same {data.gmp_comparable_candidates} quote-matched eligible offer(s). This comparison is saved with each scheduled priority snapshot.</p>}
    <p className="fine-print">Higher priority means a better modelled reward/downside trade-off, not a buy instruction. {data.claim} This screen compares all eligible open offers, not the order you selected them.</p>
    {data.ranked?.length > 0 && <div className="table-wrap"><table><thead><tr><th>Priority / IPO</th><th>Central return</th><th>Lower / upper scenarios</th><th>Risk score</th><th>Bid evidence</th><th></th></tr></thead><tbody>{data.ranked.map(row => <tr key={row.company_name}><td><strong>#{row.rank} · {row.company_name}</strong><small>{row.priority_label === 'BELOW RISK HURDLE' ? 'Below risk hurdle' : 'Research priority'}</small></td><td>{percent(row.return_forecast.central_return_pct)}</td><td>{percent(row.return_forecast.lower_return_pct)} / {percent(row.return_forecast.upper_return_pct)}</td><td>{Number(row.priority_score_pp).toFixed(1)} pp</td><td>{stamp(row.return_forecast.source_asof_ist)}<small>Cutoff {stamp(row.cutoff_at_ist)}</small></td><td><button className="secondary-button" onClick={() => onResearch(row.company_name)}>Research</button></td></tr>)}</tbody></table></div>}
    <p className="fine-print">Score = central return − half the lower-band loss below zero − a 0.5 percentage-point illustrative cost. Returns are conditional on allotment, before actual taxes/fees. Allotment odds and profit per rupee blocked are not estimated.</p>
    {data.excluded?.length > 0 && <details><summary>{data.excluded.length} open offers not ranked</summary>{data.excluded.map(row => <p className="fine-print" key={row.company_name}>{row.company_name}: {row.reason}.</p>)}</details>}
    <small className="fine-print">Checked {stamp(data.as_of_ist)} · books older than 20 minutes are not ranked.</small>
  </section>;
}

export function ReturnBenchmark({report}) {
  if (!report) return null;
  const audit = report.exchange_outcome_audit;
  const auditedTrack = audit?.status === 'verified_source_recheck' ? audit.tracks?.bids_plus_gmp : null;
  const track = auditedTrack || report.tracks?.bids_plus_gmp || report.tracks?.bids_only;
  const paired = track?.paired_2024;
  const rank = track?.ranking_2024;
  if (!track) return null;
  return <section className="panel return-benchmark">
    <div className="section-heading"><div><span className="eyebrow">RETURN MODEL VS GMP-ONLY</span><h3>Do we actually select better IPOs?</h3></div></div>
    <p>{report.conclusion}</p>
    <div className="return-metrics">
      <div><span>Paired 2024 IPOs</span><strong>{paired?.n || 0}</strong></div>
      <div><span>Model return error</span><strong>{paired?.mae_pp?.toFixed(2) || '—'} pp</strong></div>
      <div><span>GMP-only return error</span><strong>{paired?.gmp_mae_pp?.toFixed(2) || '—'} pp</strong></div>
      <div><span>Same-date selection panels</span><strong>{rank?.panels || 0}</strong></div>
    </div>
    {paired?.mae_improvement_bootstrap_95_pp && <p className="fine-print">Average error improvement: {paired.mae_improvement_over_gmp_pp.toFixed(2)} pp; bootstrap 95% interval {paired.mae_improvement_bootstrap_95_pp.map(n => n.toFixed(2)).join(' to ')} pp. An interval crossing zero does not establish an improvement.</p>}
    <p className="fine-print">Earlier-year selection: 2022–2023. Retrospective confirmation: 2024; no 2025–2026 training. Daily GMP must precede the bid date because its publication clock is unknown. Historical pages may have been edited.</p>
    {auditedTrack ? <p className="fine-print"><strong>Exchange-verified openings:</strong> the headline scores above use the same saved predictions, without retraining, rescored against official exchange archives. {audit.price_policy} This source correction does not establish a selection advantage.</p> : <p className="fine-print text-warning">The opening-price source audit is not current for this forecast ledger. These are the original research-label scores, not an exchange-verified recheck.</p>}
    <p className="fine-print">The nominal 80% residual band covered {(track.confirmation_2024.interval_80_coverage*100).toFixed(1)}% of 2024 outcomes. It is not sufficiently calibrated to call an 80% live prediction interval.</p>
    {rank?.panels > 0 && <p className="fine-print">Selection comparison: model wins {rank.model_wins}, GMP wins {rank.gmp_wins}, ties {rank.ties}. Model minus GMP conditional opening return: {rank.paired_difference_pp.toFixed(2)} pp. These are not realised portfolio profits.</p>}
    <p className="fine-print">Prospective return test: {report.prospective?.recorded_ipos || 0} IPOs with saved return forecasts; {report.prospective?.verified_outcomes || 0} verified listing outcomes. Missing outcomes are pending, not counted as successes.</p>
  </section>;
}
