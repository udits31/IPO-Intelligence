"""Compare fixed historical bid reconstructions with verified NSE listing opens.

No model fitting or threshold search occurs here. Missing/ambiguous cases stay
in the ledger; only clean, matured pairs enter the primary descriptive score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.historical_bid_service import evidence_severity
from services.module2_recent_exchange import ARCHIVES, _archive_rows, _identity_matches


def classify(probability: float) -> str:
    return 'HIGH' if probability >= .6 else 'LOW' if probability <= .4 else 'MIXED'


def compare_row(item: dict, *, as_of: date, archives: Path = ARCHIVES) -> dict:
    result = dict(item)
    result.update({'outcome_status': 'pending', 'eligible_for_primary_comparison': False})
    warnings = item.get('inputs', {}).get('evidence_warnings') or []
    result['evidence_quality'] = 'clean_reported_timestamp' if not warnings else 'needs_verification'
    probability = item.get('positive_listing_probability')
    if probability is not None:
        result['signal'] = classify(probability)
        result['binary_prediction'] = 'POSITIVE' if probability >= .5 else 'NON_POSITIVE'
    try:
        listing = date.fromisoformat(item['listing_date'])
        issue = float(item['issue_price'])
    except (KeyError, TypeError, ValueError):
        result['outcome_reason'] = 'listing date or issue price unavailable'
        return result
    if listing > as_of:
        result['outcome_reason'] = 'listing date has not arrived'
        return result
    archive = archives / f'{listing:%Y%m%d}.csv.zip'
    if not archive.exists():
        result['outcome_reason'] = 'official listing-day archive not available locally yet'
        return result
    symbol = item.get('nse_symbol')
    rows = _archive_rows(archive, {symbol}) if symbol else None
    name = item.get('official_issuer_name') or item['company_name']
    if rows is None or len(rows) != 1 or not _identity_matches(name, str(rows.iloc[0].FinInstrmNm)):
        result['outcome_status'] = 'identity_conflict'
        result['outcome_reason'] = 'NSE symbol/issuer does not uniquely match'
        return result
    raw = rows.iloc[0]
    opening = float(raw.OpnPric)
    if not math.isfinite(opening) or opening <= 0 or issue <= 0 or not str(raw.ISIN).startswith('INE'):
        result['outcome_status'] = 'invalid_price'
        return result
    positive = opening > issue
    result.update({'outcome_status': 'verified_nse_archive', 'actual_open_price': opening,
                   'actual_open_return_pct': (opening / issue - 1) * 100,
                   'actual_direction': 'POSITIVE' if positive else 'NON_POSITIVE',
                   'actual_flat': opening == issue, 'isin': str(raw.ISIN),
                   'listing_archive_sha256': str(raw.archive_sha256),
                   'listing_archive_url': 'https://nsearchives.nseindia.com/content/cm/'
                        f'BhavCopy_NSE_CM_0_0_0_{listing:%Y%m%d}_F_0000.csv.zip'})
    if probability is not None:
        result['direction_correct'] = (probability >= .5) == positive
        result['band_result'] = ('ABSTAINED' if result['signal'] == 'MIXED' else
                                'RIGHT' if (result['signal'] == 'HIGH') == positive else 'WRONG')
        result['eligible_for_primary_comparison'] = bool(not evidence_severity(item.get('inputs', {}))
            and item.get('closing_day_timestamp_found') and item.get('source_reported_at_ist'))
    return result


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {'n': 0}
    tp = sum(r['binary_prediction'] == 'POSITIVE' and r['actual_direction'] == 'POSITIVE' for r in rows)
    tn = sum(r['binary_prediction'] == 'NON_POSITIVE' and r['actual_direction'] == 'NON_POSITIVE' for r in rows)
    fp = sum(r['binary_prediction'] == 'POSITIVE' and r['actual_direction'] == 'NON_POSITIVE' for r in rows)
    fn = n - tp - tn - fp
    positives = tp + fn
    negatives = tn + fp
    decisive = [r for r in rows if r['signal'] != 'MIXED']
    # Wilson interval, not a promise that this selected cohort generalizes.
    rate, z = (tp + tn) / n, 1.959963984540054
    centre = (rate + z*z/(2*n)) / (1+z*z/n)
    radius = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {'n': n, 'correct': tp+tn, 'accuracy': rate,
            'accuracy_wilson_95': [max(0, centre-radius), min(1, centre+radius)],
            'confusion': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
            'positive_recall': tp/positives if positives else None,
            'non_positive_recall': tn/negatives if negatives else None,
            'always_positive_accuracy': positives/n,
            'brier_score': sum((r['positive_listing_probability']-(r['actual_direction']=='POSITIVE'))**2 for r in rows)/n,
            'decisive_n': len(decisive), 'mixed_n': n-len(decisive),
            'decisive_correct': sum(r['band_result']=='RIGHT' for r in decisive)}


def render(report: dict) -> str:
    primary = report['primary_clean_reconstruction_score']
    overall = report['all_recovered_matured_cases_diagnostic_only']
    lines = ['# IPO pre-close data recovery and listing-open comparison', '',
             f"Generated: {report['computed_at_ist']}", '',
             '## What this measures', '',
             'This is a retrospective replay with the current fixed model, not predictions issued live on the historical dates. '
             'The target is NSE listing opening price strictly above issue price; an exactly flat opening is NON_POSITIVE. '
             'Probability is not a predicted return or price. No model parameters or thresholds were tuned on these results.', '',
             'Binary direction uses a fixed 50% threshold. The interface bands remain HIGH ≥60%, LOW ≤40%, MIXED otherwise; '
             'MIXED is an abstention for the band score, not an automatic correct prediction.', '',
             '## Evidence coverage and scores', '',
             f"- Cohort: {len(report['ipos'])} IPOs, including all nine previously unresolved records.",
             f"- Matured official-opening comparisons: {overall['n']}.",
             f"- Clean, closing-day, pre-17:00 source-reported timestamp pairs: {primary['n']}."]
    if primary['n']:
        ci = primary['accuracy_wilson_95']
        lines += [f"- Primary descriptive direction accuracy: {primary['correct']}/{primary['n']} = {primary['accuracy']:.1%}; "
                  f"Wilson 95% interval {ci[0]:.1%}–{ci[1]:.1%}.",
                  f"- Always-positive baseline on the identical clean cases: {primary['always_positive_accuracy']:.1%}.",
                  f"- Confusion counts: {primary['confusion']}; Brier score {primary['brier_score']:.4f}.",
                  f"- Band decisions: {primary['decisive_correct']}/{primary['decisive_n']} decisive calls right; "
                  f"{primary['mixed_n']} MIXED/abstained."]
    if overall['n']:
        lines += [f"- All recovered matured candidates, INCLUDING warning-flagged inputs: {overall['correct']}/{overall['n']} "
                  f"= {overall['accuracy']:.1%}. Diagnostic only; do not report as clean or prospective accuracy."]
    lines += ['', 'This is a small, recovered, mixed-decision-time cohort with incomplete independent timestamp proof. '
              'It is not an unbiased holdout, a general accuracy claim, or proof of an implementable historical trading strategy.', '',
              '## Every IPO: inputs, model result and actual opening', '',
              '| IPO | Closing-day bids as of IST | QIB / NII / Retail | P(positive open) · band | Issue → NSE open | Opening return | 50% direction / band call | Evidence |',
              '|---|---|---|---|---|---|---|---|']
    for row in report['ipos']:
        inputs = row.get('inputs', {})
        bids = ' / '.join(f'{inputs[k]:.2f}x' if inputs.get(k) is not None else '—'
                          for k in ['qib_subscription','hni_subscription','retail_subscription'])
        p = row.get('positive_listing_probability')
        score = f"{p:.1%} · {row['signal']}" if p is not None else 'Unscored'
        price = f"₹{row['issue_price']:g} → ₹{row['actual_open_price']:g}" if row.get('actual_open_price') else 'Pending'
        gain = f"{row['actual_open_return_pct']:+.2f}%" if row.get('actual_open_return_pct') is not None else '—'
        outcome = ('RIGHT' if row['direction_correct'] else 'WRONG') + ' / ' + row['band_result'] if 'direction_correct' in row else row.get('outcome_reason', 'Pending')
        stamp = row.get('source_reported_at_ist') or 'Missing'
        lines.append(f"| {row['company_name']} | {stamp} | {bids} | {score} | {price} | {gain} | {outcome} | {row['evidence_quality']} |")
    lines += ['', '## Misses and evidence limitations', '']
    for row in report['ipos']:
        warnings = row.get('inputs', {}).get('evidence_warnings') or []
        if row.get('direction_correct') is False:
            lines += [f"- {row['company_name']}: model {row['positive_listing_probability']:.1%} positive probability; "
                      f"actual NSE opening {row['actual_open_return_pct']:+.2f}%. Descriptive miss; "
                      'this comparison alone cannot establish the causal reason.']
        for warning in warnings:
            lines += [f"- {row['company_name']}: {warning}. Excluded from the primary clean-input score."]
    lines += ['', '## Sources and reproducibility', '',
              'Bid times below are explicitly reported by publishers. Some are much earlier than 17:00; '
              'they are the latest eligible recoveries found, not asserted to be the latest bids that actually existed. '
              'Earlier and warning-flagged observations remain in the append-only source ledger.', '',
              f"- Model file SHA-256: `{report['model_file_sha256']}`.",
              '- Official offer identities/dates: NSE public-past-issues response; its SHA-256 is retained per matched IPO.',
              '- Actual prices: listing-day NSE UDiFF `OpnPric`, checked against issuer name, symbol, date and ISIN. '
              'Neither day close nor a later session open is substituted.', '']
    for row in report['ipos']:
        links = [f"[bid evidence]({s['url']}) (SHA-256 `{s['raw_sha256']}`)" for s in row.get('sources', [])]
        if row.get('listing_archive_url'):
            links += [f"[NSE listing archive]({row['listing_archive_url']}) (SHA-256 `{row['listing_archive_sha256']}`)"]
        lines += [f"- {row['company_name']}: " + '; '.join(links)]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'output/preclose_bid_recovery_coverage.json')
    parser.add_argument('--as-of', type=date.fromisoformat)
    args = parser.parse_args()
    now = datetime.now(ZoneInfo('Asia/Kolkata'))
    data = json.loads(args.input.read_text(encoding='utf-8'))
    bundle = json.loads((ROOT/'ml/model_bundle.json').read_text(encoding='utf-8'))
    bundle_digest = hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()
    if any(row.get('positive_listing_probability') is not None and row.get('model_sha256') != bundle_digest
           for row in data['ipos']):
        raise ValueError('Recovery ledger lacks the current model fingerprint; rerun recovery before comparing.')
    rows = [compare_row(row, as_of=args.as_of or now.date()) for row in data['ipos']]
    matured = [row for row in rows if 'direction_correct' in row]
    report = {'computed_at_ist': now.isoformat(), 'prospective_accuracy_claim': False,
              'model_file_sha256': hashlib.sha256((ROOT/'ml/model_bundle.json').read_bytes()).hexdigest(),
              'primary_clean_reconstruction_score': metrics([r for r in matured if r['eligible_for_primary_comparison']]),
              'all_recovered_matured_cases_diagnostic_only': metrics(matured), 'ipos': rows}
    (ROOT/'output/preclose_listing_open_comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (ROOT/'output/preclose_listing_open_comparison.md').write_text(render(report),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='ipos'},indent=2))


if __name__ == '__main__':
    main()
