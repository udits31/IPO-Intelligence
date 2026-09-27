"""Experimental return estimates, frozen pre-close and ranked with risk visible."""
from __future__ import annotations

import hashlib
import json
import math
from db import ledger as sqlite3
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from ml.return_pipeline import BUNDLE, META, REPORT, VERSION, feature_frame, predict_return, risk_score

IST = ZoneInfo('Asia/Kolkata')
OUTCOME_AUDIT = Path(__file__).resolve().parents[1]/'output/module1_exchange_outcome_replay_20261021.json'


def clock_value(value):
    try:
        stamp = datetime.fromisoformat(str(value))
        return stamp.astimezone(IST) if stamp.tzinfo else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def load_return_model():
    metadata = json.loads(META.read_text(encoding='utf-8'))
    import sklearn
    training_version=metadata.get('runtime_versions',metadata.get('reproducibility',{})).get('sklearn')
    if training_version is None:
        training_version=metadata.get('runtime',{}).get('sklearn')
    if training_version and training_version!=sklearn.__version__:
        raise ImportError('Archived return challenger requires its original scikit-learn runtime; active frozen direction model is unaffected')
    if (metadata['model_version'] != VERSION or metadata['training_years'] != [2021, 2022, 2023, 2024]
            or hashlib.sha256(BUNDLE.read_bytes()).hexdigest() != metadata['artifact_sha256']):
        raise ValueError('Return artifact failed its version/cohort/hash contract')
    artifact = joblib.load(BUNDLE)  # Local, hashed artifact produced by our trainer only.
    artifact['artifact_sha256'] = metadata['artifact_sha256']
    return artifact


def make_return_forecast(item, values, *, source_asof, decision_at, artifact=None, publisher_asof=None):
    """Compute now, never reconstruct an allegedly issued past return forecast."""
    source = clock_value(source_asof)
    if decision_at.tzinfo is None or source is None or source > decision_at:
        return None
    from services.snapshot_service import application_forecast_window
    window = application_forecast_window(item, decision_at)
    if not window['is_open'] or source < datetime.fromisoformat(window['opens_at_ist']):
        return None
    features = {key: values.get(key) for key in ('qib_subscription', 'hni_subscription', 'retail_subscription')}
    features.update(issue_price=item.get('issue_price'), issue_size_cr=item.get('issue_size_cr'))
    try:
        if any(v is None or not math.isfinite(float(v)) or float(v) < 0 for v in features.values()):
            return None
        if float(features['issue_price']) <= 0 or float(features['issue_size_cr']) <= 0:
            return None
        artifact = artifact or load_return_model()
        # A later decision cannot be used as evidence that this model existed
        # during its own historical training period.
        if decision_at.date().isoformat() <= artifact['trained_through_date']:
            return None
        gmp = item.get('gmp_pct')
        gmp_clock = clock_value(item.get('gmp_observed_at'))
        # Legacy fetch timestamps lacked an offset; do not silently assume it.
        usable_gmp = (gmp is not None and math.isfinite(float(gmp))
                      and gmp_clock is not None and 0 <= (decision_at-gmp_clock).total_seconds() <= 1200
                      and not item.get('gmp_is_stale') and bool(item.get('gmp_source_url')))
        track = 'bids_plus_gmp' if usable_gmp and 'bids_plus_gmp' in artifact['tracks'] else 'bids_only'
        fitted = artifact['tracks'][track]
        features['gmp_pct'] = float(gmp) if usable_gmp else None
        frame = pd.DataFrame([features])
        x = feature_frame(frame, fitted['with_gmp'])
        centre = float(predict_return(fitted['regressor'], frame, fitted['with_gmp'])[0])
        lower = max(-100., centre+fitted['residual_bounds'][0])
        upper = centre+fitted['residual_bounds'][1]
        probability = float(fitted['classifier'].predict_proba(x)[0, 1])
        outside = [col for col in x if not fitted['feature_min'][col] <= float(x.iloc[0][col]) <= fitted['feature_max'][col]]
        return {
            'model_version': artifact['model_version'], 'track': track,
            'artifact_sha256': artifact.get('artifact_sha256'),
            'status': artifact['status'], 'training_stage': artifact['training_stage'],
            'source_timestamp_kind': 'publisher' if publisher_asof else 'capture_only',
            'computed_at_ist': decision_at.isoformat(), 'source_asof_ist': source_asof,
            'central_return_pct': centre, 'lower_return_pct': lower, 'upper_return_pct': upper,
            'central_open_price': float(features['issue_price'])*(1+centre/100),
            'positive_listing_probability': probability,
            'priority_score_pp': float(risk_score(centre, lower)),
            'risk_weight': artifact['risk_weight'], 'cost_scenario_pp': artifact['cost_scenario_pp'],
            'input_features': features, 'gmp_observed_at': gmp_clock.isoformat() if usable_gmp else None,
            'gmp_source_url': item.get('gmp_source_url') if usable_gmp else None,
            'gmp_baseline_return_pct': float(gmp) if usable_gmp else None,
            'outside_training_features': outside,
            'better_than_gmp_established': artifact['better_than_gmp_established'],
            'interval_note': 'Earlier-year 10th–90th residual band; coverage is measured in the benchmark, not guaranteed.',
            'stage_warning': 'Trained on penultimate-session daily books; live intraday and final-day performance remains experimental.',
            'ranking_eligible': bool(publisher_asof) and not outside and (decision_at-source).total_seconds() <= 1200,
        }
    except (FileNotFoundError,ImportError):
        return None  # Existing direction model remains usable if challenger has not been trained.


def rank_forecasts(candidates, *, now):
    rows, excluded = [], []
    for item in candidates:
        forecast = item.get('return_forecast')
        cutoff = clock_value(item.get('cutoff_at_ist'))
        opening = clock_value(item.get('opens_at_ist'))
        source = clock_value((forecast or {}).get('source_asof_ist'))
        issued = clock_value((forecast or {}).get('computed_at_ist'))
        reason = None
        if not cutoff or not opening or not opening <= now < cutoff:
            reason = 'Offer not accepting applications'
        elif not forecast:
            reason = 'No saved return estimate yet'
        elif not source or not issued or not opening <= source <= issued <= now or issued >= cutoff:
            reason = 'Invalid or future observation/decision clock'
        elif (now-source).total_seconds() > 1200:
            reason = 'Bid reading older than 20 minutes; estimate retained but not prioritised'
        elif forecast.get('source_timestamp_kind') == 'capture_only':
            reason = 'Publisher update clock unverified; capture time alone cannot prove fresh bids'
        elif not forecast.get('ranking_eligible'):
            reason = 'Inputs outside training support'
        if reason:
            excluded.append({'company_name': item['company_name'], 'reason': reason})
            continue
        rows.append({**item, 'priority_score_pp': forecast['priority_score_pp'],
                     'age_minutes': (now-source).total_seconds()/60})
    rows.sort(key=lambda x: (-x['priority_score_pp'], x['company_name']))
    for i, row in enumerate(rows):
        row['rank'] = i+1
        row['priority_label'] = 'RESEARCH PRIORITY' if row['priority_score_pp'] > 0 else 'BELOW RISK HURDLE'
    skip = not rows or rows[0]['priority_score_pp'] <= 0
    paired = [r for r in rows if r['return_forecast'].get('gmp_baseline_return_pct') is not None]
    gmp_best = sorted(paired, key=lambda r: (-r['return_forecast']['gmp_baseline_return_pct'], r['company_name']))[0] if paired else None
    gmp_choice = gmp_best['company_name'] if gmp_best and gmp_best['return_forecast']['gmp_baseline_return_pct'] > .5 else None
    return {'as_of_ist': now.isoformat(), 'status': 'experimental', 'ranked': rows, 'excluded': excluded,
            'skip_all': skip, 'top_priority': None if skip else rows[0]['company_name'],
            'gmp_only_top_priority_on_paired_candidates': gmp_choice,
            'gmp_comparable_candidates': len(paired),
            'policy': 'Central return − 0.5 × lower-band downside − 0.5pp illustrative cost; fresh pre-close books only.',
            'claim': 'Research ranking; an advantage over GMP-only has not been established prospectively.'}


def current_priorities(*, cache=None, now=None, db_path=None):
    from scrapers.live_ipo_scraper import load_scraped_cache
    from services.snapshot_service import application_forecast_window, get_latest_application_forecast, DB_PATH
    clock = now or datetime.now(IST)
    cache = load_scraped_cache() if cache is None else cache
    candidates, seen = [], set()
    for item in cache.values():
        name = item.get('company_name')
        if not name or name.casefold() in seen:
            continue
        window = application_forecast_window(item, clock)
        if not window['is_open']:
            continue
        seen.add(name.casefold())
        saved = get_latest_application_forecast(name, db_path=str(db_path or DB_PATH))
        candidates.append({'company_name': name, 'cutoff_at_ist': window['cutoff_at_ist'],
                           'opens_at_ist': window['opens_at_ist'],
                           'return_forecast': (saved or {}).get('inputs', {}).get('return_forecast')})
    return rank_forecasts(candidates, now=clock)


def return_benchmark():
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    # Separate a source-correction audit from a refitted research challenger.
    # Never display refit scores as if they belonged to the deployed artifact.
    if OUTCOME_AUDIT.exists():
        audit = json.loads(OUTCOME_AUDIT.read_text(encoding='utf-8'))
        root = Path(__file__).resolve().parents[1]
        source_ledger = root/'output/module1_return_prediction_ledger.csv'
        expected = audit.get('input_sha256',{})
        matches = (audit.get('audit_for_model_version') == report.get('model_version')
                   and expected.get('output/module1_return_benchmark.json') == hashlib.sha256(REPORT.read_bytes()).hexdigest()
                   and source_ledger.exists()
                   and expected.get('output/module1_return_prediction_ledger.csv') == hashlib.sha256(source_ledger.read_bytes()).hexdigest())
        report['exchange_outcome_audit'] = ({
            'status':'verified_source_recheck', 'forecasts_refitted':False,
            'tracks':audit['original_return_forecasts_rescored'],
            'price_policy':'NSE opening at the IPO date, or exact-code BSE opening when not listed on NSE then.',
            'note':'The same saved historical predictions are rescored against exchange archives; no model was promoted.'}
            if matches else {'status':'outdated','note':'Outcome audit does not match the current benchmark/forecast ledger.'})
    return report


def record_priority_snapshot(*, cache, now, slot, db_path):
    snapshot = current_priorities(cache=cache, now=now, db_path=db_path)
    with closing(sqlite3.connect(db_path, timeout=5)) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS module1_priority_v1 (
            slot_key TEXT PRIMARY KEY, recorded_at_ist TEXT NOT NULL,
            model_version TEXT NOT NULL, snapshot_json TEXT NOT NULL)''')
        conn.execute('INSERT OR IGNORE INTO module1_priority_v1 VALUES (?, ?, ?, ?)',
                     (slot, now.isoformat(), VERSION, json.dumps(snapshot, sort_keys=True)))
        conn.commit()
    return snapshot


def prospective_returns(db_path):
    """One frozen latest-before-cutoff estimate per IPO, only official opens."""
    from services.listing_outcome_service import verified_listing_outcomes
    from services.snapshot_service import preferred_forecast
    outcomes = verified_listing_outcomes(db_path)
    with closing(sqlite3.connect(db_path, timeout=5)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            raw = [dict(r) for r in conn.execute("SELECT * FROM automatic_decision_v1 WHERE decision_kind='live_bid_forecast'")]
        except sqlite3.OperationalError:
            raw = []
    groups = {}
    for row in raw:
        row['inputs'] = json.loads(row['input_json'])
        forecast = row['inputs'].get('return_forecast')
        cutoff = clock_value(row['inputs'].get('application_cutoff_at_ist'))
        issued = clock_value(row.get('recorded_at_ist'))
        if forecast and cutoff and issued and issued < cutoff:
            groups.setdefault(row['ipo_id'], []).append(row)
    rows = []
    for ipo_id, candidates in groups.items():
        saved = preferred_forecast(candidates)
        forecast, outcome = saved['inputs']['return_forecast'], outcomes.get(ipo_id)
        actual = (outcome['nse_open_price']/outcome['issue_price']-1)*100 if outcome else None
        rows.append({'ipo_id': ipo_id, 'company_name': saved['company_name'],
                     'recorded_at_ist': saved['recorded_at_ist'], 'model_version': forecast['model_version'],
                     'central_return_pct': forecast['central_return_pct'], 'actual_return_pct': actual,
                     'absolute_error_pp': abs(actual-forecast['central_return_pct']) if actual is not None else None,
                     'gmp_baseline_return_pct': forecast.get('gmp_baseline_return_pct'),
                     'gmp_absolute_error_pp': abs(actual-forecast['gmp_baseline_return_pct'])
                       if actual is not None and forecast.get('gmp_baseline_return_pct') is not None else None})
    resolved = [r for r in rows if r['actual_return_pct'] is not None]
    paired = [r for r in resolved if r['gmp_absolute_error_pp'] is not None]
    return {'recorded_ipos': len(rows), 'verified_outcomes': len(resolved), 'paired_gmp_outcomes': len(paired),
            'return_mae_pp': float(np.mean([r['absolute_error_pp'] for r in resolved])) if resolved else None,
            'paired_model_mae_pp': float(np.mean([r['absolute_error_pp'] for r in paired])) if paired else None,
            'paired_gmp_mae_pp': float(np.mean([r['gmp_absolute_error_pp'] for r in paired])) if paired else None,
            'records': rows, 'policy': 'Actual pre-cutoff frozen estimates; one latest primary source per IPO. No historical backdating.'}
