"""Teacher-facing direction accountability; never recompute a saved forecast."""
import hashlib
import json
import math
import re
from db import ledger as sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from services.snapshot_service import get_intraday_forecast_scorecard, get_prelisting_outcome_scorecard
from services.listing_outcome_service import verified_listing_outcomes

ROOT = Path(__file__).resolve().parents[1]


def identity_name(value):
    words = re.findall(r'[a-z0-9]+', str(value).lower())
    return ''.join(word for word in words if word not in {'limited', 'ltd', 'private', 'pvt'})


def training_manifest():
    """Reconstruct actual fit membership and check its saved ID fingerprint."""
    from ml.model_pipeline import prepare_dataset
    bundle = json.loads((ROOT/'ml/model_bundle.json').read_text(encoding='utf-8'))
    for name, digest in bundle['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Training source no longer matches the saved model')
    data = prepare_dataset()
    train = data[data.year.isin(bundle['training_years']) & ~data.id.isin(bundle['excluded_non_corporate_ids'])]
    ids = sorted(int(value) for value in train.id)
    digest = hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()
    if digest != bundle['training_ipo_ids_sha256'] or len(ids) != bundle['training_samples']:
        raise ValueError('Training membership does not match the fitted model')
    return dict(model_version=bundle['model_version'], trained_at_utc=bundle['trained_at_utc'],
        training_n=len(ids), training_ids_sha256=digest, ids=set(map(str,ids)),
        names={identity_name(name) for name in train.company_name})


def saved_live_forecasts():
    from services.snapshot_service import DB_PATH
    with closing(sqlite3.connect(DB_PATH, timeout=5)) as connection:
        connection.row_factory = sqlite3.Row
        records = connection.execute("""SELECT ipo_id, company_name, recorded_at_ist,
            source_asof_ist, model_version, positive_listing_probability, input_json
            FROM automatic_decision_v1 WHERE decision_kind = 'live_bid_forecast'
            ORDER BY recorded_at_ist DESC, id DESC""").fetchall()
    return [{**dict(row), 'inputs':json.loads(row['input_json'])} for row in records]


def prospective_results(records, outcomes, manifest):
    """Choose latest eligible pre-close call, never latest post-close rewrite."""
    rows, seen = [], set()
    for record in sorted(records, key=lambda row: row['recorded_at_ist'], reverse=True):
        ipo_id = str(record['ipo_id'])
        if ipo_id in seen or record['model_version'] != manifest['model_version']:
            continue
        name = identity_name(record['company_name'])
        if not name or name in manifest['names'] or ipo_id in manifest['ids']:
            continue
        inputs = record['inputs']
        outcome = outcomes.get(ipo_id, {})
        try:
            saved = pd.Timestamp(record['recorded_at_ist'])
            source = pd.Timestamp(record['source_asof_ist'])
            cutoff = pd.Timestamp(inputs['application_cutoff_at_ist'])
            trained = pd.Timestamp(manifest['trained_at_utc'])
            if any(pd.isna(value) or value.tzinfo is None for value in (saved,source,cutoff,trained)):
                continue
            if not trained <= saved <= cutoff or source > saved:
                continue
            if outcome.get('listing_date') and saved.tz_convert('Asia/Kolkata').date() >= pd.Timestamp(outcome['listing_date']).date():
                continue
        except (KeyError, TypeError, ValueError):
            continue
        seen.add(ipo_id)
        row = result_row(company=record['company_name'], probability=record['positive_listing_probability'],
            issue=outcome.get('issue_price') or inputs.get('issue_price'), opening=outcome.get('nse_open_price'),
            predicted_at=record['recorded_at_ist'], source_at=record['source_asof_ist'],
            listing_date=outcome.get('listing_date'), evidence_kind='Saved before application cutoff',
            model_version=record['model_version'], source_hash=outcome.get('archive_sha256'))
        row.update(ipo_id=ipo_id, used_in_training=False, training_check='Excluded from verified training membership',
                   application_cutoff_at_ist=inputs['application_cutoff_at_ist'])
        rows.append(row)
    return rows


def result_row(*, company, probability, issue=None, opening=None, predicted_at=None,
               source_at=None, listing_date=None, evidence_kind, model_version=None, source_hash=None):
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid saved probability')
    observed = all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (issue, opening))
    change = (opening / issue - 1) * 100 if observed else None
    actual = 'Positive' if observed and opening > issue else 'Negative' if observed and opening < issue else 'Flat' if observed else 'Pending'
    correct = (probability >= .5) == (opening > issue) if observed else None
    return dict(company_name=company, predicted_at_ist=predicted_at, source_asof_ist=source_at,
        listing_date=listing_date, positive_probability_pct=round(probability * 100, 2),
        predicted_direction='Positive' if probability >= .5 else 'Flat or negative',
        issue_price=issue, listing_open_price=opening, actual_return_pct=round(change, 3) if observed else None,
        actual_direction=actual, result='Correct' if correct else 'Incorrect' if correct is False else 'Pending',
        evidence_kind=evidence_kind, model_version=model_version, outcome_source_sha256=source_hash)


def prediction_results():
    outcomes = verified_listing_outcomes()
    live = get_intraday_forecast_scorecard()
    locked = get_prelisting_outcome_scorecard()
    recent, seen = [], set()
    for record in sorted(live['daily_records'], key=lambda row: row['recorded_at_ist'], reverse=True):
        if record['ipo_id'] in seen:
            continue
        seen.add(record['ipo_id'])
        outcome = outcomes.get(str(record['ipo_id']), {})
        inputs = record['inputs']
        cutoff = inputs.get('application_cutoff_at_ist')
        before_cutoff = (cutoff and pd.Timestamp(record['recorded_at_ist']) <= pd.Timestamp(cutoff)
                         and record.get('source_asof_ist') and pd.Timestamp(record['source_asof_ist']) <= pd.Timestamp(cutoff))
        # Never use a table name or an old 'pre_deadline' flag alone as proof.
        kind = 'Saved before application cutoff' if before_cutoff else 'Timing not verified'
        recent.append(result_row(company=record['company_name'], probability=record['positive_listing_probability'],
            issue=outcome.get('issue_price') or inputs.get('issue_price'), opening=outcome.get('nse_open_price'),
            predicted_at=record['recorded_at_ist'], source_at=record['source_asof_ist'],
            listing_date=outcome.get('listing_date'), evidence_kind=kind, model_version=record['model_version'],
            source_hash=outcome.get('archive_sha256')))
    for record in locked['records']:
        if record['ipo_id'] in seen:
            continue
        seen.add(record['ipo_id'])
        outcome = outcomes.get(str(record['ipo_id']), {})
        before_listing = pd.Timestamp(record['recorded_at_ist']).date() < pd.Timestamp(record['listing_date']).date()
        kind = 'Before-listing replay; after bidding' if before_listing else 'Later reconstruction'
        recent.append(result_row(company=record['company_name'], probability=record['positive_listing_probability'],
            issue=outcome.get('issue_price'), opening=outcome.get('nse_open_price'),
            predicted_at=record['recorded_at_ist'], source_at=record['captured_at_ist'],
            listing_date=record['listing_date'], evidence_kind=kind, model_version=record['model_version'],
            source_hash=outcome.get('archive_sha256')))
    historical, historical_status = [], 'unavailable'
    audit_path = ROOT/'output/module1_exchange_outcome_replay_20261021.json'
    ledger_path = audit_path.with_name(audit_path.stem+'_direction_ledger.csv')
    if audit_path.exists() and ledger_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        matching = all((ROOT/name).exists() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
                       for name, sha in audit.get('input_sha256', {}).items())
        if matching and audit.get('input_sha256'):
            from ml.model_pipeline import prepare_dataset, FEATURE_COLS
            data = prepare_dataset().set_index('id')
            bundle = json.loads((ROOT/'ml/model_bundle.json').read_text(encoding='utf-8'))
            matching = all((ROOT/name).exists() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
                           for name, sha in bundle['input_sha256'].items())
            ledger = pd.read_csv(ledger_path)
            ledger = ledger[ledger.year.isin([2025, 2026])]
            openings = pd.read_csv(ROOT/'data/audit/module1_exchange_listing_open_audit.csv').set_index('ipo_id')
            openings = openings[openings.status.eq('verified_daily_exchange_archive') & openings.exchange_open.gt(0)]
            expected_ids = set(openings.index) & set(data.index[data.year.isin([2025, 2026])])
            matching = matching and not ledger.id.duplicated().any() and set(ledger.id) == expected_ids
            # Saved later probabilities must replay the actual deployed bundle.
            for record in ledger.itertuples():
                feature = data.loc[record.id]
                z = float(bundle['intercept']) + sum((float(feature[col])-mean)/scale*coef
                    for col, mean, scale, coef in zip(FEATURE_COLS, bundle['scaler_mean'], bundle['scaler_scale'], bundle['coefficients']))
                p = 1/(1+math.exp(-z))
                if (not math.isclose(p, record.probability_original_model, rel_tol=0, abs_tol=1e-9)
                    or record.trained_through != 2024
                    or not math.isclose(record.issue_price, float(feature.issue_price), rel_tol=0, abs_tol=1e-8)
                    or not math.isclose(record.exchange_open, float(openings.loc[record.id].exchange_open), rel_tol=0, abs_tol=1e-8)):
                    matching = False
                    break
            if matching:
                historical_status = 'verified_saved_model_replay'
                for record in ledger.itertuples():
                    historical.append(result_row(company=record.company_name, probability=record.probability_original_model,
                        issue=float(record.issue_price), opening=float(record.exchange_open),
                        listing_date=data.loc[record.id].listing_date.date().isoformat(),
                        evidence_kind='Retrospective final-book test', model_version=bundle['model_version'],
                        source_hash=str(openings.loc[record.id].archive_sha256)))
            else:
                historical_status = 'outdated_model_or_prediction_ledger'
        else:
            historical_status = 'outdated_source_audit'
    def summary(rows):
        resolved = [r for r in rows if r['result'] != 'Pending']
        return dict(total=len(rows), verified_openings=len(resolved), pending=len(rows)-len(resolved),
                    correct=sum(r['result'] == 'Correct' for r in resolved),
                    accuracy_pct=round(100*sum(r['result'] == 'Correct' for r in resolved)/len(resolved), 1) if resolved else None)
    genuine = [r for r in recent if r['evidence_kind'] == 'Saved before application cutoff']
    prospective, membership_status, membership = [], 'unavailable', None
    try:
        manifest = training_manifest()
        prospective = prospective_results(saved_live_forecasts(), outcomes, manifest)
        membership_status = 'verified'
        membership = {key:manifest[key] for key in ('model_version','training_n','training_ids_sha256')}
    except (OSError, ValueError, KeyError):
        # Fail closed: don't call an IPO unseen when fit membership is unknown.
        pass
    return dict(direction_threshold=.5, score_definition='Probability of opening above issue price; not predicted percentage gain.',
        actual_definition='Exchange listing OPEN compared with issue price, not current price or listing-day close.',
        recent_records=recent, historical_records=historical, historical_status=historical_status,
        prospective_records=prospective, prospective_summary=summary(prospective),
        training_membership_status=membership_status, training_membership=membership,
        summaries=dict(saved_preclose=summary(genuine), recent_replays=summary([r for r in recent if r not in genuine]),
                       historical=summary(historical)),
        note='Latest saved forecast per company; pending outcomes are not counted. Flat openings are non-positive for the binary comparison. Replays are not live pre-close predictions.')
