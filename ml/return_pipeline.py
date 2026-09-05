"""Pre-close return regression and paired GMP-selection evaluation.

This challenger uses real penultimate-session books, NOT final subscriptions.
2022/23 select a candidate; 2024 confirms it. No post-2024 training records.
The historical daily pages can have been revised, so this is retrospective
research and not proof of an investor-time trading edge.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.research_daywise_model import prepare
from ml.research_early_book_context import attach_decision_context

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'module1-preclose-return-2021-2024-v1'
BUNDLE = ROOT / 'ml/artifacts/preclose_return.joblib'
META = ROOT / 'ml/artifacts/preclose_return_metadata.json'
REPORT = ROOT / 'output/module1_return_benchmark.json'
BIDS = ['qib_subscription', 'hni_subscription', 'retail_subscription']
FEATURES = ['log_' + x for x in BIDS] + ['log_issue_price', 'log_issue_size_cr']
RISK_WEIGHT = .5  # Fixed disclosed preference, not tuned on confirmation results.
COST_SCENARIO_PP = .5  # Illustrative round-trip drag, not an actual fee calculation.


def feature_frame(frame: pd.DataFrame, with_gmp=False) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for col in BIDS + ['issue_price', 'issue_size_cr']:
        value = pd.to_numeric(frame[col], errors='coerce')
        if value.isna().any() or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f'Missing or invalid observed feature: {col}')
        result['log_' + col] = np.log1p(value)
    if with_gmp:
        result['gmp_pct'] = pd.to_numeric(frame['gmp_pct'], errors='coerce')
        if not np.isfinite(result.gmp_pct).all():
            raise ValueError('The paired GMP model requires a real dated quote')
    return result


def prepare_returns() -> tuple[pd.DataFrame, dict]:
    canonical = pd.read_csv(ROOT / 'data/canonical_listed_ipos.csv')
    frame, audit = prepare(pd.read_csv(ROOT / 'data/daywise_subscription_research_cohort.csv'), canonical)
    frame = frame.merge(canonical[['id', 'issue_size_cr']], left_on='ipo_id', right_on='id', validate='one_to_one')
    frame = attach_decision_context(frame, pd.read_csv(ROOT / 'data/nse_verified_offer_dates.csv'),
                                    pd.read_csv(ROOT / 'data/raw_daily_market_time_series.csv'))
    frame['target_return'] = (frame.listing_open_price / frame.issue_price - 1) * 100
    frame['gmp_pct'] = np.nan
    frame['gmp_date'] = None
    frame['gmp_source_url'] = None
    path = ROOT / 'data/preclose_gmp_history.csv'
    if path.exists():
        quotes = pd.read_csv(path)
        quotes['gmp_date'] = pd.to_datetime(quotes.gmp_date)
        for idx, row in frame.iterrows():
            decision = pd.Timestamp(row.decision_market_date)
            # No clock on daily GMP: same-day and future quotes are forbidden.
            eligible = quotes[(quotes.ipo_id == row.ipo_id) & (quotes.gmp_date < decision)
                              & (quotes.gmp_date >= decision-pd.Timedelta(days=7))]
            if eligible.empty:
                continue
            quote = eligible.sort_values('gmp_date').iloc[-1]
            frame.loc[idx, 'gmp_pct'] = float(quote.gmp_rs) / float(row.issue_price) * 100
            frame.loc[idx, 'gmp_date'] = quote.gmp_date.strftime('%Y-%m-%d')
            frame.loc[idx, 'gmp_source_url'] = quote.source_url
    valid = frame[BIDS + ['issue_price', 'issue_size_cr', 'target_return']].notna().all(axis=1)
    valid &= frame[BIDS].ge(0).all(axis=1) & frame[['issue_price', 'issue_size_cr']].gt(0).all(axis=1)
    frame = frame[valid].sort_values(['listing_date', 'ipo_id']).copy()
    audit.update(return_rows=len(frame), paired_gmp_rows=int(frame.gmp_pct.notna().sum()),
                 rows_by_year={str(k): int(v) for k, v in frame.groupby('year').size().items()})
    return frame, audit


def estimator(name: str):
    if name == 'ridge':
        reg = Ridge(alpha=20)
    elif name == 'forest':
        reg = RandomForestRegressor(n_estimators=240, min_samples_leaf=8, max_features=1.0,
                                    random_state=42, n_jobs=2)
    elif name == 'boosted_huber':
        reg = GradientBoostingRegressor(loss='huber', n_estimators=100, learning_rate=.04,
                                        max_depth=2, min_samples_leaf=10, random_state=42)
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), reg)


def fit_return(train, name, with_gmp=False):
    model = estimator(name)
    x = feature_frame(train, with_gmp)
    # With GMP, learn the correction to the same observed quote, not its future value.
    offset = train.gmp_pct.to_numpy() if with_gmp else 0
    model.fit(x, train.target_return.to_numpy() - offset)
    return model


def predict_return(model, test, with_gmp=False):
    offset = test.gmp_pct.to_numpy() if with_gmp else 0
    return np.maximum(-100, model.predict(feature_frame(test, with_gmp)) + offset)


def risk_score(centre, lower):
    return np.asarray(centre) - RISK_WEIGHT * np.maximum(0, -np.asarray(lower)) - COST_SCENARIO_PP


def summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {'n': 0}
    y, p = rows.target_return.to_numpy(), rows.prediction.to_numpy()
    return {'n': len(rows), 'mae_pp': float(mean_absolute_error(y, p)),
            'rmse_pp': float(np.sqrt(mean_squared_error(y, p))),
            'direction_accuracy': float(((p > 0) == (y > 0)).mean()),
            'interval_80_coverage': float(((y >= rows.lower) & (y <= rows.upper)).mean()),
            'average_interval_width_pp': float((rows.upper-rows.lower).mean()),
            'positive_probability_brier': float(brier_score_loss(y > 0, rows.probability)),
            'gmp_mae_pp': float(mean_absolute_error(y, rows.gmp_pct)) if rows.gmp_pct.notna().all() else None}


def ranking_comparison(rows: pd.DataFrame) -> dict:
    """Same-date, same-IPO choice sets only; never compare different-date books."""
    panels, pair_wins, pair_total = [], 0., 0
    eligible = rows[rows.gmp_pct.notna()]
    if 'ranking_eligible' in eligible:
        eligible = eligible[eligible.ranking_eligible]
    for day, group in eligible.groupby('decision_market_date'):
        if len(group) < 2:
            continue
        ranked = group.assign(priority=risk_score(group.prediction, group.lower))
        # Deterministic ID ties, independent of realized outcomes.
        chosen = ranked.sort_values(['priority', 'company_name'], ascending=[False, True]).iloc[0]
        gmp = ranked.sort_values(['gmp_pct', 'company_name'], ascending=[False, True]).iloc[0]
        if chosen.priority <= 0:
            chosen_name, model_result = 'SKIP ALL', 0.
        else:
            chosen_name, model_result = chosen.company_name, float(chosen.target_return)-COST_SCENARIO_PP
        if gmp.gmp_pct <= COST_SCENARIO_PP:
            gmp_name, gmp_result = 'SKIP ALL', 0.
        else:
            gmp_name, gmp_result = gmp.company_name, float(gmp.target_return)-COST_SCENARIO_PP
        panels.append({'decision_date': day, 'ipos': len(group),
                       'model_choice': chosen_name, 'gmp_choice': gmp_name,
                       'model_conditional_return_pct': model_result,
                       'gmp_conditional_return_pct': gmp_result,
                       'difference_pp': model_result-gmp_result})
        records = ranked.to_dict('records')
        for i, a in enumerate(records):
            for b in records[i+1:]:
                true_diff = a['target_return'] - b['target_return']
                if true_diff == 0:
                    continue
                guessed = a['priority'] - b['priority']
                pair_wins += .5 if guessed == 0 else float(guessed * true_diff > 0)
                pair_total += 1
    if not panels:
        return {'panels': 0, 'choices': [], 'better_than_gmp_established': False}
    delta = np.array([p['difference_pp'] for p in panels])
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    ci = np.quantile(boot, [.025, .975]).tolist()
    return {'panels': len(panels), 'choices': panels,
            'model_mean_conditional_return_pct': float(np.mean([p['model_conditional_return_pct'] for p in panels])),
            'gmp_mean_conditional_return_pct': float(np.mean([p['gmp_conditional_return_pct'] for p in panels])),
            'paired_difference_pp': float(delta.mean()), 'paired_bootstrap_95_pp': ci,
            'model_wins': int((delta > 0).sum()), 'gmp_wins': int((delta < 0).sum()),
            'ties': int((delta == 0).sum()), 'pairwise_ranking_accuracy': pair_wins/pair_total if pair_total else None,
            'better_than_gmp_established': bool(len(panels) >= 20 and ci[0] > 0)}


def evaluate_track(frame: pd.DataFrame, with_gmp=False):
    if with_gmp:
        frame = frame[frame.gmp_pct.notna()].copy()
    candidates, models = {}, {}
    for name in ('ridge', 'forest', 'boosted_huber'):
        development = []
        for year in (2022, 2023):
            train, test = frame[frame.year < year], frame[frame.year == year]
            if min(len(train), len(test)) < 10:
                continue
            model = fit_return(train, name, with_gmp)
            pred = predict_return(model, test, with_gmp)
            part = test.copy()
            part['prediction'], part['residual'] = pred, test.target_return.to_numpy()-pred
            development.append(part)
        if development:
            dev = pd.concat(development)
            candidates[name] = float(mean_absolute_error(dev.target_return, dev.prediction))
            models[name] = dev
    if not candidates:
        raise ValueError('Insufficient earlier-year data for honest model selection')
    selected = min(candidates, key=candidates.get)
    residuals = models[selected].residual.to_numpy()
    bounds = np.quantile(residuals, [.1, .9])
    bounds = [float(min(bounds[0], 0)), float(max(bounds[1], 0))]
    train, test = frame[frame.year < 2024], frame[frame.year == 2024].copy()
    confirmation_model = fit_return(train, selected, with_gmp)
    test['prediction'] = predict_return(confirmation_model, test, with_gmp)
    test['lower'] = np.maximum(-100, test.prediction + bounds[0])
    test['upper'] = test.prediction + bounds[1]
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=1000, random_state=42))
    classifier.fit(feature_frame(train, with_gmp), train.target_return > 0)
    test['probability'] = classifier.predict_proba(feature_frame(test, with_gmp))[:, 1]
    train_x, test_x = feature_frame(train, with_gmp), feature_frame(test, with_gmp)
    test['ranking_eligible'] = test_x.ge(train_x.min()).all(axis=1) & test_x.le(train_x.max()).all(axis=1)
    paired = test[test.gmp_pct.notna()].copy()
    paired_result = summary(paired)
    if len(paired):
        # Positive = the model has lower error. Resample IPOs, not forecasts.
        improvement = (paired.target_return-paired.gmp_pct).abs().to_numpy() - (paired.target_return-paired.prediction).abs().to_numpy()
        rng = np.random.default_rng(42)
        paired_result['mae_improvement_over_gmp_pp'] = float(improvement.mean())
        paired_result['mae_improvement_bootstrap_95_pp'] = np.quantile(
            [rng.choice(improvement, len(improvement), replace=True).mean() for _ in range(2000)], [.025, .975]).tolist()
    test_prior = np.repeat(float(train.target_return.median()), len(test))
    final_model = fit_return(frame, selected, with_gmp)
    classifier.fit(feature_frame(frame, with_gmp), frame.target_return > 0)
    artifact = {'regressor': final_model, 'classifier': classifier, 'residual_bounds': bounds,
                'with_gmp': with_gmp, 'candidate': selected, 'train_rows': len(frame),
                'feature_min': feature_frame(frame, with_gmp).min().to_dict(),
                'feature_max': feature_frame(frame, with_gmp).max().to_dict()}
    result = {'selected_candidate': selected, 'development_mae_pp': candidates,
              'residual_calibration_rows': len(residuals), 'residual_bounds_pp': bounds,
              'confirmation_2024': summary(test), 'paired_2024': paired_result,
              'training_median_baseline_mae_pp': float(mean_absolute_error(test.target_return, test_prior)),
              'confirmation_ranking_supported_rows': int(test.ranking_eligible.sum()),
              'largest_return_misses': test.assign(error=(test.target_return-test.prediction).abs()).nlargest(8, 'error')[
                  ['company_name', 'prediction', 'target_return', 'gmp_pct', 'error']].replace({np.nan: None}).to_dict('records'),
              'ranking_2024': ranking_comparison(test),
              'training_rows': len(frame)}
    return artifact, result, test


def render(report):
    lines = ['# Module 1 — Pre-close return forecasts and IPO priority test', '',
             '## What was built', '',
             'A separate return regressor and early-book probability classifier, trained only on 2021–2024 IPOs. '
             'Inputs are penultimate-session QIB/HNI/retail, issue price and size. The paired variant learns '
             'a correction to a dated GMP quote. The existing final-book classifier is unchanged.', '',
             'Candidates are fixed ridge, random forest and Huber boosting. Selection uses earlier-year '
             '2022/2023 predictions; 2024 is retrospective confirmation, not a blind or prospective holdout. '
             'The 80% band uses earlier-year out-of-fold residuals; observed coverage is reported, not assumed.', '',
             '## Results', '',
             '| Track | 2024 IPOs | Return MAE (percentage points) | GMP-only MAE on same cases | 80% band coverage | Selection panels | Model minus GMP |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in report['tracks'].items():
        s, r = row['confirmation_2024'], row['ranking_2024']
        gmp_mae = f"{s['gmp_mae_pp']:.2f}" if s['gmp_mae_pp'] is not None else 'See paired track'
        delta = f"{r['paired_difference_pp']:+.2f} pp" if r['panels'] else 'Not estimable'
        lines.append(f"| {name} ({row['selected_candidate']}) | {s['n']} | {s['mae_pp']:.2f} | {gmp_mae} | {s['interval_80_coverage']:.1%} | {r['panels']} | {delta} |")
    lines += ['', '## Does it beat GMP-only selection?', '', report['conclusion'], '',
              'Selection compares only IPOs with the same snapshot date and a valid preceding-date GMP quote. '
              'Same-day GMP is excluded because its update clock is unknown. Both selectors use the same '
              'feature-supported IPOs; exclusions remain in the prediction ledger. Risk score = central return '
              'minus 0.5 × downside-to-zero at the lower band minus a 0.5 percentage-point cost scenario. '
              'The top positive score is selected, otherwise skip all. GMP picks its highest positive quote '
              'after the same cost hurdle. No parameters were tuned on 2024 selection profits.', '',
              'These are conditional-on-allotment opening returns, NOT portfolio profits. The 0.5% cost '
              'is a sensitivity assumption, not an exchange fee schedule. No allotment probabilities, '
              'capital reuse or guaranteed execution are invented.', '']
    preferred = report['tracks'].get('bids_plus_gmp', report['tracks']['bids_only'])
    paired = preferred['paired_2024']
    if paired['n']:
        lines += [f"Paired return errors: model {paired['mae_pp']:.2f} pp versus GMP {paired['gmp_mae_pp']:.2f} pp "
                  f"on {paired['n']} IPOs; model improvement {paired['mae_improvement_over_gmp_pp']:+.2f} pp, "
                  f"bootstrap 95% interval {paired['mae_improvement_bootstrap_95_pp'][0]:+.2f} to "
                  f"{paired['mae_improvement_bootstrap_95_pp'][1]:+.2f} pp.", '']
    lines += ['## Every paired choice', '',
              '| Date | IPOs | Model selection | GMP selection | Model return after scenario | GMP return after scenario |',
              '|---|---:|---|---|---:|---:|']
    for p in preferred['ranking_2024']['choices']:
        lines.append(f"| {p['decision_date']} | {p['ipos']} | {p['model_choice']} | {p['gmp_choice']} | {p['model_conditional_return_pct']:+.2f}% | {p['gmp_conditional_return_pct']:+.2f}% |")
    lines += ['', '## Limits that matter', ''] + ['- '+x for x in report['limitations']]
    lines += ['', '## Biggest return misses in the paired model', '',
              '| IPO | Predicted central return | Actual opening return | Dated GMP | Model error |',
              '|---|---:|---:|---:|---:|']
    for miss in preferred['largest_return_misses']:
        gmp_text = f"{miss['gmp_pct']:+.2f}%" if miss['gmp_pct'] is not None else 'Missing'
        lines.append(f"| {miss['company_name']} | {miss['prediction']:+.2f}% | {miss['target_return']:+.2f}% | {gmp_text} | {miss['error']:.2f} pp |")
    lines += ['', 'These are observed errors, not causal explanations. The model cannot infer demand shocks '
              'or opening-auction supply solely from earlier subscription multiples and an unofficial quote. '
              'The complete ledger retains every evaluated IPO, including these misses.']
    return '\n'.join(lines)+'\n'


def main():
    frame, audit = prepare_returns()
    artifacts, tracks, ledgers = {}, {}, []
    for name, with_gmp in (('bids_only', False), ('bids_plus_gmp', True)):
        subset = frame[frame.gmp_pct.notna()] if with_gmp else frame
        if any(len(subset[subset.year == y]) < 10 for y in (2021, 2022, 2023, 2024)):
            if with_gmp:
                continue
            raise ValueError('Insufficient source coverage')
        artifacts[name], tracks[name], ledger = evaluate_track(frame, with_gmp)
        ledger['track'] = name
        ledgers.append(ledger)
    verdict = tracks.get('bids_plus_gmp', tracks['bids_only'])['ranking_2024']
    proven = verdict['better_than_gmp_established']
    conclusion = ('A positive retrospective paired selection difference passed the minimum-panel and uncertainty gate. '
                  'Prospective superiority is still unproven.' if proven else
                  'A reliable advantage over GMP-only selection has NOT been established. '
                  'Use the new forecasts as experimental research, not a proven superior allocation strategy.')
    files = ['data/canonical_listed_ipos.csv', 'data/daywise_subscription_research_cohort.csv',
             'data/nse_verified_offer_dates.csv', 'data/raw_daily_market_time_series.csv',
             'data/preclose_gmp_history.csv']
    hashes = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).exists()}
    metadata = {'model_version': VERSION, 'training_years': [2021, 2022, 2023, 2024],
                'trained_through_date': '2024-12-31', 'feature_cols': FEATURES,
                'training_stage': 'penultimate_session_end', 'input_sha256': hashes,
                'status': 'experimental_retrospective_challenger',
                'risk_weight': RISK_WEIGHT, 'cost_scenario_pp': COST_SCENARIO_PP,
                'better_than_gmp_established': proven, 'conclusion': conclusion}
    import sklearn
    metadata['runtime'] = {'sklearn': sklearn.__version__, 'numpy': np.__version__,
                           'pandas': pd.__version__, 'seed': 42}
    report = metadata | {'generated_at_ist': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
                         'cohort_audit': audit, 'tracks': tracks, 'limitations': [
        'Daily subscription and GMP archive pages can have been edited later; not immutable investor-time evidence.',
        'GMP coverage is incomplete and selected. Quotes are at least a day older than the bid snapshot.',
        'Training is penultimate-session end-of-day, not a validated day-one or final-hour model.',
        'Live intraday or closing-day use is a stage extrapolation; prospective accuracy remains unmeasured.',
        'A residual band is empirical and can miss; the upper bound is not a cap or guarantee.',
        'Small simultaneous choice sets and repeated overlapping offer exposures limit inference.',
        'No paper numbers or final-book classifier were replaced by this experiment.'
    ]}
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata | {'tracks': artifacts}, BUNDLE)
    metadata['artifact_sha256'] = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    META.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    REPORT.with_suffix('.md').write_text(render(report), encoding='utf-8')
    pd.concat(ledgers).to_csv(ROOT/'output/module1_return_prediction_ledger.csv', index=False)
    print(json.dumps({'cohort': audit, 'tracks': tracks, 'conclusion': conclusion}, indent=2))


if __name__ == '__main__':
    main()
