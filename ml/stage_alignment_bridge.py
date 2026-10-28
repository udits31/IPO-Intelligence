"""Stage-labelled recovery and training-only bridges; estimates never become observations."""
import gzip, hashlib, html, json, re, sqlite3
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from services.saved_preclose_service import ROOT
from services.historical_bid_service import name_key
from ml.recover_progressive_totals import cached_pages

CATS = ('qib', 'hni', 'retail')
LOGS = ['log_' + c for c in CATS]
GROWTH = ['log_' + c + '_growth' for c in CATS]


def number(value):
    from services.subscription_numbers import subscription_multiple
    return subscription_multiple(value)


def parse_ipoji(content):
    match = re.search(r'<script[^>]*id=["\']ipo-subscription-info-for-chart["\'][^>]*>(.*?)</script>', content, re.S)
    if not match:
        return []
    try:
        payload = json.loads(html.unescape(match[1]))
    except ValueError:
        return []
    out = []
    for item in payload:
        match = re.search(r'Day\s*(\d+)\s*-\s*(\d{2}-\d{2}-\d{2,4})\s+(\d{1,2}:\d{2}\s*[AP]M)', item.get('key', ''), re.I)
        if not match:
            continue
        fmt = '%d-%m-%y %I:%M %p' if len(match[2]) == 8 else '%d-%m-%Y %I:%M %p'
        stamp = pd.to_datetime(match[2] + ' ' + match[3], format=fmt).tz_localize('Asia/Kolkata')
        v = item.get('value', {})
        out.append(dict(day=int(match[1]), timestamp=stamp.isoformat(),
                        qib=number(v.get('QIB')), hni=number(v.get('NII')), retail=number(v.get('RII')),
                        total=number(v.get('Total')), clock_quality='publisher_reported_recovered_later'))
    return out


def complete(row):
    return all(number(row.get(c)) is not None for c in CATS)


def bridge_x(rows, direction):
    prefix = 'day2_' if direction == 'forward' else 'cutoff_'
    x = pd.DataFrame({c: np.log1p(rows[prefix + c]) for c in CATS}, index=rows.index)
    x['log_issue_size'] = rows.log_issue_size
    x['gmp_positive'] = (rows.gmp_pct > 0).astype(float).where(rows.gmp_pct.notna())
    x['qib_retail_mix'] = x.qib - x.retail
    # Pooled ridge interactions avoid fitting unstable tiny stratum regressions.
    x['large_issue'] = (rows.log_issue_size >= np.log1p(1000)).astype(float)
    for c in CATS:
        x[c + '_size_interaction'] = x[c] * x.large_issue
        x[c + '_gmp_interaction'] = x[c] * x.gmp_positive
    return x


class StageBridge:
    def fit(self, pairs, *, allowed_ids):
        if not set(pairs.issuer_key) <= set(allowed_ids):
            raise ValueError('Bridge fitting attempted outside training issuers')
        if pairs.issuer_key.duplicated().any():
            raise ValueError('Bridge pairs must represent distinct training issuers')
        self.training_ids = pairs.issuer_key.tolist()
        self.models = {}
        self.n_pairs = len(pairs)
        if len(pairs) < 20:
            self.status = 'insufficient_distinct_training_pairs'
            return self
        for direction, target in [('forward', 'cutoff_'), ('inverse', 'day2_')]:
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=10.))
            model.fit(bridge_x(pairs, direction), np.log1p(pairs[[target + c for c in CATS]].to_numpy()))
            self.models[direction] = model
        self.status = 'fitted_training_only'
        return self

    def predict(self, rows, direction):
        if direction not in self.models:
            raise ValueError('No fitted bridge: missing genuine training pairs cannot be invented')
        return np.maximum(0., np.expm1(self.models[direction].predict(bridge_x(rows, direction))))


def recover(frame):
    """Read-only cache/ledger search, retaining independent source choices and conflicts."""
    histories = defaultdict(list)
    by_name = {name_key(r.company_name): r.issuer_key for r in frame.itertuples()}
    def add(key, items, source, sha):
        for item in items:
            histories[key].append({**item, 'source_url': source, 'sha256': sha})
    paths = []
    for directory in ('module1_later_holdout_20261024', 'module1_other_sources_20261024', 'module1_public_financials_20261024','final_lock_day2_20261026'):
        paths += list((ROOT / 'data/archive' / directory).glob('*.json.gz'))
    ids = set(frame.issuer_key)
    for path in paths:
        key = path.name.split('.')[0]
        if key not in ids:
            continue
        try:
            data = json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError):
            continue
        if path.parent.name=='final_lock_day2_20261026' and not data.get('audit',{}).get('identity_offer_dates_and_cap_verified'):
            continue
        for content, url in cached_pages(data):
            add(key, parse_ipoji(content), url, hashlib.sha256(content.encode()).hexdigest())
    recent = json.loads((ROOT / 'output/listed_closing_day_backtest.json').read_text())['records']
    for r in recent:
        key = by_name.get(name_key(r['company_name']))
        if key is None:
            continue
        for h in r.get('sources', {}).get('growth_history', []):
            stamp = pd.to_datetime(h['date'] + ' ' + h['displayed_time']).tz_localize('Asia/Kolkata')
            add(key, [dict(day=h['day'], timestamp=stamp.isoformat(), qib=h['qib_subscription'],
                          hni=h['hni_subscription'], retail=h['retail_subscription'], total=None,
                          clock_quality='publisher_reported_recovered_later')], h.get('url'), h.get('raw_sha256'))
    # Include cached secondary pages in SQLite, without writing to the ledger.
    with sqlite3.connect('file:' + str(ROOT / 'data/telemetry/snapshots.db') + '?mode=ro', uri=True) as conn:
        for url, blob, sha in conn.execute('SELECT source_url,raw_html_gzip,raw_sha256 FROM historical_subscription_evidence_v1'):
            try:
                content = gzip.decompress(blob).decode('utf8', errors='replace')
            except OSError:
                continue
            normalized = name_key(html.unescape(re.sub('<[^>]+>', ' ', content)))
            matches = [key for name, key in by_name.items() if name in normalized]
            if len(matches) == 1:
                add(matches[0], parse_ipoji(content), url, sha)
        captures = pd.read_sql_query('SELECT company_name,fetched_at_ist,qib_x,nii_x,retail_x,status,raw_sha256,subscription_url FROM subscription_capture_v2', conn)
    observations = pd.read_csv(ROOT / 'data/subscription_observations.csv')
    premium = {r['issuer_key']: r for r in map(json.loads, (ROOT / 'data/audit/ipopremium_recovered_20261026.jsonl').read_text().splitlines()) if r['identity_verified']}
    audit = []; out = frame.copy(); pairs = []; final_pairs = []
    for index, row in frame.iterrows():
        key = row.issuer_key; cutoff = pd.Timestamp(row.cutoff_at_ist)
        candidates = [h for h in histories[key] if complete(h) and pd.Timestamp(h['timestamp']) <= cutoff]
        day2 = [h for h in candidates if h['day'] == 2]
        chosen = max(day2, key=lambda h: h['timestamp']) if day2 else None
        quality = 'publisher_reported_recovered_later' if chosen else None
        # Day labels remain an explicit stage-only research assumption, never a fabricated clock.
        p = premium.get(key)
        close_date = row.issue_close_date if pd.notna(row.issue_close_date) else row.decision_date
        if chosen is None and p and (row.get('bid_training_stage') == 'penultimate_session' or cutoff.date() == pd.Timestamp(close_date).date()):
            d = next((d for d in p['daywise'] if d['day'] == 2), None)
            if d:
                v = d['values']; proposed = dict(qib=v.get('QIBs'), hni=v.get('HNIs'), retail=v.get('Retail'), total=v.get('Total'))
                if complete(proposed):
                    chosen = proposed; quality = 'source_reported_day2_no_individual_clock'
        # Prefer a real late-Day2 capture if one exists. A midday reading is not Day2 EOD.
        capture_matches = captures[captures.company_name.map(name_key).eq(name_key(row.company_name))]
        real_day2 = []
        day2_date = pd.Timestamp(chosen['timestamp']).date() if chosen and chosen.get('timestamp') else None
        for capture in capture_matches.to_dict('records'):
            stamp = pd.to_datetime(capture['fetched_at_ist'], errors='coerce')
            v = dict(qib=number(capture['qib_x']),hni=number(capture['nii_x']),retail=number(capture['retail_x']))
            if pd.notna(stamp) and stamp.tzinfo is not None and stamp<=cutoff and stamp.date()==day2_date and 16<=stamp.hour<17 and complete(v):
                real_day2.append(dict(**v,total=None,timestamp=stamp.isoformat(),source_url=capture['subscription_url'],sha256=capture['raw_sha256'],day=2))
        if real_day2:
            chosen=max(real_day2,key=lambda h:h['timestamp']);quality='actual_local_capture_before_Day2_session_close'
        for c in CATS:
            out.loc[index, 'day2_' + c] = chosen.get(c) if chosen else np.nan
            out.loc[index, 'cutoff_' + c] = np.expm1(row['log_' + c])
        if chosen:
            out.loc[index, 'day2_total'] = chosen.get('total')
        stamp = chosen.get('timestamp') if chosen else None
        out.loc[index, 'day2_timestamp'] = stamp
        first = [h for h in candidates if h['day'] == 1 and (not stamp or pd.Timestamp(h['timestamp']) < pd.Timestamp(stamp))]
        first = max(first, key=lambda h: h['timestamp']) if first else None
        for c in CATS:
            out.loc[index, 'day1_' + c] = first[c] if first else np.nan
        same_day2 = bool(stamp and pd.Timestamp(stamp).date() == cutoff.date()
                         and all(np.isclose(chosen[c],np.expm1(row['log_'+c]),atol=.02,rtol=.0001) for c in CATS)) or row.get('bid_training_stage') == 'penultimate_session'
        stage = 'day2' if same_day2 else 'closing_pre_cutoff' if cutoff.date() == pd.Timestamp(row.issue_close_date if pd.notna(row.issue_close_date) else row.decision_date).date() else 'unresolved_interim'
        out.loc[index, 'verified_stage'] = stage
        out.loc[index, 'is_final_postclose_book'] = 0. if stage in ('day2','closing_pre_cutoff') else np.nan
        out.loc[index, 'day2_quality'] = quality
        # Only true closing-day pairs, not Day2/Day2 ratios of 1, can train the bridge.
        match = [h for h in candidates if pd.Timestamp(h['timestamp']).date() == cutoff.date()
                 and pd.Timestamp(h['timestamp']).hour >= 14 and pd.Timestamp(h['timestamp']).hour < 17
                 and all(np.isclose(h[c], np.expm1(row['log_' + c]), atol=.02, rtol=.0001) for c in CATS)]
        usable = bool(chosen and stamp and pd.Timestamp(stamp).date() < cutoff.date() and match and stage == 'closing_pre_cutoff')
        if usable:
            pairs.append({**row.to_dict(), **{'day2_' + c: chosen[c] for c in CATS}, **{'cutoff_' + c: np.expm1(row['log_' + c]) for c in CATS}})
        # Fix category trajectory in research inputs only. Growth means current minus actual previous day.
        if chosen and stage == 'closing_pre_cutoff':
            for c in CATS:
                out.loc[index, 'log_' + c + '_growth'] = row['log_' + c] - np.log1p(chosen[c])
            out.loc[index, 'trajectory_basis'] = quality
        same_name = captures[captures.company_name.map(name_key).eq(name_key(row.company_name))]
        valid_captures = int(same_name[['qib_x','nii_x','retail_x']].notna().all(axis=1).sum())
        obs = observations[observations.ipo_id.astype(str).eq(key)]
        audit.append(dict(issuer_key=key,company_name=row.company_name,cohort=row.cohort,year=row.year,
                          previous_total_ratio_present=pd.notna(row.day2_to_cutoff_conviction),day2_categories_recovered=chosen is not None,
                          day2_quality=quality,day2_timestamp=stamp,day2_source_url=chosen.get('source_url') if chosen else p.get('source_url') if p else None,
                          stage=stage,eligible_bridge_pair=usable,local_capture_rows_with_numbers=valid_captures,
                          observation_csv_rows=len(obs),independent_intraday_clock_verified=False))
    pair_frame = pd.DataFrame(pairs)
    if pair_frame.empty:
        pair_frame = pd.DataFrame(columns=['issuer_key','year','cohort','log_issue_size','gmp_pct'] + ['day2_'+c for c in CATS] + ['cutoff_'+c for c in CATS])
    return out, pd.DataFrame(audit), pair_frame


def run():
    frame = pd.read_csv(ROOT / 'data/audit/module1_inputs_after_ipopremium_20261026.csv', dtype={'issuer_key':str})
    out,audit,pairs = recover(frame)
    training = pairs[pairs.cohort.eq('historical') & pairs.year.le(2025)]
    allowed = frame[frame.cohort.eq('historical') & frame.year.le(2025)].issuer_key
    bridge = StageBridge().fit(training, allowed_ids=allowed)
    out.to_csv(ROOT / 'data/audit/module1_stage_aligned_inputs_20261026.csv',index=False)
    audit.to_csv(ROOT / 'output/module1_day2_recovery_audit_20261026.csv',index=False)
    pairs.to_csv(ROOT / 'data/audit/module1_stage_bridge_pairs_20261026.csv',index=False)
    report=dict(bridge_status=bridge.status,training_pair_n=len(training),previous_total_ratio_n=161,
                previous_total_ratios_equal_one=int(np.isclose(frame.loc[frame.cohort.eq('historical') & frame.year.le(2025),'day2_to_cutoff_conviction'].dropna(),1.).sum()),
                coverage={name:dict(n=int(mask.sum()),day2_categories=int(out.loc[mask,'day2_qib'].notna().sum()),
                    closing_stage=int(out.loc[mask,'verified_stage'].eq('closing_pre_cutoff').sum()))
                  for name,mask in [('training_290',out.cohort.eq('historical') & out.year.le(2025)),('test_69',out.cohort.eq('historical') & out.year.eq(2026)),('recent_24',out.cohort.eq('recent'))]},
                final_book_deflation_status='not_fitted_without_paired_preclose_and_postclose_training_books',
                limitations=['Total-ratio coverage does not establish distinct Day2/Day3 category pairs.',
                  'Recovered publisher clocks are not independent archived capture clocks.',
                  'Day-label-only inputs are stage assumptions and not timestamp-certified observations.',
                  'No statutory quota fallback inferred without the issuer eligibility/reservation regime.'])
    (ROOT/'output/module1_stage_alignment_bridge_20261026.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return out,audit,pairs,bridge

if __name__=='__main__':run()
