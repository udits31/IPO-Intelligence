"""Present the locked historical benchmark in research without inventing live calls."""
import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import pandas as pd
from services.historical_bid_service import name_key

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json'
INPUTS=ROOT/'data/audit/module1_stage_aligned_inputs_20261026.csv'
OPENINGS=ROOT/'data/audit/module1_exchange_listing_open_audit.csv'


def number(value):
    try:return float(value) if math.isfinite(float(value)) else None
    except (TypeError,ValueError):return None


def text(value):return None if pd.isna(value) else str(value)


@lru_cache(maxsize=3)
def _catalog(lock_version,input_version,opening_version):
    from services.prospective_module1 import bundles
    from ml.research_module1_forward_update import SETS
    from services.listing_outcome_service import verified_listing_outcomes
    from services.two_track_evidence import present
    locked=json.loads(LOCK.read_text(encoding='utf8'));protocol,model,_=bundles()
    if locked['active']['module1_artifact_sha256']!=protocol['model']['sha256']:
        raise ValueError('Historical benchmark uses a different model artifact')
    required={k for config in model['expert_configs'].values() for k in SETS.get(config['feature_set'],['gmp_pct'])}
    inputs=pd.read_csv(INPUTS,dtype={'issuer_key':str}).set_index('issuer_key')
    audit=pd.read_csv(OPENINGS,dtype={'ipo_id':str})
    audit={r['ipo_id']:r for r in audit.to_dict('records') if r.get('status')=='verified_daily_exchange_archive'}
    live={name_key(r['company_name']):r for r in verified_listing_outcomes().values()}
    input_sha=hashlib.sha256(INPUTS.read_bytes()).hexdigest();lock_sha=hashlib.sha256(LOCK.read_bytes()).hexdigest()
    records=[]
    for original in locked['module1']['records']:
        key=str(original['issuer_key'])
        if key not in inputs.index:continue
        row=inputs.loc[key]
        if isinstance(row,pd.DataFrame):raise ValueError('Ambiguous benchmark input issuer key: '+key)
        if name_key(row.company_name)!=name_key(original['company_name']):raise ValueError('Benchmark input identity mismatch')
        features={k:float(row[k]) for k in required if k in row and number(row[k]) is not None}
        for k in ('context_total_mult','context_bnii_mult'):
            if k in row and number(row[k]) is not None:features[k]=float(row[k])
        issue=math.expm1(features['log_offer_cap_price'])
        source=audit.get(key) or live.get(name_key(original['company_name'])) or {}
        source_issue=number(source.get('issue_price'))
        opening=number(source.get('exchange_open') or source.get('exchange_open_price') or source.get('nse_open_price'))
        outcome=None
        if source_issue and opening and math.isclose(issue,source_issue,abs_tol=.01):
            actual=100*(opening/source_issue-1)
            if math.isclose(actual,float(original['actual_return_pct']),abs_tol=.001):
                raw_path=source.get('raw_archive_path')
                raw_sha=source.get('archive_sha256')
                if raw_path and isinstance(raw_path,str) and raw_sha:
                    file=ROOT/raw_path
                    if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=raw_sha:
                        raise ValueError('Opening archive failed checksum: '+key)
                outcome=dict(actual_open=opening,issue_price=source_issue,return_pct=actual,positive=opening>source_issue,
                    listing_date=text(source.get('listing_date') or row.get('listing_date')),
                    source_url=text(source.get('source_url')),archive_sha256=text(raw_sha))
        stage='Day 2 close' if str(row.get('verified_stage','')).lower()=='day2' else 'Historical dated input snapshot'
        asof=text(row.get('day2_timestamp')) if stage=='Day 2 close' else text(row.get('cutoff_at_ist'))
        bid=dict(qib=math.expm1(features['log_qib']),hni=math.expm1(features['log_hni']),retail=math.expm1(features['log_retail']),
            observed_at_ist=asof,source=text(row.get('bid_source_url') or row.get('source_url')),
            raw_sha256=text(row.get('bid_source_sha256')),stage=stage,
            time_basis='nominal_eod_stage_exact_publisher_clock_unverified' if stage=='Day 2 close' else 'historical_source')
        gmp=dict(gmp_rs=number(row.get('gmp_rs')),gmp_date=text(row.get('gmp_date')),url=text(row.get('gmp_source_url')))
        total=number(row.get('day2_total')) if stage=='Day 2 close' else number(row.get('context_total_mult'))
        record=dict(company_name=original['company_name'],issuer_key=key,cohort=original['cohort'],
            mode='historical_benchmark',probability=original['probability'],predicted_positive=original['standard_positive'],
            correct=original['standard_positive']==outcome['positive'] if outcome else None,
            outcome=outcome,features=features,missing_features=sorted(required-set(features)),
            quality='complete' if required<=features.keys() else 'provisional_missing_inputs',
            bid=bid,sources=dict(gmp=gmp,market_date=text(row.get('macro_source_date'))),
            gmp_probability=original['gmp_probability'],capital_plan=original['plan'],
            model_sha=protocol['model']['sha256'],input_snapshot_sha256=input_sha,locked_evidence_sha256=lock_sha,
            input_stage=stage,total_subscription=total,offer_terms=dict(ofs_ratio=number(row.get('ofs_ratio')),
                retail_quota_pct=number(row.get('retail_quota_pct')),lot_size=number(row.get('lot_size_shares'))),
            listing_date=text(row.get('listing_date')),issue_price=issue,
            computed_at_ist=locked['created_at_utc'],
            limitation='Same locked historical test displayed on Model Performance, using recovered dated inputs. This is a retrospective benchmark estimate, not a call recorded while this IPO was open. The input stage is shown explicitly; a Day-2 book is not the final closing-day book.')
        record.update(present(record['probability'],outcome['positive'] if outcome else None,original['plan'].get('tier')))
        records.append(record)
    return records


def catalog():
    return _catalog(LOCK.stat().st_mtime_ns,INPUTS.stat().st_mtime_ns,OPENINGS.stat().st_mtime_ns)


@lru_cache(maxsize=256)
def verify_prediction(key,snapshot_sha,model_sha):
    from services.prospective_module1 import predict
    row=next(r for r in catalog() if r['issuer_key']==key)
    replay,_=predict(row['features'])
    if not math.isclose(replay,row['probability'],abs_tol=1e-9,rel_tol=0):
        raise ValueError('Historical inputs do not reproduce the locked model score: '+key)
    return True


def lookup(item):
    candidates=[r for r in catalog() if name_key(r['company_name'])==name_key(item.get('company_name') or '')]
    if len(candidates)!=1:return None
    record=candidates[0];price=number(item.get('issue_price'))
    if price is not None and not math.isclose(price,record['issue_price'],abs_tol=.01):return None
    date=item.get('listing_date')
    if date and record.get('listing_date') and str(date)[:10]!=record['listing_date'][:10]:return None
    verify_prediction(record['issuer_key'],record['input_snapshot_sha256'],record['model_sha'])
    record=copy.deepcopy(record)
    group='test_69' if record['cohort']=='test_69' else 'recent_24_verified'
    members=[r for r in catalog() if (r['cohort']=='test_69')==(group=='test_69')]
    resolved=[r for r in members if r['correct'] is not None]
    correct=sum(r['correct'] for r in resolved)
    table=[{k:copy.deepcopy(r.get(k)) for k in ('company_name','probability','predicted_positive','correct','outcome','application_screen','screen_result')} for r in members]
    record.update(comparison_records=table,benchmark_cohort=group,
        summary=dict(correct=correct,evaluated=len(resolved),accuracy_pct=100*correct/len(resolved) if resolved else None,
            always_positive_correct=sum(r['outcome']['positive'] for r in resolved),
            gmp_correct=sum((r['gmp_probability']>=.625)==r['outcome']['positive'] for r in resolved),
            gmp_paired_count=len(resolved),model_correct_on_gmp_pairs=correct,pending_openings=len(members)-len(resolved)))
    return record
