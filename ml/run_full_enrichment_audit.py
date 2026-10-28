"""Rebuild enrichment benchmarks and verify the preserved forecast prefix.

Default is an offline rebuild using existing exchange caches. Network requests
must be explicitly budgeted; no frozen model or forecast is replaced.
"""
import argparse,hashlib,json,math,sqlite3,re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from services.point_in_time import eligible

OUT=ROOT/'output/full_enrichment_audit_report_20261025.json'


def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y,p,apply_threshold=.82,decision_threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);call=p>=decision_threshold;apply=p>=apply_threshold;loser=y==0
    n=len(y);correct=int((call==y).sum());rate=correct/n;z=1.96
    center=(rate+z*z/(2*n))/(1+z*z/n);width=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n,correct=correct,accuracy_pct=100*rate,accuracy_wilson_95_pct=[100*(center-width),100*(center+width)],
        positive_calls=int(call.sum()),nonpositive_calls=int((~call).sum()),wrong_positive_calls=int((call&loser).sum()),missed_positive_calls=int((~call&(y==1)).sum()),
        decision_threshold=decision_threshold,apply_threshold=apply_threshold,apply_n=int(apply.sum()),apply_precision_pct=100*float(y[apply].mean()) if apply.any() else None,
        skip_recall_on_losers_pct=100*float((~call)[loser].mean()) if loser.any() else None)


def integrity():
    baseline=read('output/full_enrichment_baseline_20261025.json');files={};ledgers={}
    for name,expected in baseline['files'].items():
        path=ROOT/name;actual=sha(path) if path.exists() else None
        files[name]=dict(unchanged=actual==expected,expected_sha256=expected,actual_sha256=actual)
        # Explicitly authorised live-policy versioning preserves the old replay
        # source under its original hash; model/ledger exceptions are forbidden.
        release=ROOT/'output/active_policy_lock_20261026.json'
        if name=='core/monitor_engine.py' and actual!=expected and release.exists():
            lock=json.loads(release.read_text());archive=ROOT/'core/monitor_r0_frozen_20261026.py'
            files[name]['authorized_versioned_policy_update']=bool(lock.get('module2_policy')=='M2_R1_WITH_DATED_FILINGS'
                and lock.get('active_monitor_sha256')==actual and archive.exists() and sha(archive)==expected)
            files[name]['original_replay_source']='core/monitor_r0_frozen_20261026.py'
    with sqlite3.connect(ROOT/'data/telemetry/snapshots.db',timeout=15) as conn:
        conn.row_factory=sqlite3.Row
        for table,record in baseline['immutable_prefixes'].items():
            primary='id' if table in ('frozen_module1_forecast_v1','automatic_decision_v1') else 'ipo_id'
            original=[]
            for key in record['keys']:
                row=conn.execute(f'SELECT * FROM {table} WHERE {primary}=?',(key,)).fetchone()
                original.append(dict(row) if row else None)
            actual=hashlib.sha256(json.dumps(original,sort_keys=True,default=str).encode()).hexdigest()
            total=conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            ledgers[table]=dict(original_rows=len(original),current_rows=total,original_prefix_unchanged=actual==record['sha256'],expected_sha256=record['sha256'],actual_sha256=actual)
    return dict(protected_files=files,forecast_ledgers=ledgers,all_preserved=all(r['unchanged'] or r.get('authorized_versioned_policy_update',False) for r in files.values()) and all(r['original_prefix_unchanged'] for r in ledgers.values()))


def coverage():
    deals=pd.read_csv(ROOT/'data/deal_structure_enriched.csv',dtype={'ipo_id':str})
    frame=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv');ids=set(frame.ipo_id.astype(int).astype(str))
    cutoffs={str(int(r.ipo_id)):str(r.decision_market_date)+'T17:00:00+05:30' for r in frame.itertuples()}
    for r in read('output/listed_closing_day_backtest.json')['records']:
        cutoffs['recent_'+hashlib.sha256(r['company_name'].encode()).hexdigest()[:10]]=r['cutoff_at_ist']
    fields=['fresh_issue_cr','ofs_issue_cr','ofs_ratio','qib_quota_pct','nii_quota_pct','retail_quota_pct','snii_subscription_mult','bnii_subscription_mult','lot_size_shares','anchor_total_cr','anchor_domestic_mf_pct','anchor_30d_unlock_date','anchor_90d_unlock_date','brlm_names','brlm_prior_listing_win_rate']
    groups={}
    for label,group in [('historical_359',deals[deals.ipo_id.isin(ids)]),('recent_24',deals[deals.ipo_id.str.startswith('recent_')]),('all_extracted',deals)]:
        values={}
        for field in fields:
            present=group[field].notna()
            if field=='brlm_names':present &= ~group[field].isin(['[]',''])
            admitted=0
            for row in group[present].to_dict('records'):
                verified=row.get(field+'_pit_verified')==True
                admitted+=int(eligible(dict(source_timestamp=row.get(field+'_source_timestamp'),source_timestamp_verified=verified),cutoffs.get(row['ipo_id'])))
            values[field]=dict(extracted=int(present.sum()),eligible_by_decision=admitted,missing=int((~present).sum()))
        groups[label]=dict(issuers=len(group),fields=values)
    return groups


def comparisons():
    enriched=read('output/module1_full_enrichment_benchmark_20261025.json')
    frozen=pd.read_csv(ROOT/'output/module1_standard_classifier_audit_20261025_2026_predictions.csv')
    challenger=pd.read_csv(ROOT/'output/module1_enriched_2026_predictions.csv')
    joined=frozen[['ipo_id','standard_probability']].merge(challenger,on='ipo_id',validate='one_to_one')
    if len(joined)!=69:raise ValueError('Module 1 fixed 69-case comparison changed')
    y=joined.target_binary
    later={name:metrics(y,p) for name,p in dict(frozen_model=joined.standard_probability,enriched_calibrated=joined.enriched_probability,gmp_fitted=joined.p_gmp_fitted,gmp_sign=joined.p_gmp_sign,always_positive=np.ones(len(joined))).items()}
    baseline_cutoff=read('output/prospective_module1_protocol.json')['gmp_baseline']['cutoff']
    later['gmp_fitted']=metrics(y,joined.p_gmp_fitted,decision_threshold=baseline_cutoff)
    recent=read('output/listed_closing_day_backtest.json')['records'];new={r['company_name']:r for r in enriched['recent']['records']}
    cases=[r for r in recent if r.get('outcome')]
    labels=[r['outcome']['positive'] for r in cases]
    recent_metrics={name:metrics(labels,p) for name,p in dict(frozen_model=[r['probability'] for r in cases],enriched_calibrated=[new[r['company_name']]['probability'] for r in cases],gmp_fitted=[r['gmp_probability'] for r in cases],gmp_sign=[int(r['gmp_simple_positive']) for r in cases],always_positive=np.ones(len(cases))).items()}
    recent_metrics['gmp_fitted']=metrics(labels,[r['gmp_probability'] for r in cases],decision_threshold=baseline_cutoff)
    errors=[]
    for r in joined.to_dict('records'):
        if (r['standard_probability']>=.5)!=bool(r['target_binary']) or (r['enriched_probability']>=.5)!=bool(r['target_binary']):
            errors.append(dict(cohort='69_historical_2026',company_name=r['company_name'],actual_open=r['exchange_open'],issue_price=r['issue_price'],actual_positive=bool(r['target_binary']),frozen_probability=r['standard_probability'],enriched_probability=r['enriched_probability'],gmp_pct=r['gmp_pct']))
    for r in cases:
        if not r['correct'] or not new[r['company_name']]['correct']:
            errors.append(dict(cohort='recent_23_verified',company_name=r['company_name'],actual_open=r['outcome']['actual_open'],issue_price=r['outcome']['issue_price'],actual_positive=r['outcome']['positive'],frozen_probability=r['probability'],enriched_probability=new[r['company_name']]['probability'],missing_inputs='; '.join(r.get('missing_features',[]))))
    pd.DataFrame(errors).to_csv(ROOT/'output/full_enrichment_module1_errors_20261025.csv',index=False)
    return dict(cohort_69=later,recent_23=recent_metrics,recent_total=len(recent),pending_outcomes=[r['company_name'] for r in recent if not r.get('outcome')],
        training_n=enriched['training_n'],training_end_year=enriched['training_end_year'],training_test_overlap=enriched['training_test_overlap'],
        enriched_selected_policy=enriched['test_69'],enriched_raw=enriched['raw_test_69'],return_cqr=enriched['return_cqr'],observed_new_training_features=enriched['observed_extra_training'],
        input_scope_note='New deal-structure fields are timestamp gated. Reused legacy financial panels are fiscal-period checked, but their original publication instants are not certified. This is not a fully as-published reconstruction of every input.',
        live_promotion=False,reason='The calibrated challenger did not beat the frozen model on the fixed 69 cases. Recent outcomes have already been inspected; neither cohort is a new untouched test.')


def run(*,reports_only=False,network_requests=0):
    if not reports_only:
        from ml.extract_cached_deal_structure import run as extract
        from scrapers.exchange_enrichment_scraper import run as backfill
        from ml.research_module1_enriched import run as module1
        from ml.train_module2_enriched import run as module2
        extract();backfill(network_requests);module1();module2()
    protection=integrity()
    exchange=read('output/exchange_enrichment_backfill.json')
    statuses={}
    for attempt in exchange.get('attempts',[]):statuses[attempt['status']]=statuses.get(attempt['status'],0)+1
    module2=read('output/module2_enriched_benchmark_20261025.json')
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='implemented_with_data_and_performance_gates',
        integrity=protection,cached_field_coverage=coverage(),module1=comparisons(),module2=module2,
        exchange_backfill={**{k:v for k,v in exchange.items() if k!='attempts'},'request_status_counts':statuses},
        capital_economics=dict(implemented=True,categories=['retail','snii','bnii'],exact_allotment_adjusted_portfolio_return=None,
            valid_application_count_coverage=0,status='Exact portfolio score unavailable: verified valid application counts and category allocation basis were not recovered.',
            scenario_endpoint='/api/v1/module1/capital-scenario',rules=['Charge financing on all borrowed blocked capital, including failed allotments.','Share oversubscription reciprocals and the bNII factor 5 are explicit scenarios, not exact lottery mechanics.','A category below 1x triggers a conservative optional gate, not a claim of certain loss.']),
        outstanding=['Recover versioned pre-close RHP/anchor/BRLM evidence and bind category split tables to their own update clocks.','Verify BSE-only NSE issuer opening: archive access has returned HTTP 403; no NSE self-listing substitute is used.','BSE delivery, filings and deals need a verified exchange adapter.','Historical delivery/bulk original publication clocks and institutional client classifications remain incomplete.','Full filing attachments are not extracted by this exchange-summary collector.','Trendlyne metrics without historical publication clocks remain supplementary context.','Do not promote an enriched monitor until it beats baselines under a frozen prospective protocol.'],
        non_promoted=True,notes=['No git push performed.','Original prediction rows are checked by their saved keys; later legitimate appends do not invalidate the prefix check.','69-case and recent-case comparisons are shown separately; stage/selection differences prevent a single pooled headline.','New rules are proposals under test, not scientifically established trading laws.'])
    tests=ROOT/'output/full_enrichment_test_run.log'
    match=re.search(r'(\d+) passed',tests.read_text(encoding='utf8')) if tests.exists() else None
    smoke=ROOT/'output/enrichment_runtime_smoke_20261025.json'
    report['verification']=dict(targeted_tests_passed=int(match[1]) if match else None,runtime_smoke=json.loads(smoke.read_text()) if smoke.exists() else None)
    bse_status=ROOT/'data/telemetry/bse_outcome_fetch_status.json'
    report['bse_outcome_status']=json.loads(bse_status.read_text()) if bse_status.exists() else None
    manifests=['data/deal_structure_enriched.csv','output/module1_full_enrichment_benchmark_20261025.json','output/module2_enriched_benchmark_20261025.json','output/exchange_enrichment_backfill.json','services/point_in_time.py','services/enriched_monitor.py','services/retail_hni_economics.py','services/enrichment_runtime.py','services/bse_listing_outcomes.py','ml/run_full_enrichment_audit.py','ml/train_module2_enriched.py','ml/research_module1_enriched.py','ml/extract_cached_deal_structure.py','scrapers/exchange_enrichment_scraper.py','ml/artifacts/module1_full_enrichment_research_20261025.joblib','ml/artifacts/module2_enriched_research_20261025.joblib']
    report['input_sha256']={p:sha(ROOT/p) for p in manifests}
    OUT.write_text(json.dumps(report,indent=2,default=str,allow_nan=False),encoding='utf8')
    lines=['# Full enrichment audit — 25 October 2026','','The new extraction, exchange backfill, model challengers, capital calculator and dated monitor preview are implemented. The challengers have not earned promotion.','',f"Original models and forecast rows preserved: **{protection['all_preserved']}**",'','## Module 1: same-case comparisons','','| Cohort | Frozen model | Enriched calibrated | GMP-only fitted | Always positive |','|---|---:|---:|---:|---:|']
    for label,key in [('69 historical 2026 IPOs','cohort_69'),('23 recent verified openings','recent_23')]:
        stats=report['module1'][key];lines.append('| '+label+' | '+' | '.join(f"{stats[k]['correct']}/{stats[k]['n']} ({stats[k]['accuracy_pct']:.1f}%)" for k in ['frozen_model','enriched_calibrated','gmp_fitted','always_positive'])+' |')
    lines+=['','The fixed historical cohorts have already been inspected. These are retrospective comparisons, not new prospective accuracy.','', '## Module 2: same 132 matured IPOs','']
    for label,key in [('Existing rule engine','old_rule_accuracy_pct'),('Always sell','always_sell_accuracy_pct'),('Enriched rules','enriched_rule_accuracy_pct'),('Enriched classifier','enriched_ml_accuracy_pct')]:lines.append(f"- {label}: **{module2[key]:.2f}%**")
    lines+=['','## Remaining evidence gaps','']+['- '+s for s in report['outstanding']]+['','No exact allotment-adjusted portfolio return is claimed without the application counts. The calculator supports separately labeled assumptions and financing costs.','', 'Detailed field coverage, failed requests, cohort membership and integrity hashes are in the JSON report.']
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(OUT),integrity_preserved=protection['all_preserved'],module1_69=report['module1']['cohort_69'],module2_accuracy=module2['enriched_ml_accuracy_pct'])),flush=True)
    if not protection['all_preserved']:raise RuntimeError('Protected baseline changed; inspect audit before promotion')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reports-only',action='store_true');parser.add_argument('--network-requests',type=int,default=0)
    args=parser.parse_args();run(reports_only=args.reports_only,network_requests=args.network_requests)
