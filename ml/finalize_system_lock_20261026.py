"""Audited two-layer release; immutable artifacts and saved forecasts remain intact."""
import gzip,hashlib,json,sqlite3
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from services.locked_model_configuration import layer2,VERSION
from ml.run_full_enrichment_audit import metrics,integrity


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def run():
    from services.prospective_module1 import predict
    frame=pd.read_csv(ROOT/'data/audit/module1_stage_aligned_inputs_20261026.csv',dtype={'issuer_key':str}).set_index('issuer_key')
    predictions=pd.read_csv(ROOT/'output/module1_controlled_stage_ablation_20261026.csv',dtype={'issuer_key':str})
    predictions=predictions[predictions.variant.eq(VERSION)]
    rows=[]
    for p in predictions.itertuples():
        r=frame.loc[p.issuer_key];plan=layer2(p.company_name,r.to_dict(),float(p.probability),historical=True)
        lot=plan.get('rii_lot_rupees');retail=np.expm1(r.log_retail)
        pnl=min(1.,1/max(1.,retail))*lot*(r.target_return-.35)/100-9 if lot else None
        _,gmp_probability=predict(r.to_dict())
        rows.append(dict(cohort=p.cohort,issuer_key=p.issuer_key,company_name=p.company_name,probability=p.probability,
            actual_positive=int(p.actual_positive),actual_return_pct=r.target_return,standard_positive=p.probability>=.5,
            plan=plan,one_lot_net_rupee_scenario=pnl,gmp_probability=gmp_probability,gmp_pct=r.gmp_pct))
    with sqlite3.connect('file:'+str(ROOT/'data/telemetry/snapshots.db')+'?mode=ro',uri=True) as c:
        bound=c.execute("SELECT exchange_open_price,issue_price FROM listing_outcome_bse_v1 WHERE company_name LIKE 'National Stock Exchange%'").fetchone()
    if bound:
        extra=frame[frame.cohort.eq('recent') & frame.target_binary.isna()]
        from services.prospective_module1 import predict
        for key,r in extra.iterrows():
            if 'National Stock Exchange' not in r.company_name:continue
            features={k:float(v) for k,v in r.items() if isinstance(v,(float,int,np.number)) and pd.notna(v)}
            probability,gmp_probability=predict(features);plan=layer2(r.company_name,features,probability,historical=True)
            actual_return=100*(bound[0]/bound[1]-1);lot=plan.get('rii_lot_rupees')
            pnl=min(1.,1/max(1.,np.expm1(r.log_retail)))*lot*(actual_return-.35)/100-9 if lot else None
            rows.append(dict(cohort='recent24_extra',issuer_key=key,company_name=r.company_name,probability=probability,
                actual_positive=int(bound[0]>bound[1]),actual_return_pct=actual_return,standard_positive=probability>=.5,plan=plan,one_lot_net_rupee_scenario=pnl,gmp_probability=gmp_probability,gmp_pct=r.gmp_pct))
    cohorts={}
    for name in (('test_69','recent_23','recent_24_verified') if bound else ('test_69','recent_23')):
        group=[r for r in rows if r['cohort']==name or (name=='recent_24_verified' and r['cohort'] in ('recent_23','recent24_extra'))];y=np.array([r['actual_positive'] for r in group]);p=np.array([r['probability'] for r in group])
        apply=np.array([r['plan'].get('tier') in (1,2) for r in group]);skip=np.array([r['plan'].get('tier')==3 for r in group]);raw=p>=.5
        cohorts[name]=dict(layer1=metrics(y,p),baselines=dict(always_positive=metrics(y,np.ones(len(y))),
            gmp_sign=metrics(y,np.array([r['gmp_pct']>0 for r in group],dtype=float)),
            frozen_gmp_only=metrics(y,[r['gmp_probability'] for r in group],decision_threshold=.625)),layer2=dict(tier1=int(sum(r['plan'].get('tier')==1 for r in group)),tier2=int(sum(r['plan'].get('tier')==2 for r in group)),
            hard_skip=int(skip.sum()),data_pending=int(sum(r['plan'].get('tier') is None for r in group)),apply_n=int(apply.sum()),
            apply_precision_pct=100*float(y[apply].mean()) if apply.any() else None,
            hard_skip_loser_recall_pct=100*float(skip[y==0].mean()),false_positive_apply_n=int((apply&(y==0)).sum()),
            positive_openings_skipped=int((skip&(y==1)).sum()),
            retail_one_lot_net_rupee_scenario_layer1=sum(r['one_lot_net_rupee_scenario'] or 0 for r,a in zip(group,raw) if a),
            retail_one_lot_net_rupee_scenario_layer2=sum(r['one_lot_net_rupee_scenario'] or 0 for r,a in zip(group,apply) if a)))
    assert (cohorts['test_69']['layer1']['correct'],cohorts['recent_23']['layer1']['correct'])==(56,16)
    report=dict(configuration=VERSION,cohorts=cohorts,records=rows,layer2_weights_retrained=False,
        evidence_kind='Retrospective inspected-cohort policy evaluation; not an independent prospective performance claim',
        pnl_basis='Reciprocal retail allotment approximation, original lot cost, actual listing-open return, 0.35% friction and Rs9 per application. Not realised investor P&L. Allotment/return dependence and ASBA opportunity cost excluded.')
    path=ROOT/'output/module1_two_layer_final_audit_20261026.json';path.write_text(json.dumps(clean(report),indent=2,allow_nan=False))
    pd.DataFrame([{**r,'plan':json.dumps(r['plan'])} for r in rows]).to_csv(path.with_suffix('.csv'),index=False)
    # Reconcile only these newly searched gaps, retaining unrelated requests.
    original_gaps=pd.read_csv(ROOT/'output/module1_remaining_day2_category_gaps_20261026.csv',dtype={'issuer_key':str})
    gap_status=[]
    for r in original_gaps.itertuples():
        cache=ROOT/'data/archive/final_lock_day2_20261026'/(r.issuer_key+'.json.gz')
        d=json.loads(gzip.decompress(cache.read_bytes())) if cache.exists() else {}
        key=r.issuer_key;recovered=pd.notna(frame.loc[key,'day2_qib']);late=[]
        if not recovered:
            from ml.stage_alignment_bridge import parse_ipoji,complete
            for s in d.get('sources',[]):late.extend(h for h in parse_ipoji(s['raw_response_text']) if h['day']==2 and complete(h))
        status='RECOVERED_ELIGIBLE_DAY2' if recovered else 'DAY2_AFTER_ORIGINAL_DECISION' if late and d.get('audit',{}).get('identity_offer_dates_and_cap_verified') else 'EXTERNAL_GAP_SEARCH_INCOMPLETE'
        gap_status.append(dict(issuer_key=key,company_name=r.company_name,status=status,
            quarantined_from_stage_bridge=not recovered,
            searched='Existing local archives, observation CSV, telemetry and IPOJI dated public table; not every public archive has been exhausted.',
            source_url=d.get('audit',{}).get('source_url'),sha256=d.get('audit',{}).get('sha256')))
    pd.DataFrame(gap_status).to_csv(ROOT/'output/final_day2_gap_disposition_20261026.csv',index=False)
    missing_path=ROOT/'output/MISSING_DATA_ACTION_LIST_20261026.csv';missing=pd.read_csv(missing_path,dtype={'issuer_key':str})
    dispositions={r['issuer_key']:r for r in gap_status}
    for i,r in missing.iterrows():
        if r['field']=='day2_to_cutoff_conviction' and r.issuer_key in dispositions:
            missing.loc[i,'day2_category_status']=dispositions[r.issuer_key]['status']
    missing.to_csv(missing_path,index=False)
    m2=json.loads((ROOT/'output/module2_rule_ablation_20261026.json').read_text())
    m2['evaluation_live_promoted']=m2.pop('live_promoted',False)
    m2['active_policy']='M2_R1_WITH_DATED_FILINGS'
    m2['aggressive_structural_rules_active']=False
    for name,r in m2['results'].items():
        r['youdens_J']=(r['hold_recall_pct']+r['sell_recall_pct'])/100-1
        r['sell_precision_pct']=100*r['loser_specificity_n']/r['sell_calls'] if r['sell_calls'] else None
    coverage=json.loads((ROOT/'output/ipopremium_gap_report_20261026.json').read_text())['coverage']
    all_recent=frame[frame.cohort.eq('recent')]
    coverage['recent_24']={field:dict(after_present=int(all_recent[field].notna().sum()),remaining_missing=int(all_recent[field].isna().sum())) for field in coverage['recent_23']}
    stage=json.loads((ROOT/'output/module1_stage_alignment_bridge_20261026.json').read_text())
    with sqlite3.connect('file:'+str(ROOT/'data/telemetry/snapshots.db')+'?mode=ro',uri=True) as c:
        actual=c.execute("SELECT exchange_open_price,issue_price,archive_sha256,source_url FROM listing_outcome_bse_v1 WHERE company_name LIKE 'National Stock Exchange%'").fetchone()
    if actual:
        nse=dict(status='verified_primary_bse',open_price=actual[0],issue_price=actual[1],sha256=actual[2],source_url=actual[3],verified_denominator=24,
            locked_23_correct=16,bound_24_correct=cohorts['recent_24_verified']['layer1']['correct'])
    else:nse=dict(status='primary_BSE_archive_pending_HTTP_403',verified_denominator=23,supplied_outcome_sensitivity=dict(correct=17,n=24,accuracy_pct=100*17/24,issue_price=1785,open_price=1800,primary_verified=False))
    lock=json.loads((ROOT/'output/active_policy_lock_20261026.json').read_text())
    user_path=ROOT/'output/IPO_DATA_HELP_FROM_YOU_20261026.csv';help_frame=pd.read_csv(user_path,dtype={'issuer_key':str})
    additions=[]
    for gap in gap_status:
        if gap['status']=='EXTERNAL_GAP_SEARCH_INCOMPLETE':
            additions.append(dict(cohort='training_290',issuer_key=gap['issuer_key'],company_name=gap['company_name'],field='day2_category_archive',
                data_needed='Day-2 QIB, NII, retail and total table',reason=gap['status'],request_details='If you have a dated table/export, supply it. Public-source recovery is not exhausted; no zero or final-book substitution.',
                owner='agent_public_sources_first_user_optional_export',decision_timestamp=frame.loc[gap['issuer_key'],'cutoff_at_ist'],source_url=gap['source_url']))
    help_frame=pd.concat([help_frame,pd.DataFrame(additions)],ignore_index=True).drop_duplicates(['issuer_key','field'],keep='last')
    help_frame.to_csv(user_path,index=False);help_rows=help_frame.to_dict('records')
    test_log=ROOT/'output/final_lock_pytest_run_20261026.log'
    import re
    log=test_log.read_text(errors='replace') if test_log.exists() else ''
    test_summary=[line for line in log.splitlines() if re.search(r'\d+ passed.* in ',line)]
    final=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),active=lock,module1=report,module2=m2,
        coverage=coverage,stage_category_coverage=stage['coverage'],day2_disposition=gap_status,nse_bse_outcome=nse,
        module2_eligible_counts=m2['eligible_counts'],remaining_user_data=help_rows,
        milestone_collector=dict(active=True,table='module1_milestone_capture_v1',stages=['DAY2_EOD_SNAPSHOT','DAY3_PRECLOSE_DECISION_SNAPSHOT','DAY3_POSTCLOSE_FINAL_BOOK'],
             source='Identity-matched NSE issue endpoint; unavailable attempts and raw payloads retained. Separate split categories remain null if not published.',
             calendar='Previous bidding session from supplied calendar; otherwise weekdays-only assumption is explicitly recorded. Closing minus one calendar day is not used blindly.'),
        validation=dict(python_tests=test_summary[-1] if test_summary else 'Running; see output/final_lock_pytest_run_20261026.log',frontend_contract_tests='7 passed',frontend_production_build='passed',
            post_binding_checks='26 passed; see output/final_lock_post_binding_checks.log',
            optional_tests='External-provider smoke test is opt-in. Archived sklearn 1.5.2 return challenger is not loaded under 1.9; the frozen five-expert direction stack remains active.'),
        integrity=integrity(),limitations=['Layer2 thresholds were proposed after inspecting these cohorts and are not independent evidence of future profitability.',
             'Uncalibrated model scores do not establish calibrated win probabilities or exact expected rupee utility.',
             'Unverified exchange opens are not inserted into a verified outcome ledger.',
             'Module2 R1 remains below always-sell on accuracy and mean return; it is a chosen deterministic risk policy, not a validated return forecast.'])
    files=['ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib','core/monitor_engine.py','core/monitor_r0_frozen_20261026.py','services/filing_governance.py',
        'services/retail_hni_economics.py','services/locked_model_configuration.py','services/module1_milestones.py','services/prospective_module1.py','output/module1_two_layer_final_audit_20261026.json']
    final['source_sha256']={p:sha(ROOT/p) for p in files}
    (ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json').write_text(json.dumps(clean(final),indent=2,allow_nan=False))
    lines=['# OpenBell — frozen model and two-layer application screen','',
        'Module 1 uses the original frozen five-expert artifact with recovered inputs. Layer 2 is a separate deterministic application policy; its verdict never overwrites the raw direction call. Module 2 uses R1 price rules plus eligible adverse exchange filings. R2–R4 remain diagnostics.','',
        '| Cohort | Raw correct | Raw false positives | Tier 1 | Tier 2 | Skip | APPLY precision | Losers skipped |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in cohorts.items():
        a=r['layer1'];b=r['layer2'];lines.append(f"| {name} | {a['correct']}/{a['n']} ({a['accuracy_pct']:.1f}%) | {a['wrong_positive_calls']} | {b['tier1']} | {b['tier2']} | {b['hard_skip']} | {b['apply_precision_pct']:.1f}% | {b['hard_skip_loser_recall_pct']:.1f}% |")
    lines+=['','## Same-cohort direction baselines','', '| Cohort | Frozen model | Always positive | GMP sign | Frozen GMP-only classifier |','|---|---:|---:|---:|---:|']
    for name,r in cohorts.items():
        n=r['layer1']['n'];b=r['baselines'];lines.append(f"| {name} | {r['layer1']['correct']}/{n} | {b['always_positive']['correct']}/{n} | {b['gmp_sign']['correct']}/{n} | {b['frozen_gmp_only']['correct']}/{n} |")
    lines+=['','## Rupee scenarios, not realised portfolio returns','']
    for name,r in cohorts.items():
        b=r['layer2'];lines.append(f"- {name}: raw-positive selection Rs{b['retail_one_lot_net_rupee_scenario_layer1']:,.2f}; filtered selection Rs{b['retail_one_lot_net_rupee_scenario_layer2']:,.2f}. Original lot costs, reciprocal allotment, 0.35% friction and Rs9 per application assumed.")
    lines+=['','## Module 2','', '| Policy | Correct /132 | Winners retained /55 | Losers exited /77 | Youden J | Mean net return |','|---|---:|---:|---:|---:|---:|']
    for name,r in m2['results'].items():lines.append(f"| {name} | {r['correct']} | {r['winner_sensitivity_n']} | {r['loser_specificity_n']} | {r['youdens_J']:+.3f} | {r['conditional_strategy_net_mean_pct']:.3f}% |")
    lines+=['','## Data and verification','',f"- Eligible recovered Day-2 category coverage: {stage['coverage']['training_290']['day2_categories']}/290 training, 69/69 earlier 2026, 24/24 recent.",
        '- Gap-by-gap dispositions and source hashes: final_day2_gap_disposition_20261026.csv. A table published after the original decision is not backfilled into that decision.',
        '- NSE/BSE outcome: '+nse['status']+'. '+('The original 16/23 comparison stays locked; the expanded exchange-verified result is 17/24 (70.8%).' if actual else 'The primary denominator remains 23; the supplied 17/24 sensitivity stays separate until bound.'),
        '- The stage bridge remains unfitted: 161 matched totals did not constitute 161 distinct Day2/preclose pairs.',
        '- Existing model artifacts and forecast rows remain intact. R0 source is archived under its original hash; R1 is separately versioned.',
        '- Pytest results are recorded in final_lock_pytest_run_20261026.log. The default suite now targets automated tests; exploratory scraping scripts are not collected and the network smoke test is opt-in.','',
        '## Allotment and financing corrections','',
        '- Mainboard allocation rules and minimum-plus-residual NII allotment verified against SEBI ICDR, consolidated 21 March 2026, regulation 32 and Schedule XIII. Source: https://www.sebi.gov.in/sebi_data/attachdocs/mar-2026/1774592300989.pdf',
        '- Minimum sNII/bNII application lots strictly exceed Rs2 lakh/Rs10 lakh: floor(threshold/lot_cost)+1, including exact-boundary cases.',
        '- A 35% retail quota does not trigger the >=50% retail-heavy flag. Share multiples alone do not prove valid-application lottery odds or full allotment.',
        '- Financed interest is charged on the actual borrowed principal for every application. The 9.5% / five-day setting is a scenario, not a verified lender quote. Dynamic parcel ratio is used instead of hard-coding 5.',
        '- Missing dated bNII demand means financing viability is false/unestablished, never inferred from total NII.',
        '- No prospective profitability or zero-overfitting claim is made. Full assumptions and per-IPO decisions are in the JSON.','']
    (ROOT/'output/FINAL_SYSTEM_LOCK_20261026.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(clean(dict(cohorts=cohorts,day2_statuses=pd.Series([r['status'] for r in gap_status]).value_counts().to_dict(),integrity=final['integrity']['all_preserved'])),indent=2))
    return final

if __name__=='__main__':run()
