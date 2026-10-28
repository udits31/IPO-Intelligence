"""Update the inventory, not model weights or immutable prediction ledgers."""
import hashlib,json,math,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from ml.run_full_enrichment_audit import integrity

FIELDS={
 'gmp_pct':('Dated pre-decision GMP','A quote dated before this IPO decision. A present-day GMP is not a substitute.'),
 'ofs_ratio':('Fresh issue / OFS split','Original offer component shares/amounts, ideally with an RHP page reference.'),
 'quota_weighted_book_mult':('Category reservation denominators','Exact offered shares for QIB, HNI and retail, with anchor/employee portions identified.'),
 'day2_to_cutoff_conviction':('Matched Day-2 and cutoff total','A timestamped total and category table compatible with the saved decision-time book.'),
 'pub_ipo_earnings_yield_pct':('Pre-offer annual EPS','Annual pre-offer EPS and fiscal year, or a documented IPO valuation panel.'),
 'pub_log_ipo_price_book':('Pre-offer NAV / book valuation','IPO NAV per share or a documented IPO P/B ratio; zero placeholder P/B is not usable.'),
 'lot_size_shares':('Original IPO lot size','Original IPO application lot, rather than a later split-adjusted trading lot.')}


def run():
    base=pd.read_csv(ROOT/'output/module1_progressive_ablation_20261026_features.csv',dtype={'issuer_key':str})
    source=[json.loads(s) for s in (ROOT/'data/audit/ipopremium_recovered_20261026.jsonl').read_text(encoding='utf8').splitlines()]
    recovered={r['issuer_key']:r for r in source if r['identity_verified']};seen={r['issuer_key']:r for r in source}
    queue=json.loads((ROOT/'output/ipopremium_recovery_queue_20261026.json').read_text(encoding='utf8'))
    unmatched=json.loads((ROOT/'output/ipopremium_unmatched_20261026.json').read_text(encoding='utf8'))
    statuses=pd.read_csv(ROOT/'output/ipopremium_recovery_status_20261026.csv',dtype={'issuer_key':str}).set_index('issuer_key')
    all_targets={r['issuer_key']:r for r in [*queue,*unmatched]}
    docs_path=ROOT/'data/audit/ipopremium_documents_recovered_20261026.jsonl'
    documents=[json.loads(s) for s in docs_path.read_text(encoding='utf8').splitlines()] if docs_path.exists() else []
    doc_by_key={r['issuer_key']:r for r in documents if r.get('anchor_domestic_mf_pct') is not None}
    gaps=[];audit=[];fills=[];out=base.copy()
    for index,row in base.iterrows():
        key=row.issuer_key;record=recovered.get(key);raw_day2=None
        if record:
            financial=record['financial'];offer=record['offer'];new=dict(ofs_ratio=offer.get('ofs_ratio'),lot_size_shares=record.get('lot_size_shares'))
            for field in ('ipo_earnings_yield_pct','log_ipo_price_book','profit_margin_pct','return_on_assets_pct','revenue_growth_asinh','annual_loss'):
                new['pub_'+field]=financial.get(field)
            reservation=record.get('reservation',{});denoms=[reservation.get(k,{}).get('shares') for k in ('QIB','HNI','Retail')]
            if all(v is not None and v>0 for v in denoms):
                denominator=sum(denoms)
                new['quota_weighted_book_mult']=sum(v/denominator*np.expm1(float(row[k])) for v,k in zip(denoms,('log_qib','log_hni','log_retail')))
            days=record['daywise'];second=next((d for d in days if d['day']==2),None)
            if second:raw_day2=second['values'].get('Total')
            out.loc[index,'ipopremium_day2_total_raw']=raw_day2
            out.loc[index,'ipopremium_daily_history_days']=len(days)
            out.loc[index,'ipopremium_daily_history_clock_status']='day_labels_only_individual_update_times_unknown'
            out.loc[index,'ipopremium_last_table_update_ist']=record['last_table_update_ist']
            out.loc[index,'ipopremium_source_url']=record['source_url']
            if key in doc_by_key:
                out.loc[index,'anchor_domestic_mf_pct']=doc_by_key[key]['anchor_domestic_mf_pct']
                out.loc[index,'anchor_mf_quality']='native_anchor_document_percentage_publication_time_not_certified'
            for field,value in new.items():
                if value is not None and pd.notna(value) and (field not in out or pd.isna(out.loc[index,field])):
                    out.loc[index,field]=value;out.loc[index,field+'__ipopremium_quality']='recovered_ipo_context_not_certified_original_publication'
                    fills.append(dict(issuer_key=key,company_name=row.company_name,field=field,value=float(value),source_url=record['source_url'],source_sha256=record['sha256']))
        relevant=row.cohort=='historical' or pd.notna(row.target_binary)
        if not relevant:continue
        cohort='training_290' if row.cohort=='historical' and row.year<=2025 else 'test_69' if row.cohort=='historical' else 'recent_23'
        for field,(label,needed) in FIELDS.items():
            if pd.notna(out.loc[index,field]):continue
            status='not_recovered_from_ipopremium';owner='agent_other_public_sources_first'
            if key not in {r['issuer_key'] for r in queue}:status='IPO_not_in_public_mainboard_directory'
            elif key not in recovered:
                status=str(statuses.loc[key,'status']) if key in statuses.index else 'not_yet_captured'
            if field=='day2_to_cutoff_conviction' and raw_day2 is not None:
                status='Day2_total_recovered_cutoff_total_or_matching_clock_still_missing';owner='user_if_timestamped_capture_available'
            if field=='gmp_pct':owner='user_if_dated_quote_export_available'
            if field=='day2_to_cutoff_conviction' and pd.notna(row.get('issue_open_date')) and pd.Timestamp(row.decision_date)<=pd.Timestamp(row.issue_open_date):status='not_available_at_decision_do_not_supply_future_Day2';owner='not_applicable'
            gaps.append(dict(cohort=cohort,issuer_key=key,company_name=row.company_name,field=field,data_needed=label,reason=status,request_details=needed,owner=owner,
                decision_timestamp=row.cutoff_at_ist,source_url=record['source_url'] if record else all_targets.get(key,{}).get('url')))
    out.to_csv(ROOT/'data/audit/module1_inputs_after_ipopremium_20261026.csv',index=False)
    for key,target in all_targets.items():
        r=recovered.get(key)
        status='recovered_identity_matched' if r else str(statuses.loc[key,'status']) if key in statuses.index else 'not_in_public_directory' if key in {x['issuer_key'] for x in unmatched} else 'not_yet_captured'
        audit.append(dict(issuer_key=key,company_name=target['company_name'],status=status,source_url=target.get('url'),
            daywise_days=len(r['daywise']) if r else 0,ofs_ratio=r['offer'].get('ofs_ratio') if r else None,
            annual_eps=r['financial'].get('ipo_pre_offer_eps') if r else None,book_value=r['financial'].get('nav_per_share') if r else None,
            anchor_mf_pct=doc_by_key.get(key,{}).get('anchor_domestic_mf_pct')))
    original=ROOT/'output/MISSING_DATA_ACTION_LIST_20261026.csv';backup=original.with_name('MISSING_DATA_ACTION_LIST_BEFORE_IPOPREMIUM_20261026.csv')
    if original.exists() and not backup.exists():shutil.copyfile(original,backup)
    gap_frame=pd.DataFrame(gaps);gap_frame.to_csv(original,index=False)
    pd.DataFrame(audit).to_csv(ROOT/'output/IPO_PREMIUM_ALL_COHORT_COVERAGE_20261026.csv',index=False)
    pd.DataFrame(fills).to_csv(ROOT/'output/IPO_PREMIUM_RECOVERED_FIELDS_20261026.csv',index=False)
    user_rows=[g for g in gaps if g['owner'].startswith('user_')]
    # Every listed-case replay already has publisher-reported readings. The
    # remaining request is independent timestamp evidence, not invented values.
    for row in base[base.cohort.eq('recent')].itertuples():
        record=recovered.get(row.issuer_key)
        user_rows.append(dict(cohort='recent_listed_24',issuer_key=row.issuer_key,company_name=row.company_name,field='independent_preclose_capture',
            data_needed='Closing-day pre-17:00 subscription archive',reason='Existing publisher-reported replay needs independent capture-time evidence',
            request_details='If available: original screenshot/archive/export with date/time, QIB, HNI, retail and total before '+str(row.cutoff_at_ist)+'. Do not send a final post-close table as an earlier reading.',owner='user_optional_archive_evidence',decision_timestamp=row.cutoff_at_ist,source_url=record['source_url'] if record else None))
    pd.DataFrame(user_rows).to_csv(ROOT/'output/IPO_DATA_HELP_FROM_YOU_20261026.csv',index=False)
    summary={}
    for label,mask in [('training_290',out.cohort.eq('historical')&out.year.le(2025)),('test_69',out.cohort.eq('historical')&out.year.eq(2026)),('recent_23',out.cohort.eq('recent')&out.target_binary.notna())]:
        summary[label]={field:dict(before_present=int(base.loc[mask,field].notna().sum()),after_present=int(out.loc[mask,field].notna().sum()),remaining_missing=int(out.loc[mask,field].isna().sum())) for field in FIELDS}
    protected=integrity()
    if not protected['all_preserved']:raise ValueError('Frozen model or original forecast prefix changed')
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),scope_issuers=len(all_targets),captured_pages=len(source),identity_matched=len(recovered),
        directory_unmatched=len(unmatched),coverage=summary,new_field_cells=len(fills),remaining_field_cells=len(gaps),user_help_rows=len(user_rows),
        anchor_documents_with_mf_pct=len(doc_by_key),retraining_performed=False,integrity_preserved=True,
        assumptions=['Recovered static/valuation figures are IPO-context facts; original publication/revision times remain unverified.','Day-wise history and final closing books do not establish a pre-17:00 intraday observation.','Reservation denominators distinguish the three bidding categories from separate anchor/employee shares.','Raw source facts and derived research overlays do not overwrite frozen model inputs or saved predictions.'])
    (ROOT/'output/ipopremium_gap_report_20261026.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
    lines=['# IPOPremium recovery and remaining data — 26 October 2026','',f"Scope: {len(all_targets)} distinct IPOs across Module 1 and Module 2 training/evaluation cohorts, including 24 recent listed IPOs. No retraining was performed.",'','## Coverage after recovery','','| Input | Training: before → after / 290 | 2026: before → after / 69 | Recent: before → after / 23 |','|---|---:|---:|---:|']
    for field,(label,_) in FIELDS.items():lines.append('| '+label+' | '+' | '.join(f"{summary[g][field]['before_present']} → {summary[g][field]['after_present']}" for g in ('training_290','test_69','recent_23'))+' |')
    lines+=['','Static offer facts and valuation fills use explicitly labelled IPO-context assumptions. This is recovery coverage, not proof of historical publication or a new accuracy result.','','## Exact lists','','- `IPO_PREMIUM_ALL_COHORT_COVERAGE_20261026.csv`: one row per IPO, with source/capture status.','- `IPO_PREMIUM_RECOVERED_FIELDS_20261026.csv`: each newly filled field and its source hash.','- `MISSING_DATA_ACTION_LIST_20261026.csv`: remaining model-input gaps, by company, reason and owner.','- `IPO_DATA_HELP_FROM_YOU_20261026.csv`: dated GMP quotes or compatible cutoff tables still useful from an existing export, plus optional independent pre-close archive evidence for the 24 listed cases. Public facts still recoverable from other sources remain agent work.','','## Do not confuse these gaps','','A blank day-wise table on an already-listed IPO is missing source coverage, not zero demand. A blocked/throttled page is an incomplete collection, not an absent fact. A final Day-3 table updated after 17:00 is historical closing evidence, not a saved application-time forecast. An approximate application count is not an official final allotment basis.','','All original model files and prediction-ledger prefixes remain unchanged.']
    (ROOT/'output/IPO_PREMIUM_RECOVERY_AND_DATA_NEEDED_20261026.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('coverage','assumptions')}),flush=True)
    return report


if __name__=='__main__':run()
