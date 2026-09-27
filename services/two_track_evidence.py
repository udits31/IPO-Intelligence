"""Presentation-only separation of frozen model direction and application selection."""
import hashlib,json
from datetime import datetime,timezone
from services.saved_preclose_service import ROOT

LOCK=ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json'
OUT=ROOT/'output/module1_two_track_evidence_20261026.json'


def present(probability,actual_positive,tier):
    call=float(probability)>=.5;known=actual_positive is not None;selected=tier in (1,2);skip=tier==3
    screen_result='Pending opening' if not known else ('Winner Selected' if actual_positive else 'Loser Admitted') if selected else ('Winner Skipped (Opportunity Cost)' if actual_positive else 'Loser Avoided') if skip else 'Not evaluated'
    return dict(model_call='Positive' if call else 'Flat-Negative',
        model_outcome=('Right' if call==bool(actual_positive) else 'Missed') if known else 'Pending',
        model_direction_correct=call==bool(actual_positive) if known else None,
        application_screen='Apply Tier 1' if tier==1 else 'Apply Tier 2' if tier==2 else 'Skip' if skip else 'Data pending',
        screen_result=screen_result,skipped_winner_opportunity_cost=bool(skip and actual_positive) if known else None)


def annotate(row):
    p=float(row['probability']);positive=bool(row['actual_positive']);call=p>=.5
    plan=row['plan'];tier=plan.get('tier');selected=tier in (1,2);skip=tier==3
    screen_result=('Winner Selected' if positive else 'Loser Admitted') if selected else ('Winner Skipped (Opportunity Cost)' if positive else 'Loser Avoided') if skip else 'Not evaluated'
    return dict(issuer_key=row['issuer_key'],company_name=row['company_name'],actual_positive=positive,
        actual_return_pct=row['actual_return_pct'],model_probability=p,
        model_call='Positive' if call else 'Flat-Negative',model_outcome='Right' if call==positive else 'Missed',
        model_direction_correct=call==positive,model_predicted_positive=call,
        application_screen='Apply Tier 1' if tier==1 else 'Apply Tier 2' if tier==2 else 'Skip' if skip else 'Data pending',
        screen_result=screen_result,application_tier=tier,selected_apply=selected,skipped=skip,
        skipped_winner_opportunity_cost=skip and positive,
        raw_positive_call_rupee_scenario=row.get('one_lot_net_rupee_scenario') if call else 0.,
        selected_apply_rupee_scenario=row.get('one_lot_net_rupee_scenario') if selected else 0.,
        screen_flags=plan.get('flags',{}),evidence_kind='Frozen historical replay; not a saved investor-time prediction')


def summarize(rows):
    tp=sum(r['model_predicted_positive'] and r['actual_positive'] for r in rows)
    fp=sum(r['model_predicted_positive'] and not r['actual_positive'] for r in rows)
    fn=sum(not r['model_predicted_positive'] and r['actual_positive'] for r in rows)
    tn=sum(not r['model_predicted_positive'] and not r['actual_positive'] for r in rows)
    selected=[r for r in rows if r['selected_apply']];skips=[r for r in rows if r['skipped']]
    winners=tp+fn;losers=fp+tn;sw=sum(r['actual_positive'] for r in selected)
    skipped_winners=sum(r['actual_positive'] for r in skips);avoided=len(skips)-skipped_winners
    raw_rupees=sum(r['raw_positive_call_rupee_scenario'] or 0 for r in rows)
    filtered_rupees=sum(r['selected_apply_rupee_scenario'] or 0 for r in rows)
    return dict(n=len(rows),model_direction=dict(
        predicted_positive_opened_positive=tp,predicted_positive_opened_flat_or_negative=fp,
        predicted_flat_or_negative_opened_positive=fn,predicted_flat_or_negative_opened_flat_or_negative=tn,
        model_direction_correct=tp+tn,model_direction_missed=fp+fn,accuracy_pct=100*(tp+tn)/len(rows),
        raw_positive_calls=tp+fp,raw_negative_calls=fn+tn),
        application_screen=dict(selected_apply_total=len(selected),tier1=sum(r['application_tier']==1 for r in selected),tier2=sum(r['application_tier']==2 for r in selected),
            selected_apply_winners=sw,selected_apply_losers_admitted=len(selected)-sw,apply_precision_pct=100*sw/len(selected) if selected else None,
            skipped_total=len(skips),skipped_losers_avoided=avoided,total_actual_losers=losers,skip_loser_recall_pct=100*avoided/losers if losers else None,
            skipped_winners_opportunity_cost=skipped_winners,total_actual_winners=winners,
            skipped_model_true_positives_vetoed=sum(r['actual_positive'] and r['model_predicted_positive'] for r in skips),
            skipped_model_false_negatives=sum(r['actual_positive'] and not r['model_predicted_positive'] for r in skips),
            data_pending=len(rows)-len(selected)-len(skips)),
        rupee_scenarios=dict(raw_positive_total=raw_rupees,filtered_apply_total=filtered_rupees,
            raw_positive_call_count=tp+fp,selected_apply_call_count=len(selected),
            per_raw_positive_call=raw_rupees/(tp+fp) if tp+fp else None,
            per_applied_call=filtered_rupees/len(selected) if selected else None,
            realised_profit=False,basis='Locked reciprocal-allotment, original-lot, actual-opening and assumed-cost scenario; no new allocation assumptions.'))


def build():
    raw=LOCK.read_bytes();source=json.loads(raw);records=source['module1']['records'];cohorts={}
    for name in ('test_69','recent_23','recent_24_verified'):
        members=[r for r in records if r['cohort']==name or (name=='recent_24_verified' and r['cohort'] in ('recent_23','recent24_extra'))]
        rows=[annotate(r) for r in members];summary=summarize(rows)
        locked=source['module1']['cohorts'][name]
        if summary['model_direction']['model_direction_correct']!=locked['layer1']['correct']:raise ValueError('Direction changed while presenting the screen')
        if summary['application_screen']['selected_apply_total']!=locked['layer2']['apply_n']:raise ValueError('Application policy changed')
        cohorts[name]=dict(**summary,records=rows)
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(),locked_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        model_weights_changed=False,application_rules_changed=False,cohorts=cohorts,
        corrections=['The frozen records contain 20 raw-positive calls in recent_23 and 21 in recent_24_verified; averages use those denominators.',
        'Direction errors remain errors even when the application screen skips the IPO.',
        'An APPLY winner rate is selection precision, not all-IPO accuracy or a safety guarantee.'])


def export():
    report=build();OUT.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    lines=['# Module 1: two separate scorecards','',
        '| Cohort | Model right | Model missed (FP / FN) | Apply winners / selected | Losers admitted | Losers avoided | Winners skipped |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in report['cohorts'].items():
        m=r['model_direction'];s=r['application_screen'];lines.append(f"| {name} | {m['model_direction_correct']}/{r['n']} | {m['model_direction_missed']} ({m['predicted_positive_opened_flat_or_negative']} / {m['predicted_flat_or_negative_opened_positive']}) | {s['selected_apply_winners']}/{s['selected_apply_total']} | {s['selected_apply_losers_admitted']} | {s['skipped_losers_avoided']}/{s['total_actual_losers']} | {s['skipped_winners_opportunity_cost']}/{s['total_actual_winners']} |")
    lines+=['','## Rupee scenarios per application','', '| Cohort | Raw positive calls | Total raw scenario | Average raw | Apply calls | Total filtered scenario | Average filtered |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in report['cohorts'].items():
        p=r['rupee_scenarios'];lines.append(f"| {name} | {p['raw_positive_call_count']} | Rs{p['raw_positive_total']:,.2f} | Rs{p['per_raw_positive_call']:,.2f} | {p['selected_apply_call_count']} | Rs{p['filtered_apply_total']:,.2f} | Rs{p['per_applied_call']:,.2f} |")
    lines+=['','The 20 and 21 raw-positive denominators come from TP+FP in the frozen records. The supplied 19/20 denominators were inconsistent with those confusion matrices. These are assumed-allotment scenarios, not realised profits.','',
        'Every per-IPO JSON record reports Model Call, Model Outcome, Application Screen and Screen Result independently. Skipped winners are opportunity cost; admitted losers remain visible. No locked benchmark, model, capital rule or forecast row was changed.','']
    OUT.with_suffix('.md').write_text('\n'.join(lines),encoding='utf8');return report


if __name__=='__main__':export()
