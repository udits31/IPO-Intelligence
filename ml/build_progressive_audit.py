"""Publish the complete fixed-protocol experiment, including failed gates."""
import hashlib,json,re
from datetime import datetime,timezone
from services.saved_preclose_service import ROOT
from services.progressive_engine_service import refresh_regime_state
from ml.run_full_enrichment_audit import integrity


def run():
    module1=json.loads((ROOT/'output/module1_progressive_ablation_20261026.json').read_text())
    module2=json.loads((ROOT/'output/module2_prove_it_benchmark_20261026.json').read_text())
    protected=integrity()
    if not protected['all_preserved']:raise ValueError('Protected model or original forecast prefix changed')
    state=refresh_regime_state();tests=(ROOT/'output/progressive_test_run_20261026.log').read_text()
    passed=re.search(r'(\d+) passed',tests)
    paths=['core/features.py','core/prove_it_monitor_engine.py','services/ipo_regime_service.py','services/progressive_engine_service.py','services/retail_hni_economics.py','ml/research_module1_progressive.py','ml/recover_progressive_totals.py','ml/evaluate_module2_prove_it.py','ml/build_progressive_audit.py','data/deal_structure_enriched.csv','output/module1_progressive_ablation_20261026_features.csv','output/module1_progressive_ablation_20261026.json','output/module2_prove_it_benchmark_20261026.json']
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),module1=module1,module2=module2,regime=state,integrity=protected,
        tests_passed=int(passed[1]) if passed else None,production_models_changed=False,
        slow_loop='Half-life weighting implemented and evaluated. No component passed the full promotion gate; quarterly review due state is recorded. Automatic expanding refit/promotion is not enabled against a cohort that would become training.',
        fast_loop='Backend tracker refreshes every 15 minutes and incorporates newly verified outcomes and saved pre-close GMP inputs. Rejected regime adjustments do not alter the frozen model.',
        capital_allocator='Three-tier policy and full-allotment loss stress implemented as explicit scenarios. Fixed loss/fee assumptions and reciprocal multiples do not establish exact expected investor profit.',
        input_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
        sources={'NSE_special_preopen':'https://www.nseindia.com/static/products-services/equity-market-special-pre-open-session','NSE_bulk_disclosure':'https://archives.nseindia.com/content/circulars/cmtr4808.htm','NSE_allotment_example':'https://nsearchives.nseindia.com/web/sites/default/files/inline-files/Advt_2.2.pdf'})
    out=ROOT/'output/progressive_engine_audit_20261026.json';out.write_text(json.dumps(report,indent=2,default=str,allow_nan=False),encoding='utf8')
    lines=['# OpenBell progressive-engine experiment — 26 October 2026','','No candidate replaced the frozen model. All original saved prediction rows and protected model hashes are unchanged.','','## Module 1: same five experts, fixed 50% direction rule','','Training: 290 IPOs from 2021–2025. Tests: the same 69 2026 IPOs and 23 recent verified openings. No issuer overlaps training and test. No post-hoc calibration was added.','','| Variant | Correct / 69 | Wrong positive / 23 losers | Correct / 23 recent | Pass all gates? |','|---|---:|---:|---:|---|']
    for name,r in module1['results'].items():lines.append(f"| {name} | {r['cohort_69']['correct']} | {r['cohort_69']['wrong_positive_calls']} | {r['recent_23']['correct']} | {'Yes' if r['passes_user_gate'] else 'No'} |")
    lines+=['','The OFS/conviction candidate reached 85.5% on the older cohort but worsened the recent cohort. It was rejected. None of the four isolated components passed all requested gates, so V5 retains V0. Combining rejected components would violate the requested selection rule.','','## Coverage before fitting','', '| Feature | Training 290 | 2026 test 69 | Recent 23 |','|---|---:|---:|---:|']
    for field in module1['preflight']['training_290']:
        lines.append('| '+field+' | '+' | '.join(str(module1['preflight'][g][field]) for g in ('training_290','test_69','recent_23'))+' |')
    lines+=['','Static terms are admitted under an explicit presumed-original-offer assumption. Fetch time is retained separately; the assumed opening-day availability is not presented as independently verified publication. Dated total-demand rows must match the frozen category readings. Missing/conflicting totals remain unavailable.','','## Module 2: same 132 matured IPOs','','The 31st-close checkpoint, 31→61 direction label and next-observed-close execution remain unchanged. Archived delivery uses only earlier sessions; filings use actual broadcast timestamps. Allotment calendars recovered later are explicitly assumed historical calendars.','','| Policy | Correct / 132 | Accuracy | Winner recall | Loser recall | Governance return |','|---|---:|---:|---:|---:|']
    for name,key in [('Existing monitor','original_monitor'),('Always sell','always_sell'),('Prove-it rules','new_prove_it'),('Verified-clock sensitivity','strict_clock_sensitivity')]:
        r=module2[key];lines.append(f"| {name} | {r['correct']} | {r['accuracy_pct']:.2f}% | {r['hold_recall_pct']:.2f}% | {r['sell_recall_pct']:.2f}% | {r['conditional_strategy_net_mean_pct']:.3f}% |")
    costs=module2['consistent_exit_cost_net_mean_pct'];lines+=['',f"Applying the same 0.30% exit cost to both immediate exits and eventual held exits gives: prove-it {costs['new_prove_it']:.3f}%, existing monitor {costs['old_monitor']:.3f}%, always-sell {costs['always_sell']:.3f}%. The original governance return is also retained above for exact comparison.",'','The prove-it policy does not beat always-sell. With strictly verified historical clocks, required evidence gaps make every call defensive; that is the always-sell baseline, not a predictive edge.','','## Capital scenarios and live tracking','','The capital calculator uses actual price × lot size and category thresholds. The proposed 6.5% loss, 0.35% friction, Rs9 fee and seven financing days are assumptions. Reciprocal subscription odds are labeled proxies. A separate full-allotment-loss stress can reject funded HNI exposure even when the retail scenario looks attractive. No orders are placed.','','The backend regime tracker updates every 15 minutes. The next quarterly review is recorded, but no automatic model refit or promotion is enabled after all four components failed the gate. A future expanding-window refit must reserve a new future test cohort.','','## Listing-day timing correction','','A first-15-minute price-floor strategy cannot be evaluated at 09:45 during IPO special pre-open. NSE order entry ends by 09:45, followed by matching and a buffer to 10:00. The guarded implementation uses the first 15 completed continuous minutes, normally 10:00–10:15. Historical intraday bars are unavailable, so no intraday accuracy or fill is claimed. [NSE session rules](https://www.nseindia.com/static/products-services/equity-market-special-pre-open-session).','',f"Targeted checks passed: {report['tests_passed']}. Original ledger/model integrity preserved: {protected['all_preserved']}.",'','These rules and ablations were proposed after the later outcomes had been inspected. They are exploratory comparisons; a gate pass would still need prospective confirmation.']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(out),tests=report['tests_passed'],integrity=protected['all_preserved'],accepted_components=module1['accepted_components'],module2_correct=module2['new_prove_it']['correct'])),flush=True)
    return report


if __name__=='__main__':run()
